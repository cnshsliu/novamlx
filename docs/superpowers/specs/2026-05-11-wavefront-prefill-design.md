# Overlapped Pipeline Wavefront Prefill Design

**Date**: 2026-05-11
**Status**: Draft — pending user review
**Branch**: `feature/distributed-inference`
**Depends on**: Distributed inference skeleton (Tasks 1–14)

---

## 1. Problem Statement

Sequential pipeline prefill has a bubble: each rank waits for the previous rank to finish the entire prompt before starting its own computation. For a 2-node cluster with a 32K-token prompt, rank 1 is idle while rank 0 processes all 32K tokens through its layers, then rank 1 processes all 32K tokens through its layers. Total prefill time = `rank0_time + rank1_time`.

Overlapped wavefront prefill splits the prompt into chunks and staggers them across ranks so computation overlaps with communication. Rank 0 starts chunk 0 immediately, sends the activation to rank 1, then moves to chunk 1 while rank 1 processes chunk 0. Total prefill time ≈ `n_chunks × chunk_time + (worldSize - 1) × chunk_time` instead of `n_chunks × chunk_time × worldSize`.

**Expected improvement**: 25–45% TTFT reduction on long prompts (8K–32K tokens) with 3–4 node Thunderbolt 5 clusters.

## 2. Hard Constraints

1. **Sequential fallback preserved**: Prompts shorter than `minWavefrontTokens` use the existing sequential `prefill()` path. No overhead for short prompts.
2. **Decode unchanged**: Wavefront only applies to prefill. Decode remains sequential (one token per step).
3. **ComputePolicy unchanged**: `compute(input:cache:)` signature stays the same. Pipeline layer wrapping happens during model construction in `bindWeights()`.
4. **Local KV cache**: Every rank populates only its own layers' KV. No cross-node KV transfer.
5. **Zero standalone overhead**: All wavefront code is dormant when no cluster is configured.

## 3. Algorithm

### 3.1 Chunking

```
chunkSize = max(baseStepSize / worldSize, minChunkSize)
```

- `baseStepSize`: configurable, default 4096 tokens
- `minChunkSize`: configurable floor, default 512 tokens
- Prevents pathological tiny chunks on large clusters

The prompt (minus the last token) is split into `nReal` chunks of `chunkSize` tokens. The last chunk may be smaller.

```
nReal = ceil((promptLen - 1) / chunkSize)
```

### 3.2 Staggered Wavefront

Each rank R processes `nReal + worldSize - 1` iterations:

- **Leading dummies**: `R` iterations — pure no-ops, no forward pass. Pad the loop so rank R starts its real work at iteration R.
- **Real chunks**: `nReal` iterations — forward pass through assigned layers with real tokens.
- **Trailing dummies**: `worldSize - 1 - R` iterations — pure no-ops. Drain the pipeline.

```
Time →       iter0           iter1           iter2           iter3
Rank 0:    [real chunk0]   [real chunk1]   [real chunk2]   [dummy]
Rank 1:    [dummy]         [real chunk0]   [real chunk1]   [real chunk2]
```

Synchronization is implicit: when rank 1 processes its first real chunk, `PipelineFirstLayer.recv` blocks until rank 0's `PipelineLastLayer` sends from iteration 0. No explicit barriers.

### 3.3 Send Batching

During prefill, `PipelineLastLayer` queues sends in a buffer instead of transmitting immediately. After each real chunk's full forward pass, `flushPrefillSends()` fires all queued sends via `MLX.asyncEval`. This batches inter-node communication so compute and transfer overlap.

During decode, sends are immediate (no batching — single token, no benefit from batching).

### 3.4 Final Two Passes

After the wavefront loop, two single-token forward passes process the last prompt token:

1. **Pass 1**: Run `prompt[-1:]` through the full pipeline (completes prompt processing)
2. **Pass 2**: Run `prompt[-1:]` again (generates first response token, matching `stream_generate` behavior)

Both passes include `flushPrefillSends()` and a final `MLX.eval()` on all cache state.

### 3.5 Activation Threshold

If `promptLen < minWavefrontTokens`, fall back to existing sequential `prefill()`. Configurable `minWavefrontTokens`, default 4096.

## 4. New Components

### 4.1 PrefillConfig

Added to `DistributedTypes.swift`.

```swift
struct PrefillConfig: Codable, Sendable {
    var baseStepSize: Int = 4096
    var minChunkSize: Int = 512
    var minWavefrontTokens: Int = 4096
}
```

Embedded in `ClusterConfig`. Configurable via admin API `POST /admin/api/cluster/config`.

### 4.2 PipelineLayer

New file: `Sources/NovaMLXDistributed/PipelineLayer.swift`

Two layer wrappers that replace the first and last layers of each shard during model construction:

**PipelineFirstLayer** (wraps first layer of each shard):
- Rank 0: passes through (no recv needed — generates embeddings from token IDs)
- Other ranks: `recv` activation from `rank - 1`, then forward through the wrapped layer
- `mx.eval(activation)` before recv to materialize the tensor and split the compute graph

**PipelineLastLayer** (wraps last layer of each shard):
- Last rank: passes through (returns activation for sampling)
- Other ranks during prefill (`queueSends = true`): queue send to `rank + 1` in `prefillSendQueue`
- Other ranks during decode (`queueSends = false`): immediate `send` to `rank + 1`
- `mx.eval(output)` before sending to split the compute graph

**Send queue**:
```swift
struct PendingSend {
    let output: MLXArray
    let destination: Int
    let group: DistributedGroup
}
```

Thread-safe via `NSLock` (hot path — no DispatchQueue/actor overhead).

**Flush/clear**:
- `flushPrefillSends()`: drain queue, fire all sends via `MLX.asyncEval`, clear queue
- `clearPrefillSends()`: discard pending sends without transmitting (error/cancellation path)

### 4.3 ShardEngine.wavefrontPrefill

New method on `ShardEngine` alongside existing `prefill()`.

```swift
func wavefrontPrefill(
    tokens: MLXArray,
    config: PrefillConfig
) async throws -> MLXArray
```

Logic:
1. Compute `chunkSize`, `nReal`, `nLeading`, `nTrailing`
2. If `promptLen < config.minWavefrontTokens`, delegate to `prefill(tokens:)`
3. Set `queueSends = true` on PipelineLastLayer instances
4. Run loop: `nLeading` no-ops → `nReal` chunks (forward + flush) → `nTrailing` no-ops
5. Two final single-token passes with flush
6. `MLX.eval()` on all cache state
7. Set `queueSends = false`

### 4.4 WavefrontStats

Observability struct returned from `wavefrontPrefill` (logged + exposed via admin API).

```swift
struct WavefrontStats {
    let chunkSize: Int
    let nRealChunks: Int
    let nLeadingDummies: Int
    let nTrailingDummies: Int
    let promptTokens: Int
    let prefillCommBytes: UInt64
}
```

## 5. Integration with Existing Code

### 5.1 ComputePolicy

`compute(input:cache:)` signature unchanged. Pipeline layer wrapping happens in `FitInMemoryPolicy.bindWeights()` — when constructing the model, wrap the first layer with `PipelineFirstLayer` and the last layer with `PipelineLastLayer`. This is transparent to the compute call.

### 5.2 ShardEngine Routing

`ShardEngine` exposes a unified `prefill` entry point that selects sequential vs wavefront:

```swift
public func prefill(tokens: MLXArray, config: PrefillConfig? = nil) async throws -> MLXArray {
    let cfg = config ?? PrefillConfig()
    let tokenCount = tokens.shape.reduce(1, *)
    if group.size > 1 && tokenCount >= cfg.minWavefrontTokens {
        return try await wavefrontPrefill(tokens: tokens, config: cfg)
    }
    return try await sequentialPrefill(tokens: tokens)
}
```

The existing `prefill()` method is renamed to `sequentialPrefill()`. No callers break — the public API is the same method with an optional config parameter.

### 5.3 Decode Unchanged

Decode remains sequential. No `PipelineFirstLayer`/`PipelineLastLayer` involvement beyond normal send/recv.

### 5.4 Fault Recovery

If a rank crashes mid-wavefront:
1. `recv` in `PipelineFirstLayer` fails with a distributed error
2. Error propagates to `wavefrontPrefill()`
3. `clearPrefillSends()` discards any pending sends
4. Error rethrown to caller (coordinator's inference handler)
5. Coordinator's heartbeat timeout triggers L1/L2/L3 fault recovery as usual

No special wavefront-specific recovery logic needed — the existing fault recovery handles it.

## 6. Files Changed

| File | Change |
|------|--------|
| `Sources/NovaMLXDistributed/PipelineLayer.swift` | **New**: PipelineFirstLayer, PipelineLastLayer, PendingSend, flush/clear |
| `Sources/NovaMLXDistributed/ShardEngine.swift` | Add `wavefrontPrefill()`, rename `prefill()` → `sequentialPrefill()`, unified `prefill(config:)` entry |
| `Sources/NovaMLXDistributed/DistributedTypes.swift` | Add `PrefillConfig`, `WavefrontStats` |
| `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift` | Add wavefront stats to cluster status |
| `Tests/NovaMLXDistributedTests/PipelineLayerTests.swift` | **New**: send queue, flush, clear, layer wrapping |
| `Tests/NovaMLXDistributedTests/ShardEngineTests.swift` | Add wavefront-specific tests |

## 7. Single-Machine Guarantee

All wavefront code is dormant when no cluster is configured:
- `PrefillConfig` is only created when `ClusterConfig` is present
- `PipelineLayer` wrappers are only installed during `bindWeights()` when cluster is active
- `wavefrontPrefill()` is never called when `group.size <= 1`
- The unified `prefill(config:)` entry falls through to `sequentialPrefill()` when `group.size <= 1`

No additional imports, no static initializers, no runtime checks on the standalone path.

## 8. Comparison with EXO

| Dimension | EXO | NovaMLX |
|---|---|---|
| Chunk size formula | `4096 / min(4, worldSize)` | `max(baseStepSize / worldSize, minChunkSize)` — configurable floor |
| Activation threshold | Hardcoded 4096 | Configurable `minWavefrontTokens` |
| Dummy semantics | Pure no-ops (callback only) | Pure no-ops |
| Send batching | `_pending_prefill_sends` list + `flush` | `prefillSendQueue` (NSLock-protected) + `flushPrefillSends()` |
| Async eval | `mx.async_eval(sent)` | `MLX.asyncEval()` |
| Final passes | 2 single-token passes | 2 single-token passes (same) |
| Layer wrapping | `PipelineFirstLayer` / `PipelineLastLayer` replacing layers in model | Same pattern, Swift implementation |
| Stats/observability | `distributed_prompt_progress_callback` | `WavefrontStats` struct + admin API |
