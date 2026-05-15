# Distributed Inference Performance v2: 2-Stage Async Pipeline

## Problem

Distributed inference over Thunderbolt (M4 Max + M4 Mac Mini) runs at **7.9 tok/s** with Qwen3.6-27B-4bit — slower than single-node M4 Max (~12-15 tok/s). The previous optimization round added bfloat16 transport, shard rebalancing, and a speculative async pipeline, but the speculation achieved only 20-28% draft accuracy — a net negative.

Root cause: the decode loop is fundamentally serialized. Coord computes activation → TCP → Worker computes → TCP → Coord runs head. Even with bfloat16 and head offloading, the serial dependency caps throughput at 1/(31ms + 29ms + 6ms) ≈ 14 tok/s.

## Target

Beat single-node speed: **18+ tok/s** (from 7.9 tok/s).

## Design: 2-Stage Async Pipeline Without Speculation

Two interlocking optimizations.

### 1. Compute-Ratio-Aware Layer Rebalancing

**Current**: 53/13 layer split (proportional to memory). Worker (M4 base, 10 GPU cores) is 4.3x slower per layer than coordinator (M4 Max, 40 GPU cores), so 13 layers takes 33ms vs 31ms for coordinator's 53 layers.

**Proposed**: Factor GPU compute ratio into `ShardPlan`. The coordinator already knows each worker's CPU model (from heartbeat). Map known Apple Silicon variants to relative compute ratios:
- M4 Max (40 GPU cores): 1.0
- M4 Pro (20 GPU cores): 0.5
- M4 base (10 GPU cores): 0.23
- M3/M2/etc: 0.2 (conservative default)

The `ShardPlan` weights layer assignment by compute ratio so both sides have similar wall-clock GPU time. For Qwen3.6-27B-4bit (66 layers):
- Coordinator: 55 layers × 0.60ms = 33ms
- Worker (M4 base): 11 layers × 2.6ms = 28.6ms

This is close enough for the overlap pipeline. The worker still meets `minLayersPerShard: 8`.

**Files**: `Sources/NovaMLXDistributed/DistributedTypes.swift` (ShardPlan), `Sources/NovaMLXDistributed/ClusterModelManager.swift` (compute ratio lookup)

### 2. 2-Stage Async Pipeline Decode Loop

Replace the speculative decode loop with a clean 2-stage overlap pipeline. No speculation, no draft prediction — just ensure coord and worker are always busy.

**Current serial flow** (per token):
```
coord computeLayersOnly → TCP → worker compute → TCP → computeHeadOnly
[    31ms            ] [3ms] [    29ms      ] [3ms] [    2ms       ]
Total: ~68ms → ~14 tok/s ceiling
```

**New 2-stage overlap flow** (per token, after warmup):
```
Iteration N:
  await pendingWorker        // worker result from prev fire (~0ms if overlapped)
  computeHeadOnly(result)    // ~2ms
  computeLayersOnly(token)   // ~33ms (coord busy)
  fire worker(activation)    // worker starts computing for next iter

Meanwhile, worker has been computing during the coord's computeLayersOnly.
When loop returns to `await`, worker result is already ready.
```

Throughput = 1/max(coord_head+coord_layers, worker_compute+TCP) = 1/max(35ms, 32ms) = **~28 tok/s** theoretical, **18-22 tok/s** realistic.

**Decode loop pseudocode** (generate method):
```
// Prefill → activation
// computeHeadOnly(activation) → firstToken
activation = computeLayersOnly(firstToken)
pendingWorker = Task { worker.compute(activation) }

while more tokens:
    workerHidden = await pendingWorker.value
    token = computeHeadOnly(workerHidden)
    if eos: break
    append token
    activation = computeLayersOnly(token)
    pendingWorker = Task { worker.compute(activation) }
```

The overlap is implicit: `pendingWorker` from the bottom of iteration N is awaited at the top of iteration N+1. During the coord's 33ms `computeLayersOnly`, the worker runs its ~29ms compute in parallel.

**Stream method**: Same pattern, but `append token` becomes `yieldToken(token)`.

**Removing speculation complexity**: Delete from decode loop:
- `draftTokens()` calls
- `rollbackCache()` calls
- `recordPrediction` / adaptive disable logic
- `speculationEnabled` / `adaptiveWindow` / `disableThreshold` variables
- Speculative verification inner loop (K>1 draft tokens)

The `draftTokens()` and `rollbackCache()` methods stay on `SlicedForwardPolicy` for future use — just not called from the distributed decode loop.

### 3. Prefill: No Changes

The pipelined prefill (coord chunk N+1 overlaps worker chunk N) already works correctly with `computeLayersOnly()`. Short prompts (<256 tokens) use the simple pipeline with `computeLayersOnly()` on both sides. No changes needed.

### 4. Performance Monitoring

Add per-token overlap measurement. Every 20 tokens, log:
```
[Pipeline] token 40: 34ms total, coord=33ms, worker_wait=1ms, overlap=97%
```

This verifies the pipeline is actually overlapping. The `worker_wait` time should be near 0ms when overlap is working. Add `overlapMs` and `overlapPct` to `DistributedInferenceStats`.

## Files to Modify

| File | Change |
|------|--------|
| `NovaMLXDistributed/DistributedTypes.swift` | Add compute ratio to `NodeSpec`, weight in `ShardPlan` |
| `NovaMLXDistributed/ClusterModelManager.swift` | Look up compute ratio from CPU model |
| `NovaMLXDistributed/DistributedInferenceRunner.swift` | Rewrite decode loops (generate + stream) with 2-stage overlap, remove speculation |
| `NovaMLXDistributed/ClusterManager.swift` | Report compute ratio from worker heartbeat |

## Verification

1. Single-node baseline: measure Qwen3.6-27B-4bit tok/s on M4 Max alone
2. Distributed: activate model, run 200-token non-streaming test
3. Target: distributed > single-node (18+ tok/s)
4. Check logs for overlap percentage (target: >90%)
5. Streaming comparison: verify stream() matches generate() tok/s
6. Check layer split in logs: should be 55/11, not 53/13
