# NovaMLX Distributed Inference Architecture Design

**Date**: 2026-05-10
**Status**: Draft — pending user review
**Branch**: `feature/distributed-inference`

---

## 1. Problem Statement

When a model's memory requirement exceeds a single Mac's RAM (e.g., a 96GB model on a 64GB Mac mini), users must either use a smaller model or give up. NovaMLX should transparently distribute inference across multiple Macs on a Thunderbolt network, while keeping single-machine operation completely unaffected.

**Primary use case**: Models that barely don't fit on one machine (the "last 20%" problem). A 2-node cluster (M4 Max 128GB + Mac mini M5 64GB, Thunderbolt 5) is the baseline scenario.

## 2. Hard Constraints

1. **Zero overhead when no cluster configured**: All distributed code is compiled in but dormant. No runtime checks on the inference hot path when running standalone. No feature flags, no compile-time branches.
2. **Transparent to inference API users**: Clients hitting `:6590` see no difference — same OpenAI/Anthropic compatible endpoints, same streaming behavior.
3. **Cluster management on admin port**: All cluster-aware operations live on `:6591` (admin API).

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Coordinator Node                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ API Server│  │ClusterManager│  │  ShardEngine      │  │
│  │ :6590    │  │ (Bonjour +   │  │  (layers 0..N)    │  │
│  │ :6591    │  │  heartbeat)  │  │                   │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│         │               │              │                 │
│         └───────────────┼──────────────┘                 │
│                         │                                │
│              ┌─────────────────────────────┐              │
│              │ MLX Distributed              │              │
│              │ (JACCL/RDMA or Ring/TCP)     │              │
│              └─────────────────────────────┘              │
│                         │                                │
└─────────────────────────┼────────────────────────────────┘
                          │ Thunderbolt 5 (RDMA) / LAN (TCP)
┌─────────────────────────┼────────────────────────────────┐
│               ┌───────────────────┐                      │
│               │  WorkerService    │                      │
│               │  (Bonjour +       │                      │
│               │   heartbeat)      │                      │
│               └───────────────────┘                      │
│                         │                                │
│               ┌───────────────────┐                      │
│               │  ShardEngine      │                      │
│               │  (layers N+1..M)  │                      │
│               └───────────────────┘                      │
│                         │                                │
│              ┌─────────────────────────────┐             │
│              │ MLX Distributed              │             │
│              │ (JACCL/RDMA or Ring/TCP)     │             │
│              └─────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

## 4. Module Design

New module: `Sources/NovaMLXDistributed/`

### 4.1 MLXDistributed (Swift Wrappers)

Wraps the MLX C distributed API (`mlx_distributed_send`, `mlx_distributed_recv`, `mlx_distributed_all_gather`, `mlx_distributed_group`) in Swift. ~200-300 lines following the existing `MLXFast.swift` bridging pattern.

**Backend auto-selection**: On initialization, detect the best available backend:
- `mlx_distributed_is_available("jaccl")` — JACCL (RDMA over Thunderbolt, macOS 26.2+, requires IBV library). Lowest latency, highest bandwidth. Compiled in when macOS SDK >= 26.2 (our SDK is 26.4).
- Falls back to `"ring"` — TCP ring backend (plain POSIX sockets). Works everywhere, including LAN.
- This detection runs **once** at cluster init time, not per-token. Standalone mode never triggers it.

```swift
// Example API surface
public class DistributedGroup {
    let ctx: mlx_distributed_group
    public var rank: Int { mlx_distributed_group_rank(ctx) }
    public var size: Int { mlx_distributed_group_size(ctx) }
}

public func distributedSend(_ array: MLXArray, to dst: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray
public func distributedRecv(shape: [Int], dtype: Dtype, from src: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray
```

Bootstrapping: `MLX_HOSTFILE` env var (JSON with ip:port pairs) + `MLX_RANK` env var for ring. JACCL uses `MLX_IBV_DEVICES` + `MLX_JACCL_COORDINATOR` + `MLX_RANK`.

### 4.2 ClusterManager (Coordinator-side)

Responsibilities:
- **Bonjour discovery**: Advertise `_novamlx._tcp` service, listen for worker registrations
- **Role negotiation**: Config-based (default=worker). First-boot-wins for coordinator conflict
- **Heartbeat**: Periodic health checks on all registered workers
- **Shard plan**: Compute layer distribution based on per-layer memory analysis (via ModelAnalyzer), user-overridable via admin API
- **Fault detection**: Trigger recovery when worker heartbeat lost

**Bonjour on Thunderbolt**: Uses `NWBrowser` + `NWParameters` with explicit interface selection. Prioritizes Thunderbolt bridge interfaces (`bridge*`, Thunderbolt-specific `en*`) for discovery. Falls back to all interfaces. A `GET /admin/api/cluster/discovery-debug` endpoint exposes discovered interfaces, Bonjour state, and connectivity status for troubleshooting.

```swift
struct ClusterConfig {
    var role: ClusterRole  // .coordinator or .worker
    var coordinatorHost: String
    var coordinatorPort: Int
    var strategy: ClusterStrategy  // .minNodes or .spread
}

struct ShardPlan {
    var assignments: [ShardAssignment]  // nodeId -> layer range
    var totalLayers: Int
    var strategy: ClusterStrategy
}

struct ShardAssignment {
    var nodeId: String
    var startLayer: Int
    var endLayer: Int  // exclusive
    var memoryEstimate: UInt64
}
```

### 4.3 WorkerService (Worker-side)

Responsibilities:
- **Registration**: Find coordinator via Bonjour (explicit Thunderbolt interface selection), register with device specs (RAM, compute capability)
- **Heartbeat**: Periodic heartbeat to coordinator
- **Shard execution**: Receive shard plan, load assigned layers, execute forward pass
- **Reshard**: Handle coordinator resharding requests (load new layer range)

### 4.4 ModelAnalyzer

Analyzes model structure for precise shard planning. Avoids the "memory ratio only" pitfall where embedding/lm_head layers are 2-3× larger than middle layers, or MoE expert layers vary wildly.

```swift
struct LayerProfile {
    var layerIndex: Int
    var parameterCount: UInt64
    var estimatedMemoryBytes: UInt64
    var layerType: LayerType  // .embedding, .transformer, .output, .moe
}

class ModelAnalyzer {
    func analyze(modelPath: String) throws -> [LayerProfile]
    func computeShardPlan(profiles: [LayerProfile], nodes: [NodeSpec], strategy: ClusterStrategy) -> ShardPlan
}
```

- Parses safetensors headers to get per-layer parameter counts (no need to load full weights)
- Combines with node RAM and compute capability for weighted allocation
- Preserves contiguous layer ranges (pipeline-friendly)
- Includes `estimatedKVPerToken` for load-time validation against `max_tokens` setting

### 4.5 ShardEngine (Both sides)

The core inference engine wrapper. Each node runs one ShardEngine instance.

```swift
protocol ComputePolicy {
    func bindWeights(for layers: Range<Int>, model: MLXModel) throws
    func compute(input: MLXArray, layers: Range<Int>, cache: inout [KVCache]) throws -> MLXArray
    func releaseWeights(for layers: Range<Int>)
}

class ShardEngine {
    let group: DistributedGroup
    let assignment: ShardAssignment
    let policy: ComputePolicy  // FitInMemoryPolicy initially, OffloadPolicy future
    var kvCache: [KVCache]

    func prefill(tokens: MLXArray) async throws -> MLXArray
    func decode(token: MLXArray) async throws -> MLXArray
}
```

**ComputePolicy** is an explicit extension point:
- `FitInMemoryPolicy`: All assigned layers' weights loaded in memory (initial implementation)
- `OffloadPolicy`: Windowed weight residency with disk streaming (future, inspired by DNet)

### 4.6 WeightDistributor

Handles model weight distribution to worker nodes. The model loading flow is:

1. Worker receives shard plan → checks if model file exists locally
2. **Path A (local access)**: Model file present → mmap assigned layers directly
3. **Path B (auto-download)**: Model file missing → coordinator streams model via parallel HTTP chunked transfer to worker's cache directory. During download, model status is `"syncing"` and inference requests are rejected. After download, proceeds as Path A.

Path B is transparent but not invisible — admin API exposes download progress. Download is one-time per model per worker; subsequent loads always use Path A. For shared-storage setups (Thunderbolt-mounted volumes, NFS), all workers naturally hit Path A.

## 5. Inference Data Flow

### 5.1 Cluster Setup

```
1. User configures cluster role in config.json (or defaults to worker)
2. Coordinator starts → advertises via Bonjour (_novamlx._tcp)
3. Workers discover coordinator → register with device specs
4. User loads model via admin API
5. ClusterManager computes ShardPlan via ModelAnalyzer (per-layer memory analysis)
6. Each node checks for model file locally; if missing, downloads from coordinator (status: "syncing")
7. Each node loads assigned layers into its ShardEngine
8. MLX Distributed initialized — auto-selects JACCL (RDMA) if available, falls back to Ring (TCP)
```

### 5.2 Prefill (Sequential)

```
Client → POST /v1/chat/completions
         │
         v
    [Coordinator: API Server]
         │ tokenize prompt → MLXArray
         │
         v
    [Shard 0 (Coordinator)]
         │ embed tokens
         │ forward through layers 0..N
         │ distributed.send(activation, to: rank+1)
         v
    [Shard 1 (Worker)]
         │ distributed.recv(from: rank-1) → activation
         │ forward through layers N+1..M
         │ [last shard] → normalize → lm_head → logits
         │ sample token → return token_id (4 bytes)
         v
    [Coordinator: API Server]
         │ receive token_id from last shard
         │ add to sequence, continue decode loop
```

**Key details**:
- KV cache is **local per shard** — only caches its own layers. No cross-node KV transfer.
- Prefix cache (SSD) works independently on each shard for its own layer range.
- Streaming to client works identically — coordinator emits SSE chunks as tokens arrive.
- Token return path: last shard sends token_id back to coordinator via MLX Ring `send` to rank 0. For 2-node clusters this is a single hop. For larger clusters the token traverses intermediate ranks (acceptable overhead for a 4-byte payload).

### 5.3 Decode (Per-token)

Same flow as prefill, but input is a single token `[token_id]` instead of the full prompt. Each token traverses all shards sequentially.

### 5.4 Sampling Location

**Last shard samples directly**, returning only the token ID (4 bytes) to the coordinator. This avoids transmitting the full logits tensor (vocab_size × float, typically hundreds of KB) across the network.

## 6. Cluster Sizing Strategy

Two strategies, configurable via admin API:

- **`min_nodes`** (default): Use the minimum number of nodes needed to fit the model. Remaining nodes form a spare pool for fault recovery. Minimizes per-token latency (fewer network hops).
- **`spread`**: Distribute layers across all available nodes. Maximizes spare capacity per node, useful for running multiple models simultaneously or when preparing for frequent node failures.

## 7. Fault Recovery

Three levels, all implemented from day 1:

### L1: Transient Disconnect (30s window)
- Worker heartbeat lost → coordinator enters 30-second grace period
- Pending inference requests are paused (not failed)
- If worker reconnects within 30s → resume normally
- If not → escalate to L2

### L2: Spare Node Swap
- Coordinator selects a spare node from the pool
- New node loads the failed node's layer range
- Shard plan updated, MLX Ring reconfigured
- Pending requests replayed on new shard
- If no spare available → escalate to L3

### L3a: Auto-Reshard (automatic)
- Coordinator redistributes layers across remaining active nodes (if they have enough combined memory)
- All nodes reload their new layer ranges, MLX Ring reconfigured
- Pending requests replayed on new shard layout
- If combined memory insufficient → escalate to L3b
- Shares most code with L2 (new node load + shard plan update + ring reconfigure) — incremental complexity is modest

### L3b: Hard Fail or Manual Reshard
- **Hard fail (automatic)**: Unload model, notify admin API user, log error
- **Manual reshard**: Admin can trigger `POST /admin/api/models/{id}/cluster/reshard` to attempt resharding with new topology (e.g., after adding a replacement node)

## 8. API Surface

### Inference Port (:6590) — Unchanged
All existing endpoints work identically. Clients see no difference.

### Admin Port (:6591) — Cluster-Aware Additions

Cluster-level endpoints:
```
GET  /admin/api/cluster/status              — cluster health, node list
POST /admin/api/cluster/config              — set strategy (min_nodes/spread)
POST /admin/api/cluster/nodes/{id}/drain    — gracefully remove a node
GET  /admin/api/cluster/discovery-debug     — Bonjour state, discovered interfaces, connectivity
```

Model-level cluster endpoints (scoped per model):
```
GET  /admin/api/models/{id}/cluster/shard-plan       — current shard plan for this model
POST /admin/api/models/{id}/cluster/shard-plan       — override shard plan (user-specified layer split)
POST /admin/api/models/{id}/cluster/reshard          — trigger manual reshard
GET  /admin/api/models/{id}/cluster/sync-status      — model download/sync progress per worker
```

## 9. Single-Machine Guarantee

All distributed code is compiled into every binary. When `ClusterConfig.role` is not set (or no coordinator is discovered):

- `ClusterManager` is never instantiated
- `WorkerService` never starts
- `ShardEngine` is never used — inference follows the existing `MLXEngine` path
- `MLXDistributed` wrappers are never called
- **Zero runtime overhead**: No additional function calls, no conditionals on the inference hot path

The trigger is purely configuration-driven: if no cluster is configured, the distributed code paths are dead code from the runtime's perspective.

**Isolation mechanism**: `MLXEngine` (the existing inference engine) has zero knowledge of `NovaMLXDistributed`. The routing decision happens once at `APIServer` startup — if cluster config is present, request handling delegates to `ShardEngine`; otherwise, the existing `MLXEngine` path runs unchanged. This is a one-time routing check in the API layer, not per-token. Module boundaries are enforced via protocols (not concrete types), preventing import-chain contamination of the hot path.

**Validation**: Use Instruments Time Profiler to compare standalone vs cluster prefill/decode latency. Any measurable difference is a bug.

## 10. Future Work (Explicitly Deferred)

These features are **not** in the initial release but are designed as extension points:

1. **EXO-style overlapped pipeline wavefront prefill**: Staggered chunk processing across ranks for faster prefill. Most impactful with 3+ nodes. See memory: `project-distributed-future-exo-prefill.md`.

2. **DNet-style disk streaming**: Windowed weight residency with LRU eviction and madvise prefetch. Enables running models exceeding cluster RAM. `ComputePolicy` protocol is the extension point. See memory: `project-distributed-future-dnet-disk-streaming.md`.

3. **Tensor Parallelism**: Column-shard attention projections, row-shard output projections, all-reduce after each layer. Requires MLX Distributed all-reduce support. Significant additional complexity; not justified for the 2-node baseline.

4. **Batched inference across shards**: Multiple concurrent requests flowing through the pipeline simultaneously.

## 11. Comparison with EXO and DNet

| Dimension | EXO | DNet | NovaMLX |
|---|---|---|---|
| **Language** | Python | Python | Swift |
| **Communication** | MLX Distributed (TCP/RDMA) | gRPC + custom codec | MLX Distributed (JACCL RDMA auto / Ring TCP fallback) |
| **Activation transfer** | MLX array direct (zero-copy) | Serialize→protobuf→deserialize | MLX array direct (zero-copy) |
| **Parallelism** | Pipeline + Tensor | Pipeline ring | Pipeline (initial) |
| **Prefill** | Overlapped wavefront | Sequential (no optimization) | Sequential (initial) |
| **Sampling** | Rank 0 (all_gather logits) | Last shard (direct) | Last shard (direct) |
| **KV cache** | Per-shard, prefix cache with LRU | Per-nonce, TTL-based | Per-shard, SSD prefix cache |
| **Disk streaming** | No | Yes (windowed residency) | No (ComputePolicy extension point) |
| **Sharding algorithm** | Proportional + cycle filtering | MILP solver (HALDA) | Per-layer memory analysis (ModelAnalyzer) |
| **Discovery** | Gossipsub (libp2p) | Manual config | Bonjour/mDNS (Thunderbolt-aware) |
| **Fault recovery** | Runner restart | Unknown | 3-level (L1/L2/L3a+L3b) |
