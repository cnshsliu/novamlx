# NovaMLX Distributed Inference Plan

## Vision

NovaMLX runs standalone on a single Mac **or** as a cluster of Macs connected via Thunderbolt/Ethernet. When multiple NovaMLX nodes detect each other on the network, they automatically form a cluster and pool their unified memory + compute for LLM inference.

**Goal:** Run models that don't fit on one machine. Speed up models that do.

---

## Architecture

### Cluster Topology

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  NovaMLX #1  │◄───────►│  NovaMLX #2  │◄───────►│  NovaMLX #3  │
│  (Coordinator│  TB/ETH  │   (Worker)   │  TB/ETH  │   (Worker)   │
│   + Worker)  │         │              │         │              │
│  48GB unified│         │  48GB unified│         │  48GB unified│
└──────────────┘         └──────────────┘         └──────────────┘
        │
        ▼
  OpenAI-compatible API
  (external clients)
```

- **Coordinator**: The node that receives the API request. Runs tokenizer, orchestrates pipeline, runs sampling.
- **Worker**: Loads its shard of model weights, processes assigned layers, forwards hidden states.
- Any node can be coordinator. First node to boot becomes coordinator (or elected via Raft-style consensus).

### Pipeline Parallelism (Today — Thunderbolt 4 / Ethernet)

Split model layers across N nodes. Hidden state flows through nodes like an assembly line.

```
Token decode flow:

Client Request
      │
      ▼
┌─────────────┐   hidden_state   ┌─────────────┐   hidden_state   ┌─────────────┐
│   Node 0    │ ──── (TCP) ────► │   Node 1    │ ──── (TCP) ────► │   Node 2    │
│ Layers 0-31 │    ~16KB/tok     │ Layers 32-63│    ~16KB/tok     │ Layers 64-95│
│ + Tokenizer │                  │             │                  │ + Sampling  │
│ + KV cache  │                  │ + KV cache  │                  │ + KV cache  │
└─────────────┘                  └─────────────┘                  └─────────────┘
      │                                                                │
      └──────────── sampled token back to coordinator ─────────────────┘
```

- Each node holds 1/N of model weights + KV cache for its own layers
- Hidden state transfer: ~16 KB/token (negligible, ~14μs over TB4)
- Works over **any** IP network: Thunderbolt 4, Thunderbolt 5, 10GbE, even WiFi

### Tensor Parallelism (Future — Thunderbolt 5 + macOS 26.2+ RDMA)

Shard each layer's matrix ops across all nodes. Requires JACCL (Apple's RDMA backend).

```
Token decode flow (tensor parallel):

┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Node 0    │◄──all───►│   Node 1    │◄──all───►│   Node 2    │
│  1/3 of     │  reduce  │  1/3 of     │  reduce  │  1/3 of     │
│  each layer │ (TB5 RDMA)│  each layer │ (TB5 RDMA)│  each layer │
└─────────────┘         └─────────────┘         └─────────────┘

All nodes run ALL layers, but each node only computes 1/3 of each matmul.
```

- Requires: Thunderbolt 5 hardware, macOS 26.2+, RDMA enabled in Recovery Mode
- ~80 Gb/s bandwidth, 5-14μs latency via JACCL
- Lower single-token latency than pipeline parallelism
- **We design the software now, activate when hardware arrives.**

---

## Module Design

### New Swift Modules

```
NovaMLXDistributed/           ← New module
├── ClusterDiscovery.swift    ← Bonjour/mDNS node auto-discovery
├── ClusterNode.swift         ← Node identity, capabilities, health
├── ClusterCoordinator.swift  ← Orchestrates pipeline, assigns shards
├── PipelineWorker.swift      ← Loads shard, runs layers, sends/receives hidden states
├── ModelSharder.swift        ← Splits model weights into N contiguous layer chunks
├── ShardTransport.swift      ← TCP/RDMA abstraction for hidden state transfer
├── DistributedKVCache.swift  ← KV cache management across pipeline stages
└── ClusterConfig.swift       ← Cluster settings, node priorities, failover
```

### Dependency on Existing Modules

```
NovaMLXDistributed
    ├── NovaMLXCore        (types, protocols, InferenceRequest/Result)
    ├── NovaMLXEngine      (ModelContainer, tokenization, sampling)
    ├── NovaMLXUtils       (logging, metrics)
    └── MLX / MLXLM        (tensor operations, model loading)
```

---

## Component Details

### 1. ClusterDiscovery (Bonjour/mDNS)

```swift
// Auto-detect other NovaMLX nodes on the local network
class ClusterDiscovery {
    // Broadcast: "I am NovaMLX node, here's my IP, port, memory, compute capability"
    // Listen: Discover other nodes advertising the same service type
    // Elect: First node becomes coordinator (or highest-memory node wins)

    func startBroadcast(identity: NodeIdentity)
    func discoverPeers() -> AsyncStream<[ClusterNode]>
    func electCoordinator(nodes: [ClusterNode]) -> ClusterNode
}

struct NodeIdentity {
    let id: UUID
    let hostname: String
    let ipAddress: String
    let port: UInt16
    let totalMemoryGB: Int        // unified memory size
    let availableMemoryGB: Int    // currently free
    let computeTier: ComputeTier  // M4 Pro, M4 Max, M3 Ultra, etc.
    let thunderboltVersion: Int   // 4 or 5
}
```

### 2. ModelSharder

```swift
// Split model into N contiguous layer chunks for pipeline parallelism
class ModelSharder {
    // At model download time: download full model, split weights into shards
    // Each shard contains: model config + contiguous layers + partial tokenizer (coordinator only)

    func planShards(modelConfig: ModelConfig, nodeCount: Int) -> [ShardPlan]
    func splitWeights(modelPath: URL, plans: [ShardPlan]) -> [URL]  // saves shard files
    func loadShard(path: URL) -> ModelShard  // load only this node's layers
}

struct ShardPlan {
    let nodeIndex: Int
    let startLayer: Int
    let endLayer: Int
    let estimatedMemoryGB: Double
    let outputFilePath: String
}
```

### 3. PipelineWorker

```swift
// Each node runs one PipelineWorker
// Handles: receive hidden state → run layers → send hidden state
actor PipelineWorker {
    let shard: ModelShard
    let transport: ShardTransport
    let kvCache: DistributedKVCache

    // Pipeline parallel decode loop
    func runPipelineStage(
        input: AsyncStream<HiddenState>,    // from previous node (or tokenizer)
        output: AsyncStream<HiddenState>     // to next node (or sampler)
    ) async throws

    // Prefill: process prompt tokens through this node's layers
    func prefillStage(tokens: [Int]) async throws -> HiddenState
}
```

### 4. ShardTransport

```swift
// Abstraction over transport layer — TCP today, RDMA tomorrow
protocol ShardTransport {
    func send(hiddenState: MLXArray, to node: ClusterNode) async throws
    func receive(from node: ClusterNode) async throws -> MLXArray
    var latency: Duration { get }     // estimated one-way latency
    var bandwidth: Double { get }     // bytes/sec
}

// Phase 1: TCP/IP (works on any network)
class TCPTransport: ShardTransport { ... }

// Phase 2: Thunderbolt 5 RDMA (macOS 26.2+, JACCL)
class RDMATransport: ShardTransport { ... }  // Future
```

### 5. ClusterCoordinator

```swift
// Runs on the coordinator node. Ties everything together.
actor ClusterCoordinator {
    let discovery: ClusterDiscovery
    let sharder: ModelSharder
    let nodes: [ClusterNode]
    let localWorker: PipelineWorker

    // Main inference entry point (called by NovaMLXAPI)
    func generate(_ request: InferenceRequest) async throws -> InferenceResult
    func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error>

    // Cluster lifecycle
    func formCluster(discoveredNodes: [ClusterNode]) async throws
    func dissolveCluster() async
    func handleNodeFailure(node: ClusterNode) async  // failover / re-shard
}
```

### 6. Integration with Existing NovaMLX

```swift
// In InferenceService — route to cluster or standalone
extension InferenceService {
    func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        if clusterCoordinator.isClusterActive {
            return try await clusterCoordinator.generate(request)  // distributed
        } else if CloudBackend.isCloudModel(resolvedId) {
            return try await CloudBackend.shared.proxy(request)    // cloud
        } else {
            return try await localGenerate(request)                // standalone
        }
    }
}
```

---

## Execution Modes

### Standalone Mode (Default — Zero Config)

NovaMLX works exactly as it does today. No cluster code runs. No network ports opened. No performance overhead.

Detection: If `ClusterDiscovery` finds no peers within 5 seconds of startup → standalone mode.

### Cluster Mode (Auto-Detected)

When multiple NovaMLX instances see each other on the network:

1. **Discovery** — Bonjour broadcasts, peers respond with capabilities
2. **Election** — Highest-memory node becomes coordinator
3. **Shard Planning** — Coordinator decides how to split the model across nodes
4. **Shard Loading** — Each node downloads/loads its shard of model weights
5. **Pipeline Ready** — Coordinator exposes normal OpenAI-compatible API
6. **Inference** — Requests flow through the pipeline

### Graceful Degradation

- If a node drops mid-inference: coordinator re-shards to remaining nodes (slower but continues)
- If coordinator drops: remaining nodes elect new coordinator
- If only 1 node remains: falls back to standalone mode automatically

---

## Implementation Phases

### Phase 1: Foundation (Future Sprint)
- [ ] Create `NovaMLXDistributed` Swift module
- [ ] Implement `ClusterDiscovery` with Bonjour/mDNS
- [ ] Implement `NodeIdentity` and `ClusterNode` types
- [ ] Implement basic `TCPTransport` for hidden state transfer
- [ ] Add cluster/standalone detection to `InferenceService` routing

### Phase 2: Pipeline Parallelism
- [ ] Implement `ModelSharder` — split weights by layer count
- [ ] Implement `PipelineWorker` — receive → run layers → forward
- [ ] Implement `ClusterCoordinator` — orchestrate the pipeline
- [ ] Implement `DistributedKVCache` — each node manages its own KV cache
- [ ] Wire up to NovaMLXAPI — distributed inference behind OpenAI API

### Phase 3: Robustness
- [ ] Node failure detection + automatic re-sharding
- [ ] Coordinator election / failover
- [ ] Cluster metrics dashboard (memory per node, throughput, pipeline utilization)
- [ ] Multi-model cluster (different models on different node groups)

### Phase 4: Tensor Parallelism (Hardware Dependent)
- [ ] Implement `RDMATransport` using JACCL backend
- [ ] Implement tensor-parallel layer execution (shard matmuls across nodes)
- [ ] All-reduce communication pattern
- [ ] Hybrid pipeline + tensor parallelism for very large clusters

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Transport (Phase 1) | TCP/IP over Thunderbolt | Works on TB4/TB5/Ethernet, no special OS config |
| Transport (Phase 4) | RDMA over TB5 (JACCL) | Apple's official path for tensor parallelism |
| Parallelism strategy | Pipeline first, tensor later | Pipeline works today; tensor needs TB5+RDMA |
| Discovery | Bonjour/mDNS | Apple-native, zero-config, works on any LAN |
| Model sharding | Contiguous layer chunks | Simple, minimal communication, proven approach |
| Shard storage | Per-node shard files on disk | Download once, reuse; no re-download on restart |
| Failover | Re-shard to remaining nodes | No data loss, graceful degradation |
| Coordinator election | Highest available memory | Gives heaviest role to most capable node |

---

## Memory Math for Reference

### 3x Mac Mini M4 Pro (48GB each) = 144GB cluster

| Model | Quant | Total | Per Node | Fits? | Est. Speed |
|-------|-------|-------|----------|-------|------------|
| Llama 3.1 70B | Q4 | ~40 GB | ~13 GB | Yes | ~15 tok/s |
| Qwen 2.5 72B | Q4 | ~42 GB | ~14 GB | Yes | ~15 tok/s |
| Llama 3.1 70B | Q8 | ~70 GB | ~23 GB | Yes | ~8 tok/s |
| Llama 3.1 120B | Q4 | ~68 GB | ~23 GB | Yes | ~8 tok/s |
| Qwen3 235B | Q4 | ~135 GB | ~45 GB | Tight | ~3 tok/s |
| Llama 3.1 70B | FP16 | ~140 GB | ~47 GB | Tight | ~4 tok/s |

### 3x Mac Mini M4 Max (128GB each) = 384GB cluster

| Model | Quant | Total | Per Node | Fits? | Est. Speed |
|-------|-------|-------|----------|-------|------------|
| DeepSeek 671B | Q4 | ~380 GB | ~127 GB | Tight | ~1-2 tok/s |
| Llama 3.1 405B | Q4 | ~230 GB | ~77 GB | Yes | ~3-5 tok/s |
| Any sub-200B | FP16 | varies | varies | Yes | varies |

---

## References

- [MLX Distributed Communication Docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [Apple TN3205: RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
- [WWDC25: Get Started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [Jeff Geerling: 1.5TB VRAM Mac Studio Cluster](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/)
- [arXiv:2601.19139 — vLLM-MLX Paper](https://arxiv.org/abs/2601.19139)
- [exo-explore/exo](https://github.com/exo-explore/exo) — reference architecture for P2P cluster
- [firstbatchxyz/dnet](https://github.com/firstbatchxyz/dnet) — reference for device profiling
