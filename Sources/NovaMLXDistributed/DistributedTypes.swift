import Foundation

// MARK: - ClusterRole

/// Role a node plays in the distributed cluster.
public enum ClusterRole: String, Codable, Sendable, Equatable {
    case coordinator
    case worker
}

// MARK: - ClusterStrategy

/// Strategy for distributing model layers across nodes.
public enum ClusterStrategy: String, Codable, Sendable, Equatable {
    /// Use the minimum number of nodes that fit the model.
    case minNodes
    /// Spread layers across all nodes, respecting min layers per shard.
    case spread
}

// MARK: - ClusterConfig

/// Configuration for a node joining a distributed cluster.
public struct ClusterConfig: Codable, Sendable, Equatable {
    public let role: ClusterRole
    public let coordinatorHost: String
    public let coordinatorPort: Int
    public let strategy: ClusterStrategy
    public var prefill: PrefillConfig

    /// Minimum layers assigned to any single shard.
    /// Prevents over-splitting where communication overhead dominates.
    /// Default: 32 (roughly 8-16 transformer blocks).
    public let minLayersPerShard: Int

    /// Whether to attempt using MLX Ring transport (instead of custom TCP).
    /// Ring is currently experimental and only recommended after assigning
    /// stable private IPs via Scripts/setup-thunderbolt-ring.sh.
    public var enableRingTransport: Bool

    public init(
        role: ClusterRole,
        coordinatorHost: String,
        coordinatorPort: Int = 6591,
        strategy: ClusterStrategy = .minNodes,
        prefill: PrefillConfig = PrefillConfig(),
        minLayersPerShard: Int = 32,
        enableRingTransport: Bool = false
    ) {
        self.role = role
        self.coordinatorHost = coordinatorHost
        self.coordinatorPort = coordinatorPort
        self.strategy = strategy
        self.prefill = prefill
        self.minLayersPerShard = minLayersPerShard
        self.enableRingTransport = enableRingTransport
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(ClusterRole.self, forKey: .role)
        coordinatorHost = try container.decode(String.self, forKey: .coordinatorHost)
        coordinatorPort = try container.decodeIfPresent(Int.self, forKey: .coordinatorPort) ?? 6591
        strategy = try container.decodeIfPresent(ClusterStrategy.self, forKey: .strategy) ?? .minNodes
        prefill = try container.decodeIfPresent(PrefillConfig.self, forKey: .prefill) ?? PrefillConfig()
        minLayersPerShard = try container.decodeIfPresent(Int.self, forKey: .minLayersPerShard) ?? 32
        enableRingTransport = try container.decodeIfPresent(Bool.self, forKey: .enableRingTransport) ?? false
    }
}

// MARK: - NodeSpec

/// Specification of a node in the cluster.
public struct NodeSpec: Codable, Sendable, Equatable {
    public let nodeId: String
    public let totalMemoryBytes: UInt64
    public let computeCapability: Double
    public let hostname: String
    public let port: Int
    public let cpuModel: String

    /// Reachable IP address for TCP connections (set by Coordinator polling).
    /// Falls back to `hostname` if not set.
    public var networkHost: String?

    /// Fingerprint of the binary the worker is running (for version consistency check).
    public var binaryFingerprint: String?

    /// Hash of the authoritative cluster configuration (mainly cluster-policy.json).
    public var configHash: String?

    public init(
        nodeId: String,
        totalMemoryBytes: UInt64,
        computeCapability: Double,
        hostname: String,
        port: Int,
        cpuModel: String = "",
        networkHost: String? = nil,
        binaryFingerprint: String? = nil,
        configHash: String? = nil
    ) {
        self.nodeId = nodeId
        self.totalMemoryBytes = totalMemoryBytes
        self.computeCapability = computeCapability
        self.hostname = hostname
        self.port = port
        self.cpuModel = cpuModel
        self.networkHost = networkHost
        self.binaryFingerprint = binaryFingerprint
        self.configHash = configHash
    }
}

/// Map Apple Silicon CPU model string to relative GPU compute ratio.
/// Used by ShardPlan to weight layer assignment by GPU capability.
/// M4 Max = 1.0 baseline (40 GPU cores).
public func computeRatio(for cpuModel: String) -> Double {
    if cpuModel.contains("M4 Max") { return 1.0 }
    if cpuModel.contains("M4 Pro") { return 0.50 }
    if cpuModel.contains("M4") { return 0.23 }      // base M4 (10 cores)
    if cpuModel.contains("M3 Max") { return 0.75 }
    if cpuModel.contains("M3 Pro") { return 0.38 }
    if cpuModel.contains("M3") { return 0.19 }
    if cpuModel.contains("M2 Max") { return 0.75 }
    if cpuModel.contains("M2 Pro") { return 0.38 }
    if cpuModel.contains("M2") { return 0.19 }
    return 0.2  // conservative default for unknown chips
}

// MARK: - LayerType

/// Type of a model layer, used for profiling and sharding decisions.
public enum LayerType: String, Codable, Sendable, Equatable {
    case embedding
    case transformer
    case output
    case moe
}

// MARK: - LayerProfile

/// Profile of a single model layer, including memory and parameter estimates.
public struct LayerProfile: Codable, Sendable, Equatable {
    public let layerIndex: Int
    public let parameterCount: UInt64
    public let estimatedMemoryBytes: UInt64
    public let layerType: LayerType

    public init(
        layerIndex: Int,
        parameterCount: UInt64,
        estimatedMemoryBytes: UInt64,
        layerType: LayerType
    ) {
        self.layerIndex = layerIndex
        self.parameterCount = parameterCount
        self.estimatedMemoryBytes = estimatedMemoryBytes
        self.layerType = layerType
    }
}

// MARK: - ShardAssignment

/// Assignment of a contiguous range of layers to a specific node.
public struct ShardAssignment: Codable, Sendable, Equatable {
    public let nodeId: String
    public let startLayer: Int
    /// Exclusive upper bound.
    public let endLayer: Int
    public let memoryEstimate: UInt64

    /// Number of layers in this assignment.
    public var layerCount: Int {
        endLayer - startLayer
    }

    public init(
        nodeId: String,
        startLayer: Int,
        endLayer: Int,
        memoryEstimate: UInt64
    ) {
        self.nodeId = nodeId
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.memoryEstimate = memoryEstimate
    }
}

// MARK: - ShardPlan

/// A complete plan for distributing model layers across cluster nodes.
public struct ShardPlan: Codable, Sendable, Equatable {
    public let assignments: [ShardAssignment]
    public let totalLayers: Int
    public let strategy: ClusterStrategy

    /// Direct initializer with pre-computed assignments.
    public init(
        assignments: [ShardAssignment],
        totalLayers: Int,
        strategy: ClusterStrategy
    ) {
        self.assignments = assignments
        self.totalLayers = totalLayers
        self.strategy = strategy
    }

    /// Compute a shard plan by allocating layers to nodes.
    ///
    /// - **minNodes**: Use fewest nodes that can hold the model. Packs nodes
    ///   greedily by memory until all layers are assigned.
    /// - **spread**: Minimize sequential pipeline latency. Fastest node gets
    ///   maximum layers, slowest get minimum (`minLayersPerShard`). Memory-capped.
    ///
    /// Both strategies enforce `minLayersPerShard` to prevent over-splitting
    /// where communication overhead would dominate compute.
    public init(
        profiles: [LayerProfile],
        nodes: [NodeSpec],
        strategy: ClusterStrategy,
        minLayersPerShard: Int = 32
    ) {
        let totalLayers = profiles.count
        let minPerShard = max(1, minLayersPerShard)

        // Cap the number of active nodes: each must get at least minPerShard layers.
        let maxNodes = max(1, totalLayers / minPerShard)
        let activeNodes = Array(nodes.prefix(maxNodes))

        precondition(
            activeNodes.count <= totalLayers,
            "Cannot distribute \(totalLayers) layers across \(activeNodes.count) nodes"
        )

        var assignments: [ShardAssignment] = []
        var currentLayer = 0

        switch strategy {
        case .minNodes:
            // Greedy: pack layers into the biggest nodes first until done.
            let sorted = activeNodes.map { n in
                (node: n, memory: n.totalMemoryBytes)
            }.sorted { $0.memory > $1.memory }

            var remaining = totalLayers
            for entry in sorted {
                guard remaining > 0 else { break }
                // Give this node as many remaining layers as possible, at least minPerShard
                let layersForNode = max(minPerShard, remaining)
                let assigned = min(layersForNode, remaining)
                let start = totalLayers - remaining
                let end = start + assigned

                let memEstimate = profiles[min(start, profiles.endIndex)..<min(end, profiles.endIndex)]
                    .reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }
                assignments.append(ShardAssignment(
                    nodeId: entry.node.nodeId,
                    startLayer: start,
                    endLayer: end,
                    memoryEstimate: memEstimate
                ))
                remaining -= assigned
            }

        case .spread:
            // Sequential pipeline optimization: minimize total latency.
            // total_time = sum(layers_i * time_per_layer_i)
            // Since faster nodes have lower time_per_layer, giving them more layers
            // reduces total time. Optimal: give minPerShard to slow nodes, rest to fastest.
            let sortedIndices = activeNodes.indices.sorted { i, j in
                computeRatio(for: activeNodes[i].cpuModel) > computeRatio(for: activeNodes[j].cpuModel)
            }
            var layerAllocation = [Int](repeating: minPerShard, count: activeNodes.count)
            var remaining = totalLayers - activeNodes.count * minPerShard

            // Greedy: fill fastest nodes first, respecting memory
            for idx in sortedIndices {
                let node = activeNodes[idx]
                let alreadyAllocated = layerAllocation[idx]
                if idx == sortedIndices[0] {
                    // Fastest node gets all extra layers on top of its minimum
                    layerAllocation[idx] = alreadyAllocated + remaining
                    remaining = 0
                }
                // Memory sanity: if node can't hold its allocation, reduce and redistribute
                let profileMemory = profiles.reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }
                let memPerLayer = totalLayers > 0 ? profileMemory / UInt64(totalLayers) : 0
                if memPerLayer > 0 {
                    let maxLayersByMem = Int(node.totalMemoryBytes / max(1, memPerLayer))
                    if layerAllocation[idx] > maxLayersByMem && maxLayersByMem >= minPerShard {
                        let excess = layerAllocation[idx] - maxLayersByMem
                        layerAllocation[idx] = maxLayersByMem
                        remaining += excess
                    }
                }
            }
            // If fastest node was capped, give excess to next-fastest
            if remaining > 0 {
                for i in 1..<sortedIndices.count {
                    let idx = sortedIndices[i]
                    let add = min(remaining, totalLayers - layerAllocation.reduce(0, +))
                    if add > 0 {
                        layerAllocation[idx] += add
                        remaining -= add
                    }
                }
            }

            for (index, node) in activeNodes.enumerated() {
                let layersForNode = layerAllocation[index]
                let endLayer = min(currentLayer + layersForNode, totalLayers)
                let memEstimate = profiles[currentLayer..<endLayer]
                    .reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }
                assignments.append(ShardAssignment(
                    nodeId: node.nodeId,
                    startLayer: currentLayer,
                    endLayer: endLayer,
                    memoryEstimate: memEstimate
                ))
                currentLayer = endLayer
            }
        }

        self.assignments = assignments
        self.totalLayers = totalLayers
        self.strategy = strategy
    }
}

// MARK: - ClusterModelState

/// Cluster-wide model activation state.
/// Only one model can be active at a time in distributed mode.
public enum ClusterModelState: String, Codable, Sendable, Equatable {
    /// No model activated. Inference requests will be rejected.
    case idle
    /// Model is being loaded across nodes. Inference requests will be rejected.
    case activating
    /// All nodes loaded their shards. Ready for inference.
    case ready
    /// Activation failed on one or more nodes.
    case failed
}

// MARK: - ShardReadiness

/// Readiness status of a single node's shard.
public struct ShardReadiness: Codable, Sendable, Equatable {
    public let nodeId: String
    public let hostname: String
    /// Layer range assigned to this node.
    public let startLayer: Int
    public let endLayer: Int
    /// Current loading status.
    public var status: ShardLoadStatus
    /// Memory used by loaded shard (0 if not loaded).
    public var memoryUsedBytes: UInt64
    /// Error message if status is failed.
    public var errorMessage: String?

    public init(
        nodeId: String,
        hostname: String,
        startLayer: Int,
        endLayer: Int,
        status: ShardLoadStatus = .pending,
        memoryUsedBytes: UInt64 = 0,
        errorMessage: String? = nil
    ) {
        self.nodeId = nodeId
        self.hostname = hostname
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.status = status
        self.memoryUsedBytes = memoryUsedBytes
        self.errorMessage = errorMessage
    }

    /// Number of layers in this shard.
    public var layerCount: Int { endLayer - startLayer }
}

// MARK: - ShardLoadStatus

/// Status of a shard loading on a node.
public enum ShardLoadStatus: String, Codable, Sendable, Equatable {
    /// Waiting for load command.
    case pending
    /// Currently loading model and binding weights.
    case loading
    /// Shard loaded and ready for inference.
    case ready
    /// Loading failed.
    case failed
}

// MARK: - ClusterModelStatus

/// Full status snapshot for the cluster's active model.
public struct ClusterModelStatus: Codable, Sendable {
    public let activeModel: String?
    public let state: ClusterModelState
    public let shardPlan: ShardPlan?
    public let nodes: [ShardReadiness]

    /// How many nodes are ready vs total.
    public var readinessFraction: (ready: Int, total: Int) {
        let ready = nodes.filter { $0.status == .ready }.count
        return (ready, nodes.count)
    }

    public init(
        activeModel: String? = nil,
        state: ClusterModelState = .idle,
        shardPlan: ShardPlan? = nil,
        nodes: [ShardReadiness] = []
    ) {
        self.activeModel = activeModel
        self.state = state
        self.shardPlan = shardPlan
        self.nodes = nodes
    }
}

// MARK: - PrefillConfig

/// Configuration for overlapped wavefront prefill.
public struct PrefillConfig: Codable, Sendable, Equatable {
    /// Base step size in tokens. Divided by `worldSize` to get actual chunk size.
    public var baseStepSize: Int

    /// Minimum chunk size in tokens. Prevents pathological tiny chunks on large clusters.
    public var minChunkSize: Int

    /// Minimum prompt length to activate wavefront prefill.
    public var minWavefrontTokens: Int

    public init(
        baseStepSize: Int = 4096,
        minChunkSize: Int = 512,
        minWavefrontTokens: Int = 4096
    ) {
        self.baseStepSize = baseStepSize
        self.minChunkSize = minChunkSize
        self.minWavefrontTokens = minWavefrontTokens
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        baseStepSize = try container.decodeIfPresent(Int.self, forKey: .baseStepSize) ?? 4096
        minChunkSize = try container.decodeIfPresent(Int.self, forKey: .minChunkSize) ?? 512
        minWavefrontTokens = try container.decodeIfPresent(Int.self, forKey: .minWavefrontTokens) ?? 4096
    }
}

// MARK: - WavefrontStats

/// Observability stats from a wavefront prefill execution.
public struct WavefrontStats: Sendable, Equatable {
    public let chunkSize: Int
    public let nRealChunks: Int
    public let nLeadingDummies: Int
    public let nTrailingDummies: Int
    public let promptTokens: Int
    public let prefillCommBytes: UInt64

    public init(
        chunkSize: Int,
        nRealChunks: Int,
        nLeadingDummies: Int,
        nTrailingDummies: Int,
        promptTokens: Int,
        prefillCommBytes: UInt64
    ) {
        self.chunkSize = chunkSize
        self.nRealChunks = nRealChunks
        self.nLeadingDummies = nLeadingDummies
        self.nTrailingDummies = nTrailingDummies
        self.promptTokens = promptTokens
        self.prefillCommBytes = prefillCommBytes
    }
}
