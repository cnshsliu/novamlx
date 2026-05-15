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

    public init(
        role: ClusterRole,
        coordinatorHost: String,
        coordinatorPort: Int = 6591,
        strategy: ClusterStrategy = .minNodes,
        prefill: PrefillConfig = PrefillConfig(),
        minLayersPerShard: Int = 32
    ) {
        self.role = role
        self.coordinatorHost = coordinatorHost
        self.coordinatorPort = coordinatorPort
        self.strategy = strategy
        self.prefill = prefill
        self.minLayersPerShard = minLayersPerShard
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(ClusterRole.self, forKey: .role)
        coordinatorHost = try container.decode(String.self, forKey: .coordinatorHost)
        coordinatorPort = try container.decodeIfPresent(Int.self, forKey: .coordinatorPort) ?? 6591
        strategy = try container.decodeIfPresent(ClusterStrategy.self, forKey: .strategy) ?? .minNodes
        prefill = try container.decodeIfPresent(PrefillConfig.self, forKey: .prefill) ?? PrefillConfig()
        minLayersPerShard = try container.decodeIfPresent(Int.self, forKey: .minLayersPerShard) ?? 32
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

    public init(
        nodeId: String,
        totalMemoryBytes: UInt64,
        computeCapability: Double,
        hostname: String,
        port: Int,
        cpuModel: String = "",
        networkHost: String? = nil
    ) {
        self.nodeId = nodeId
        self.totalMemoryBytes = totalMemoryBytes
        self.computeCapability = computeCapability
        self.hostname = hostname
        self.port = port
        self.cpuModel = cpuModel
        self.networkHost = networkHost
    }
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
    /// - **spread**: Distribute evenly across nodes, but each shard gets at
    ///   least `minLayersPerShard` layers. Excess nodes are left idle.
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

        let totalMemory = activeNodes.reduce(UInt64(0)) { $0 + $1.totalMemoryBytes }

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
            // Even distribution proportional to memory, respecting minPerShard.
            for (index, node) in activeNodes.enumerated() {
                let isLast = index == activeNodes.count - 1
                var layersForNode: Int

                if isLast {
                    layersForNode = totalLayers - currentLayer
                } else {
                    let remainingNodes = activeNodes.count - index
                    let remainingLayers = totalLayers - currentLayer
                    let evenSplit = remainingLayers / remainingNodes
                    if totalMemory > 0 {
                        let ratio = Double(node.totalMemoryBytes) / Double(totalMemory)
                        layersForNode = max(minPerShard, Int(ratio * Double(totalLayers)))
                    } else {
                        layersForNode = max(minPerShard, evenSplit)
                    }
                    // Don't overshoot: leave room for remaining nodes
                    let maxAllowed = remainingLayers - (remainingNodes - 1) * minPerShard
                    layersForNode = min(layersForNode, maxAllowed)
                    layersForNode = max(minPerShard, layersForNode)
                }

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
