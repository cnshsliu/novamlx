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
    /// Use the minimum number of nodes needed.
    case minNodes
    /// Spread layers evenly across all available nodes.
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

    public init(
        role: ClusterRole,
        coordinatorHost: String,
        coordinatorPort: Int = 6591,
        strategy: ClusterStrategy = .minNodes,
        prefill: PrefillConfig = PrefillConfig()
    ) {
        self.role = role
        self.coordinatorHost = coordinatorHost
        self.coordinatorPort = coordinatorPort
        self.strategy = strategy
        self.prefill = prefill
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(ClusterRole.self, forKey: .role)
        coordinatorHost = try container.decode(String.self, forKey: .coordinatorHost)
        coordinatorPort = try container.decodeIfPresent(Int.self, forKey: .coordinatorPort) ?? 6591
        strategy = try container.decodeIfPresent(ClusterStrategy.self, forKey: .strategy) ?? .minNodes
        prefill = try container.decodeIfPresent(PrefillConfig.self, forKey: .prefill) ?? PrefillConfig()
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

    public init(
        nodeId: String,
        totalMemoryBytes: UInt64,
        computeCapability: Double,
        hostname: String,
        port: Int
    ) {
        self.nodeId = nodeId
        self.totalMemoryBytes = totalMemoryBytes
        self.computeCapability = computeCapability
        self.hostname = hostname
        self.port = port
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

    /// Compute a shard plan by allocating layers to nodes proportional to their memory.
    ///
    /// Each node receives layers proportional to `totalMemoryBytes`. The last node
    /// gets all remaining layers to guarantee full coverage.
    public init(profiles: [LayerProfile], nodes: [NodeSpec], strategy: ClusterStrategy) {
        let totalLayers = profiles.count
        precondition(
            nodes.count <= totalLayers,
            "Cannot distribute \(totalLayers) layers across \(nodes.count) nodes: each node needs at least 1 layer"
        )
        let totalMemory = nodes.reduce(UInt64(0)) { $0 + $1.totalMemoryBytes }

        var assignments: [ShardAssignment] = []
        var currentLayer = 0

        for (index, node) in nodes.enumerated() {
            let isLast = index == nodes.count - 1
            let layersForNode: Int

            if isLast {
                layersForNode = totalLayers - currentLayer
            } else if totalMemory > 0 {
                let ratio = Double(node.totalMemoryBytes) / Double(totalMemory)
                layersForNode = max(1, Int(ratio * Double(totalLayers)))
            } else {
                layersForNode = max(1, totalLayers / nodes.count)
            }

            let endLayer = min(currentLayer + layersForNode, totalLayers)
            let memoryEstimate = profiles[currentLayer..<endLayer]
                .reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }

            assignments.append(ShardAssignment(
                nodeId: node.nodeId,
                startLayer: currentLayer,
                endLayer: endLayer,
                memoryEstimate: memoryEstimate
            ))
            currentLayer = endLayer
        }

        self.assignments = assignments
        self.totalLayers = totalLayers
        self.strategy = strategy
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
