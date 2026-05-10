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

    public init(
        role: ClusterRole,
        coordinatorHost: String,
        coordinatorPort: Int = 6591,
        strategy: ClusterStrategy = .minNodes
    ) {
        self.role = role
        self.coordinatorHost = coordinatorHost
        self.coordinatorPort = coordinatorPort
        self.strategy = strategy
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(ClusterRole.self, forKey: .role)
        coordinatorHost = try container.decode(String.self, forKey: .coordinatorHost)
        coordinatorPort = try container.decodeIfPresent(Int.self, forKey: .coordinatorPort) ?? 6591
        strategy = try container.decodeIfPresent(ClusterStrategy.self, forKey: .strategy) ?? .minNodes
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
        let totalMemory = nodes.reduce(UInt64(0)) { $0 + $1.totalMemoryBytes }
        let perLayerMemory = totalLayers > 0
            ? profiles.reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes } / UInt64(totalLayers)
            : UInt64(0)

        var assignments: [ShardAssignment] = []
        var currentLayer = 0

        for (index, node) in nodes.enumerated() {
            let isLast = index == nodes.count - 1
            let layersForNode: Int

            if isLast {
                // Last node gets all remaining layers to guarantee full coverage
                layersForNode = totalLayers - currentLayer
            } else if totalMemory > 0 {
                // Proportional allocation based on memory
                let ratio = Double(node.totalMemoryBytes) / Double(totalMemory)
                layersForNode = max(1, Int(ratio * Double(totalLayers)))
            } else {
                // Equal split fallback
                layersForNode = max(1, totalLayers / nodes.count)
            }

            let endLayer = min(currentLayer + layersForNode, totalLayers)
            let memoryEstimate = UInt64(endLayer - currentLayer) * perLayerMemory

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
