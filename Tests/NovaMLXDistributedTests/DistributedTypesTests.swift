import Foundation
import Testing
@testable import NovaMLXDistributed

@Suite("Distributed Types")
struct DistributedTypesTests {

    @Test("ClusterConfig decodes with defaults")
    func clusterConfigDefaults() throws {
        let json = """
        {"role": "worker", "coordinatorHost": "192.168.1.1", "coordinatorPort": 6591}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ClusterConfig.self, from: json)
        #expect(config.role == .worker)
        #expect(config.coordinatorHost == "192.168.1.1")
        #expect(config.coordinatorPort == 6591)
        #expect(config.strategy == .minNodes)
    }

    @Test("ShardPlan with 2 nodes covers all layers")
    func shardPlanTwoNodes() {
        let profiles = (0..<80).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let nodes = [
            NodeSpec(nodeId: "mac-a", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "mac-a.local", port: 6591),
            NodeSpec(nodeId: "mac-b", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 0.6, hostname: "mac-b.local", port: 6591),
        ]
        let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .spread, minLayersPerShard: 4)
        let totalCovered = plan.assignments.reduce(0) { $0 + ($1.endLayer - $1.startLayer) }
        #expect(totalCovered == 80)
        let totalMemory = plan.assignments.reduce(UInt64(0)) { $0 + $1.memoryEstimate }
        #expect(totalMemory == 80 * 4_000_000)
    }

    @Test("ShardPlan enforces minLayersPerShard — caps active nodes")
    func shardPlanMinLayers() {
        // 100 layers, 100 nodes, min 32 per shard → max 3 active nodes
        let profiles = (0..<100).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let nodes = (0..<100).map { i in
            NodeSpec(nodeId: "node-\(i)", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "node-\(i)", port: 6591)
        }
        let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .spread, minLayersPerShard: 32)
        #expect(plan.assignments.count <= 3) // 100/32 = 3 max
        let totalCovered = plan.assignments.reduce(0) { $0 + ($1.endLayer - $1.startLayer) }
        #expect(totalCovered == 100)
        // Every shard has at least 32 layers (except possibly the last)
        for a in plan.assignments.dropLast() {
            #expect(a.endLayer - a.startLayer >= 32)
        }
    }

    @Test("ShardPlan minNodes packs into 1 node when model fits")
    func shardPlanMinNodesPacksOne() {
        // 64 layers × 4MB = 256MB fits in any 64GB node → 1 node
        let profiles = (0..<64).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let nodes = (0..<10).map { i in
            NodeSpec(nodeId: "node-\(i)", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "node-\(i)", port: 6591)
        }
        let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .minNodes, minLayersPerShard: 32)
        #expect(plan.assignments.count == 1)
        #expect(plan.assignments[0].endLayer - plan.assignments[0].startLayer == 64)
    }

    @Test("ClusterRole codable round-trip")
    func clusterRoleRoundTrip() throws {
        for role in [ClusterRole.coordinator, .worker] {
            let encoded = try JSONEncoder().encode(role)
            let decoded = try JSONDecoder().decode(ClusterRole.self, from: encoded)
            #expect(decoded == role)
        }
    }

    @Test("PrefillConfig has correct defaults")
    func prefillConfigDefaults() {
        let config = PrefillConfig()
        #expect(config.baseStepSize == 4096)
        #expect(config.minChunkSize == 512)
        #expect(config.minWavefrontTokens == 4096)
    }

    @Test("PrefillConfig codable round-trip")
    func prefillConfigCodable() throws {
        let config = PrefillConfig(baseStepSize: 2048, minChunkSize: 256, minWavefrontTokens: 8192)
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(PrefillConfig.self, from: data)
        #expect(decoded == config)
    }

    @Test("PrefillConfig decodes with missing fields using defaults")
    func prefillConfigPartialDecode() throws {
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(PrefillConfig.self, from: json)
        #expect(config.baseStepSize == 4096)
        #expect(config.minChunkSize == 512)
        #expect(config.minWavefrontTokens == 4096)
    }

    @Test("WavefrontStats stores correct values")
    func wavefrontStatsValues() {
        let stats = WavefrontStats(
            chunkSize: 2048,
            nRealChunks: 4,
            nLeadingDummies: 1,
            nTrailingDummies: 0,
            promptTokens: 8192,
            prefillCommBytes: 65536
        )
        #expect(stats.chunkSize == 2048)
        #expect(stats.nRealChunks == 4)
        #expect(stats.nLeadingDummies == 1)
        #expect(stats.nTrailingDummies == 0)
        #expect(stats.promptTokens == 8192)
        #expect(stats.prefillCommBytes == 65536)
    }

    @Test("ClusterConfig with PrefillConfig decodes correctly")
    func clusterConfigWithPrefill() throws {
        let json = """
        {"role": "coordinator", "coordinatorHost": "192.168.1.1", "coordinatorPort": 6591, "prefill": {"baseStepSize": 2048, "minChunkSize": 256, "minWavefrontTokens": 8192}}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ClusterConfig.self, from: json)
        #expect(config.prefill.baseStepSize == 2048)
        #expect(config.prefill.minChunkSize == 256)
        #expect(config.prefill.minWavefrontTokens == 8192)
    }

    @Test("ClusterConfig without PrefillConfig uses defaults")
    func clusterConfigWithoutPrefill() throws {
        let json = """
        {"role": "coordinator", "coordinatorHost": "192.168.1.1", "coordinatorPort": 6591}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ClusterConfig.self, from: json)
        #expect(config.prefill.baseStepSize == 4096)
        #expect(config.prefill.minChunkSize == 512)
    }
}

@Suite("Distributed Group Wrappers")
struct DistributedGroupTests {

    @Test("Backend availability check does not crash")
    func backendAvailabilityCheck() {
        let ringAvailable = MLXDistributedWrapper.isBackendAvailable("ring")
        #expect(type(of: ringAvailable) == Bool.self)
    }

    @Test("DistributedGroup wraps C handle")
    func groupWrapsHandle() {
        let group = DistributedGroup.uninitialized
        #expect(group.rank == -1)
        #expect(group.size == 0)
        #expect(group.isValid == false)
    }

    @Test("Uninitialized group equality")
    func uninitializedEquality() {
        let a = DistributedGroup.uninitialized
        let b = DistributedGroup.uninitialized
        #expect(a == b)
    }

    @Test("Initialize without strict returns a valid single-node group")
    func initializeWithoutStrict() {
        let group = MLXDistributedWrapper.initialize(strict: false, backend: nil)
        // With distributed backend compiled in, non-strict returns a valid single-node group.
        #expect(group.isValid == true)
        #expect(group.rank == 0)
        #expect(group.size == 1)
    }

    @Test("bestAvailableBackend returns a non-empty string")
    func bestAvailableBackendReturnsString() {
        let backend = MLXDistributedWrapper.bestAvailableBackend()
        #expect(!backend.isEmpty)
    }

    @Test("isCBBackendAvailable is Bool")
    func isCBackendAvailableIsBool() {
        let available = MLXDistributedWrapper.isCBBackendAvailable
        #expect(type(of: available) == Bool.self)
    }
}
