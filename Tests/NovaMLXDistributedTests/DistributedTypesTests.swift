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

    @Test("ShardPlan computes correct layer ranges for 2 nodes")
    func shardPlanTwoNodes() {
        let profiles = (0..<40).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let nodes = [
            NodeSpec(nodeId: "mac-a", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "mac-a.local", port: 6591),
            NodeSpec(nodeId: "mac-b", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 0.6, hostname: "mac-b.local", port: 6591),
        ]
        let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .minNodes)
        #expect(plan.assignments.count == 2)
        let totalCovered = plan.assignments.reduce(0) { $0 + ($1.endLayer - $1.startLayer) }
        #expect(totalCovered == 40)
    }

    @Test("ClusterRole codable round-trip")
    func clusterRoleRoundTrip() throws {
        for role in [ClusterRole.coordinator, .worker] {
            let encoded = try JSONEncoder().encode(role)
            let decoded = try JSONDecoder().decode(ClusterRole.self, from: encoded)
            #expect(decoded == role)
        }
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

    @Test("Initialize returns uninitialized without backend")
    func initializeWithoutBackend() {
        // Without a compiled distributed backend, initialize should return uninitialized.
        let group = MLXDistributedWrapper.initialize(strict: false, backend: nil)
        #expect(group.isValid == false)
        #expect(group.rank == -1)
        #expect(group.size == 0)
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
