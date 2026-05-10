import Foundation
import Testing
@testable import NovaMLXDistributed

@Suite("Fault Recovery")
struct FaultRecoveryTests {

    // MARK: - L1 Transient Disconnect

    @Test("L1: node is in grace period within 30 seconds of disconnect")
    func gracePeriodWithinThreshold() {
        let manager = FaultRecoveryManager()
        let disconnectTime = Date()

        manager.trackDisconnect(nodeId: "node-1", at: disconnectTime)

        // 15 seconds later — still in grace period
        let checkTime = disconnectTime.addingTimeInterval(15)
        #expect(manager.isInGracePeriod(nodeId: "node-1", at: checkTime))
    }

    @Test("L1: node is NOT in grace period after 30 seconds")
    func gracePeriodExpired() {
        let manager = FaultRecoveryManager()
        let disconnectTime = Date()

        manager.trackDisconnect(nodeId: "node-1", at: disconnectTime)

        // 31 seconds later — grace period expired
        let checkTime = disconnectTime.addingTimeInterval(31)
        #expect(!manager.isInGracePeriod(nodeId: "node-1", at: checkTime))
    }

    @Test("L1: clearDisconnect removes tracking")
    func clearDisconnectRemovesTracking() {
        let manager = FaultRecoveryManager()
        let disconnectTime = Date()

        manager.trackDisconnect(nodeId: "node-1", at: disconnectTime)
        manager.clearDisconnect(nodeId: "node-1")

        // Immediately after — should NOT be in grace period since cleared
        #expect(!manager.isInGracePeriod(nodeId: "node-1", at: disconnectTime))
    }

    @Test("L1: untracked node is not in grace period")
    func untrackedNodeNotInGracePeriod() {
        let manager = FaultRecoveryManager()
        let now = Date()

        #expect(!manager.isInGracePeriod(nodeId: "unknown", at: now))
    }

    // MARK: - L2 Spare Node Swap

    @Test("L2: selects largest capacity spare that can handle memory")
    func selectSparePicksLargestCapacityMatch() {
        let manager = FaultRecoveryManager()

        let failedAssignment = ShardAssignment(
            nodeId: "failed-node",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 40 * 1024 * 1024  // 40 MB
        )

        let spares = [
            NodeSpec(nodeId: "spare-32", totalMemoryBytes: 32 * 1024 * 1024, computeCapability: 1.0, hostname: "s32.local", port: 6591),
            NodeSpec(nodeId: "spare-64", totalMemoryBytes: 64 * 1024 * 1024, computeCapability: 1.0, hostname: "s64.local", port: 6591),
        ]

        let selected = manager.selectSpareFor(failedAssignment: failedAssignment, spares: spares)
        #expect(selected?.nodeId == "spare-64")
    }

    @Test("L2: returns nil when no spare has enough memory")
    func selectSpareReturnsNilWhenInsufficientMemory() {
        let manager = FaultRecoveryManager()

        let failedAssignment = ShardAssignment(
            nodeId: "failed-node",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 100 * 1024 * 1024  // 100 MB
        )

        let spares = [
            NodeSpec(nodeId: "spare-32", totalMemoryBytes: 32 * 1024 * 1024, computeCapability: 1.0, hostname: "s32.local", port: 6591),
            NodeSpec(nodeId: "spare-64", totalMemoryBytes: 64 * 1024 * 1024, computeCapability: 1.0, hostname: "s64.local", port: 6591),
        ]

        let selected = manager.selectSpareFor(failedAssignment: failedAssignment, spares: spares)
        #expect(selected == nil)
    }

    @Test("L2: returns nil for empty spares list")
    func selectSpareReturnsNilForEmptySpares() {
        let manager = FaultRecoveryManager()

        let failedAssignment = ShardAssignment(
            nodeId: "failed-node",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 40 * 1024 * 1024
        )

        let selected = manager.selectSpareFor(failedAssignment: failedAssignment, spares: [])
        #expect(selected == nil)
    }

    // MARK: - L3a Auto-Reshard

    @Test("L3a: can reshard when remaining nodes have enough combined memory")
    func canReshardWithSufficientMemory() {
        // 1 node with 128 GB, model is 160 MB with overhead 1.3 = 208 MB
        // 128 GB >> 208 MB, so can reshard
        let remainingNodes = [
            NodeSpec(nodeId: "node-a", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "a.local", port: 6591),
        ]
        let totalModelMemory: UInt64 = 160 * 1024 * 1024  // 160 MB

        let result = FaultRecoveryManager.canReshard(
            remainingNodes: remainingNodes,
            totalModelMemory: totalModelMemory,
            overheadFactor: 1.3
        )
        #expect(result)
    }

    @Test("L3a: cannot reshard when remaining nodes lack sufficient memory")
    func cannotReshardWithInsufficientMemory() {
        // 1 node with 100 MB, model is 160 MB with overhead 1.3 = 208 MB
        // 100 MB < 208 MB, cannot reshard
        let remainingNodes = [
            NodeSpec(nodeId: "node-a", totalMemoryBytes: 100 * 1024 * 1024, computeCapability: 1.0, hostname: "a.local", port: 6591),
        ]
        let totalModelMemory: UInt64 = 160 * 1024 * 1024  // 160 MB

        let result = FaultRecoveryManager.canReshard(
            remainingNodes: remainingNodes,
            totalModelMemory: totalModelMemory,
            overheadFactor: 1.3
        )
        #expect(!result)
    }

    @Test("L3a: cannot reshard with no remaining nodes")
    func cannotReshardWithNoNodes() {
        let totalModelMemory: UInt64 = 160 * 1024 * 1024

        let result = FaultRecoveryManager.canReshard(
            remainingNodes: [],
            totalModelMemory: totalModelMemory,
            overheadFactor: 1.3
        )
        #expect(!result)
    }

    @Test("L3a: computeReshardPlan returns valid plan")
    func computeReshardPlanReturnsValidPlan() throws {
        let manager = FaultRecoveryManager()

        let remainingNodes = [
            NodeSpec(nodeId: "node-a", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "a.local", port: 6591),
            NodeSpec(nodeId: "node-b", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 0.8, hostname: "b.local", port: 6591),
        ]

        let profiles = (0..<40).map { i in
            LayerProfile(
                layerIndex: i,
                parameterCount: 1_000_000,
                estimatedMemoryBytes: 4_000_000,
                layerType: .transformer
            )
        }

        let plan = manager.computeReshardPlan(remainingNodes: remainingNodes, profiles: profiles)
        #expect(plan != nil)

        let unwrappedPlan = try #require(plan)
        #expect(unwrappedPlan.assignments.count == 2)
        #expect(unwrappedPlan.totalLayers == 40)

        let totalCovered = unwrappedPlan.assignments.reduce(0) { $0 + ($1.endLayer - $1.startLayer) }
        #expect(totalCovered == 40)
    }

    @Test("L3a: computeReshardPlan returns nil for empty nodes")
    func computeReshardPlanReturnsNilForEmptyNodes() {
        let manager = FaultRecoveryManager()

        let profiles = (0..<10).map { i in
            LayerProfile(
                layerIndex: i,
                parameterCount: 1_000_000,
                estimatedMemoryBytes: 4_000_000,
                layerType: .transformer
            )
        }

        let plan = manager.computeReshardPlan(remainingNodes: [], profiles: profiles)
        #expect(plan == nil)
    }

    // MARK: - L3b Hard Fail

    @Test("L3b: handleHardFail does not crash")
    func handleHardFailDoesNotCrash() {
        let manager = FaultRecoveryManager()
        // Should not throw or crash — placeholder implementation
        manager.handleHardFail(modelId: "model-x", reason: "node unresponsive")
    }
}
