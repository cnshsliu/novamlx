import Foundation

// MARK: - FaultRecoveryManager

/// Manages fault recovery for a distributed inference cluster.
///
/// Recovery levels:
/// - **L1 — Transient Disconnect**: Tracks disconnect timestamps and provides a 30-second
///   grace period during which a node is expected to reconnect.
/// - **L2 — Spare Node Swap**: Selects a spare node from a pool to replace a failed assignment,
///   choosing the largest-capacity spare that satisfies the memory requirement.
/// - **L3a — Auto-Reshard**: Re-distributes layers across remaining healthy nodes when a spare
///   is unavailable.
/// - **L3b — Hard Fail**: Last resort when no recovery is possible; triggers admin notification.
///
/// This type is **not** a singleton; instantiate with `init()` for testability.
public final class FaultRecoveryManager: @unchecked Sendable {

    /// Grace period window in seconds. Nodes disconnecting within this window are
    /// considered transiently unavailable.
    public static let gracePeriodSeconds: TimeInterval = 30

    /// Disconnect timestamps keyed by node identifier.
    private var disconnectTimes: [String: Date] = [:]

    public init() {}

    // MARK: - L1 Transient Disconnect

    /// Record the time at which a node disconnected.
    ///
    /// - Parameters:
    ///   - nodeId: Identifier of the disconnected node.
    ///   - at: Timestamp of the disconnect event.
    public func trackDisconnect(nodeId: String, at: Date) {
        disconnectTimes[nodeId] = at
    }

    /// Check whether a node is still within the grace period for transient disconnects.
    ///
    /// - Parameters:
    ///   - nodeId: Identifier of the node to check.
    ///   - at: The current time to evaluate against.
    /// - Returns: `true` if the node disconnected within the last ``gracePeriodSeconds`` seconds.
    public func isInGracePeriod(nodeId: String, at: Date) -> Bool {
        guard let disconnectTime = disconnectTimes[nodeId] else {
            return false
        }
        return at.timeIntervalSince(disconnectTime) <= Self.gracePeriodSeconds
    }

    /// Remove disconnect tracking for a node (e.g., after it reconnects or is replaced).
    ///
    /// - Parameter nodeId: Identifier of the node to clear.
    public func clearDisconnect(nodeId: String) {
        disconnectTimes.removeValue(forKey: nodeId)
    }

    // MARK: - L2 Spare Node Swap

    /// Select the best spare node to take over a failed shard assignment.
    ///
    /// Filters spares to those with enough total memory to handle the failed assignment's
    /// memory estimate, then returns the spare with the largest capacity.
    ///
    /// - Parameters:
    ///   - failedAssignment: The shard assignment that needs replacement.
    ///   - spares: Available spare nodes.
    /// - Returns: The largest-capacity spare that can handle the memory, or `nil` if none qualify.
    public func selectSpareFor(
        failedAssignment: ShardAssignment,
        spares: [NodeSpec]
    ) -> NodeSpec? {
        let eligible = spares.filter { $0.totalMemoryBytes >= failedAssignment.memoryEstimate }
        return eligible.max(by: { $0.totalMemoryBytes < $1.totalMemoryBytes })
    }

    // MARK: - L3a Auto-Reshard

    /// Check whether the remaining healthy nodes can collectively hold the model.
    ///
    /// - Parameters:
    ///   - remainingNodes: Nodes still available in the cluster.
    ///   - totalModelMemory: Total memory (in bytes) required by the model weights.
    ///   - overheadFactor: Multiplier applied to `totalModelMemory` to account for KV cache,
    ///     activations, and runtime overhead. Defaults to 1.3.
    /// - Returns: `true` if the combined memory of remaining nodes meets the requirement.
    public static func canReshard(
        remainingNodes: [NodeSpec],
        totalModelMemory: UInt64,
        overheadFactor: Double = 1.3
    ) -> Bool {
        guard !remainingNodes.isEmpty else { return false }
        let combinedMemory = remainingNodes.reduce(UInt64(0)) { $0 + $1.totalMemoryBytes }
        let required = Double(totalModelMemory) * overheadFactor
        return Double(combinedMemory) >= required
    }

    /// Compute a new shard plan distributing layers across the remaining nodes.
    ///
    /// Uses the existing ``ShardPlan/init(profiles:nodes:strategy:)`` initializer to
    /// compute a proportional distribution.
    ///
    /// - Parameters:
    ///   - remainingNodes: Healthy nodes available for the new plan.
    ///   - profiles: Layer profiles describing the model's layer structure.
    /// - Returns: A new ``ShardPlan``, or `nil` if no nodes are available.
    public func computeReshardPlan(
        remainingNodes: [NodeSpec],
        profiles: [LayerProfile]
    ) -> ShardPlan? {
        guard !remainingNodes.isEmpty else { return nil }
        return ShardPlan(profiles: profiles, nodes: remainingNodes, strategy: .minNodes)
    }

    // MARK: - L3b Hard Fail

    /// Handle an unrecoverable failure for a model.
    ///
    /// This is a placeholder for admin notification logic (e.g., logging, alerting).
    ///
    /// - Parameters:
    ///   - modelId: Identifier of the model that failed.
    ///   - reason: Human-readable description of the failure.
    public func handleHardFail(modelId: String, reason: String) {
        // Placeholder: in production this would send an admin notification,
        // log to a monitoring system, etc.
    }
}
