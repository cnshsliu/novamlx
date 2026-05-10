import Foundation

/// Admin API routes for cluster observability.
///
/// Provides JSON snapshots of cluster state, worker discovery info,
/// and model weight sync progress. These routes are wired into the
/// admin HTTP server by ``NovaMLXAPI/APIServer``.
public final class ClusterAdminRoutes: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = ClusterAdminRoutes()

    private init() {}

    /// Aggregate cluster status: running state, configuration, and active workers.
    public func clusterStatus() -> [String: Any] {
        let manager = ClusterManager.shared
        var status: [String: Any] = [:]
        status["isRunning"] = manager.isRunning
        status["config"] = manager.config.map { [
            "role": $0.role.rawValue,
            "coordinatorHost": $0.coordinatorHost,
            "coordinatorPort": $0.coordinatorPort,
            "strategy": $0.strategy.rawValue,
        ] }
        status["workers"] = manager.activeWorkers.map { [
            "nodeId": $0.nodeId,
            "status": $0.status.rawValue,
            "memory": $0.spec.totalMemoryBytes,
            "lastHeartbeat": ISO8601DateFormatter().string(from: $0.lastHeartbeat),
        ] }
        return status
    }

    /// Detailed Bonjour discovery and worker registration debug info.
    public func discoveryDebug() -> [String: Any] {
        return ClusterManager.shared.discoveryDebugInfo()
    }

    /// Download progress for a model weight sync operation.
    ///
    /// - Parameter modelId: The model identifier to query.
    /// - Returns: A dictionary with sync progress, or `["status": "not_syncing"]`
    ///   if no download is in progress for the given model.
    public func modelSyncStatus(modelId: String) -> [String: Any] {
        guard let progress = WeightDistributor.shared.syncStatus(modelId: modelId) else {
            return ["status": "not_syncing"]
        }
        return [
            "status": progress.isComplete ? "complete" : "syncing",
            "bytesDownloaded": progress.bytesDownloaded,
            "totalBytes": progress.totalBytes,
            "progress": String(format: "%.1f%%", progress.fraction * 100),
        ]
    }

    /// Placeholder for the current shard plan of a model.
    ///
    /// Will be populated once the ShardEngine is wired to live inference.
    public func currentShardPlan(modelId: String) -> [String: Any]? {
        return nil
    }
}
