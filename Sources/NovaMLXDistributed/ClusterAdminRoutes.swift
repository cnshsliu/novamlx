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
    /// Works for both coordinator and worker nodes.
    public func clusterStatus() -> [String: Any] {
        let manager = ClusterManager.shared
        let worker = WorkerService.shared
        var status: [String: Any] = [:]

        if let config = manager.config {
            // Coordinator node
            status["isRunning"] = manager.isRunning
            status["config"] = [
                "role": config.role.rawValue,
                "coordinatorHost": config.coordinatorHost,
                "coordinatorPort": config.coordinatorPort,
                "strategy": config.strategy.rawValue,
            ]
            // Show all non-disconnected workers (ready + active + loading + syncing)
            let allWorkers = manager.workers.values
                .filter { $0.status != .disconnected && $0.status != .failed }
                .sorted { $0.registeredAt < $1.registeredAt }
            status["workers"] = allWorkers.map { [
                "nodeId": $0.nodeId,
                "status": $0.status.rawValue,
                "hostname": $0.spec.hostname,
                "port": $0.spec.port,
                "memory": $0.spec.totalMemoryBytes,
                "cpuModel": $0.spec.cpuModel,
                "lastHeartbeat": ISO8601DateFormatter().string(from: $0.lastHeartbeat),
            ] }
        } else if worker.isRunning {
            // Worker node — report own status (passive mode, Coordinator polls us)
            let spec = worker.collectLocalSpec()
            status["isRunning"] = true
            status["config"] = [
                "role": "worker",
            ]
            status["localSpec"] = [
                "nodeId": spec.nodeId,
                "hostname": spec.hostname,
                "port": spec.port,
                "memory": spec.totalMemoryBytes,
                "cpuModel": spec.cpuModel,
            ]
            status["workers"] = []
        } else {
            status["isRunning"] = false
            status["workers"] = []
        }

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

    /// Current shard plan for a model, if one has been computed.
    public func currentShardPlan(modelId: String) -> [String: Any]? {
        guard let cached = DistributedInferenceRunnerCache.shared.lastPlan,
              cached.modelId == modelId else {
            return nil
        }
        let plan = cached.plan
        return [
            "modelId": modelId,
            "totalLayers": plan.totalLayers,
            "strategy": plan.strategy.rawValue,
            "shards": plan.assignments.map { [
                "nodeId": $0.nodeId,
                "startLayer": $0.startLayer,
                "endLayer": $0.endLayer,
                "layerCount": $0.layerCount,
                "memoryEstimate": $0.memoryEstimate,
            ] }
        ]
    }

    /// Wavefront prefill stats (placeholder until wired to live inference).
    public func wavefrontStats() -> [String: Any] {
        return ["status": "not_available"]
    }

    // MARK: - Model Activation

    /// Activate a model for distributed inference across the cluster.
    public func activateModel(modelId: String) async -> [String: Any] {
        do {
            let status = try await ClusterModelManager.shared.activateModel(modelId: modelId)
            return encodeModelStatus(status)
        } catch {
            return ["error": true, "message": error.localizedDescription]
        }
    }

    /// Deactivate the current distributed model.
    public func deactivateModel() async -> [String: Any] {
        do {
            let status = try await ClusterModelManager.shared.deactivateModel()
            return encodeModelStatus(status)
        } catch {
            return ["error": true, "message": error.localizedDescription]
        }
    }

    /// Get current cluster model status (active model, readiness, shard plan).
    public func modelStatus() -> [String: Any] {
        let status = ClusterModelManager.shared.getStatus()
        return encodeModelStatus(status)
    }

    private func encodeModelStatus(_ status: ClusterModelStatus) -> [String: Any] {
        var result: [String: Any] = [
            "activeModel": status.activeModel as Any,
            "state": status.state.rawValue,
        ]
        if let plan = status.shardPlan {
            result["shardPlan"] = [
                "totalLayers": plan.totalLayers,
                "strategy": plan.strategy.rawValue,
                "shards": plan.assignments.map { [
                    "nodeId": $0.nodeId,
                    "startLayer": $0.startLayer,
                    "endLayer": $0.endLayer,
                    "layerCount": $0.layerCount,
                    "memoryEstimate": $0.memoryEstimate,
                ] as [String: Any] }
            ]
        }
        let (ready, total) = status.readinessFraction
        result["readiness"] = ["ready": ready, "total": total]
        result["nodes"] = status.nodes.map { [
            "nodeId": $0.nodeId,
            "hostname": $0.hostname,
            "startLayer": $0.startLayer,
            "endLayer": $0.endLayer,
            "layerCount": $0.layerCount,
            "status": $0.status.rawValue,
            "memoryUsedBytes": $0.memoryUsedBytes,
            "errorMessage": $0.errorMessage as Any,
        ] as [String: Any] }
        return result
    }
}
