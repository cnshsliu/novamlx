import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine

// MARK: - ClusterModelError

public enum ClusterModelError: Error, LocalizedError {
    case modelAlreadyActive(String)
    case noModelActive
    case noWorkersAvailable
    case activationInProgress(String)
    case activationFailed(String)
    case notCoordinator

    public var errorDescription: String? {
        switch self {
        case .modelAlreadyActive(let m):
            "Model '\(m)' is already active. Deactivate it first."
        case .noModelActive:
            "No model is activated for distributed inference. Activate a model first."
        case .noWorkersAvailable:
            "No workers available in the cluster"
        case .activationInProgress(let m):
            "Model '\(m)' is currently activating. Please wait."
        case .activationFailed(let reason):
            "Activation failed: \(reason)"
        case .notCoordinator:
            "This node is not the coordinator"
        }
    }
}

// MARK: - ClusterModelManager

/// Manages cluster-wide model lifecycle: activation, readiness tracking, and deactivation.
///
/// Only one model can be active at a time. The activation flow:
/// 1. Coordinator computes ShardPlan via ModelAnalyzer
/// 2. Coordinator loads its own shard (SlicedForwardPolicy) into MLXEngine
/// 3. Coordinator sends shard assignments + model info to all Workers
/// 4. Each Worker loads its assigned layers and reports readiness
/// 5. When all nodes are ready, state transitions to `.ready`
///
/// Thread safety: all mutable state guarded by `lock`.
public final class ClusterModelManager: @unchecked Sendable {

    public static let shared = ClusterModelManager()

    private let lock = NSLock()

    /// Currently active model ID (nil if idle).
    public private(set) var activeModel: String?

    /// Current state machine state.
    public private(set) var state: ClusterModelState = .idle

    /// Per-node readiness status.
    public private(set) var nodeReadiness: [String: ShardReadiness] = [:]

    /// The shard plan for the active model.
    public private(set) var shardPlan: ShardPlan?

    /// Pre-created shard engines (populated after activation).
    public private(set) var shardEngines: [ShardEngine] = []

    /// Engine reference for local shard loading.
    private weak var engine: MLXEngine?

    /// Tokenizer provider for distributed runner.
    private var tokenizerProvider: ((String) -> DistributedTokenizer?)?

    /// Model path provider.
    private var modelPathProvider: ((String) -> String?)?

    private init() {}

    /// Configure with engine and providers. Called during app startup.
    public func configure(
        engine: MLXEngine,
        tokenizerProvider: @escaping (String) -> DistributedTokenizer?,
        modelPathProvider: @escaping (String) -> String?
    ) {
        lock.withLock {
            self.engine = engine
            self.tokenizerProvider = tokenizerProvider
            self.modelPathProvider = modelPathProvider
        }
    }

    // MARK: - Public API

    /// Activate a model for distributed inference across all cluster nodes.
    public func activateModel(modelId: String) async throws -> ClusterModelStatus {
        let (currentState, currentModel) = lock.withLock { (state, activeModel) }

        guard currentState == .idle || currentState == .failed else {
            if currentState == .activating {
                throw ClusterModelError.activationInProgress(currentModel ?? modelId)
            }
            throw ClusterModelError.modelAlreadyActive(currentModel ?? "unknown")
        }

        guard let engine = engine else {
            throw ClusterModelError.activationFailed("Engine not configured")
        }

        guard let modelPath = modelPathProvider?(modelId) else {
            throw ClusterModelError.activationFailed("Model path not found: \(modelId)")
        }

        lock.withLock {
            self.activeModel = modelId
            self.state = .activating
            self.nodeReadiness = [:]
            self.shardEngines = []
        }

        NovaMLXLog.info("[ClusterModel] Activating model: \(modelId)")

        do {
            let status = try await performActivation(modelId: modelId, modelPath: modelPath, engine: engine)
            return status
        } catch {
            lock.withLock {
                self.state = .failed
            }
            NovaMLXLog.error("[ClusterModel] Activation failed: \(error)")
            throw error
        }
    }

    /// Deactivate the current model. Releases all shards on all nodes.
    public func deactivateModel() async throws -> ClusterModelStatus {
        let (currentState, currentModel) = lock.withLock { (state, activeModel) }

        guard currentModel != nil else {
            throw ClusterModelError.noModelActive
        }

        NovaMLXLog.info("[ClusterModel] Deactivating model: \(currentModel ?? "unknown")")

        // Release local shard engines
        for shard in shardEngines {
            shard.policy.releaseWeights()
        }

        // Send releaseWeights to remote workers
        // TODO: This is handled by shard engine release — for now just clear state

        lock.withLock {
            self.activeModel = nil
            self.state = .idle
            self.nodeReadiness = [:]
            self.shardPlan = nil
            self.shardEngines = []
        }

        NovaMLXLog.info("[ClusterModel] Model deactivated")
        return getStatus()
    }

    /// Get current cluster model status.
    public func getStatus() -> ClusterModelStatus {
        lock.withLock {
            ClusterModelStatus(
                activeModel: activeModel,
                state: state,
                shardPlan: shardPlan,
                nodes: Array(nodeReadiness.values).sorted { $0.nodeId < $1.nodeId }
            )
        }
    }

    /// Check if cluster is ready for inference on a specific model.
    public func isReady(for modelId: String) -> Bool {
        lock.withLock {
            guard activeModel == modelId, state == .ready else { return false }
            return nodeReadiness.values.allSatisfy { $0.status == .ready }
        }
    }

    /// Get the pre-created shard engines for an active model.
    /// Returns nil if model is not active or not ready.
    public func getShardEngines(for modelId: String) -> [ShardEngine]? {
        lock.withLock {
            guard activeModel == modelId, state == .ready else { return nil }
            return shardEngines
        }
    }

    /// Update a node's readiness status (called when Worker reports back).
    public func updateNodeReadiness(nodeId: String, status: ShardLoadStatus, memoryUsedBytes: UInt64 = 0, errorMessage: String? = nil) {
        lock.withLock {
            guard var readiness = nodeReadiness[nodeId] else { return }
            readiness.status = status
            readiness.memoryUsedBytes = memoryUsedBytes
            readiness.errorMessage = errorMessage
            nodeReadiness[nodeId] = readiness

            // Check if all nodes are ready
            if status == .ready {
                let allReady = nodeReadiness.values.allSatisfy { $0.status == .ready }
                if allReady && state == .activating {
                    state = .ready
                    NovaMLXLog.info("[ClusterModel] All nodes ready! Model \(activeModel ?? "?") is active.")
                }
            }

            if status == .failed {
                let anyFailed = nodeReadiness.values.contains { $0.status == .failed }
                if anyFailed && state == .activating {
                    state = .failed
                    NovaMLXLog.error("[ClusterModel] Node \(nodeId) failed: \(errorMessage ?? "unknown")")
                }
            }
        }
    }

    // MARK: - Private

    private func performActivation(modelId: String, modelPath: String, engine: MLXEngine) async throws -> ClusterModelStatus {
        // 1. Profile model layers
        let profiles: [LayerProfile]
        do {
            profiles = try await ModelAnalyzer.shared.analyze(modelPath: modelPath)
        } catch {
            throw ClusterModelError.activationFailed("Model analysis failed: \(error)")
        }
        guard !profiles.isEmpty else {
            throw ClusterModelError.activationFailed("No layer profiles produced")
        }

        // 2. Ensure model is loaded in main engine for local shard
        if engine.getContainer(for: modelId) == nil {
            NovaMLXLog.info("[ClusterModel] Loading model \(modelId) into main engine...")
            let modelDir = URL(fileURLWithPath: modelPath)
            let config = ModelConfig(identifier: ModelIdentifier(id: modelId, family: .qwen))
            _ = try await engine.loadModel(from: modelDir, config: config)
            NovaMLXLog.info("[ClusterModel] Model \(modelId) loaded in main engine")
        }

        // 3. Build node list: Coordinator (rank 0) + available workers
        let clusterConfig = ClusterManager.shared.config
        let availableWorkers = ClusterManager.shared.workers.values
            .filter { $0.status == .ready || $0.status == .active }

        let localMemory = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? ProcessInfo.processInfo.physicalMemory
        let coordinatorSpec = NodeSpec(
            nodeId: "local-coordinator",
            totalMemoryBytes: localMemory,
            computeCapability: 1.0,
            hostname: "127.0.0.1",
            port: clusterConfig?.coordinatorPort ?? 6591
        )
        let effectiveNodes = [coordinatorSpec] + availableWorkers.map(\.spec)

        guard effectiveNodes.count >= 1 else {
            throw ClusterModelError.noWorkersAvailable
        }

        NovaMLXLog.info("[ClusterModel] Nodes: \(effectiveNodes.count) (coordinator=\(bytesFormatted(localMemory)), workers=\(availableWorkers.count))")

        // 4. Compute shard plan
        let strategy = clusterConfig?.strategy ?? .spread
        let minLayers = clusterConfig?.minLayersPerShard ?? 8
        let plan = ShardPlan(
            profiles: profiles,
            nodes: effectiveNodes,
            strategy: strategy,
            minLayersPerShard: minLayers
        )

        lock.withLock { self.shardPlan = plan }

        for (i, a) in plan.assignments.enumerated() {
            NovaMLXLog.info("[ClusterModel] Shard \(i): \(a.nodeId) layers \(a.startLayer)..<\(a.endLayer) (\(a.layerCount) layers, \(bytesFormatted(a.memoryEstimate)))")
        }

        // 5. Initialize node readiness tracking
        var readinessMap: [String: ShardReadiness] = [:]
        for (index, assignment) in plan.assignments.enumerated() {
            let hostname: String
            if assignment.nodeId == "local-coordinator" {
                hostname = "127.0.0.1"
            } else {
                hostname = availableWorkers.first { $0.spec.nodeId == assignment.nodeId }?.spec.hostname ?? assignment.nodeId
            }
            readinessMap[assignment.nodeId] = ShardReadiness(
                nodeId: assignment.nodeId,
                hostname: hostname,
                startLayer: assignment.startLayer,
                endLayer: assignment.endLayer,
                status: .pending
            )
        }
        lock.withLock { self.nodeReadiness = readinessMap }

        // 6. Initialize distributed group
        let group: DistributedGroup
        if MLXDistributedWrapper.isCBBackendAvailable {
            let backend = MLXDistributedWrapper.bestAvailableBackend()
            group = MLXDistributedWrapper.initialize(strict: false, backend: backend)
        } else {
            group = .uninitialized
        }

        // 7. Create shard engines and bind weights
        var engines: [ShardEngine] = []
        for (index, assignment) in plan.assignments.enumerated() {
            let isFirst = index == 0
            let isLast = index == plan.assignments.count - 1
            let policy: ComputePolicy

            if assignment.nodeId == "local-coordinator" {
                // Coordinator shard — load locally
                updateNodeReadiness(nodeId: assignment.nodeId, status: .loading)

                policy = SlicedForwardPolicy(
                    assignment: assignment,
                    engine: engine,
                    modelId: modelId,
                    isFirst: isFirst,
                    isLast: isLast
                )
                do {
                    try await policy.bindWeights()
                    updateNodeReadiness(nodeId: assignment.nodeId, status: .ready,
                                       memoryUsedBytes: assignment.memoryEstimate)
                    NovaMLXLog.info("[ClusterModel] Local shard ready: layers \(assignment.startLayer)..<\(assignment.endLayer)")
                } catch {
                    updateNodeReadiness(nodeId: assignment.nodeId, status: .failed,
                                       errorMessage: error.localizedDescription)
                    throw ClusterModelError.activationFailed("Local shard bind failed: \(error)")
                }
            } else {
                // Remote worker shard — send commands over TCP
                updateNodeReadiness(nodeId: assignment.nodeId, status: .loading)

                let worker = availableWorkers.first { $0.spec.nodeId == assignment.nodeId }
                let host = worker?.spec.networkHost ?? worker?.spec.hostname ?? assignment.nodeId
                let endpoint = NodeEndpoint(
                    nodeId: assignment.nodeId,
                    host: host,
                    port: 7010
                )
                policy = RemoteShardPolicy(
                    assignment: assignment,
                    workerEndpoint: endpoint,
                    modelId: modelId,
                    modelPath: modelPath,
                    isFirst: isFirst,
                    isLast: isLast
                )
                do {
                    try await policy.bindWeights()
                    updateNodeReadiness(nodeId: assignment.nodeId, status: .ready,
                                       memoryUsedBytes: assignment.memoryEstimate)
                    NovaMLXLog.info("[ClusterModel] Remote shard ready: \(host):7010 layers \(assignment.startLayer)..<\(assignment.endLayer)")
                } catch {
                    updateNodeReadiness(nodeId: assignment.nodeId, status: .failed,
                                       errorMessage: error.localizedDescription)
                    NovaMLXLog.error("[ClusterModel] Remote shard failed: \(host):7010 — \(error)")
                    // Don't throw — mark as failed but continue (partial activation)
                }
            }

            engines.append(ShardEngine(
                group: group,
                assignment: assignment,
                policy: policy
            ))
        }

        lock.withLock {
            self.shardEngines = engines
        }

        // 8. Check if all nodes ready
        let finalStatus = getStatus()
        if finalStatus.state == .ready {
            NovaMLXLog.info("[ClusterModel] Model \(modelId) activation complete — all shards ready")
        } else {
            NovaMLXLog.warning("[ClusterModel] Model \(modelId) activation partial — some shards failed")
        }

        return finalStatus
    }

    private func bytesFormatted(_ bytes: UInt64) -> String {
        String(format: "%.1fGB", Double(bytes) / 1e9)
    }
}
