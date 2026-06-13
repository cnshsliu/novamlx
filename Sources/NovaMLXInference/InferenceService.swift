import Foundation
import MLX
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXModelManager
import NovaMLXDistributed
import AsyncAlgorithms

/// Per-model load dedup. Concurrent callers for the same modelId await a
/// single underlying task. Cleanup is synchronous in the actor's isolation
/// context so a failed load releases its slot before the error propagates,
/// allowing immediate retry by piled-up callers.
actor LoadDedup {
    private var inFlight: [String: Task<Void, Error>] = [:]

    /// Number of in-flight loads (test/observability hook).
    var inFlightCount: Int { inFlight.count }

    func ensureSingle(
        modelId: String,
        work: @Sendable @escaping () async throws -> Void
    ) async throws {
        if let existing = inFlight[modelId] {
            try await existing.value
            return
        }
        let task = Task<Void, Error> { try await work() }
        inFlight[modelId] = task
        defer { inFlight.removeValue(forKey: modelId) }
        try await task.value
    }
}

public final class InferenceService: @unchecked Sendable {
    public let engine: MLXEngine
    private let batcher: ContinuousBatcher
    private let fusedScheduler: FusedBatchScheduler
    public let settingsManager: ModelSettingsManager
    public let transcriptionService: TranscriptionService
    public let ttsService: TTSService
    public let imageGenerationService: ImageGenerationService

    // Worker subprocess mode
    public let workerMode: Bool
    private var worker: WorkerSupervisor?
    private var workerLoadedModels: Set<String> = []
    private var workerModelTypes: [String: ModelType] = [:]
    private var workerHybridModels: Set<String> = []
    private let loadDedup = LoadDedup()
    private var ttlSweepTask: Task<Void, Never>?

    // Cluster distributed inference mode
    private let clusterMode: Bool
    private var distributedRunner: DistributedInferenceRunner?

    public init(engine: MLXEngine, settingsManager: ModelSettingsManager, maxBatchSize: Int = 8, workerMode: Bool = false, workerBinaryPath: String? = nil, clusterMode: Bool = false, clusterConfig: ClusterConfig? = nil) {
        self.engine = engine
        self.batcher = ContinuousBatcher(engine: engine, maxBatchSize: maxBatchSize)
        self.fusedScheduler = FusedBatchScheduler(engine: engine, maxConcurrentPerModel: 4)
        self.settingsManager = settingsManager
        self.transcriptionService = TranscriptionService()
        self.ttsService = TTSService()
        self.imageGenerationService = ImageGenerationService()
        self.workerMode = workerMode
        self.clusterMode = clusterMode

        if workerMode, let path = workerBinaryPath {
            self.worker = WorkerSupervisor(workerBinaryPath: path)
        }

        if clusterMode, let config = clusterConfig {
            self.distributedRunner = DistributedInferenceRunner(
                clusterConfig: config,
                tokenizerProvider: { [weak self] modelId in
                    guard let self = self else { return nil }
                    let container = self.engine.getContainer(for: modelId)
                    guard let tokenizer = container?.tokenizer else { return nil }
                    return DistributedTokenizer(
                        encode: { text in tokenizer.encode(text) },
                        decode: { tokens in tokenizer.decode(tokens) }
                    )
                },
                modelPathProvider: { modelId in
                    let path = NovaMLXPaths.modelsDir.appendingPathComponent(modelId).path
                    var isDir: ObjCBool = false
                    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
                        return nil
                    }
                    return path
                },
                engine: engine
            )
        }

        engine.settingsProvider = { [settingsManager] modelId in
            settingsManager.getSettings(modelId)
        }

        // TTL sweep: auto-evict models that exceed their idle TTL every 60s
        ttlSweepTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                guard let self = self else { return }
                self.checkTTLExpirations()
            }
        }
    }

    // MARK: - Worker Lifecycle

    public func startWorker() throws {
        guard workerMode, let worker = worker else { return }
        worker.onCrash = { [weak self] in
            guard let self = self else { return }
            NovaMLXLog.warning("[InferenceService] Worker crashed — clearing in-memory loaded models state")
            self.workerLoadedModels.removeAll()
            self.workerModelTypes.removeAll()
            self.workerHybridModels.removeAll()
            // Do NOT saveLoadedModelsList() here — on crash/exit, the persisted file
            // should keep the model list so restoreModels() can reload on next launch.
            // The terminationHandler fires on normal app exit too, which would wipe the list.
        }
        try worker.start()
        NovaMLXLog.info("[InferenceService] Worker mode started")
    }

    public func stopWorker() {
        worker?.stop()
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let settings = settingsManager.getSettings(resolvedId)
        var finalRequest = settings.applySamplingOverrides(to: request)
        finalRequest = InferenceRequest(
            id: finalRequest.id, model: resolvedId, messages: finalRequest.messages,
            tools: finalRequest.tools,
            temperature: finalRequest.temperature, maxTokens: finalRequest.maxTokens,
            topP: finalRequest.topP, topK: finalRequest.topK, minP: finalRequest.minP,
            frequencyPenalty: finalRequest.frequencyPenalty, presencePenalty: finalRequest.presencePenalty,
            repetitionPenalty: finalRequest.repetitionPenalty, seed: finalRequest.seed,
            stream: finalRequest.stream, stop: finalRequest.stop,
            sessionId: finalRequest.sessionId,
            responseFormat: finalRequest.responseFormat,
            jsonSchemaDef: finalRequest.jsonSchemaDef,
            regexPattern: finalRequest.regexPattern,
            gbnfGrammar: finalRequest.gbnfGrammar,
            thinkingBudget: finalRequest.thinkingBudget,
            enableThinking: finalRequest.enableThinking,
            preserveThinking: finalRequest.preserveThinking,
            draftModel: finalRequest.draftModel,
            numDraftTokens: finalRequest.numDraftTokens
        )
        finalRequest = autoInjectDraftModel(finalRequest)

        // Cluster mode: check readiness, then route to distributed inference
        if clusterMode, let runner = distributedRunner {
            let modelManager = ClusterModelManager.shared
            let clusterState = modelManager.getStatus()

            switch clusterState.state {
            case .idle:
                // No model activated for distributed — fall through to local inference
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster idle, using local inference")
            case .activating:
                let (ready, total) = clusterState.readinessFraction
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Rejected: cluster preparing (\(ready)/\(total) nodes ready)")
                throw DistributedInferenceError.shardPlanFailed("Cluster is preparing model '\(clusterState.activeModel ?? "?")' (\(ready)/\(total) nodes ready). Please wait.")
            case ClusterModelState.failed:
                // Failed — fall through to local inference
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster failed, using local inference")
            case .ready:
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] -> Distributed (model=\(resolvedId))")
                return try await runner.generate(request: finalRequest)
            }
        }

        // Worker mode: route through subprocess
        if workerMode, let worker = worker {
            let result = try await worker.sendGenerate(finalRequest)
            if result.completionTokens > 0 {
                engine.metricsStore.recordRequest(
                    model: resolvedId,
                    tokens: UInt64(result.completionTokens),
                    inferenceTime: result.tokensPerSecond > 0 ? Double(result.completionTokens) / result.tokensPerSecond : 0
                )
            }
            return result
        }

        let reqTag = finalRequest.id.uuidString.prefix(8)

        // Context window check moved to engine/scheduler layer after tokenization,
        // where we have accurate token counts (includes chat template tokens).
        let container = engine.getContainer(for: resolvedId)

        // VLM models use ContinuousBatcher — they have complex internal position state
        // (3D mRoPE, precomputedPositionIds, ropeDeltas) that our fused decode step
        // can't replicate with a simple model(token, cache:) call.
        let isVLM = container?.config.modelType == .vlm || workerModelTypes[resolvedId] == .vlm

        // Hybrid linear attention models (e.g. Qwen3.5) mix MambaCache + KVCacheSimple layers.
        // FusedBatchScheduler only supports KVCacheSimple — hybrid models must use ContinuousBatcher.
        let hasLinearAttention = container?.config.hasLinearAttention == true

        // Session, grammar, VLM, hybrid, and draft-model paths use engine directly (specialized execution)
        let hasDraftModel = finalRequest.draftModel != nil
        let needsSpecialized = finalRequest.sessionId != nil ||
            finalRequest.jsonSchemaDef != nil ||
            finalRequest.responseFormat == .jsonObject ||
            finalRequest.regexPattern != nil ||
            finalRequest.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention ||
            hasDraftModel

        if needsSpecialized {
            let reason = isVLM ? "VLM" : (hasLinearAttention ? "hybrid" : (hasDraftModel ? "draft-model" : (finalRequest.sessionId != nil ? "session" : "grammar")))
            NovaMLXLog.info("[Route:\(reqTag)] → ContinuousBatcher (reason=\(reason), model=\(resolvedId))")
            return try await batcher.submit(finalRequest)
        }

        // LLM standard path: fused batch scheduler (shared GPU forward passes)
        NovaMLXLog.info("[Route:\(reqTag)] → FusedBatchScheduler (model=\(resolvedId))")
        return try await fusedScheduler.submit(finalRequest)
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let settings = settingsManager.getSettings(resolvedId)
        var finalRequest = settings.applySamplingOverrides(to: request)
        finalRequest = InferenceRequest(
            id: finalRequest.id, model: resolvedId, messages: finalRequest.messages,
            tools: finalRequest.tools,
            temperature: finalRequest.temperature, maxTokens: finalRequest.maxTokens,
            topP: finalRequest.topP, topK: finalRequest.topK, minP: finalRequest.minP,
            frequencyPenalty: finalRequest.frequencyPenalty, presencePenalty: finalRequest.presencePenalty,
            repetitionPenalty: finalRequest.repetitionPenalty, seed: finalRequest.seed,
            stream: true, stop: finalRequest.stop,
            sessionId: finalRequest.sessionId,
            responseFormat: finalRequest.responseFormat,
            jsonSchemaDef: finalRequest.jsonSchemaDef,
            regexPattern: finalRequest.regexPattern,
            gbnfGrammar: finalRequest.gbnfGrammar,
            thinkingBudget: finalRequest.thinkingBudget,
            enableThinking: finalRequest.enableThinking,
            preserveThinking: finalRequest.preserveThinking,
            draftModel: finalRequest.draftModel,
            numDraftTokens: finalRequest.numDraftTokens
        )
        finalRequest = autoInjectDraftModel(finalRequest)

        // Cluster mode: route to distributed streaming when cluster is ready
        if clusterMode, let runner = distributedRunner {
            let modelManager = ClusterModelManager.shared
            let clusterState = modelManager.getStatus()

            switch clusterState.state {
            case .idle:
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster idle, using local inference for stream")
            case .activating:
                let (ready, total) = clusterState.readinessFraction
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Rejected: cluster preparing (\(ready)/\(total) nodes ready)")
                return AsyncThrowingStream { $0.finish(throwing: DistributedInferenceError.shardPlanFailed("Cluster is preparing model '\(clusterState.activeModel ?? "?")' (\(ready)/\(total) nodes ready). Please wait.")) }
            case .failed:
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster failed, using local inference for stream")
            case .ready:
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] -> Distributed stream (model=\(resolvedId))")
                return runner.stream(request: finalRequest)
            }
        }

        // Worker mode: route through subprocess
        if workerMode, let worker = worker {
            let tracker = StreamTracker(model: resolvedId, metricsStore: engine.metricsStore)
            let upstream = worker.sendStream(finalRequest)
            return AsyncThrowingStream { continuation in
                let task = Task { @Sendable in
                    do {
                        for try await token in upstream {
                            tracker.increment()
                            continuation.yield(token)
                        }
                        tracker.finish()
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        }

        let reqTag = finalRequest.id.uuidString.prefix(8)

        // Context window check moved to engine/scheduler layer after tokenization,
        // where we have accurate token counts (includes chat template tokens).
        let container = engine.getContainer(for: resolvedId)

        // VLM models use ContinuousBatcher — they have complex internal position state
        // (3D mRoPE, precomputedPositionIds, ropeDeltas) that our fused decode step
        // can't replicate with a simple model(token, cache:) call.
        let isVLM = container?.config.modelType == .vlm || workerModelTypes[resolvedId] == .vlm

        // Hybrid linear attention models (e.g. Qwen3.5) mix MambaCache + KVCacheSimple layers.
        // FusedBatchScheduler only supports KVCacheSimple — hybrid models must use ContinuousBatcher.
        let hasLinearAttention = container?.config.hasLinearAttention == true

        // Session, grammar, VLM, hybrid, and draft-model paths use engine directly (specialized execution)
        let hasDraftModel = finalRequest.draftModel != nil
        let needsSpecialized = finalRequest.sessionId != nil ||
            finalRequest.jsonSchemaDef != nil ||
            finalRequest.responseFormat == .jsonObject ||
            finalRequest.regexPattern != nil ||
            finalRequest.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention ||
            hasDraftModel

        if needsSpecialized {
            let reason = isVLM ? "VLM" : (hasLinearAttention ? "hybrid" : (hasDraftModel ? "draft-model" : (finalRequest.sessionId != nil ? "session" : "grammar")))
            NovaMLXLog.info("[Route:\(reqTag)] → ContinuousBatcher stream (reason=\(reason), model=\(resolvedId))")
            return batcher.submitStream(finalRequest)
        }

        // LLM standard path: fused batch scheduler (shared GPU forward passes)
        NovaMLXLog.info("[Route:\(reqTag)] → FusedBatchScheduler stream (model=\(resolvedId))")
        return fusedScheduler.submitStream(finalRequest)
    }

    public func abort(requestId: UUID) async {
        if workerMode, let worker = worker {
            worker.sendAbort(requestId: requestId)
        } else {
            engine.abort(requestId: requestId)
            batcher.abort(requestId: requestId)
            fusedScheduler.abort(requestId: requestId)
        }
    }

    // MARK: - Auto Speculative Decoding (Speed Boost)

    /// Automatically inject draft model for speculative decoding when a compatible
    /// draft model is loaded. Skipped if user explicitly sets draftModel, or for
    /// hybrid/distributed/cluster requests.
    private func autoInjectDraftModel(_ request: InferenceRequest) -> InferenceRequest {
        // User explicitly specified — respect their choice
        guard request.draftModel == nil else { return request }

        // Skip when model is actively served by distributed runner.
        // When cluster is idle (no model activated), requests fall through to
        // local inference where speculative decoding works fine.
        if clusterMode, let runner = distributedRunner {
            let clusterState = ClusterModelManager.shared.getStatus()
            if clusterState.state == .ready, clusterState.activeModel == settingsManager.resolveModelId(request.model) {
                return request
            }
        }

        guard isModelLoaded(request.model) else { return request }
        guard !isHybridModel(request.model) else { return request }

        // Look up recommendation from registry
        let family: ModelFamily
        if let container = engine.getContainer(for: request.model) {
            family = container.identifier.family
        } else {
            // Worker mode: engine pool is empty, infer family from model ID
            let id = settingsManager.resolveModelId(request.model).lowercased()
            if id.contains("qwen") { family = .qwen }
            else if id.contains("llama") { family = .llama }
            else if id.contains("gemma") { family = .gemma }
            else if id.contains("mistral") { family = .mistral }
            else if id.contains("phi") { family = .phi }
            else { return request }
        }

        guard let candidate = DraftModelRegistry.shared.recommendation(
            family: family,
            isHybrid: false
        ) else { return request }

        guard isModelLoaded(candidate.draftModelId) else { return request }

        // Validate vocab_size match from config.json on disk
        let mainId = settingsManager.resolveModelId(request.model)
        let mainDir = NovaMLXPaths.modelsDir.appendingPathComponent(mainId)
        let draftDir = NovaMLXPaths.modelsDir.appendingPathComponent(candidate.draftModelId)
        guard let mainVocab = DraftModelRegistry.readVocabSize(from: mainDir),
              let draftVocab = DraftModelRegistry.readVocabSize(from: draftDir),
              mainVocab == draftVocab else {
            NovaMLXLog.warning("[SpecBoost] Vocab mismatch for \(request.model), skipping auto-injection")
            return request
        }

        NovaMLXLog.info("[SpecBoost] Auto-injecting draft '\(candidate.draftModelId)' for '\(request.model)'")
        return InferenceRequest(
            id: request.id, model: request.model, messages: request.messages,
            tools: request.tools,
            temperature: request.temperature, maxTokens: request.maxTokens,
            topP: request.topP, topK: request.topK, minP: request.minP,
            frequencyPenalty: request.frequencyPenalty, presencePenalty: request.presencePenalty,
            repetitionPenalty: request.repetitionPenalty, seed: request.seed,
            stream: request.stream, stop: request.stop,
            sessionId: request.sessionId,
            responseFormat: request.responseFormat,
            jsonSchemaDef: request.jsonSchemaDef,
            regexPattern: request.regexPattern,
            gbnfGrammar: request.gbnfGrammar,
            thinkingBudget: request.thinkingBudget,
            enableThinking: request.enableThinking,
            preserveThinking: request.preserveThinking,
            draftModel: candidate.draftModelId,
            numDraftTokens: request.numDraftTokens ?? 4
        )
    }

    public func loadModel(at url: URL, config: ModelConfig, progress: (@Sendable (LoadPhase) -> Void)? = nil) async throws {
        let modelId = config.identifier.id

        // Audio models bypass engine entirely — route to specialized services
        if config.modelType == .audio {
            NovaMLXLog.info("[InferenceService] Loading audio model: \(modelId), family=\(config.identifier.family), url=\(url.path)")
            try await loadDedup.ensureSingle(modelId: modelId) {
                let family = config.identifier.family
                if family == .qwen3Tts || family == .dotsTts {
                    NovaMLXLog.info("[InferenceService] Routing to TTSService for \(modelId)")
                    do {
                        try await self.ttsService.loadModel(from: url)
                        NovaMLXLog.info("[InferenceService] TTS model loaded successfully: \(modelId)")
                    } catch {
                        NovaMLXLog.error("[InferenceService] TTS model load FAILED for \(modelId): \(error)")
                        throw error
                    }
                    self.saveLoadedModelsList()
                    return
                }
                NovaMLXLog.info("[InferenceService] Routing to TranscriptionService for \(modelId)")
                _ = try await self.transcriptionService.loadModel(from: url, config: config, progress: progress)
                self.saveLoadedModelsList()
            }
            return
        }

        try await loadDedup.ensureSingle(modelId: modelId) { [self] in
            if self.workerMode, let worker = self.worker {
                let isHybrid = try await worker.sendLoad(modelId: modelId, path: url.path, config: config, progress: progress)
                self.workerLoadedModels.insert(modelId)
                self.workerModelTypes[modelId] = config.modelType
                if isHybrid {
                    self.workerHybridModels.insert(modelId)
                }
            } else {
                _ = try await self.engine.loadModel(from: url, config: config, progress: progress)
                let settings = self.settingsManager.getSettings(modelId)
                if settings.isPinned {
                    self.engine.pool.pin(modelId)
                }
            }
            self.saveLoadedModelsList()
        }
    }

    public func unloadModel(_ identifier: ModelIdentifier) async {
        // TTS models
        if identifier.family == .qwen3Tts {
            ttsService.unloadModel()
            saveLoadedModelsList()
            return
        }
        if workerMode, let worker = worker {
            try? await worker.sendUnload(modelId: identifier.id)
            workerLoadedModels.remove(identifier.id)
            workerModelTypes.removeValue(forKey: identifier.id)
            workerHybridModels.remove(identifier.id)
        } else {
            engine.unloadModel(identifier)
        }
        saveLoadedModelsList()
    }

    public func isModelLoaded(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        if workerMode {
            return workerLoadedModels.contains(resolvedId)
        }
        if engine.getContainer(for: resolvedId)?.isLoaded == true { return true }
        if transcriptionService.isLoaded(resolvedId) { return true }
        if ttsService.listLoadedModels().contains(resolvedId) { return true }
        if imageGenerationService.isLoaded(resolvedId) { return true }
        return false
    }

    public func isHybridModel(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        if workerMode {
            return workerHybridModels.contains(resolvedId)
        }
        return engine.getContainer(for: resolvedId)?.config.hasLinearAttention ?? false
    }

    /// Check if a model can be loaded given current memory constraints.
    /// Works in both direct and worker mode — uses Metal device info directly.
    public func checkMemoryFeasibility(modelId: String, sizeBytes: UInt64, localURL: URL) async -> MemoryFeasibility? {
        if isModelLoaded(modelId) { return nil }

        let maxGPU = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? 0
        guard maxGPU > 0 else { return nil }

        let estimatedBytes = MLXEngine.estimateModelWeightSize(at: localURL) ?? sizeBytes
        let currentBytes = UInt64(MLX.Memory.activeMemory)
        let available = currentBytes < maxGPU ? maxGPU - currentBytes : 0

        return MemoryFeasibility.evaluate(
            modelId: modelId,
            modelSizeBytes: estimatedBytes,
            currentlyAvailableBytes: available,
            gpuBudgetBytes: maxGPU
        )
    }

    public func listLoadedModels() -> [String] {
        var models: [String]
        if workerMode {
            models = Array(workerLoadedModels)
        } else {
            models = engine.listLoadedModels()
        }
        // Include audio models loaded in transcriptionService
        models.append(contentsOf: transcriptionService.listLoadedModels().filter { !models.contains($0) })
        // Include TTS models
        models.append(contentsOf: ttsService.listLoadedModels().filter { !models.contains($0) })
        // Include image models
        models.append(contentsOf: imageGenerationService.listLoadedModels().filter { !models.contains($0) })
        return models
    }

    public func checkTTLExpirations() {
        let allInfo = engine.pool.allModelInfo()
        let now = Date()
        for info in allInfo {
            if info.pinned { continue }

            if let prd = info.perRequestDeadline {
                if prd > now { continue }
                let identifier = ModelIdentifier(id: info.id, family: .other)
                engine.unloadModel(identifier)
                NovaMLXLog.debug("Per-request keep_alive expired for \(info.id), unloaded")
                continue
            }

            let settings = settingsManager.getSettings(info.id)
            guard let ttl = settings.ttlSeconds, ttl > 0 else { continue }
            let idleTime = now.timeIntervalSince(info.lastAccessed)
            if idleTime >= Double(ttl) {
                let identifier = ModelIdentifier(id: info.id, family: .other)
                engine.unloadModel(identifier)
                NovaMLXLog.debug("TTL expired for \(info.id), unloaded after \(ttl)s idle")
            }
        }
        saveLoadedModelsList()
    }

    // MARK: - Loaded Models Persistence

    public func saveLoadedModelsList() {
        let ids = listLoadedModels()
        try? NovaDB.shared.loadedModelsStore.replaceAll(with: ids)
    }

    private func loadLoadedModelsList() -> [String] {
        (try? NovaDB.shared.loadedModelsStore.list()) ?? []
    }

    public func restoreModels(modelManager: ModelManager) async {
        let ids = loadLoadedModelsList()
        guard !ids.isEmpty else { return }
        NovaMLXLog.info("[InferenceService] Restoring \(ids.count) previously loaded model(s)...")
        for modelId in ids {
            guard let record = modelManager.getRecord(modelId) else {
                NovaMLXLog.warning("[InferenceService] Skipping restore of '\(modelId)' — not found in registry")
                continue
            }
            let config = ModelConfig(
                identifier: ModelIdentifier(id: modelId, family: record.family),
                modelType: record.modelType
            )
            do {
                try await loadModel(at: record.localURL, config: config)
                NovaMLXLog.info("[InferenceService] Restored model: \(modelId) (type: \(record.modelType))")
            } catch {
                NovaMLXLog.warning("[InferenceService] Failed to restore model \(modelId): \(error)")
            }
        }
        saveLoadedModelsList()
    }

    public var stats: InferenceStats {
        let gpuMem: UInt64
        if workerMode, let worker = worker, let memStats = worker.latestMemoryStats {
            gpuMem = memStats.currentBytes
        } else {
            gpuMem = engine.gpuActiveMemory
        }
        let modelCount = workerMode ? workerLoadedModels.count : engine.loadedModelCount
        let activeReqs = workerMode
            ? (worker?.activeRequestCount ?? 0)
            : (batcher.activeRequests + fusedScheduler.activeRequestCount)
        let workerCpu = workerMode ? (worker?.latestMemoryStats?.cpuUsage ?? 0) : 0
        return InferenceStats(
            loadedModels: modelCount,
            activeRequests: activeReqs,
            gpuMemoryUsed: gpuMem,
            recentTokensPerSecond: engine.metricsStore.recentTokensPerSecond,
            totalTokensGenerated: engine.metricsStore.metrics.totalTokensAllTime,
            workerCpuUsage: workerCpu
        )
    }

    public var batcherMetrics: BatcherMetrics {
        batcher.metrics
    }

    public var fusedSchedulerMetrics: FusedSchedulerMetrics {
        fusedScheduler.metrics
    }

    public func resolveModelId(_ input: String) -> String {
        settingsManager.resolveModelId(input)
    }

    public func forkSession(from sourceId: String, into targetId: String, modelId: String) async throws {
        let resolvedModelId = settingsManager.resolveModelId(modelId)
        try await engine.forkSession(from: sourceId, into: targetId, modelId: resolvedModelId)
    }

    public func countTokens(model: String, messages: [ChatMessage]) -> Int? {
        let resolvedId = settingsManager.resolveModelId(model)
        guard let container = engine.getContainer(for: resolvedId),
              let tokenizer = container.tokenizer else { return nil }
        var total = 0
        for msg in messages {
            if let content = msg.content {
                total += tokenizer.encode(content).count
            }
        }
        return total
    }

    public func getContextWindow(for modelId: String) -> Int? {
        let resolvedId = settingsManager.resolveModelId(modelId)
        let settings = settingsManager.getSettings(resolvedId)
        return settings.maxContextWindow
    }
}

private final class StreamTracker: @unchecked Sendable {
    let model: String
    let metricsStore: MetricsStore
    let startTime = Date()
    var tokenCount = 0
    private var lastLiveUpdate: Date = Date()

    init(model: String, metricsStore: MetricsStore) {
        self.model = model
        self.metricsStore = metricsStore
    }

    func increment() {
        tokenCount += 1
        let now = Date()
        // Update live TPS every second during streaming to keep the status chart fresh
        if now.timeIntervalSince(lastLiveUpdate) >= 1.0 {
            let elapsed = now.timeIntervalSince(startTime)
            if elapsed > 0.01 {
                metricsStore.updateLiveTps(Double(tokenCount) / elapsed)
            }
            lastLiveUpdate = now
        }
    }

    func finish() {
        let elapsed = Date().timeIntervalSince(startTime)
        if tokenCount > 0 && elapsed > 0 {
            metricsStore.recordRequest(model: model, tokens: UInt64(tokenCount), inferenceTime: elapsed)
        }
    }
}

public struct InferenceStats: Sendable {
    public let loadedModels: Int
    public let activeRequests: Int
    public let gpuMemoryUsed: UInt64
    public let recentTokensPerSecond: Double
    public let totalTokensGenerated: UInt64
    public let workerCpuUsage: Double
    public init(loadedModels: Int = 0, activeRequests: Int = 0, gpuMemoryUsed: UInt64 = 0, recentTokensPerSecond: Double = 0, totalTokensGenerated: UInt64 = 0, workerCpuUsage: Double = 0) {
        self.loadedModels = loadedModels
        self.activeRequests = activeRequests
        self.gpuMemoryUsed = gpuMemoryUsed
        self.recentTokensPerSecond = recentTokensPerSecond
        self.totalTokensGenerated = totalTokensGenerated
        self.workerCpuUsage = workerCpuUsage
    }
}
