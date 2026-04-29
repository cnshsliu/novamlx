import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXModelManager
import AsyncAlgorithms

public final class InferenceService: @unchecked Sendable {
    public let engine: MLXEngine
    private let batcher: ContinuousBatcher
    private let fusedScheduler: FusedBatchScheduler
    public let settingsManager: ModelSettingsManager
    private let loadedModelsFile: URL

    // Worker subprocess mode
    public let workerMode: Bool
    private var worker: WorkerSupervisor?
    private var workerLoadedModels: Set<String> = []

    public init(engine: MLXEngine, settingsManager: ModelSettingsManager, maxBatchSize: Int = 8, workerMode: Bool = false, workerBinaryPath: String? = nil) {
        self.engine = engine
        self.batcher = ContinuousBatcher(engine: engine, maxBatchSize: maxBatchSize)
        self.fusedScheduler = FusedBatchScheduler(engine: engine, maxConcurrentPerModel: 4)
        self.settingsManager = settingsManager
        self.loadedModelsFile = NovaMLXPaths.loadedModelsFile
        self.workerMode = workerMode

        if workerMode, let path = workerBinaryPath {
            self.worker = WorkerSupervisor(workerBinaryPath: path)
        }

        engine.settingsProvider = { [settingsManager] modelId in
            settingsManager.getSettings(modelId)
        }
    }

    // MARK: - Worker Lifecycle

    public func startWorker() throws {
        guard workerMode, let worker = worker else { return }
        worker.onCrash = { [weak self] in
            guard let self = self else { return }
            NovaMLXLog.warning("[InferenceService] Worker crashed — clearing loaded models state")
            self.workerLoadedModels.removeAll()
            self.saveLoadedModelsList()
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
            thinkingBudget: finalRequest.thinkingBudget
        )

        // Cloud mode: proxy to remote inference engine
        if CloudBackend.isCloudModel(resolvedId) {
            NovaMLXLog.info("[Cloud] → proxy generate (model=\(resolvedId))")
            return try await CloudBackend.shared.proxy(finalRequest)
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

        // Context window check — fail fast if prompt exceeds model's context length.
        // This gives the client an immediate error instead of hanging through prefill.
        let container = engine.getContainer(for: resolvedId)
        if let tokenizer = container?.tokenizer {
            let promptTokens = engine.tokenizeMessages(finalRequest.messages, tokenizer: tokenizer)
            let contextLength = settings.maxContextWindow ?? container?.config.contextLength ?? 4096
            let maxTokens = finalRequest.maxTokens ?? container?.config.maxTokens ?? 4096
            if promptTokens.count + maxTokens > contextLength {
                NovaMLXLog.warning("[ContextWindow:\(reqTag)] Prompt (\(promptTokens.count) tokens) + maxTokens (\(maxTokens)) exceeds context length (\(contextLength))")
                throw NovaMLXError.contextWindowExceeded(
                    promptTokens: promptTokens.count, maxTokens: maxTokens, contextLength: contextLength
                )
            }
        }

        // VLM models use ContinuousBatcher — they have complex internal position state
        // (3D mRoPE, precomputedPositionIds, ropeDeltas) that our fused decode step
        // can't replicate with a simple model(token, cache:) call.
        let isVLM = container?.config.modelType == .vlm

        // Hybrid linear attention models (e.g. Qwen3.5) mix MambaCache + KVCacheSimple layers.
        // FusedBatchScheduler only supports KVCacheSimple — hybrid models must use ContinuousBatcher.
        let hasLinearAttention = container?.config.hasLinearAttention == true

        // Session, grammar, and VLM paths use engine directly (specialized execution)
        let needsSpecialized = finalRequest.sessionId != nil ||
            finalRequest.jsonSchemaDef != nil ||
            finalRequest.responseFormat == .jsonObject ||
            finalRequest.regexPattern != nil ||
            finalRequest.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention

        if needsSpecialized {
            let reason = isVLM ? "VLM" : (hasLinearAttention ? "hybrid" : (finalRequest.sessionId != nil ? "session" : "grammar"))
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
            thinkingBudget: finalRequest.thinkingBudget
        )

        // Cloud mode: proxy stream to remote inference engine
        if CloudBackend.isCloudModel(resolvedId) {
            NovaMLXLog.info("[Cloud] → proxy stream (model=\(resolvedId))")
            return CloudBackend.proxyStreamStatic(finalRequest)
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

        // Context window check — fail fast with an error stream instead of hanging.
        let container = engine.getContainer(for: resolvedId)
        if let tokenizer = container?.tokenizer {
            let promptTokens = engine.tokenizeMessages(finalRequest.messages, tokenizer: tokenizer)
            let contextLength = settings.maxContextWindow ?? container?.config.contextLength ?? 4096
            let maxTokens = finalRequest.maxTokens ?? container?.config.maxTokens ?? 4096
            if promptTokens.count + maxTokens > contextLength {
                NovaMLXLog.warning("[ContextWindow:\(reqTag)] Prompt (\(promptTokens.count) tokens) + maxTokens (\(maxTokens)) exceeds context length (\(contextLength))")
                return AsyncThrowingStream {
                    $0.finish(throwing: NovaMLXError.contextWindowExceeded(
                        promptTokens: promptTokens.count, maxTokens: maxTokens, contextLength: contextLength
                    ))
                }
            }
        }

        // VLM models use ContinuousBatcher — they have complex internal position state
        // (3D mRoPE, precomputedPositionIds, ropeDeltas) that our fused decode step
        // can't replicate with a simple model(token, cache:) call.
        let isVLM = container?.config.modelType == .vlm

        // Hybrid linear attention models (e.g. Qwen3.5) mix MambaCache + KVCacheSimple layers.
        // FusedBatchScheduler only supports KVCacheSimple — hybrid models must use ContinuousBatcher.
        let hasLinearAttention = container?.config.hasLinearAttention == true

        // Session, grammar, and VLM paths use engine directly (specialized execution)
        let needsSpecialized = finalRequest.sessionId != nil ||
            finalRequest.jsonSchemaDef != nil ||
            finalRequest.responseFormat == .jsonObject ||
            finalRequest.regexPattern != nil ||
            finalRequest.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention

        if needsSpecialized {
            let reason = isVLM ? "VLM" : (hasLinearAttention ? "hybrid" : (finalRequest.sessionId != nil ? "session" : "grammar"))
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
        }
    }

    public func loadModel(at url: URL, config: ModelConfig) async throws {
        let modelId = config.identifier.id

        if workerMode, let worker = worker {
            try await worker.sendLoad(modelId: modelId, path: url.path, config: config)
            workerLoadedModels.insert(modelId)
        } else {
            _ = try await engine.loadModel(from: url, config: config)
            let settings = settingsManager.getSettings(modelId)
            if settings.isPinned {
                engine.pool.pin(modelId)
            }
        }
        saveLoadedModelsList()
    }

    public func unloadModel(_ identifier: ModelIdentifier) async {
        if workerMode, let worker = worker {
            try? await worker.sendUnload(modelId: identifier.id)
            workerLoadedModels.remove(identifier.id)
        } else {
            engine.unloadModel(identifier)
        }
        saveLoadedModelsList()
    }

    public func isModelLoaded(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        // Cloud models are always "loaded" (available remotely)
        // Check both with and without :cloud suffix (Anthropic clients may omit it)
        if CloudBackend.isCloudModel(resolvedId) || CloudBackend.isCloudModel(resolvedId + ":cloud") {
            return true
        }
        if workerMode {
            return workerLoadedModels.contains(resolvedId)
        }
        guard let container = engine.getContainer(for: resolvedId) else { return false }
        return container.isLoaded
    }

    public func listLoadedModels() -> [String] {
        var models: [String]
        if workerMode {
            models = Array(workerLoadedModels)
        } else {
            models = engine.listLoadedModels()
        }
        return models
    }

    /// Return cloud model names (with ":cloud" suffix) from cached discovery
    public func listCloudModels() async -> [String] {
        await CloudBackend.shared.getModels()
    }

    public func checkTTLExpirations() {
        let allInfo = engine.pool.allModelInfo()
        let now = Date()
        for info in allInfo {
            if info.pinned { continue }
            let settings = settingsManager.getSettings(info.id)
            guard let ttl = settings.ttlSeconds, ttl > 0 else { continue }
            let idleTime = now.timeIntervalSince(info.lastAccessed)
            if idleTime >= Double(ttl) {
                let identifier = ModelIdentifier(id: info.id, family: .other)
                engine.unloadModel(identifier)
                NovaMLXLog.info("TTL expired for \(info.id), unloaded after \(ttl)s idle")
            }
        }
        saveLoadedModelsList()
    }

    // MARK: - Loaded Models Persistence

    private func saveLoadedModelsList() {
        let ids = workerMode ? Array(workerLoadedModels) : engine.pool.loadedModelIds
        guard let data = try? JSONEncoder().encode(ids) else { return }
        try? data.write(to: loadedModelsFile, options: .atomic)
    }

    private func loadLoadedModelsList() -> [String] {
        guard let data = try? Data(contentsOf: loadedModelsFile),
              let ids = try? JSONDecoder().decode([String].self, from: data) else { return [] }
        return ids
    }

    public func restoreModels(modelManager: ModelManager) async {
        let ids = loadLoadedModelsList()
        guard !ids.isEmpty else { return }
        NovaMLXLog.info("Restoring \(ids.count) previously loaded model(s)...")
        for modelId in ids {
            guard let record = modelManager.getRecord(modelId) else {
                NovaMLXLog.warning("Skipping restore of '\(modelId)' — not found in registry")
                continue
            }
            let config = ModelConfig(
                identifier: ModelIdentifier(id: modelId, family: record.family),
                modelType: record.modelType
            )
            do {
                try await loadModel(at: record.localURL, config: config)
                NovaMLXLog.info("Restored model: \(modelId)")
            } catch {
                NovaMLXLog.warning("Failed to restore model \(modelId): \(error)")
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
