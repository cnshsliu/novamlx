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
    public let settingsManager: ModelSettingsManager
    private let loadedModelsFile: URL

    public init(engine: MLXEngine, settingsManager: ModelSettingsManager, maxBatchSize: Int = 8) {
        self.engine = engine
        self.batcher = ContinuousBatcher(engine: engine, maxBatchSize: maxBatchSize)
        self.settingsManager = settingsManager
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.loadedModelsFile = appSupport.appendingPathComponent("NovaMLX/loaded_models.json")
        engine.settingsProvider = { [settingsManager] modelId in
            settingsManager.getSettings(modelId)
        }
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let settings = settingsManager.getSettings(resolvedId)
        var finalRequest = settings.applySamplingOverrides(to: request)
        finalRequest = InferenceRequest(
            id: finalRequest.id, model: resolvedId, messages: finalRequest.messages,
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

        // ALL requests go through the batcher for unified memory-aware admission.
        // Session, grammar, and JSON schema paths are handled inside engine.generate()
        // which dispatches to the appropriate method based on request fields.
        return try await batcher.submit(finalRequest)
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let settings = settingsManager.getSettings(resolvedId)
        var finalRequest = settings.applySamplingOverrides(to: request)
        finalRequest = InferenceRequest(
            id: finalRequest.id, model: resolvedId, messages: finalRequest.messages,
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

        return batcher.submitStream(finalRequest)
    }

    public func abort(requestId: UUID) async {
        engine.abort(requestId: requestId)
    }

    public func loadModel(at url: URL, config: ModelConfig) async throws {
        _ = try await engine.loadModel(from: url, config: config)
        let modelId = config.identifier.id
        let settings = settingsManager.getSettings(modelId)
        if settings.isPinned {
            engine.pool.pin(modelId)
        }
        saveLoadedModelsList()
    }

    public func unloadModel(_ identifier: ModelIdentifier) async {
        engine.unloadModel(identifier)
        saveLoadedModelsList()
    }

    public func isModelLoaded(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        guard let container = engine.getContainer(for: resolvedId) else { return false }
        return container.isLoaded
    }

    public func listLoadedModels() -> [String] {
        engine.listLoadedModels()
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
        let ids = engine.pool.loadedModelIds
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
        InferenceStats(
            loadedModels: engine.loadedModelCount,
            activeRequests: batcher.activeRequests,
            gpuMemoryUsed: engine.gpuActiveMemory
        )
    }

    public var batcherMetrics: BatcherMetrics {
        batcher.metrics
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

public struct InferenceStats: Sendable {
    public let loadedModels: Int
    public let activeRequests: Int
    public let gpuMemoryUsed: UInt64
    public init(loadedModels: Int = 0, activeRequests: Int = 0, gpuMemoryUsed: UInt64 = 0) {
        self.loadedModels = loadedModels
        self.activeRequests = activeRequests
        self.gpuMemoryUsed = gpuMemoryUsed
    }
}
