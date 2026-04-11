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

    public init(engine: MLXEngine, settingsManager: ModelSettingsManager, maxBatchSize: Int = 8) {
        self.engine = engine
        self.batcher = ContinuousBatcher(engine: engine, maxBatchSize: maxBatchSize)
        self.settingsManager = settingsManager
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        let resolvedId = settingsManager.resolveModelId(request.model)

        if let sessionId = request.sessionId {
            return try await generateWithSession(request, sessionId: sessionId)
        }

        if request.responseFormat == .jsonObject {
            let resolvedRequest = InferenceRequest(
                id: request.id, model: resolvedId, messages: request.messages,
                temperature: request.temperature, maxTokens: request.maxTokens,
                topP: request.topP, topK: request.topK, minP: request.minP,
                frequencyPenalty: request.frequencyPenalty, presencePenalty: request.presencePenalty,
                repetitionPenalty: request.repetitionPenalty, seed: request.seed,
                stream: false, stop: request.stop, responseFormat: .jsonObject,
                thinkingBudget: request.thinkingBudget
            )
            return try await engine.generateWithProcessor(resolvedRequest)
        }

        let settings = settingsManager.getSettings(resolvedId)
        let adjusted = settings.applySamplingOverrides(to: request)
        let finalRequest = InferenceRequest(
            id: adjusted.id, model: resolvedId, messages: adjusted.messages,
            temperature: adjusted.temperature, maxTokens: adjusted.maxTokens,
            topP: adjusted.topP, topK: adjusted.topK, minP: adjusted.minP,
            frequencyPenalty: adjusted.frequencyPenalty, presencePenalty: adjusted.presencePenalty,
            repetitionPenalty: adjusted.repetitionPenalty, seed: adjusted.seed,
            stream: adjusted.stream, stop: adjusted.stop,
            thinkingBudget: adjusted.thinkingBudget
        )
        return try await batcher.submit(finalRequest)
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let resolvedId = settingsManager.resolveModelId(request.model)

        if let sessionId = request.sessionId {
            return streamWithSession(request, sessionId: sessionId)
        }

        if request.responseFormat == .jsonObject {
            let resolvedRequest = InferenceRequest(
                id: request.id, model: resolvedId, messages: request.messages,
                temperature: request.temperature, maxTokens: request.maxTokens,
                topP: request.topP, topK: request.topK, minP: request.minP,
                frequencyPenalty: request.frequencyPenalty, presencePenalty: request.presencePenalty,
                repetitionPenalty: request.repetitionPenalty, seed: request.seed,
                stream: true, stop: request.stop, responseFormat: .jsonObject,
                thinkingBudget: request.thinkingBudget
            )
            return engine.streamWithProcessor(resolvedRequest)
        }

        let settings = settingsManager.getSettings(resolvedId)
        let adjusted = settings.applySamplingOverrides(to: request)
        let finalRequest = InferenceRequest(
            id: adjusted.id, model: resolvedId, messages: adjusted.messages,
            temperature: adjusted.temperature, maxTokens: adjusted.maxTokens,
            topP: adjusted.topP, topK: adjusted.topK, minP: adjusted.minP,
            frequencyPenalty: adjusted.frequencyPenalty, presencePenalty: adjusted.presencePenalty,
            repetitionPenalty: adjusted.repetitionPenalty, seed: adjusted.seed,
            stream: adjusted.stream, stop: adjusted.stop,
            thinkingBudget: adjusted.thinkingBudget
        )
        return batcher.submitStream(finalRequest)
    }

    private func generateWithSession(_ request: InferenceRequest, sessionId: String) async throws -> InferenceResult {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let resolvedRequest = InferenceRequest(
            id: request.id, model: resolvedId, messages: request.messages,
            temperature: request.temperature, maxTokens: request.maxTokens,
            topP: request.topP, topK: request.topK, minP: request.minP,
            frequencyPenalty: request.frequencyPenalty, presencePenalty: request.presencePenalty,
            repetitionPenalty: request.repetitionPenalty, seed: request.seed,
            stream: request.stream, stop: request.stop, sessionId: sessionId,
            thinkingBudget: request.thinkingBudget
        )
        return try await engine.generateWithSession(resolvedRequest, sessionId: sessionId)
    }

    private func streamWithSession(_ request: InferenceRequest, sessionId: String) -> AsyncThrowingStream<Token, Error> {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let resolvedRequest = InferenceRequest(
            id: request.id, model: resolvedId, messages: request.messages,
            temperature: request.temperature, maxTokens: request.maxTokens,
            topP: request.topP, topK: request.topK, minP: request.minP,
            frequencyPenalty: request.frequencyPenalty, presencePenalty: request.presencePenalty,
            repetitionPenalty: request.repetitionPenalty, seed: request.seed,
            stream: request.stream, stop: request.stop, sessionId: sessionId,
            thinkingBudget: request.thinkingBudget
        )
        return engine.streamWithSession(resolvedRequest, sessionId: sessionId)
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
    }

    public func unloadModel(_ identifier: ModelIdentifier) async {
        engine.unloadModel(identifier)
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
