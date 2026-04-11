import Foundation
import CoreImage
import MLX
import MLXNN
import MLXRandom
import MLXLLM
import MLXLMCommon
import Tokenizers
import Hub
import NovaMLXCore
import NovaMLXUtils
import NovaMLXPrefixCache

public final class ModelContainer: @unchecked Sendable {
    public let identifier: ModelIdentifier
    public let config: ModelConfig
    public private(set) var mlxContainer: MLXLMCommon.ModelContainer?
    public private(set) var tokenizer: Tokenizer?
    public private(set) var isLoaded: Bool

    public init(identifier: ModelIdentifier, config: ModelConfig) {
        self.identifier = identifier
        self.config = config
        self.isLoaded = false
    }

    public func setLoaded(mlxContainer: MLXLMCommon.ModelContainer, tokenizer: Tokenizer) {
        self.mlxContainer = mlxContainer
        self.tokenizer = tokenizer
        self.isLoaded = true
        NovaMLXLog.info("Model loaded: \(identifier.displayName)")
    }

    public func unload() {
        mlxContainer = nil
        tokenizer = nil
        isLoaded = false
        NovaMLXLog.info("Model unloaded: \(identifier.displayName)")
    }
}

public struct Tokenizer: Sendable {
    public let encode: @Sendable (String) -> [Int]
    public let decode: @Sendable ([Int]) -> String
    public let eosToken: String?
    public let eosTokenId: Int?

    public init(
        encode: @Sendable @escaping (String) -> [Int],
        decode: @Sendable @escaping ([Int]) -> String,
        eosToken: String?,
        eosTokenId: Int?
    ) {
        self.encode = encode
        self.decode = decode
        self.eosToken = eosToken
        self.eosTokenId = eosTokenId
    }
}

public final class MLXEngine: InferenceEngineProtocol, @unchecked Sendable {
    public private(set) var isReady: Bool

    public let pool: EnginePool
    public let metricsStore: MetricsStore
    public let sessionManager: ChatSessionManager
    private var prefixCacheManagers: [String: PrefixCacheManager]
    private var activeCount: Int
    private var activeTasks: [UUID: Task<Void, Never>]
    private let lock = NovaMLXLock()

    private var totalRequests: UInt64
    private var totalTokensGenerated: UInt64
    private var totalInferenceTime: TimeInterval

    public init(kvCacheCapacity: Int = 1024, maxMemoryMB: UInt64 = 2048, maxConcurrent: Int = 8, metricsStore: MetricsStore? = nil) {
        self.isReady = false
        self.pool = EnginePool(maxMemoryMB: maxMemoryMB)
        self.metricsStore = metricsStore ?? MetricsStore(baseDirectory: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!.appendingPathComponent("NovaMLX"))
        self.sessionManager = ChatSessionManager()
        self.prefixCacheManagers = [:]
        self.activeCount = 0
        self.activeTasks = [:]
        self.totalRequests = 0
        self.totalTokensGenerated = 0
        self.totalInferenceTime = 0
        NovaMLXLog.info("MLX Engine initialized (maxMemory: \(maxMemoryMB)MB)")
    }

    public func loadModel(from url: URL, config: ModelConfig) async throws -> ModelContainer {
        let container = ModelContainer(identifier: config.identifier, config: config)
        NovaMLXLog.info("Loading model from: \(url.path)")

        let modelConfig = ModelConfiguration(directory: url)
        let mlxContainer = try await LLMModelFactory.shared.loadContainer(configuration: modelConfig)

        let mlxTokenizer = await mlxContainer.tokenizer
        let eosString = mlxTokenizer.eosToken
        let tokenizer = Tokenizer(
            encode: { text in mlxTokenizer.encode(text: text) },
            decode: { tokens in mlxTokenizer.decode(tokens: tokens) },
            eosToken: eosString,
            eosTokenId: mlxTokenizer.eosTokenId
        )

        container.setLoaded(mlxContainer: mlxContainer, tokenizer: tokenizer)
        pool.add(container)
        metricsStore.recordModelLoad()
        return container
    }

    public func unloadModel(_ identifier: ModelIdentifier) {
        _ = pool.remove(identifier.id)
        metricsStore.recordModelUnload()
        MLX.Memory.clearCache()
        NovaMLXLog.info("GPU cache cleared after unloading \(identifier.displayName)")
    }

    public func getContainer(for modelId: String) -> ModelContainer? {
        pool.get(modelId)
    }

    public func getPrefixCacheStats(for modelId: String) -> PrefixCacheStats? {
        lock.withLock { prefixCacheManagers[modelId]?.getStats() }
    }

    public func clearPrefixCache(for modelId: String) {
        lock.withLock { prefixCacheManagers[modelId]?.clear() }
    }

    private func getOrCreatePrefixCacheManager(modelId: String) -> PrefixCacheManager? {
        lock.withLock {
            if let existing = prefixCacheManagers[modelId] {
                return existing
            }
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let ssdDir = appSupport.appendingPathComponent("NovaMLX/prefix_cache/\(modelId.replacingOccurrences(of: "/", with: "_"))", isDirectory: true)
            let config = PrefixCacheConfig(
                blockSize: 64,
                maxBlocks: 4096,
                initialBlocks: 256,
                ssdCacheDir: ssdDir,
                ssdMaxSizeBytes: 100 * 1024 * 1024 * 1024
            )
            let manager = PrefixCacheManager(config: config, modelName: modelId)
            prefixCacheManagers[modelId] = manager
            return manager
        }
    }

    private func tokenizeMessages(_ messages: [ChatMessage], tokenizer: Tokenizer) -> [Int] {
        var allTokens: [Int] = []
        for msg in messages {
            if let content = msg.content {
                allTokens.append(contentsOf: tokenizer.encode(content))
            }
        }
        return allTokens
    }

    private func buildGenerateParameters(
        request: InferenceRequest, config: ModelConfig
    ) -> GenerateParameters {
        let temperature = Float(request.temperature ?? config.temperature)
        let maxTokens = request.maxTokens ?? config.maxTokens
        let topP = Float(request.topP ?? config.topP)
        let topK = request.topK ?? 0
        let minP = request.minP ?? 0.0

        var params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature > 0 ? temperature : 0.6,
            topP: topP,
            topK: topK,
            minP: minP
        )

        let repPenalty = request.repetitionPenalty ?? (config.repeatPenalty != 1.0 ? config.repeatPenalty : nil)
        if let repPenalty, repPenalty != 1.0 {
            params.repetitionPenalty = repPenalty
            params.repetitionContextSize = config.repeatLastN
        }

        if let freq = request.frequencyPenalty, freq != 0 {
            params.frequencyPenalty = freq
        }

        if let pres = request.presencePenalty, pres != 0 {
            params.presencePenalty = pres
        }

        if let kvBits = config.kvBits {
            params.kvBits = kvBits
            params.kvGroupSize = config.kvGroupSize
        }

        if let seed = request.seed ?? config.seed {
            MLXRandom.seed(seed)
        }

        return params
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        guard let container = getContainer(for: request.model) else {
            throw NovaMLXError.modelNotFound(request.model)
        }
        guard let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }

        lock.withLock { activeCount += 1 }
        defer { lock.withLock { activeCount -= 1 } }

        let startTime = Date()

        let hasImages = request.messages.contains { msg in
            guard let imgs = msg.images, !imgs.isEmpty else { return false }
            return true
        }

        let userInput: UserInput
        if hasImages {
            let chatMessages = request.messages.compactMap { msg -> Chat.Message? in
                let images: [UserInput.Image] = (msg.images ?? []).compactMap { urlString -> UserInput.Image? in
                    if urlString.hasPrefix("data:") {
                        let parts = urlString.split(separator: ",", maxSplits: 1)
                        guard parts.count == 2 else { return nil }
                        let base64Str = String(parts[1])
                        guard let data = Data(base64Encoded: base64Str) else { return nil }
                        guard let ciImage = CIImage(data: data) else { return nil }
                        return .ciImage(ciImage)
                    } else if urlString.hasPrefix("http") {
                        guard let url = URL(string: urlString) else { return nil }
                        return .url(url)
                    } else {
                        return .url(URL(fileURLWithPath: urlString))
                    }
                }
                return .user(msg.content ?? "", images: images)
            }
            userInput = UserInput(prompt: .chat(chatMessages))
        } else {
            let messages: [Message] = request.messages.map { msg in
                ["role": msg.role.rawValue, "content": msg.content ?? ""]
            }
            userInput = UserInput(prompt: .messages(messages))
        }

        let input = try await mlxContainer.prepare(input: userInput)

        let promptTokenCount = input.text.tokens.size
        NovaMLXLog.info("Encoded \(promptTokenCount) tokens via chat template")
        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Starting generation for \(request.model)")

        let prefixCacheTokens: [Int]? = if let _ = getOrCreatePrefixCacheManager(modelId: request.model), promptTokenCount > 0 {
            input.text.tokens.asArray(Int32.self).map { Int($0) }
        } else {
            nil
        }

        let parameters = buildGenerateParameters(request: request, config: container.config)
        let stream = try await mlxContainer.generate(input: input, parameters: parameters)

        var generatedText = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                completionTokens += 1
            case .info(let info):
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall: break
            }
        }

        if let tokenArray = prefixCacheTokens,
           let prefixCache = getOrCreatePrefixCacheManager(modelId: request.model) {
            prefixCache.storeCache(tokenIds: tokenArray, cache: [])
        }

        MLX.Memory.clearCache()

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        lock.withLock {
            totalRequests += 1
            totalTokensGenerated += UInt64(completionTokens)
            totalInferenceTime += elapsed
        }

        metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

        NovaMLXLog.info("Generated \(completionTokens) tokens (\(String(format: "%.1f", tps)) tok/s)")
        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Completed: \(completionTokens) tokens, \(String(format: "%.1f", tps)) tok/s")

        return InferenceResult(
            id: request.id, model: request.model, text: generatedText,
            tokensPerSecond: tps, promptTokens: promptTokenCount, completionTokens: completionTokens,
            finishReason: finishReason
        )
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let container = self.getContainer(for: request.model) else {
                        throw NovaMLXError.modelNotFound(request.model)
                    }
                    guard let mlxContainer = container.mlxContainer else {
                        throw NovaMLXError.inferenceFailed("Model not loaded")
                    }

                    self.lock.withLock { self.activeCount += 1 }

                    let hasImages = request.messages.contains { msg in
                        guard let imgs = msg.images, !imgs.isEmpty else { return false }
                        return true
                    }

                    let userInput: UserInput
                    if hasImages {
                        let chatMessages = request.messages.compactMap { msg -> Chat.Message? in
                            let images: [UserInput.Image] = (msg.images ?? []).compactMap { urlString -> UserInput.Image? in
                                if urlString.hasPrefix("data:") {
                                    let parts = urlString.split(separator: ",", maxSplits: 1)
                                    guard parts.count == 2 else { return nil }
                                    let base64Str = String(parts[1])
                                    guard let data = Data(base64Encoded: base64Str) else { return nil }
                                    guard let ciImage = CIImage(data: data) else { return nil }
                                    return .ciImage(ciImage)
                                } else if urlString.hasPrefix("http") {
                                    guard let url = URL(string: urlString) else { return nil }
                                    return .url(url)
                                } else {
                                    return .url(URL(fileURLWithPath: urlString))
                                }
                            }
                            return .user(msg.content ?? "", images: images)
                        }
                        userInput = UserInput(prompt: .chat(chatMessages))
                    } else {
                        let messages: [Message] = request.messages.map { msg in
                            ["role": msg.role.rawValue, "content": msg.content ?? ""]
                        }
                        userInput = UserInput(prompt: .messages(messages))
                    }

                    let input = try await mlxContainer.prepare(input: userInput)

                    let parameters = self.buildGenerateParameters(request: request, config: container.config)
                    let stream = try await mlxContainer.generate(input: input, parameters: parameters)

                    var finishReason: FinishReason = .stop
                    var completionTokens = 0
                    let startTime = Date()

                    for await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            completionTokens += 1
                            continuation.yield(Token(id: 0, text: text))
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall: break
                        }
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    self.lock.withLock {
                        self.totalRequests += 1
                        self.totalTokensGenerated += UInt64(completionTokens)
                        self.totalInferenceTime += elapsed
                    }

                    self.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                    continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }

                self.lock.withLock { self.activeCount -= 1 }
                MLX.Memory.clearCache()
            }

            self.lock.withLock { self.activeTasks[request.id] = task }

            continuation.onTermination = { _ in
                task.cancel()
                _ = self.lock.withLock { self.activeTasks.removeValue(forKey: request.id) }
            }
        }
    }

    public func abort(requestId: UUID) {
        lock.withLock {
            activeTasks[requestId]?.cancel()
            activeTasks.removeValue(forKey: requestId)
        }
        NovaMLXLog.info("Aborting request: \(requestId)")
    }

    public var currentActiveCount: Int { lock.withLock { activeCount } }
    public var loadedModelCount: Int { pool.loadedModelCount }
    public func listLoadedModels() -> [String] { pool.loadedModelIds }
    public var gpuActiveMemory: UInt64 { UInt64(MLX.Memory.activeMemory) }

    public func saveSession(_ sessionId: String) async throws {
        try await sessionManager.saveSession(sessionId)
    }

    public func hasCachedSession(_ sessionId: String) -> Bool {
        FileManager.default.fileExists(atPath: sessionManager.cacheURL(for: sessionId).path)
    }

    public func forkSession(from sourceId: String, into targetId: String, modelId: String) async throws {
        guard let container = getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(modelId)
        }
        try await sessionManager.fork(
            from: sourceId,
            into: targetId,
            mlxContainer: mlxContainer,
            kvBits: container.config.kvBits,
            kvGroupSize: container.config.kvGroupSize
        )
    }

    public func generateWithSession(
        _ request: InferenceRequest,
        sessionId: String
    ) async throws -> InferenceResult {
        guard let container = getContainer(for: request.model),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(request.model)
        }

        let lastUserMessage = request.messages.last(where: { $0.role == .user })?.content ?? ""
        let systemPrompt = request.messages.first(where: { $0.role == .system })?.content

        let sessionBox = sessionManager.getOrCreate(
            sessionId: sessionId,
            mlxContainer: mlxContainer,
            modelId: request.model,
            systemPrompt: systemPrompt,
            kvBits: container.config.kvBits,
            kvGroupSize: container.config.kvGroupSize
        )

        let startTime = Date()
        var generatedText = ""
        var promptTokens = 0
        var completionTokens = 0
        var finishReason: FinishReason = .stop

        let stream = sessionBox.streamDetails(to: lastUserMessage)
        for try await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall: break
            }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        lock.withLock {
            totalRequests += 1
            totalTokensGenerated += UInt64(completionTokens)
            totalInferenceTime += elapsed
        }

        metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

        return InferenceResult(
            id: request.id, model: request.model, text: generatedText,
            tokensPerSecond: tps, promptTokens: promptTokens,
            completionTokens: completionTokens, finishReason: finishReason
        )
    }

    public func streamWithSession(
        _ request: InferenceRequest,
        sessionId: String
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let engine = self

            let task = Task {
                do {
                    guard let container = engine.getContainer(for: request.model),
                          let mlxContainer = container.mlxContainer else {
                        throw NovaMLXError.modelNotFound(request.model)
                    }

                    let lastUserMessage = request.messages.last(where: { $0.role == .user })?.content ?? ""
                    let systemPrompt = request.messages.first(where: { $0.role == .system })?.content

                    let sessionBox = engine.sessionManager.getOrCreate(
                        sessionId: sessionId,
                        mlxContainer: mlxContainer,
                        modelId: request.model,
                        systemPrompt: systemPrompt,
                        kvBits: container.config.kvBits,
                        kvGroupSize: container.config.kvGroupSize
                    )

                    let startTime = Date()
                    var finishReason: FinishReason = .stop
                    var completionTokens = 0

                    let stream = sessionBox.streamDetails(to: lastUserMessage)
                    for try await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            completionTokens += 1
                            continuation.yield(Token(id: 0, text: text))
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall: break
                        }
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    engine.lock.withLock {
                        engine.totalRequests += 1
                        engine.totalTokensGenerated += UInt64(completionTokens)
                        engine.totalInferenceTime += elapsed
                    }

                    engine.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                    continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public var metrics: EngineMetrics {
        lock.withLock {
            EngineMetrics(
                totalRequests: totalRequests,
                totalTokensGenerated: totalTokensGenerated,
                totalInferenceTime: totalInferenceTime,
                averageTokensPerSecond: totalInferenceTime > 0 ? Double(totalTokensGenerated) / totalInferenceTime : 0,
                activeRequests: activeCount
            )
        }
    }

    public func resetSessionMetrics() {
        lock.withLock {
            totalRequests = 0
            totalTokensGenerated = 0
            totalInferenceTime = 0
        }
    }

    public func generateWithProcessor(
        _ request: InferenceRequest
    ) async throws -> InferenceResult {
        guard let container = getContainer(for: request.model) else {
            throw NovaMLXError.modelNotFound(request.model)
        }
        guard let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }

        let processor = JSONLogitProcessor(tokenizer: container.tokenizer!)

        var systemMessages = request.messages.filter { $0.role == .system }
        if request.responseFormat == .jsonObject {
            systemMessages.append(ChatMessage(role: .system, content: "You must respond with valid JSON only. Do not include any text, explanation, or markdown outside the JSON object."))
        }
        let allMessages = systemMessages + request.messages.filter { $0.role != .system }
        let mappedMessages: [Message] = allMessages.map { msg in
            ["role": msg.role.rawValue, "content": msg.content ?? ""]
        }

        let userInput = UserInput(prompt: .messages(mappedMessages))
        let input = try await mlxContainer.prepare(input: userInput)

        let promptTokenCount = input.text.tokens.size

        let sampler = TopPSampler(
            temperature: Float(request.temperature ?? container.config.temperature),
            topP: Float(request.topP ?? container.config.topP)
        )

        let model = await mlxContainer.perform { context in SendableBox(context.model) }
        let tokenizer = await mlxContainer.tokenizer
        let config = await mlxContainer.configuration

        let iterator = try TokenIterator(
            input: input, model: model.value,
            processor: processor, sampler: sampler,
            maxTokens: request.maxTokens ?? container.config.maxTokens
        )

        let (stream, task) = generateTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: config,
            tokenizer: tokenizer,
            iterator: iterator
        )

        let startTime = Date()
        var generatedText = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                completionTokens += 1
            case .info(let info):
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall: break
            }
        }
        _ = task

        MLX.Memory.clearCache()

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        lock.withLock {
            totalRequests += 1
            totalTokensGenerated += UInt64(completionTokens)
            totalInferenceTime += elapsed
        }

        metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

        return InferenceResult(
            id: request.id, model: request.model, text: generatedText,
            tokensPerSecond: tps, promptTokens: promptTokenCount,
            completionTokens: completionTokens, finishReason: finishReason
        )
    }

    public func streamWithProcessor(
        _ request: InferenceRequest
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let engine = self

            let task = Task {
                do {
                    guard let container = engine.getContainer(for: request.model),
                          let mlxContainer = container.mlxContainer else {
                        throw NovaMLXError.modelNotFound(request.model)
                    }

                    let processor = JSONLogitProcessor(tokenizer: container.tokenizer!)

                    var systemMessages = request.messages.filter { $0.role == .system }
                    if request.responseFormat == .jsonObject {
                        systemMessages.append(ChatMessage(role: .system, content: "You must respond with valid JSON only. Do not include any text, explanation, or markdown outside the JSON object."))
                    }
                    let allMessages = systemMessages + request.messages.filter { $0.role != .system }
                    let mappedMessages: [Message] = allMessages.map { msg in
                        ["role": msg.role.rawValue, "content": msg.content ?? ""]
                    }

                    let userInput = UserInput(prompt: .messages(mappedMessages))
                    let input = try await mlxContainer.prepare(input: userInput)
                    let promptTokenCount = input.text.tokens.size

                    let sampler = TopPSampler(
                        temperature: Float(request.temperature ?? container.config.temperature),
                        topP: Float(request.topP ?? container.config.topP)
                    )

                    let model = await mlxContainer.perform { context in SendableBox(context.model) }
                    let tokenizer = await mlxContainer.tokenizer
                    let config = await mlxContainer.configuration

                    let iterator = try TokenIterator(
                        input: input, model: model.value,
                        processor: processor, sampler: sampler,
                        maxTokens: request.maxTokens ?? container.config.maxTokens
                    )

                    let (stream, genTask) = generateTask(
                        promptTokenCount: promptTokenCount,
                        modelConfiguration: config,
                        tokenizer: tokenizer,
                        iterator: iterator
                    )

                    let startTime = Date()
                    var finishReason: FinishReason = .stop
                    var completionTokens = 0

                    for await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            completionTokens += 1
                            continuation.yield(Token(id: 0, text: text))
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall: break
                        }
                    }
                    _ = genTask

                    let elapsed = Date().timeIntervalSince(startTime)
                    engine.lock.withLock {
                        engine.totalRequests += 1
                        engine.totalTokensGenerated += UInt64(completionTokens)
                        engine.totalInferenceTime += elapsed
                    }

                    engine.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                    MLX.Memory.clearCache()

                    continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

public struct EngineMetrics: Sendable {
    public let totalRequests: UInt64
    public let totalTokensGenerated: UInt64
    public let totalInferenceTime: TimeInterval
    public let averageTokensPerSecond: Double
    public let activeRequests: Int

    public init(totalRequests: UInt64 = 0, totalTokensGenerated: UInt64 = 0, totalInferenceTime: TimeInterval = 0, averageTokensPerSecond: Double = 0, activeRequests: Int = 0) {
        self.totalRequests = totalRequests
        self.totalTokensGenerated = totalTokensGenerated
        self.totalInferenceTime = totalInferenceTime
        self.averageTokensPerSecond = averageTokensPerSecond
        self.activeRequests = activeRequests
    }
}
