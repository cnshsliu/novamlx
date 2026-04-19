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
    public var config: ModelConfig
    public private(set) var mlxContainer: MLXLMCommon.ModelContainer?
    public private(set) var tokenizer: Tokenizer?
    public private(set) var isLoaded: Bool
    public private(set) var wiredReservationTicket: WiredMemoryTicket?
    public var kvMemoryOverride: Int?

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

    public func setWiredReservation(_ ticket: WiredMemoryTicket) {
        self.wiredReservationTicket = ticket
    }

    public func unload() {
        mlxContainer = nil
        tokenizer = nil
        isLoaded = false
        wiredReservationTicket = nil
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
    public let turboQuantService: TurboQuantService
    public let adapterService: AdapterService
    public let specDecoder: SpeculativeDecoder
    public let fusedBatchDecoder: FusedBatchDecoder
    public let budgetTracker: MemoryBudgetTracker
    public lazy var batchScheduler = BatchScheduler(engine: self)
    public var settingsProvider: (@Sendable (String) -> ModelSettings?)?
    private var prefixCacheManagers: [String: PrefixCacheManager]
    private var activeCount: Int
    private var activeTasks: [UUID: Task<Void, Never>]
    private let lock = NovaMLXLock()

    private var totalRequests: UInt64
    private var totalTokensGenerated: UInt64
    private var totalInferenceTime: TimeInterval

    private var deferredClearCounter: Int = 0
    private let deferredClearThreshold: Int = 2
    private let wiredPolicy = MLXLMCommon.WiredSumPolicy()

    public init(kvCacheCapacity: Int = 1024, maxMemoryMB: UInt64 = 2048, maxConcurrent: Int = 8, metricsStore: MetricsStore? = nil) {
        // Hard cap MLX at 50% of physical RAM — allocations beyond this will fail
        // rather than consuming all memory and crashing the system.
        let physMem = ProcessInfo.processInfo.physicalMemory
        let hardLimitBytes = physMem / 2
        MLX.Memory.memoryLimit = Int(hardLimitBytes)
        MLX.Memory.cacheLimit = 512 * 1024 * 1024  // 512MB compute cache cap

        self.isReady = false
        self.pool = EnginePool(maxMemoryMB: maxMemoryMB)
        self.metricsStore = metricsStore ?? MetricsStore(baseDirectory: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!.appendingPathComponent("NovaMLX"))
        self.sessionManager = ChatSessionManager()
        self.turboQuantService = TurboQuantService()
        self.adapterService = AdapterService()
        self.specDecoder = SpeculativeDecoder()
        self.fusedBatchDecoder = FusedBatchDecoder()
        self.budgetTracker = MemoryBudgetTracker(gpuLimitBytes: hardLimitBytes)
        self.prefixCacheManagers = [:]
        self.activeCount = 0
        self.activeTasks = [:]
        self.totalRequests = 0
        self.totalTokensGenerated = 0
        self.totalInferenceTime = 0
        FusedSDPARegistration.register()
        NovaMLXLog.info("MLX Engine initialized (maxMemory: \(maxMemoryMB)MB, hardLimit: \(hardLimitBytes / 1024 / 1024)MB)")
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

        let model = await mlxContainer.perform { context in SendableBox(context.model) }

        // Read model architecture params from config.json.
        // We need head_dim for KV memory calculation and max_position_embeddings
        // for context window enforcement. Every HuggingFace model has these.
        var headDim = 128
        let configFile = url.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configFile),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Multimodal models (Qwen3.5) store attention params in text_config
            let tc = (json["text_config"] as? [String: Any]) ?? json
            if let hd = tc["head_dim"] as? Int {
                headDim = hd
            } else if let hs = tc["hidden_size"] as? Int,
                      let nah = tc["num_attention_heads"] as? Int, nah > 0 {
                headDim = hs / nah
            }

            // Auto-detect context length from max_position_embeddings.
            // This is the standard HuggingFace field that every LLM model defines.
            // Without this, all models default to 4096 which is wrong for most modern
            // models (Phi-3.5=128K, Qwen=128K, Llama-3=128K, etc).
            if let maxPos = tc["max_position_embeddings"] as? Int, maxPos > 0 {
                container.config.contextLength = maxPos
                NovaMLXLog.info("Auto-detected context length: \(maxPos) tokens for \(config.identifier.displayName)")
            }

            // Detect hybrid linear attention models (e.g. Qwen3.5).
            // These use MambaCache for linear_attention layers + KVCacheSimple for full_attention layers.
            // FusedBatchScheduler only supports KVCacheSimple, so these must use ContinuousBatcher.
            if let layerTypes = tc["layer_types"] as? [String],
               layerTypes.contains("linear_attention") {
                container.config.hasLinearAttention = true
                NovaMLXLog.info("Detected hybrid linear attention model (\(layerTypes.filter { $0 == "linear_attention" }.count)/\(layerTypes.count) linear layers) — will use ContinuousBatcher")
            }
        }

        if let kvProvider = model.value as? KVCacheDimensionProvider, container.config.kvMemoryBytesPerToken == nil {
            let numLayers = kvProvider.kvHeads.count
            let kvHeadsSum = kvProvider.kvHeads.reduce(0, +)
            // KV cache per token = 2 (K+V) × sum(kvHeads per layer) × headDim × sizeof(Float16)
            let bytesPerToken = 2 * kvHeadsSum * headDim * MemoryLayout<Float16>.size
            container.config.kvMemoryBytesPerToken = bytesPerToken
            NovaMLXLog.info("Auto-calculated KV memory: \(bytesPerToken / 1024)KB/token (\(numLayers) layers, \(kvHeadsSum) total kvHeads, headDim=\(headDim))")
        }

        let weightsBytes = MLX.Memory.activeMemory
        if weightsBytes > 0 {
            let reservationTicket = wiredPolicy.ticket(size: Int(weightsBytes), kind: MLX.WiredMemoryTicketKind.reservation)
            _ = await reservationTicket.start()
            container.setWiredReservation(reservationTicket)
            NovaMLXLog.info("Wired memory reservation: \(weightsBytes / 1024 / 1024)MB for \(config.identifier.displayName)")
        }

        pool.add(container)
        metricsStore.recordModelLoad()
        return container
    }

    public func unloadModel(_ identifier: ModelIdentifier) {
        if let container = pool.get(identifier.id),
           let ticket = container.wiredReservationTicket {
            Task { await ticket.end() }
            NovaMLXLog.info("Wired memory reservation released for \(identifier.displayName)")
        }
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

    func getOrCreatePrefixCacheManager(modelId: String) -> PrefixCacheManager? {
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

    public func tokenizeMessages(_ messages: [ChatMessage], tokenizer: Tokenizer) -> [Int] {
        var allTokens: [Int] = []
        for msg in messages {
            if let content = msg.content {
                allTokens.append(contentsOf: tokenizer.encode(content))
            }
        }
        return allTokens
    }

    func buildUserInput(request: InferenceRequest, hasImages: Bool) async throws -> UserInput {
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
            return UserInput(prompt: .chat(chatMessages))
        } else {
            let messages: [Message] = request.messages.map { msg in
                ["role": msg.role.rawValue, "content": msg.content ?? ""]
            }
            return UserInput(prompt: .messages(messages))
        }
    }

    private func deferredClearCache() {
        lock.withLock {
            deferredClearCounter += 1
            if deferredClearCounter >= deferredClearThreshold {
                deferredClearCounter = 0
                MLX.Memory.clearCache()
            }
        }
    }

    private func createGenerationTicket(promptTokens: Int, maxTokens: Int, bytesPerToken: Int = 262_144) -> WiredMemoryTicket {
        let totalTokens = promptTokens + maxTokens
        let estimatedBytes = totalTokens * bytesPerToken
        return wiredPolicy.ticket(size: estimatedBytes, kind: MLX.WiredMemoryTicketKind.active)
    }

    public func preflightCheck(modelId: String, promptTokens: Int, maxTokens: Int) async throws {
        guard let maxGPU = GPU.maxRecommendedWorkingSetBytes(), maxGPU > 0 else { return }

        guard let container = getContainer(for: modelId) else { return }

        let adminOverride = container.kvMemoryOverride
        let autoCalculated = container.config.kvMemoryBytesPerToken
        let bytesPerToken = adminOverride ?? autoCalculated ?? 262144

        // Use budgetTracker's committed bytes (accurate) instead of MLX.Memory.activeMemory (unreliable)
        let budgetMetrics = await budgetTracker.metrics
        let currentCommitted = budgetMetrics.weightsMemoryBytes + budgetMetrics.committedKVBytes
        let estimatedKVBytes = UInt64(promptTokens + maxTokens) * UInt64(bytesPerToken)
        let projectedTotal = currentCommitted + estimatedKVBytes

        // Soft cap at 45% — early warning before hitting hard 50% MLX limit
        let safeLimit = UInt64(maxGPU) * 45 / 100

        guard projectedTotal <= safeLimit else {
            let neededMB = projectedTotal / 1024 / 1024
            let limitMB = safeLimit / 1024 / 1024
            NovaMLXLog.warning("Preflight REJECTED: model=\(modelId), estimated=\(neededMB)MB, limit=\(limitMB)MB, committed=\(currentCommitted / 1024 / 1024)MB, kvEstimate=\(estimatedKVBytes / 1024 / 1024)MB, bytesPerToken=\(bytesPerToken)")
            throw NovaMLXError.inferenceFailed(
                "Insufficient GPU memory: estimated \(neededMB)MB needed, limit \(limitMB)MB. Committed: \(currentCommitted / 1024 / 1024)MB"
            )
        }

        NovaMLXLog.info("Preflight OK: model=\(modelId), projected=\(projectedTotal / 1024 / 1024)MB, limit=\(safeLimit / 1024 / 1024)MB, committed=\(currentCommitted / 1024 / 1024)MB, bytesPerToken=\(bytesPerToken)")
    }

    /// Compute effective bytesPerToken for a model, factoring in TurboQuant compression.
    /// Returns raw FP16 bytesPerToken when TurboQuant is not configured.
    public func effectiveBytesPerToken(modelId: String) -> Int {
        guard let container = getContainer(for: modelId) else { return 262_144 }
        let raw = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144

        // Factor in TurboQuant compression
        if let turboConfig = turboQuantService.getConfig(forModel: modelId) {
            let compressionRatio = Double(16) / Double(turboConfig.bits)
            return max(1, Int(Double(raw) / compressionRatio))
        }

        // Also check per-model config kvBits
        if let kvBits = container.config.kvBits, kvBits < 16 {
            let compressionRatio = Double(16) / Double(kvBits)
            return max(1, Int(Double(raw) / compressionRatio))
        }

        return raw
    }

    /// Estimate total tokens for a request (prompt + max output).
    /// Prompt tokens are already in activeMemory, so we only budget for output tokens.
    public func estimateRequestTokens(modelId: String, request: InferenceRequest) -> Int {
        let maxTokens = request.maxTokens ?? getContainer(for: modelId)?.config.maxTokens ?? 4096
        return maxTokens
    }

    func buildGenerateParameters(
        request: InferenceRequest, config: ModelConfig, settings: ModelSettings? = nil
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

        if let settings {
            turboQuantService.applyToGenerateParameters(&params, modelId: config.identifier.id, settings: settings)
        } else if let fetchedSettings = settingsProvider?(config.identifier.id) {
            turboQuantService.applyToGenerateParameters(&params, modelId: config.identifier.id, settings: fetchedSettings)
        } else {
            let familyOpt = ModelFamilyRegistry.shared.optimization(
                for: config.identifier.id, family: config.identifier.family)
            if params.kvBits == nil, let defaultBits = familyOpt.defaultKVBits {
                params.kvBits = defaultBits
                params.kvGroupSize = familyOpt.safeGroupSize()
            }
            if params.kvBits == nil, let kvBits = config.kvBits {
                params.kvBits = kvBits
                params.kvGroupSize = config.kvGroupSize
            }
        }

        let familyOpt = ModelFamilyRegistry.shared.optimization(
            for: config.identifier.id, family: config.identifier.family)
        params.prefillStepSize = familyOpt.prefillStepSize

        if let seed = request.seed ?? config.seed {
            MLXRandom.seed(seed)
        }

        return params
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        // Dispatch to specialized paths based on request fields
        if let sessionId = request.sessionId {
            return try await generateWithSession(request, sessionId: sessionId)
        }
        if request.jsonSchemaDef != nil || request.responseFormat == .jsonObject ||
            request.regexPattern != nil || request.gbnfGrammar != nil {
            return try await generateWithProcessor(request)
        }

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

        let userInput = try await buildUserInput(request: request, hasImages: hasImages)
        let input = try await mlxContainer.prepare(input: userInput)

        let promptTokenCount = input.text.tokens.size
        NovaMLXLog.info("Encoded \(promptTokenCount) tokens via chat template")
        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Starting generation for \(request.model)")

        let maxTokens = request.maxTokens ?? container.config.maxTokens
        try await preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        let prefixCacheTokens: [Int]? = if let _ = getOrCreatePrefixCacheManager(modelId: request.model), promptTokenCount > 0 {
            input.text.tokens.asArray(Int32.self).map { Int($0) }
        } else {
            nil
        }

        let parameters = buildGenerateParameters(request: request, config: container.config, settings: settingsProvider?(request.model))

        let kvCacheBox = MutableSendableBox<[KVCache]?>(nil)
        let inputBox = SendableBox(input)

        let bytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
        let ticket = createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: maxTokens, bytesPerToken: bytesPerToken)

        let stream = try await mlxContainer.perform { context in
            let cache = context.model.newCache(parameters: parameters)
            kvCacheBox.value = cache
            return try MLXLMCommon.generate(
                input: inputBox.value,
                cache: cache,
                parameters: parameters,
                context: context,
                wiredMemoryTicket: ticket
            )
        }

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
           let prefixCache = getOrCreatePrefixCacheManager(modelId: request.model),
           let kvCache = kvCacheBox.value {
            prefixCache.storeCache(tokenIds: tokenArray, cache: kvCache)
        }

        deferredClearCache()

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
        // Dispatch to specialized paths based on request fields
        if let sessionId = request.sessionId {
            return streamWithSession(request, sessionId: sessionId)
        }
        if request.jsonSchemaDef != nil || request.responseFormat == .jsonObject ||
            request.regexPattern != nil || request.gbnfGrammar != nil {
            return streamWithProcessor(request)
        }

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

                    let userInput = try await self.buildUserInput(request: request, hasImages: hasImages)

                    let input = try await mlxContainer.prepare(input: userInput)

                    let parameters = self.buildGenerateParameters(request: request, config: container.config, settings: self.settingsProvider?(request.model))
                    let promptTokenCount = input.text.tokens.size
                    let maxTokens = request.maxTokens ?? container.config.maxTokens
                    try await self.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)
                    let ticketBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262144
                    let ticket = self.createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: maxTokens, bytesPerToken: ticketBytesPerToken)
                    let stream = try await mlxContainer.generate(input: input, parameters: parameters, wiredMemoryTicket: ticket)

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
                self.deferredClearCache()
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

        let estimatedPromptTokens = tokenizeMessages(request.messages, tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil)).count
        let maxTokens = request.maxTokens ?? container.config.maxTokens
        try await preflightCheck(modelId: request.model, promptTokens: estimatedPromptTokens, maxTokens: maxTokens)

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

                    let estimatedPromptTokens = engine.tokenizeMessages(request.messages, tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil)).count
                    let maxTokens = request.maxTokens ?? container.config.maxTokens
                    try await engine.preflightCheck(modelId: request.model, promptTokens: estimatedPromptTokens, maxTokens: maxTokens)

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

        let grammarProcessor: any LogitProcessor
        if let schema = request.jsonSchemaDef {
            grammarProcessor = SchemaGuidedProcessor(schema: schema, tokenizer: container.tokenizer!)
        } else if let pattern = request.regexPattern {
            grammarProcessor = try RegexLogitProcessor(pattern: pattern, tokenizer: container.tokenizer!)
        } else if let grammar = request.gbnfGrammar {
            grammarProcessor = try GBNFLogitProcessor(grammar: grammar, tokenizer: container.tokenizer!)
        } else {
            grammarProcessor = JSONLogitProcessor(tokenizer: container.tokenizer!)
        }

        let parameters = buildGenerateParameters(request: request, config: container.config, settings: settingsProvider?(request.model))
        let penaltyProcessor = parameters.processor()
        let processor: any LogitProcessor
        if let penalty = penaltyProcessor {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty)
        } else {
            processor = grammarProcessor
        }

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
        let maxTokens = request.maxTokens ?? container.config.maxTokens
        try await preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

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

        let ticketBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
        let (stream, task) = generateTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: config,
            tokenizer: tokenizer,
            iterator: iterator,
            wiredMemoryTicket: createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: request.maxTokens ?? container.config.maxTokens, bytesPerToken: ticketBytesPerToken)
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

        deferredClearCache()

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

                    let grammarProcessor: any LogitProcessor
                    if let schema = request.jsonSchemaDef {
                        grammarProcessor = SchemaGuidedProcessor(schema: schema, tokenizer: container.tokenizer!)
                    } else if let pattern = request.regexPattern {
                        grammarProcessor = try RegexLogitProcessor(pattern: pattern, tokenizer: container.tokenizer!)
                    } else if let grammar = request.gbnfGrammar {
                        grammarProcessor = try GBNFLogitProcessor(grammar: grammar, tokenizer: container.tokenizer!)
                    } else {
                        grammarProcessor = JSONLogitProcessor(tokenizer: container.tokenizer!)
                    }

                    let parameters = engine.buildGenerateParameters(request: request, config: container.config, settings: engine.settingsProvider?(request.model))
                    let penaltyProc = parameters.processor()
                    let processor: any LogitProcessor
                    if let penalty = penaltyProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty)
                    } else {
                        processor = grammarProcessor
                    }

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
                    let maxTokens = request.maxTokens ?? container.config.maxTokens
                    try await engine.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

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

                    let sessionBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
                    let (stream, genTask) = generateTask(
                        promptTokenCount: promptTokenCount,
                        modelConfiguration: config,
                        tokenizer: tokenizer,
                        iterator: iterator,
                        wiredMemoryTicket: engine.createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: request.maxTokens ?? container.config.maxTokens, bytesPerToken: sessionBytesPerToken)
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

                    self.deferredClearCache()

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
