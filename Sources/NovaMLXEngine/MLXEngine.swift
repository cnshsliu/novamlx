import Foundation
import CoreImage
import MLX
import MLXNN
import MLXRandom
import MLXLLM
import MLXLMCommon
import MLXVLM
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
    /// Control tokens extracted from tokenizer.json + chat_template (e.g. "<|turn>", "<|im_end|>")
    public private(set) var controlTokens: [String] = []

    public init(identifier: ModelIdentifier, config: ModelConfig) {
        self.identifier = identifier
        self.config = config
        self.isLoaded = false
    }

    public func setLoaded(mlxContainer: MLXLMCommon.ModelContainer, tokenizer: Tokenizer) {
        self.mlxContainer = mlxContainer
        self.tokenizer = tokenizer
        self.isLoaded = true
        self.controlTokens = Self.extractControlTokens(for: identifier.id)
        if !controlTokens.isEmpty {
            NovaMLXLog.info("[ControlTokens] \(identifier.displayName): \(controlTokens)")
        }
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
        controlTokens = []
        NovaMLXLog.info("Model unloaded: \(identifier.displayName)")
    }

    // MARK: - Control Token Extraction

    /// Extract control/special tokens from tokenizer.json added_tokens + chat_template
    public static func extractControlTokens(for modelId: String) -> [String] {
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        var tokens = Set<String>()

        // 1. From tokenizer.json added_tokens where special=true
        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.json")
        if let data = try? Data(contentsOf: tokenizerPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let added = json["added_tokens"] as? [[String: Any]] {
            for entry in added {
                if entry["special"] as? Bool == true,
                   let content = entry["content"] as? String,
                   content.hasPrefix("<") {
                    tokens.insert(content)
                }
            }
        }

        // 2. From chat_template: extract all <|xxx> and <|xxx|> patterns
        let template = loadChatTemplate(modelDir: modelDir)
        if let template {
            // Match <|xxx> and <|xxx|> patterns
            if let regex = try? NSRegularExpression(pattern: "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>") {
                let nsRange = NSRange(template.startIndex..., in: template)
                for match in regex.matches(in: template, range: nsRange) {
                    if let range = Range(match.range, in: template) {
                        tokens.insert(String(template[range]))
                    }
                }
            }
        }

        // Remove EOS/BOS/PAD/UNK — those are structural, not output separators
        tokens.remove("")
        return tokens.sorted()
    }

    private static func loadChatTemplate(modelDir: URL) -> String? {
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: tcPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let template = json["chat_template"] as? String {
            return template
        }
        let jinjaPath = modelDir.appendingPathComponent("chat_template.jinja")
        return try? String(contentsOf: jinjaPath, encoding: .utf8)
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
        self.metricsStore = metricsStore ?? MetricsStore(baseDirectory: NovaMLXPaths.baseDir)
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

    static let tokenizerLoader = LocalTokenizerLoader()

    /// Ensure the model directory has a chat template. If missing, try to copy
    /// from another downloaded model in the same family, or inject a default template
    /// for known model families (gemma, qwen, phi, llama).
    private static func ensureChatTemplate(modelDir: URL, modelId: String, family: ModelFamily) {
        let fm = FileManager.default

        // Check if chat template already exists in tokenizer_config.json
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: tcPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           json["chat_template"] != nil {
            return // Already has a chat template
        }

        // Check standalone template files
        let jinjaPath = modelDir.appendingPathComponent("chat_template.jinja")
        let jsonPath = modelDir.appendingPathComponent("chat_template.json")
        if fm.fileExists(atPath: jinjaPath.path) || fm.fileExists(atPath: jsonPath.path) {
            return
        }

        NovaMLXLog.warning("[loadModel] \(modelId): no chat template found — attempting to inject one")

        // Try to copy from a sibling model of the same family
        let modelsDir = modelDir.deletingLastPathComponent()
        if let siblingTemplate = findChatTemplateInSiblings(modelsDir: modelsDir, currentModel: modelDir) {
            do {
                try fm.copyItem(at: siblingTemplate, to: jinjaPath)
                NovaMLXLog.info("[loadModel] \(modelId): copied chat template from sibling model")
                return
            } catch {
                NovaMLXLog.warning("[loadModel] \(modelId): failed to copy sibling template: \(error)")
            }
        }

        // Inject a default template for known families
        if let defaultTemplate = defaultChatTemplate(for: family) {
            do {
                try defaultTemplate.write(to: jinjaPath, atomically: true, encoding: String.Encoding.utf8)
                NovaMLXLog.info("[loadModel] \(modelId): injected default \(family) chat template")
            } catch {
                NovaMLXLog.warning("[loadModel] \(modelId): failed to write default template: \(error)")
            }
        } else {
            NovaMLXLog.warning("[loadModel] \(modelId): no chat template available — inference will fail for chat requests")
        }
    }

    private static func findChatTemplateInSiblings(modelsDir: URL, currentModel: URL) -> URL? {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil) else { return nil }

        for sibling in contents {
            guard sibling != currentModel else { continue }
            let jinja = sibling.appendingPathComponent("chat_template.jinja")
            if fm.fileExists(atPath: jinja.path) { return jinja }
        }
        return nil
    }

    private static func defaultChatTemplate(for family: ModelFamily) -> String? {
        switch family {
        case .gemma:
            return """
            {%- set ns = namespace(prev_message_type=None) -%}
            {%- set loop_messages = messages -%}
            {{ bos_token }}
            {%- if messages[0]['role'] in ['system', 'developer'] -%}
            {{- '<|turn>system\\n' -}}
            {{- messages[0]['content'] | trim -}}
            {{- '\\n' -}}
            {%- set loop_messages = messages[1:] -%}
            {%- endif -%}
            {%- for message in loop_messages -%}
            {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
            {{- '<|turn>' + role + '\\n' -}}
            {%- if message['content'] is string -%}
            {{- message['content'] | trim -}}
            {%- elif message['content'] is sequence -%}
            {%- for item in message['content'] -%}
            {%- if item['type'] == 'text' -%}
            {{- item['text'] | trim -}}
            {%- endif -%}
            {%- endfor -%}
            {%- endif -%}
            {{- '\\n' -}}
            {%- endfor -%}
            {%- if add_generation_prompt -%}
            {{- '<|turn>model\\n' -}}
            {%- endif -%}
            """
        default:
            return nil
        }
    }

    public func loadModel(from url: URL, config: ModelConfig) async throws -> ModelContainer {
        let container = ModelContainer(identifier: config.identifier, config: config)
        NovaMLXLog.info("Loading model from: \(url.path)")

        // Ensure chat template exists — models without it will crash on inference.
        // Check tokenizer_config.json, chat_template.jinja, chat_template.json in order.
        Self.ensureChatTemplate(modelDir: url, modelId: config.identifier.id, family: config.identifier.family)

        // VLM models must be loaded via VLMModelFactory — they need vision tower
        // and VLM-specific prepare()/callAsFunction implementations
        let isVLM = config.modelType == .vlm
        let mlxContainer: MLXLMCommon.ModelContainer
        if isVLM {
            mlxContainer = try await VLMModelFactory.shared.loadContainer(
                from: url,
                using: Self.tokenizerLoader
            )
        } else {
            mlxContainer = try await LLMModelFactory.shared.loadContainer(
                from: url,
                using: Self.tokenizerLoader
            )
        }

        let mlxTokenizer = await mlxContainer.tokenizer
        let eosString = mlxTokenizer.eosToken
        let tokenizer = Tokenizer(
            encode: { text in mlxTokenizer.encode(text: text) },
            decode: { tokens in mlxTokenizer.decode(tokenIds: tokens) },
            eosToken: eosString,
            eosTokenId: mlxTokenizer.eosTokenId
        )

        container.setLoaded(mlxContainer: mlxContainer, tokenizer: tokenizer)

        let model = await mlxContainer.perform { (context: ModelContext) in
            SendableBox(context.model)
        }

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
        // Clear prefix cache for this model
        if let cacheManager = lock.withLock({ prefixCacheManagers.removeValue(forKey: identifier.id) }) {
            cacheManager.clear()
            NovaMLXLog.info("Prefix cache cleared for \(identifier.displayName)")
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
            let ssdDir = NovaMLXPaths.prefixCacheDir(for: modelId)
            let config = PrefixCacheConfig(
                blockSize: 64,
                maxBlocks: 4096,
                initialBlocks: 256,
                ssdCacheDir: ssdDir,
                ssdMaxSizeBytes: 5 * 1024 * 1024 * 1024  // 5 GB (down from 100 GB)
            )
            let manager = PrefixCacheManager(config: config, modelName: modelId)
            prefixCacheManagers[modelId] = manager
            return manager
        }
    }

    public func cleanupOrphanedCacheDirs(downloadedModelIds: Set<String>) {
        let fm = FileManager.default
        let cacheBase = NovaMLXPaths.prefixCacheBaseDir
        guard let contents = try? fm.contentsOfDirectory(at: cacheBase, includingPropertiesForKeys: nil) else { return }
        var cleaned = 0
        for dir in contents {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            let modelId = dir.lastPathComponent.replacingOccurrences(of: "_", with: "/")
            if !downloadedModelIds.contains(modelId) {
                try? fm.removeItem(at: dir)
                cleaned += 1
            }
        }
        if cleaned > 0 {
            NovaMLXLog.info("Cleaned \(cleaned) orphaned prefix cache directories")
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

    private func toSendable(_ value: Any) -> any Sendable {
        switch value {
        case let s as String: return s
        case let i as Int: return i
        case let d as Double: return d
        case let b as Bool: return b
        case let a as [Any]: return a.map { toSendable($0) } as any Sendable
        case let d as [String: Any]: return d.mapValues { toSendable($0) } as any Sendable
        default: return "\(value)"
        }
    }

    func buildUserInput(request: InferenceRequest, hasImages: Bool) async throws -> UserInput {
        let toolSpecs: [[String: any Sendable]]? = request.tools?.map { tool in
            tool.mapValues { toSendable($0) }
        }

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
            return UserInput(prompt: .chat(chatMessages), tools: toolSpecs)
        } else {
            let messages: [Message] = request.messages.map { msg in
                var m: Message = ["role": msg.role.rawValue, "content": msg.content ?? ""]
                if let tcId = msg.toolCallId { m["tool_call_id"] = tcId }
                if let name = msg.name { m["name"] = name }
                if let tcs = msg.toolCalls {
                    m["tool_calls"] = tcs.map { tc in
                        ["function": ["name": tc.functionName, "arguments": tc.arguments]] as [String: any Sendable]
                    }
                }
                return m
            }
            return UserInput(prompt: .messages(messages), tools: toolSpecs)
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

    // MARK: - Control Token Stop Detection

    private static let controlTokenLock = NSLock()
    private nonisolated(unsafe) static var _controlTokenCache: [String: [String]] = [:]

    /// Get cached control tokens for a model, extracting if needed
    static func controlTokensForModel(modelId: String) -> [String] {
        controlTokenLock.lock()
        if let cached = _controlTokenCache[modelId] {
            controlTokenLock.unlock()
            return cached
        }
        controlTokenLock.unlock()

        let tokens = ModelContainer.extractControlTokens(for: modelId)

        controlTokenLock.lock()
        _controlTokenCache[modelId] = tokens
        controlTokenLock.unlock()
        if !tokens.isEmpty {
            NovaMLXLog.info("[ControlTokens] Detected \(tokens.count) control tokens for \(modelId)")
        }
        return tokens
    }

    private func buildTurnStopProcessor(
        modelId: String,
        tokenizer: Tokenizer,
        modelConfig: ModelConfiguration
    ) -> TurnStopProcessor? {
        let patterns = Self.controlTokensForModel(modelId: modelId)
        guard !patterns.isEmpty else { return nil }

        var eosIds = Set<Int>()
        if let eosId = tokenizer.eosTokenId { eosIds.insert(eosId) }
        eosIds = eosIds.union(modelConfig.eosTokenIds)

        return TurnStopProcessor(
            stopPatterns: patterns,
            eosTokenIds: eosIds,
            decode: tokenizer.decode
        )
    }

    /// Compose a base processor (repetition penalty) with optional turn stop processor.
    private func composeWithTurnStop(
        baseProcessor: (any LogitProcessor)?,
        modelId: String,
        tokenizer: Tokenizer,
        modelConfig: ModelConfiguration
    ) -> (any LogitProcessor)? {
        guard let tsp = buildTurnStopProcessor(
            modelId: modelId,
            tokenizer: tokenizer,
            modelConfig: modelConfig
        ) else { return baseProcessor }

        if let base = baseProcessor {
            return ComposedLogitProcessor(
                grammarProcessor: base,
                penaltyProcessor: nil,
                turnStopProcessor: tsp
            )
        }
        return tsp
    }

    private static func trimControlTokens(_ text: String, patterns: [String]) -> String {
        guard !patterns.isEmpty else { return text }
        var result = text
        for pattern in patterns {
            if let range = result.range(of: pattern) {
                result = String(result[..<range.lowerBound])
            }
        }
        return result
    }

    /// Regex-based scrub: strip any <|xxx|> or <|xxx> patterns from output.
    /// Catches multi-token control sequences (mask_start, tool_call, etc.)
    /// that aren't registered as special tokens in the tokenizer.
    private static let controlTokenRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>")
    }()

    private static func scrubControlTokens(_ text: String) -> String {
        let nsRange = NSRange(text.startIndex..., in: text)
        let matches = controlTokenRegex.matches(in: text, range: nsRange)
        guard !matches.isEmpty else { return text }
        var result = text
        for match in matches.reversed() {
            if let range = Range(match.range, in: result) {
                result.removeSubrange(range)
            }
        }
        return result
    }

    /// Filter a streaming chunk for control tokens. Returns (cleanText, shouldStop).
    /// Handles both full matches and partial matches (where TurnStopProcessor
    /// forces EOS mid-token, leaving e.g. "<|turn" without the closing ">").
    private static func filterControlInChunk(
        _ text: String,
        accumulated: inout String,
        yieldedCount: inout Int,
        patterns: [String]
    ) -> (String, Bool) {
        accumulated += text
        let totalLen = accumulated.count

        // 1. Full control token match — trim and stop
        for pattern in patterns {
            if let range = accumulated.range(of: pattern) {
                let safeEnd = accumulated.distance(from: accumulated.startIndex, to: range.lowerBound)
                let cleanText: String
                if safeEnd > yieldedCount {
                    cleanText = String(accumulated.dropFirst(yieldedCount).prefix(safeEnd - yieldedCount))
                } else {
                    cleanText = ""
                }
                yieldedCount = totalLen
                return (cleanText, true)
            }
        }

        // 2. Partial prefix check — if tail could be start of a control token, buffer it
        let maxLen = patterns.map(\.count).max() ?? 0
        let checkLen = min(totalLen, maxLen)
        var safeEnd = totalLen
        for i in 1...checkLen {
            let suffixStart = accumulated.index(accumulated.endIndex, offsetBy: -i)
            let suffix = accumulated[suffixStart...]
            for pattern in patterns {
                if pattern.hasPrefix(suffix) {
                    safeEnd = min(safeEnd, totalLen - i)
                }
            }
        }

        // Yield safe portion (everything before the potential control token prefix)
        if safeEnd > yieldedCount {
            let cleanText = String(accumulated.dropFirst(yieldedCount).prefix(safeEnd - yieldedCount))
            yieldedCount = safeEnd
            return (cleanText, false)
        }
        return ("", false)
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
        let input: LMInput
        do {
            input = try await mlxContainer.prepare(input: userInput)
        } catch {
            let desc = error.localizedDescription
            if desc.contains("chat template") {
                // VLM models can't use plain text fallback — their forward pass expects image embeddings
                if container.config.modelType == .vlm {
                    NovaMLXLog.error("[generate] \(request.model): VLM model has no chat template — cannot process request")
                    throw NovaMLXError.inferenceFailed("Model '\(request.model)' does not have a chat template and is not compatible with text-only requests.")
                }
                // LLM models: fall back to plain text encoding
                NovaMLXLog.warning("[generate] \(request.model): no chat template — falling back to plain text encoding")
                let plainText = request.messages.compactMap { $0.content }.joined(separator: "\n\n")
                let tokenizer = await mlxContainer.tokenizer
                let tokens = tokenizer.encode(text: plainText)
                input = LMInput(tokens: MLXArray(tokens))
            } else {
                throw error
            }
        }

        let promptTokenCount = input.text.tokens.size
        NovaMLXLog.info("Encoded \(promptTokenCount) tokens via chat template")
        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Starting generation for \(request.model)")

        let maxTokens = request.maxTokens ?? container.config.maxTokens
        try await preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        let parameters = buildGenerateParameters(request: request, config: container.config, settings: settingsProvider?(request.model))

        // Capture prompt tokens for prefix cache storage + loading
        let allPromptTokens: [Int]?
        if let _ = getOrCreatePrefixCacheManager(modelId: request.model), promptTokenCount > 0 {
            let tokens = input.text.tokens.asArray(Int32.self).map { Int($0) }
            allPromptTokens = tokens
        } else {
            allPromptTokens = nil
        }

        // Try prefix cache load
        let prefixResult = allPromptTokens.flatMap { tokens in
            getOrCreatePrefixCacheManager(modelId: request.model)?.fetchPrefix(tokenIds: tokens)
        }

        // Build processor (repetition penalty only)
        let baseProcessor = parameters.processor()
        let model = await mlxContainer.perform { context in SendableBox(context.model) }
        let mlxTokenizer = await mlxContainer.tokenizer
        let modelConfig = await mlxContainer.configuration

        let processor = composeWithTurnStop(
            baseProcessor: baseProcessor,
            modelId: request.model,
            tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil),
            modelConfig: modelConfig
        )

        let sampler = parameters.sampler()

        // Build effective input and cache — use prefix cache hit if valid
        var effectiveInput = input
        let kvCache: [KVCache]
        if let result = prefixResult, result.cachedTokenCount > 0, let restored = result.cache {
            let freshCaches = model.value.newCache(parameters: parameters)
            var cacheValid = restored.count == freshCaches.count
            // Validate: layer types must match, KV cache states must be 4D with 2 arrays
            if cacheValid {
                for i in 0..<restored.count {
                    let restoredType = String(describing: type(of: restored[i]))
                    let freshType = String(describing: type(of: freshCaches[i]))
                    guard restoredType == freshType else {
                        NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: L\(i) type mismatch (\(restoredType) vs \(freshType))")
                        cacheValid = false
                        break
                    }
                    let isKV = restored[i] is KVCacheSimple
                        || restored[i] is RotatingKVCache
                        || restored[i] is ChunkedKVCache
                        || restored[i] is QuantizedKVCache
                    if isKV {
                        // KV caches: must have exactly 2 state arrays (keys, values), both 4D
                        let state = restored[i].state
                        guard state.count == 2 else {
                            NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: L\(i) KV state count=\(state.count), expected 2")
                            cacheValid = false
                            break
                        }
                        for j in 0..<2 {
                            guard state[j].ndim == 4 else {
                                NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: L\(i).S\(j) ndim=\(state[j].ndim), expected 4")
                                cacheValid = false
                                break
                            }
                        }
                    }
                    // Fixed-size caches (MambaCache, etc.): type match is sufficient
                    if !cacheValid { break }
                }
            }

            if cacheValid {
                let cachedTokenCount = result.cachedTokenCount
                // RotatingKVCache (sliding window) is incompatible with prefix cache:
                // offset > maxSize breaks invariants, and shared KV across layers
                // means any mismatched layer corrupts the entire attention.
                // Fall back to full prefill if any RotatingKVCache present.
                let hasRotating = restored.contains { $0 is RotatingKVCache }
                if hasRotating {
                    NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: model has RotatingKVCache layers — skipping (incompatible with sliding window)")
                    kvCache = model.value.newCache(parameters: parameters)
                } else {
                    for layer in restored {
                        if let kv = layer as? BaseKVCache { kv.offset = cachedTokenCount }
                    }
                    NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache HIT: \(cachedTokenCount) cached, \(result.remainingTokenIds.count) remaining")
                    let remainingTokens = MLXArray(result.remainingTokenIds.map { Int32($0) })
                    effectiveInput = LMInput(text: .init(tokens: remainingTokens))
                    kvCache = restored
                }
            } else {
                NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: validation failed — using fresh cache")
                kvCache = model.value.newCache(parameters: parameters)
            }
        } else {
            kvCache = model.value.newCache(parameters: parameters)
        }
        let kvCacheBox = MutableSendableBox<[KVCache]?>(kvCache)

        // Try creating iterator with restored cache; fall back to fresh on error
        let iterator: TokenIterator
        do {
            iterator = try TokenIterator(
                input: effectiveInput, model: model.value,
                cache: kvCache,
                processor: processor, sampler: sampler,
                maxTokens: maxTokens
            )
        } catch {
            NovaMLXLog.error("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache iterator failed: \(error.localizedDescription) — falling back to full prefill")
            let freshCache = model.value.newCache(parameters: parameters)
            kvCacheBox.value = freshCache
            iterator = try TokenIterator(
                input: input, model: model.value,
                cache: freshCache,
                processor: processor, sampler: sampler,
                maxTokens: maxTokens
            )
        }

        let bytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
        let ticket = createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: maxTokens, bytesPerToken: bytesPerToken)

        let (stream, _) = generateTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfig,
            tokenizer: mlxTokenizer,
            iterator: iterator,
            wiredMemoryTicket: ticket
        )

        var generatedText = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var genStopReason: String = "none"
        var collectedToolCalls: [ToolCallResult] = []

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                completionTokens += 1
            case .info(let info):
                completionTokens = info.generationTokenCount
                genStopReason = String(describing: info.stopReason)
                if case .length = info.stopReason { finishReason = .length }
                NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Info: tokens=\(info.generationTokenCount), stopReason=\(genStopReason), tps=\(String(format: "%.1f", info.tokensPerSecond))")
            case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                collectedToolCalls.append(tcResult)
                finishReason = .toolCalls
            }
        }

        NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Done: completionTokens=\(completionTokens), stopReason=\(genStopReason)")

        // Store KV cache in prefix cache for future requests with same prompt prefix
        if let tokenArray = allPromptTokens,
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

        let trimmedText = Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: Self.controlTokensForModel(modelId: request.model)))

        return InferenceResult(
            id: request.id, model: request.model, text: trimmedText,
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
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

                    // Capture prompt tokens for prefix cache storage + loading
                    let allPromptTokens: [Int]? = if let _ = self.getOrCreatePrefixCacheManager(modelId: request.model), promptTokenCount > 0 {
                        input.text.tokens.asArray(Int32.self).map { Int($0) }
                    } else {
                        nil
                    }

                    // Try prefix cache load
                    let prefixResult = allPromptTokens.flatMap { tokens in
                        self.getOrCreatePrefixCacheManager(modelId: request.model)?.fetchPrefix(tokenIds: tokens)
                    }

                    // Build processor (repetition penalty only)
                    let baseProcessor = parameters.processor()
                    let model = await mlxContainer.perform { context in SendableBox(context.model) }
                    let mlxTokenizer = await mlxContainer.tokenizer
                    let modelConfig = await mlxContainer.configuration

                    let processor = self.composeWithTurnStop(
                        baseProcessor: baseProcessor,
                        modelId: request.model,
                        tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil),
                        modelConfig: modelConfig
                    )

                    let sampler = parameters.sampler()

                    // Build effective input and cache — use prefix cache hit if valid
                    var effectiveInput = input
                    let kvCache: [KVCache]
                    if let result = prefixResult, result.cachedTokenCount > 0, let restored = result.cache {
                        let freshCaches = model.value.newCache(parameters: parameters)
                        var cacheValid = restored.count == freshCaches.count
                        if cacheValid {
                            for i in 0..<restored.count {
                                let rType = String(describing: type(of: restored[i]))
                                let fType = String(describing: type(of: freshCaches[i]))
                                guard rType == fType else { cacheValid = false; break }
                                let isKV = restored[i] is KVCacheSimple
                                    || restored[i] is RotatingKVCache
                                    || restored[i] is ChunkedKVCache
                                    || restored[i] is QuantizedKVCache
                                if isKV {
                                    let state = restored[i].state
                                    guard state.count == 2, state.allSatisfy({ $0.ndim == 4 }) else {
                                        cacheValid = false; break
                                    }
                                }
                                // Fixed-size caches: type match is sufficient
                                if !cacheValid { break }
                            }
                        }
                        if cacheValid {
                            let cachedTokenCount = result.cachedTokenCount
                            let hasRotating = restored.contains { $0 is RotatingKVCache }
                            if hasRotating {
                                NovaMLXLog.info("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache: model has RotatingKVCache layers — skipping (incompatible with sliding window)")
                                kvCache = model.value.newCache(parameters: parameters)
                            } else {
                                for layer in restored {
                                    if let kv = layer as? BaseKVCache { kv.offset = cachedTokenCount }
                                }
                                NovaMLXLog.info("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache HIT: \(cachedTokenCount) cached, \(result.remainingTokenIds.count) remaining")
                                let remainingTokens = MLXArray(result.remainingTokenIds.map { Int32($0) })
                                effectiveInput = LMInput(text: .init(tokens: remainingTokens))
                                kvCache = restored
                            }
                        } else {
                            NovaMLXLog.warning("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache: validation failed — using fresh cache")
                            kvCache = model.value.newCache(parameters: parameters)
                        }
                    } else {
                        kvCache = model.value.newCache(parameters: parameters)
                    }
                    let kvCacheBox = MutableSendableBox<[KVCache]?>(kvCache)

                    // Try creating iterator with restored cache; fall back to fresh on error
                    let iterator: TokenIterator
                    do {
                        iterator = try TokenIterator(
                            input: effectiveInput, model: model.value,
                            cache: kvCache,
                            processor: processor, sampler: sampler,
                            maxTokens: maxTokens
                        )
                    } catch {
                        NovaMLXLog.error("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache iterator failed: \(error.localizedDescription) — falling back to full prefill")
                        let freshCache = model.value.newCache(parameters: parameters)
                        kvCacheBox.value = freshCache
                        iterator = try TokenIterator(
                            input: input, model: model.value,
                            cache: freshCache,
                            processor: processor, sampler: sampler,
                            maxTokens: maxTokens
                        )
                    }

                    let ticketBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262144
                    let ticket = self.createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: maxTokens, bytesPerToken: ticketBytesPerToken)
                    let (stream, _) = generateTask(
                        promptTokenCount: promptTokenCount,
                        modelConfiguration: modelConfig,
                        tokenizer: mlxTokenizer,
                        iterator: iterator,
                        wiredMemoryTicket: ticket
                    )

                    var finishReason: FinishReason = .stop
                    var completionTokens = 0
                    var infoStopReason: String = "none"
                    var streamAccumulated = ""
                    var streamYieldedCount = 0
                    let controlPatterns = Self.controlTokensForModel(modelId: request.model)
                    let startTime = Date()
                    let reqTag = request.id.uuidString.prefix(8).description

                    for await generation in stream {
                        if Task.isCancelled {
                            NovaMLXLog.error("[STREAM:\(reqTag)] Task cancelled during generation after \(completionTokens) tokens")
                            break
                        }
                        switch generation {
                        case .chunk(let text):
                            let (rawText, shouldStop) = Self.filterControlInChunk(text, accumulated: &streamAccumulated, yieldedCount: &streamYieldedCount, patterns: controlPatterns)
                            let cleanText = Self.scrubControlTokens(rawText)
                            if !cleanText.isEmpty {
                                completionTokens += 1
                                continuation.yield(Token(id: 0, text: cleanText))
                            }
                            if shouldStop { break }
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            infoStopReason = String(describing: info.stopReason)
                            if case .length = info.stopReason { finishReason = .length }
                            NovaMLXLog.info("[STREAM:\(reqTag)] Info: tokens=\(info.generationTokenCount), stopReason=\(infoStopReason), tps=\(String(format: "%.1f", info.tokensPerSecond))")
                        case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                continuation.yield(Token(id: 0, text: "", toolCall: tcResult))
                        }
                    }

                    NovaMLXLog.info("[STREAM:\(reqTag)] Loop ended: completionTokens=\(completionTokens), taskCancelled=\(Task.isCancelled), infoStopReason=\(infoStopReason)")

                    // Store KV cache in prefix cache for future requests
                    if let tokenArray = allPromptTokens,
                       let prefixCache = self.getOrCreatePrefixCacheManager(modelId: request.model),
                       let kvCache = kvCacheBox.value {
                        prefixCache.storeCache(tokenIds: tokenArray, cache: kvCache)
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
        var collectedToolCalls: [ToolCallResult] = []

        let stream = sessionBox.streamDetails(to: lastUserMessage)
        for try await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                collectedToolCalls.append(tcResult)
                finishReason = .toolCalls
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
            id: request.id, model: request.model, text: Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: Self.controlTokensForModel(modelId: request.model))),
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
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
                    var streamAccumulated = ""
                    var streamYieldedCount = 0
                    let controlPatterns = Self.controlTokensForModel(modelId: request.model)

                    let stream = sessionBox.streamDetails(to: lastUserMessage)
                    for try await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            let (rawText, shouldStop) = Self.filterControlInChunk(text, accumulated: &streamAccumulated, yieldedCount: &streamYieldedCount, patterns: controlPatterns)
                            let cleanText = Self.scrubControlTokens(rawText)
                            if !cleanText.isEmpty {
                                completionTokens += 1
                                continuation.yield(Token(id: 0, text: cleanText))
                            }
                            if shouldStop { break }
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                continuation.yield(Token(id: 0, text: "", toolCall: tcResult))
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
        guard let tokenizer = container.tokenizer else {
            throw NovaMLXError.inferenceFailed("Tokenizer not available for \(request.model)")
        }

        let grammarProcessor: any LogitProcessor
        if let schema = request.jsonSchemaDef {
            grammarProcessor = SchemaGuidedProcessor(schema: schema, tokenizer: tokenizer)
        } else if let pattern = request.regexPattern {
            grammarProcessor = try RegexLogitProcessor(pattern: pattern, tokenizer: tokenizer)
        } else if let grammar = request.gbnfGrammar {
            grammarProcessor = try GBNFLogitProcessor(grammar: grammar, tokenizer: tokenizer)
        } else {
            grammarProcessor = JSONLogitProcessor(tokenizer: tokenizer)
        }

        let parameters = buildGenerateParameters(request: request, config: container.config, settings: settingsProvider?(request.model))
        let penaltyProcessor = parameters.processor()

        let modelConfig = await mlxContainer.configuration

        let turnStopProc = buildTurnStopProcessor(
            modelId: request.model,
            tokenizer: tokenizer,
            modelConfig: modelConfig
        )

        let processor: any LogitProcessor
        if let penalty = penaltyProcessor, let tsp = turnStopProc {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty, turnStopProcessor: tsp)
        } else if let penalty = penaltyProcessor {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty)
        } else if let tsp = turnStopProc {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: nil, turnStopProcessor: tsp)
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
        let mlxTokenizer = await mlxContainer.tokenizer

        let iterator = try TokenIterator(
            input: input, model: model.value,
            processor: processor, sampler: sampler,
            maxTokens: request.maxTokens ?? container.config.maxTokens
        )

        let ticketBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
        let (stream, task) = generateTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfig,
            tokenizer: mlxTokenizer,
            iterator: iterator,
            wiredMemoryTicket: createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: request.maxTokens ?? container.config.maxTokens, bytesPerToken: ticketBytesPerToken)
        )

        let startTime = Date()
        var generatedText = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var collectedToolCalls: [ToolCallResult] = []

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                completionTokens += 1
            case .info(let info):
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                collectedToolCalls.append(tcResult)
                finishReason = .toolCalls
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
            id: request.id, model: request.model, text: Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: Self.controlTokensForModel(modelId: request.model))),
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
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
                    guard let tokenizer = container.tokenizer else {
                        throw NovaMLXError.inferenceFailed("Tokenizer not available for \(request.model)")
                    }

                    let grammarProcessor: any LogitProcessor
                    if let schema = request.jsonSchemaDef {
                        grammarProcessor = SchemaGuidedProcessor(schema: schema, tokenizer: tokenizer)
                    } else if let pattern = request.regexPattern {
                        grammarProcessor = try RegexLogitProcessor(pattern: pattern, tokenizer: tokenizer)
                    } else if let grammar = request.gbnfGrammar {
                        grammarProcessor = try GBNFLogitProcessor(grammar: grammar, tokenizer: tokenizer)
                    } else {
                        grammarProcessor = JSONLogitProcessor(tokenizer: tokenizer)
                    }

                    let parameters = engine.buildGenerateParameters(request: request, config: container.config, settings: engine.settingsProvider?(request.model))
                    let penaltyProc = parameters.processor()

                    let modelConfig = await mlxContainer.configuration

                    let turnStopProc = engine.buildTurnStopProcessor(
                        modelId: request.model,
                        tokenizer: tokenizer,
                        modelConfig: modelConfig
                    )

                    let processor: any LogitProcessor
                    if let penalty = penaltyProc, let tsp = turnStopProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty, turnStopProcessor: tsp)
                    } else if let penalty = penaltyProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty)
                    } else if let tsp = turnStopProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: nil, turnStopProcessor: tsp)
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
                    let mlxTokenizer = await mlxContainer.tokenizer

                    let iterator = try TokenIterator(
                        input: input, model: model.value,
                        processor: processor, sampler: sampler,
                        maxTokens: request.maxTokens ?? container.config.maxTokens
                    )

                    let sessionBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
                    let (stream, genTask) = generateTask(
                        promptTokenCount: promptTokenCount,
                        modelConfiguration: modelConfig,
                        tokenizer: mlxTokenizer,
                        iterator: iterator,
                        wiredMemoryTicket: engine.createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: request.maxTokens ?? container.config.maxTokens, bytesPerToken: sessionBytesPerToken)
                    )

                    let startTime = Date()
                    var finishReason: FinishReason = .stop
                    var completionTokens = 0
                    var streamAccumulated = ""
                    var streamYieldedCount = 0
                    let controlPatterns = Self.controlTokensForModel(modelId: request.model)

                    for await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            let (rawText, shouldStop) = Self.filterControlInChunk(text, accumulated: &streamAccumulated, yieldedCount: &streamYieldedCount, patterns: controlPatterns)
                            let cleanText = Self.scrubControlTokens(rawText)
                            if !cleanText.isEmpty {
                                completionTokens += 1
                                continuation.yield(Token(id: 0, text: cleanText))
                            }
                            if shouldStop { break }
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                continuation.yield(Token(id: 0, text: "", toolCall: tcResult))
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
