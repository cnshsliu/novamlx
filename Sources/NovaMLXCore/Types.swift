import Foundation
import Logging

public enum NovaMLX {}

public let version = "1.0.6"

public var buildTimestamp: String {
    guard let execURL = Bundle.main.executableURL,
          let attrs = try? FileManager.default.attributesOfItem(atPath: execURL.path),
          let modDate = attrs[.modificationDate] as? Date
    else { return "Unknown" }
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
    return formatter.string(from: modDate)
}

public enum NovaMLXError: Error, LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String, underlying: Error)
    case inferenceFailed(String)
    case configurationError(String)
    case cacheError(String)
    case apiError(String)
    case downloadFailed(String, underlying: Error)
    case unsupportedModel(String)
    case contextWindowExceeded(promptTokens: Int, maxTokens: Int, contextLength: Int)
    case insufficientMemory(neededMB: UInt64, availableMB: UInt64, modelId: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name): "Model not found: \(name)"
        case .modelLoadFailed(let name, let err): "Failed to load model '\(name)': \(err.localizedDescription)"
        case .inferenceFailed(let msg): "Inference failed: \(msg)"
        case .configurationError(let msg): "Configuration error: \(msg)"
        case .cacheError(let msg): "Cache error: \(msg)"
        case .apiError(let msg): "API error: \(msg)"
        case .downloadFailed(let url, let err): "Download failed for \(url): \(err.localizedDescription)"
        case .unsupportedModel(let name): "Unsupported model: \(name)"
        case .contextWindowExceeded(let promptTokens, let maxTokens, let contextLength):
            "Context window exceeded: prompt has \(promptTokens) tokens + max_tokens \(maxTokens) = \(promptTokens + maxTokens), but model context length is \(contextLength). Reduce your prompt or max_tokens."
        case .insufficientMemory(let neededMB, let availableMB, let modelId):
            "Insufficient memory to load '\(modelId)': need \(neededMB)MB but only \(availableMB)MB available under the current memory limit. Unload unused models, pin important ones, or increase maxProcessMemory."
        }
    }
}

public enum QuantizationScheme: String, Codable, Sendable {
    case affine
}

public struct ModelIdentifier: Hashable, Codable, Sendable {
    public let id: String
    public let family: ModelFamily

    public init(id: String, family: ModelFamily) {
        self.id = id
        self.family = family
    }

    public var displayName: String { id }
}

public enum ModelFamily: String, Codable, Sendable, CaseIterable {
    case llama
    case mistral
    case phi
    case qwen
    case gemma
    case starcoder
    case claude
    case bailing
    case gptOss
    case other

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ModelFamily(rawValue: raw) ?? .other
    }
}

public enum ModelType: String, Codable, Sendable {
    case llm
    case vlm
    case embedding
}

public struct ModelConfig: Codable, Sendable {
    public let identifier: ModelIdentifier
    public let modelType: ModelType
    public var hasLinearAttention: Bool
    public var contextLength: Int
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    public var repeatPenalty: Float
    public let repeatLastN: Int
    public let seed: UInt64?
    public let kvBits: Int?
    public let kvGroupSize: Int
    public var kvMemoryBytesPerToken: Int?
    public var topK: Int
    public var minP: Float
    public var frequencyPenalty: Float
    public var presencePenalty: Float

    public init(
        identifier: ModelIdentifier,
        modelType: ModelType = .llm,
        hasLinearAttention: Bool = false,
        contextLength: Int = 4096,
        maxTokens: Int = 2048,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        repeatPenalty: Float = 1.0,
        repeatLastN: Int = 64,
        seed: UInt64? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        kvMemoryBytesPerToken: Int? = nil,
        topK: Int = 0,
        minP: Float = 0.0,
        frequencyPenalty: Float = 0,
        presencePenalty: Float = 0
    ) {
        self.identifier = identifier
        self.modelType = modelType
        self.hasLinearAttention = hasLinearAttention
        self.contextLength = contextLength
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repeatPenalty = repeatPenalty
        self.repeatLastN = repeatLastN
        self.seed = seed
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.kvMemoryBytesPerToken = kvMemoryBytesPerToken
        self.topK = topK
        self.minP = minP
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
    }
}

/// Parameters parsed from HuggingFace generation_config.json.
/// All fields optional — only values present in the file are populated.
public struct GenerationConfig: Codable, Sendable {
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let frequencyPenalty: Float?
    public let presencePenalty: Float?
    public let maxNewTokens: Int?

    private enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case maxNewTokens = "max_new_tokens"
    }
}

public enum ResponseFormat: String, Codable, Sendable {
    case text
    case jsonObject = "json_object"
}

public struct ToolCallResult: Codable, Sendable {
    public let id: String
    public let functionName: String
    public let arguments: String

    public init(id: String, functionName: String, arguments: String) {
        self.id = id
        self.functionName = functionName
        self.arguments = arguments
    }
}

public struct InferenceRequest: @unchecked Sendable {
    public let id: UUID
    public let model: String
    public let messages: [ChatMessage]
    public let tools: [[String: Any]]?
    public let temperature: Double?
    public let maxTokens: Int?
    public let topP: Double?
    public let topK: Int?
    public let minP: Float?
    public let frequencyPenalty: Float?
    public let presencePenalty: Float?
    public let repetitionPenalty: Float?
    public let seed: UInt64?
    public let stream: Bool
    public let stop: [String]?
    public let sessionId: String?
    public let responseFormat: ResponseFormat?
    public let jsonSchemaDef: [String: Any]?
    public let regexPattern: String?
    public let gbnfGrammar: String?
    public let thinkingBudget: Int?
    public let enableThinking: Bool?
    /// When true, the chat template is asked to keep historical
    /// `reasoning_content` blocks in the rendered prompt (Qwen3.5/3.6's
    /// `preserve_thinking` Jinja kwarg). Templates that don't reference the
    /// variable ignore it via Jinja `is defined` semantics, so forwarding to
    /// non-Qwen families is safe.
    public let preserveThinking: Bool?
    /// Draft model ID for speculative decoding. If provided, the engine uses
    /// SpeculativeTokenIterator (draft-model verify) instead of plain autoregressive decode.
    /// Both models must share the same tokenizer.
    public let draftModel: String?
    /// Number of tokens the draft model proposes per speculation round (default: 4).
    public let numDraftTokens: Int?

    public init(
        id: UUID = UUID(),
        model: String,
        messages: [ChatMessage],
        tools: [[String: Any]]? = nil,
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        frequencyPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        repetitionPenalty: Float? = nil,
        seed: UInt64? = nil,
        stream: Bool = false,
        stop: [String]? = nil,
        sessionId: String? = nil,
        responseFormat: ResponseFormat? = nil,
        jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil,
        gbnfGrammar: String? = nil,
        thinkingBudget: Int? = nil,
        enableThinking: Bool? = nil,
        preserveThinking: Bool? = nil,
        draftModel: String? = nil,
        numDraftTokens: Int? = nil
    ) {
        self.id = id
        self.model = model
        self.messages = messages
        self.tools = tools
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stream = stream
        self.stop = stop
        self.sessionId = sessionId
        self.responseFormat = responseFormat
        self.jsonSchemaDef = jsonSchemaDef
        self.regexPattern = regexPattern
        self.gbnfGrammar = gbnfGrammar
        self.thinkingBudget = thinkingBudget
        self.enableThinking = enableThinking
        self.preserveThinking = preserveThinking
        self.draftModel = draftModel
        self.numDraftTokens = numDraftTokens
    }
}

public struct ChatMessage: Codable, Sendable {
    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    public let role: Role
    public let content: String?
    public let images: [String]?
    public let name: String?
    public let toolCallId: String?
    public let toolCalls: [ToolCallResult]?

    public init(role: Role, content: String? = nil, images: [String]? = nil, name: String? = nil, toolCallId: String? = nil, toolCalls: [ToolCallResult]? = nil) {
        self.role = role
        self.content = content
        self.images = images
        self.name = name
        self.toolCallId = toolCallId
        self.toolCalls = toolCalls
    }
}

public struct InferenceResult: Codable, Sendable {
    public let id: UUID
    public let model: String
    public let text: String
    public let toolCalls: [ToolCallResult]?
    public let tokensPerSecond: Double
    public let promptTokens: Int
    public let completionTokens: Int
    public let finishReason: FinishReason

    public init(
        id: UUID,
        model: String,
        text: String,
        toolCalls: [ToolCallResult]? = nil,
        tokensPerSecond: Double,
        promptTokens: Int,
        completionTokens: Int,
        finishReason: FinishReason
    ) {
        self.id = id
        self.model = model
        self.text = text
        self.toolCalls = toolCalls
        self.tokensPerSecond = tokensPerSecond
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.finishReason = finishReason
    }
}

public enum FinishReason: String, Codable, Sendable {
    case stop
    case length
    case toolCalls = "tool_calls"
}

public struct Token: Codable, Sendable {
    public let id: Int
    public let text: String
    public let logprob: Float?
    public let finishReason: FinishReason?
    public let toolCall: ToolCallResult?

    public init(id: Int, text: String, logprob: Float? = nil, finishReason: FinishReason? = nil, toolCall: ToolCallResult? = nil) {
        self.id = id
        self.text = text
        self.logprob = logprob
        self.finishReason = finishReason
        self.toolCall = toolCall
    }
}

public protocol TokenizerProtocol: Sendable {
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String
    var vocabularySize: Int { get }
    var bosToken: String? { get }
    var eosToken: String? { get }
}

public protocol ModelProtocol: Sendable {
    var identifier: ModelIdentifier { get }
    var config: ModelConfig { get }
    func warmup() async throws
}

public protocol InferenceEngineProtocol: Sendable {
    func generate(_ request: InferenceRequest) async throws -> InferenceResult
    func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error>
    func abort(requestId: UUID) async
    var isReady: Bool { get }
}

public protocol ModelRegistryProtocol: Sendable {
    func availableModels() async -> [ModelIdentifier]
    func loadModel(_ identifier: ModelIdentifier) async throws -> ModelProtocol
    func unloadModel(_ identifier: ModelIdentifier) async throws
    func isLoaded(_ identifier: ModelIdentifier) async -> Bool
}

public enum ProcessMemoryLimit: Codable, Sendable, Equatable {
    case auto
    case disabled
    case percent(Double)
    case bytes(UInt64)

    private static let _4GB: UInt64 = 4 * 1024 * 1024 * 1024
    private static let _8GB: UInt64 = 8 * 1024 * 1024 * 1024
    private static let _2GB: UInt64 = 2 * 1024 * 1024 * 1024

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        self = Self.parse(raw)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .auto: try container.encode("auto")
        case .disabled: try container.encode("disabled")
        case .percent(let p): try container.encode("\(Int(p))%")
        case .bytes(let b):
            if b >= 1024 * 1024 * 1024 {
                try container.encode("\(b / (1024 * 1024 * 1024))GB")
            } else {
                try container.encode("\(b / (1024 * 1024))MB")
            }
        }
    }

    public static func parse(_ raw: String) -> Self {
        let trimmed = raw.trimmingCharacters(in: .whitespaces).lowercased()
        if trimmed == "auto" { return .auto }
        if trimmed == "disabled" || trimmed == "none" || trimmed == "off" { return .disabled }
        if trimmed.hasSuffix("%"), let p = Double(trimmed.dropLast()) {
            return .percent(min(max(p, 10), 90))
        }
        if trimmed.hasSuffix("gb"), let v = Double(trimmed.dropLast(2)) {
            return .bytes(UInt64(v * 1024 * 1024 * 1024))
        }
        if trimmed.hasSuffix("mb"), let v = Double(trimmed.dropLast(2)) {
            return .bytes(UInt64(v * 1024 * 1024))
        }
        if let v = UInt64(trimmed) { return .bytes(v) }
        return .auto
    }

    /// Resolve to concrete byte limit. Returns nil for .disabled.
    /// Result is capped at physMem - 4GB (absolute minimum reserve for macOS).
    public func resolveBytes(physicalRAM: UInt64 = UInt64(ProcessInfo.processInfo.physicalMemory)) -> UInt64? {
        switch self {
        case .auto:
            let reserve = max(Self._4GB, min(Self._8GB, physicalRAM / 5))
            let limit = physicalRAM > reserve ? physicalRAM - reserve : 0
            return min(limit, physicalRAM - Self._4GB)
        case .disabled:
            return nil
        case .percent(let p):
            let limit = UInt64(Double(physicalRAM) * p / 100.0)
            return min(limit, physicalRAM - Self._4GB)
        case .bytes(let b):
            return min(b, physicalRAM - Self._4GB)
        }
    }

    /// Hard limit = max(softLimit, physMem - 4GB). Always has a floor.
    public func resolveHardLimit(
        physicalRAM: UInt64 = UInt64(ProcessInfo.processInfo.physicalMemory)
    ) -> UInt64 {
        return physicalRAM > Self._4GB ? physicalRAM - Self._4GB : physicalRAM / 2
    }
}

public struct ServerConfig: Codable, Sendable {
    public let host: String
    public let port: Int
    public let adminPort: Int
    public let apiKeys: [String]
    public let maxConcurrentRequests: Int
    public let requestTimeout: TimeInterval
    public let contextScalingTarget: Int?
    public let tlsCertPath: String?
    public let tlsKeyPath: String?
    public let tlsKeyPassword: String?
    public let maxRequestSizeMB: Double
    public let maxProcessMemory: String
    /// Operational kill switch for the prefix cache subsystem.
    /// When `false`, no SSD cache directories are created, no `fetchPrefix`
    /// or `storeCache` calls are issued, and `getOrCreatePrefixCacheManager`
    /// returns `nil` for every model.
    public let prefixCacheEnabled: Bool

    private enum CodingKeys: String, CodingKey {
        case host, port, adminPort, apiKeys, maxConcurrentRequests
        case requestTimeout, contextScalingTarget
        case tlsCertPath, tlsKeyPath, tlsKeyPassword, maxRequestSizeMB
        case maxProcessMemory
        case prefixCacheEnabled
    }

    public init(
        host: String = "127.0.0.1",
        port: Int = 6590,
        adminPort: Int = 6591,
        apiKeys: [String] = [],
        maxConcurrentRequests: Int = 16,
        requestTimeout: TimeInterval = 300,
        contextScalingTarget: Int? = nil,
        tlsCertPath: String? = nil,
        tlsKeyPath: String? = nil,
        tlsKeyPassword: String? = nil,
        maxRequestSizeMB: Double = 100,
        maxProcessMemory: String = "auto",
        prefixCacheEnabled: Bool = true
    ) {
        self.host = host
        self.port = port
        self.adminPort = adminPort
        self.apiKeys = apiKeys
        self.maxConcurrentRequests = maxConcurrentRequests
        self.requestTimeout = requestTimeout
        self.contextScalingTarget = contextScalingTarget
        self.tlsCertPath = tlsCertPath
        self.tlsKeyPath = tlsKeyPath
        self.tlsKeyPassword = tlsKeyPassword
        self.maxRequestSizeMB = maxRequestSizeMB
        self.maxProcessMemory = maxProcessMemory
        self.prefixCacheEnabled = prefixCacheEnabled
    }

    public var isTLSEnabled: Bool { tlsCertPath != nil }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        host = try container.decodeIfPresent(String.self, forKey: .host) ?? "127.0.0.1"
        port = try container.decodeIfPresent(Int.self, forKey: .port) ?? 6590
        adminPort = try container.decodeIfPresent(Int.self, forKey: .adminPort) ?? 6591
        apiKeys = try container.decodeIfPresent([String].self, forKey: .apiKeys) ?? []
        maxConcurrentRequests = try container.decodeIfPresent(Int.self, forKey: .maxConcurrentRequests) ?? 16
        requestTimeout = try container.decodeIfPresent(TimeInterval.self, forKey: .requestTimeout) ?? 300
        contextScalingTarget = try container.decodeIfPresent(Int.self, forKey: .contextScalingTarget)
        tlsCertPath = try container.decodeIfPresent(String.self, forKey: .tlsCertPath)
        tlsKeyPath = try container.decodeIfPresent(String.self, forKey: .tlsKeyPath)
        tlsKeyPassword = try container.decodeIfPresent(String.self, forKey: .tlsKeyPassword)
        maxRequestSizeMB = try container.decodeIfPresent(Double.self, forKey: .maxRequestSizeMB) ?? 100
        maxProcessMemory = try container.decodeIfPresent(String.self, forKey: .maxProcessMemory) ?? "auto"
        prefixCacheEnabled = try container.decodeIfPresent(Bool.self, forKey: .prefixCacheEnabled) ?? true
    }

    public func scaleTokenCount(_ count: Int, modelContextWindow: Int) -> Int {
        guard let target = contextScalingTarget, target > 0, modelContextWindow > 0 else { return count }
        return Int(Double(count) * Double(target) / Double(modelContextWindow))
    }
}

public struct SystemStats: Sendable {
    public let cpuUsage: Double
    public let memoryUsed: UInt64
    public let memoryTotal: UInt64
    public let gpuMemoryUsed: UInt64
    public let activeRequests: Int
    public let tokensPerSecond: Double
    public let uptime: TimeInterval

    public init(
        cpuUsage: Double = 0,
        memoryUsed: UInt64 = 0,
        memoryTotal: UInt64 = 0,
        gpuMemoryUsed: UInt64 = 0,
        activeRequests: Int = 0,
        tokensPerSecond: Double = 0,
        uptime: TimeInterval = 0
    ) {
        self.cpuUsage = cpuUsage
        self.memoryUsed = memoryUsed
        self.memoryTotal = memoryTotal
        self.gpuMemoryUsed = gpuMemoryUsed
        self.activeRequests = activeRequests
        self.tokensPerSecond = tokensPerSecond
        self.uptime = uptime
    }

    public var memoryUsagePercent: Double {
        guard memoryTotal > 0 else { return 0 }
        return Double(memoryUsed) / Double(memoryTotal) * 100
    }
}

// MARK: - Download Management

public enum DownloadStatus: Sendable, Equatable {
    case pending
    case downloading
    case completed
    case failed
}

public struct DownloadTaskInfo: Sendable {
    public let repoId: String
    public var taskId: String?
    public var status: DownloadStatus
    public var progress: Double
    public var downloadedBytes: Int64
    public var totalBytes: Int64
    public var errorMessage: String?
    public let startedAt: Date
    public var fileProgresses: [FileDownloadInfo]

    public init(repoId: String) {
        self.repoId = repoId
        self.taskId = nil
        self.status = .pending
        self.progress = 0
        self.downloadedBytes = 0
        self.totalBytes = 0
        self.errorMessage = nil
        self.startedAt = Date()
        self.fileProgresses = []
    }

    public var isActive: Bool { status == .downloading || status == .pending }
}

/// Lightweight per-file progress info for UI display
public struct FileDownloadInfo: Sendable, Identifiable {
    public var id: String { filename }
    public let filename: String
    public var downloadedBytes: Int64
    public var totalBytes: Int64
    public var status: String  // "downloading", "completed", "failed", "waiting"

    public init(filename: String, downloadedBytes: Int64 = 0, totalBytes: Int64 = 0, status: String = "waiting") {
        self.filename = filename
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.status = status
    }
}
