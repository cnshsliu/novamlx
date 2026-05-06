import Foundation
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

public struct OpenAIResponseFormat: Codable, Sendable {
    public let type: String
    public let jsonSchema: OpenAIJSONSchemaField?
    public let regex: String?
    public let gbnf: String?

    private enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
        case regex
        case gbnf
    }

    public init(type: String = "text", jsonSchema: OpenAIJSONSchemaField? = nil, regex: String? = nil, gbnf: String? = nil) {
        self.type = type
        self.jsonSchema = jsonSchema
        self.regex = regex
        self.gbnf = gbnf
    }
}

public struct OpenAIJSONSchemaField: Codable, Sendable {
    public let name: String?
    public let strict: Bool?
    public let schema: AnyCodableDict?

    private enum CodingKeys: String, CodingKey {
        case name, strict, schema
    }

    public init(name: String? = nil, strict: Bool? = nil, schema: AnyCodableDict? = nil) {
        self.name = name
        self.strict = strict
        self.schema = schema
    }
}

public struct AnyCodableDict: Codable, Sendable, Equatable {
    public let value: [String: AnyCodableValue]

    public init(_ dict: [String: Any]) {
        var result: [String: AnyCodableValue] = [:]
        for (k, v) in dict { result[k] = AnyCodableValue(v) }
        self.value = result
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicCodingKey.self)
        var result: [String: AnyCodableValue] = [:]
        for key in container.allKeys {
            result[key.stringValue] = try container.decode(AnyCodableValue.self, forKey: key)
        }
        self.value = result
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicCodingKey.self)
        for (k, v) in value {
            try container.encode(v, forKey: DynamicCodingKey(stringValue: k)!)
        }
    }

    public func toDict() -> [String: Any] {
        var result: [String: Any] = [:]
        for (k, v) in value { result[k] = v.toAny() }
        return result
    }
}

public enum AnyCodableValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AnyCodableValue])
    case dict([String: AnyCodableValue])

    public init(_ value: Any) {
        switch value {
        case is NSNull: self = .null
        case let b as Bool: self = .bool(b)
        case let i as Int: self = .int(i)
        case let d as Double: self = .double(d)
        case let s as String: self = .string(s)
        case let arr as [Any]: self = .array(arr.map { AnyCodableValue($0) })
        case let dict as [String: Any]:
            var result: [String: AnyCodableValue] = [:]
            for (k, v) in dict { result[k] = AnyCodableValue(v) }
            self = .dict(result)
        default: self = .null
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let b = try? container.decode(Bool.self) { self = .bool(b) }
        else if let i = try? container.decode(Int.self) { self = .int(i) }
        else if let d = try? container.decode(Double.self) { self = .double(d) }
        else if let s = try? container.decode(String.self) { self = .string(s) }
        else if let arr = try? container.decode([AnyCodableValue].self) { self = .array(arr) }
        else if let dict = try? container.decode([String: AnyCodableValue].self) {
            self = .dict(dict)
        } else { self = .null }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let b): try container.encode(b)
        case .int(let i): try container.encode(i)
        case .double(let d): try container.encode(d)
        case .string(let s): try container.encode(s)
        case .array(let arr): try container.encode(arr)
        case .dict(let dict): try container.encode(dict)
        }
    }

    public func toAny() -> Any {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let arr): return arr.map { $0.toAny() }
        case .dict(let dict): return dict.mapValues { $0.toAny() }
        }
    }
}

private struct DynamicCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int?
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { self.stringValue = "\(intValue)"; self.intValue = intValue }
}

public struct OpenAIStreamOptions: Codable, Sendable {
    public let includeUsage: Bool?

    private enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }

    public init(includeUsage: Bool? = nil) {
        self.includeUsage = includeUsage
    }
}

public struct OpenAIRequest: Codable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let tools: [[String: AnyCodable]]?
    public let toolChoice: AnyCodable?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let maxTokens: Int?
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let stop: [String]?
    public let n: Int?
    public let frequencyPenalty: Double?
    public let presencePenalty: Double?
    public let repetitionPenalty: Double?
    public let seed: UInt64?
    public let sessionId: String?
    public let responseFormat: OpenAIResponseFormat?
    public let thinkingBudget: Int?
    public let enableThinking: Bool?
    public let preserveThinking: Bool?
    public let chatTemplateKwargs: [String: AnyCodable]?

    private enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream, stop, n, seed, tools
        case toolChoice = "tool_choice"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case maxTokens = "max_tokens"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case repetitionPenalty = "repetition_penalty"
        case sessionId = "session_id"
        case responseFormat = "response_format"
        case streamOptions = "stream_options"
        case thinkingBudget = "thinking_budget"
        case enableThinking = "enable_thinking"
        case preserveThinking = "preserve_thinking"
        case chatTemplateKwargs = "chat_template_kwargs"
    }

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        tools: [[String: AnyCodable]]? = nil,
        toolChoice: AnyCodable? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        maxTokens: Int? = nil,
        stream: Bool? = nil,
        streamOptions: OpenAIStreamOptions? = nil,
        stop: [String]? = nil,
        n: Int? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        seed: UInt64? = nil,
        sessionId: String? = nil,
        responseFormat: OpenAIResponseFormat? = nil,
        thinkingBudget: Int? = nil,
        enableThinking: Bool? = nil,
        preserveThinking: Bool? = nil,
        chatTemplateKwargs: [String: AnyCodable]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.tools = tools
        self.toolChoice = toolChoice
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.maxTokens = maxTokens
        self.stream = stream
        self.streamOptions = streamOptions
        self.stop = stop
        self.n = n
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.sessionId = sessionId
        self.responseFormat = responseFormat
        self.thinkingBudget = thinkingBudget
        self.enableThinking = enableThinking
        self.preserveThinking = preserveThinking
        self.chatTemplateKwargs = chatTemplateKwargs
    }

    /// Resolve thinking toggle from multiple client formats:
    /// 1. `enable_thinking: false` (Qwen / DashScope)
    /// 2. `chat_template_kwargs.enable_thinking: false` (vLLM / SGLang)
    /// 3. `chat_template_kwargs.thinking: false` (alternative Qwen)
    public var resolvedEnableThinking: Bool? {
        if let v = enableThinking { return v }
        if let kwargs = chatTemplateKwargs {
            if let ac = kwargs["enable_thinking"] {
                if case .bool(let v) = ac { return v }
            }
            if let ac = kwargs["thinking"] {
                if case .bool(let v) = ac { return v }
            }
        }
        return nil
    }

    /// Resolve `preserve_thinking` toggle from multiple client formats:
    /// 1. `preserve_thinking: true` (top-level)
    /// 2. `chat_template_kwargs.preserve_thinking: true` (vLLM / SGLang)
    ///
    /// When set, the chat template is asked to keep historical
    /// `reasoning_content` blocks in the rendered prompt — useful for models
    /// that benefit from re-grounding on prior reasoning (Qwen3.5/3.6).
    /// Templates that don't reference the variable simply ignore it (Jinja
    /// `is defined` evaluates false), so forwarding to non-Qwen families is
    /// safe.
    public var resolvedPreserveThinking: Bool? {
        if let v = preserveThinking { return v }
        if let kwargs = chatTemplateKwargs {
            if let ac = kwargs["preserve_thinking"] {
                if case .bool(let v) = ac { return v }
            }
        }
        return nil
    }

}

public enum MessageContent: Codable, Sendable {
    case text(String)
    case parts([ContentPart])

    public var textValue: String? {
        switch self {
        case .text(let s): s
        case .parts(let parts):
            parts.compactMap { part -> String? in
                if case .text(let t) = part { return t } else { return nil }
            }.joined()
        }
    }

    public var imageParts: [ContentPart] {
        switch self {
        case .text: []
        case .parts(let parts): parts.filter { if case .imageUrl = $0 { return true } else { return false } }
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else {
            self = .parts(try container.decode([ContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .parts(let parts): try container.encode(parts)
        }
    }
}

public enum ContentPart: Codable, Sendable {
    case text(String)
    case imageUrl(ImageURL)

    private enum CodingKeys: String, CodingKey {
        case type, text
        case imageUrl = "image_url"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "image_url":
            let url = try container.decode(ImageURL.self, forKey: .imageUrl)
            self = .imageUrl(url)
        default:
            self = .text("")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .imageUrl(let url):
            try container.encode("image_url", forKey: .type)
            try container.encode(url, forKey: .imageUrl)
        }
    }
}

public struct ImageURL: Codable, Sendable {
    public let url: String
    public let detail: String?

    public init(url: String, detail: String? = nil) {
        self.url = url
        self.detail = detail
    }
}

public struct OpenAIChatMessage: Codable, Sendable {
    public let role: String
    public let content: MessageContent?
    public let reasoningContent: String?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallId: String?
    public let name: String?

    private enum CodingKeys: String, CodingKey {
        case role, content, name
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
    }

    public init(role: String, content: String? = nil, reasoningContent: String? = nil, toolCalls: [OpenAIToolCall]? = nil, toolCallId: String? = nil, name: String? = nil) {
        self.role = role
        self.content = content.map { .text($0) }
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
    }

    public init(role: String, content: MessageContent?, reasoningContent: String? = nil, toolCalls: [OpenAIToolCall]? = nil, toolCallId: String? = nil, name: String? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
    }
}

public struct OpenAIToolCall: Codable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall

    public init(id: String, function: OpenAIFunctionCall) {
        self.id = id
        self.type = "function"
        self.function = function
    }
}

public struct OpenAIFunctionCall: Codable, Sendable {
    public let name: String
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct OpenAIResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAIChoice]
    public let usage: OpenAIUsage?

    public init(id: String, model: String, choices: [OpenAIChoice], usage: OpenAIUsage? = nil) {
        self.id = id
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAIChoice: Codable, Sendable {
    public let index: Int
    public let message: OpenAIChatMessage
    public let finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
    }

    public init(index: Int, message: OpenAIChatMessage, finishReason: String? = nil) {
        self.index = index
        self.message = message
        self.finishReason = finishReason
    }
}

public struct OpenAIUsage: Codable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }
}

public struct OpenAIStreamChunk: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAIStreamChoice]
    public let usage: OpenAIUsage?

    public init(id: String, model: String, choices: [OpenAIStreamChoice], usage: OpenAIUsage? = nil) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAIStreamChoice: Codable, Sendable {
    public let index: Int
    public let delta: OpenAIDelta
    public let finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }

    public init(index: Int, delta: OpenAIDelta, finishReason: String? = nil) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }
}

public struct OpenAIDelta: Codable, Sendable {
    public let role: String?
    public let content: String?
    public let reasoningContent: String?
    public let toolCalls: [OpenAIToolCallDelta]?

    private enum CodingKeys: String, CodingKey {
        case role, content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    public init(role: String? = nil, content: String? = nil, reasoningContent: String? = nil, toolCalls: [OpenAIToolCallDelta]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }
}

public struct OpenAIToolCallDelta: Codable, Sendable {
    public let index: Int
    public let id: String?
    public let type: String?
    public let function: OpenAIFunctionCallDelta?

    public init(index: Int, id: String? = nil, type: String? = nil, function: OpenAIFunctionCallDelta? = nil) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct OpenAIFunctionCallDelta: Codable, Sendable {
    public let name: String?
    public let arguments: String?

    public init(name: String? = nil, arguments: String? = nil) {
        self.name = name
        self.arguments = arguments
    }
}

public struct OpenAIModelsResponse: Codable, Sendable {
    public let object: String
    public let data: [OpenAIModel]

    public init(data: [OpenAIModel]) {
        self.object = "list"
        self.data = data
    }
}

public struct OpenAIModel: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let ownedBy: String

    private enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }

    public init(id: String) {
        self.id = id
        self.object = "model"
        self.created = Int(Date().timeIntervalSince1970)
        self.ownedBy = "novamlx"
    }
}

public struct OpenAIErrorResponse: Codable, Sendable {
    public let error: OpenAIErrorDetail

    public init(error: OpenAIErrorDetail) {
        self.error = error
    }
}

public struct OpenAIErrorDetail: Codable, Sendable {
    public let message: String
    public let type: String
    public let param: String?
    public let code: String?

    public init(message: String, type: String, param: String? = nil, code: String? = nil) {
        self.message = message
        self.type = type
        self.param = param
        self.code = code
    }
}

public struct AdminModelStatus: Codable, Sendable {
    public let id: String
    public let family: String
    public let downloaded: Bool
    public let loaded: Bool
    public let sizeBytes: UInt64
    public let downloadedAt: String?
    public let memoryFeasibility: MemoryFeasibility?

    public init(
        id: String,
        family: String,
        downloaded: Bool,
        loaded: Bool,
        sizeBytes: UInt64,
        downloadedAt: Date?,
        memoryFeasibility: MemoryFeasibility? = nil
    ) {
        self.id = id
        self.family = family
        self.downloaded = downloaded
        self.loaded = loaded
        self.sizeBytes = sizeBytes
        self.memoryFeasibility = memoryFeasibility
        if let date = downloadedAt {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            self.downloadedAt = formatter.string(from: date)
        } else {
            self.downloadedAt = nil
        }
    }
}

public struct AdminDownloadRequest: Codable, Sendable {
    public let modelId: String

    public init(modelId: String) {
        self.modelId = modelId
    }
}

public struct AdminLoadRequest: Codable, Sendable {
    public let modelId: String

    public init(modelId: String) {
        self.modelId = modelId
    }
}

public struct AdminSessionForkRequest: Codable, Sendable {
    public let sourceId: String
    public let targetId: String
    public let modelId: String

    public init(sourceId: String, targetId: String, modelId: String) {
        self.sourceId = sourceId
        self.targetId = targetId
        self.modelId = modelId
    }
}

public struct AdminDownloadProgress: Codable, Sendable {
    public let modelId: String
    public let bytesDownloaded: UInt64
    public let totalBytes: UInt64
    public let fractionCompleted: Double
    public let status: String

    public init(modelId: String, progress: NovaMLXModelManager.DownloadProgress, status: String = "downloading") {
        self.modelId = modelId
        self.bytesDownloaded = progress.bytesDownloaded
        self.totalBytes = progress.totalBytes
        self.fractionCompleted = progress.fractionCompleted
        self.status = status
    }
}

public struct AdminDiscoveredModel: Codable, Sendable {
    public let id: String
    public let type: String
    public let family: String
    public let path: String

    public init(id: String, type: String, family: String, path: String) {
        self.id = id
        self.type = type
        self.family = family
        self.path = path
    }
}

public struct AdminDiscoverResponse: Codable, Sendable {
    public let discovered: [AdminDiscoveredModel]
    public let count: Int

    public init(discovered: [AdminDiscoveredModel]) {
        self.discovered = discovered
        self.count = discovered.count
    }
}

public struct AdminModelSettingsResponse: Codable, Sendable {
    public let modelId: String
    public let settings: NovaMLXCore.ModelSettings

    private enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case settings
    }

    public init(modelId: String, settings: NovaMLXCore.ModelSettings) {
        self.modelId = modelId
        self.settings = settings
    }
}

public struct ModelSettingsUpdateRequest: Codable, Sendable {
    public var maxContextWindow: Int?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Float?
    public var repetitionPenalty: Float?
    public var presencePenalty: Float?
    public var frequencyPenalty: Float?
    public var ttlSeconds: Int?
    public var modelAlias: String?
    public var isPinned: Bool?
    public var isDefault: Bool?
    public var displayName: String?
    public var description: String?
    public var thinkingBudget: Int?
    public var kvBits: Int?
    public var kvGroupSize: Int?
    public var kvMemoryBytesPerTokenOverride: Int?

    private enum CodingKeys: String, CodingKey {
        case maxContextWindow = "max_context_window"
        case maxTokens = "max_tokens"
        case temperature, topP = "top_p", topK = "top_k", minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case ttlSeconds = "ttl_seconds"
        case modelAlias = "model_alias"
        case isPinned = "is_pinned"
        case isDefault = "is_default"
        case displayName = "display_name"
        case description
        case thinkingBudget = "thinking_budget"
        case kvBits = "kv_bits"
        case kvGroupSize = "kv_group_size"
        case kvMemoryBytesPerTokenOverride = "kv_memory_bytes_per_token_override"
    }
}

public struct OpenAICompletionRequest: Codable, Sendable {
    public let model: String
    public let prompt: String
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let maxTokens: Int?
    public let stream: Bool?
    public let stop: [String]?
    public let frequencyPenalty: Double?
    public let presencePenalty: Double?
    public let repetitionPenalty: Double?
    public let seed: UInt64?
    public let n: Int?

    private enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, stream, stop, n
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case maxTokens = "max_tokens"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case repetitionPenalty = "repetition_penalty"
        case seed
    }
}

public struct OpenAICompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAICompletionChoice]
    public let usage: OpenAIUsage?

    public init(id: String, model: String, choices: [OpenAICompletionChoice], usage: OpenAIUsage? = nil) {
        self.id = id
        self.object = "text_completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAICompletionChoice: Codable, Sendable {
    public let index: Int
    public let text: String
    public let finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }
}

public enum EmbeddingInput: Codable, Sendable {
    case single(String)
    case multiple([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .single(str)
        } else {
            self = .multiple(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let str): try container.encode(str)
        case .multiple(let arr): try container.encode(arr)
        }
    }

    public var texts: [String] {
        switch self {
        case .single(let str): [str]
        case .multiple(let arr): arr
        }
    }
}

public struct OpenAIEmbeddingRequest: Codable, Sendable {
    public let model: String
    public let input: EmbeddingInput
    public let encodingFormat: String?

    private enum CodingKeys: String, CodingKey {
        case model, input
        case encodingFormat = "encoding_format"
    }

    public init(model: String, input: EmbeddingInput, encodingFormat: String? = nil) {
        self.model = model
        self.input = input
        self.encodingFormat = encodingFormat
    }
}

public struct OpenAIEmbeddingResponse: Codable, Sendable {
    public let object: String
    public let data: [OpenAIEmbeddingData]
    public let model: String
    public let usage: OpenAIEmbeddingUsage

    public init(model: String, data: [OpenAIEmbeddingData], usage: OpenAIEmbeddingUsage) {
        self.object = "list"
        self.model = model
        self.data = data
        self.usage = usage
    }
}

public struct OpenAIEmbeddingData: Codable, Sendable {
    public let object: String
    public let index: Int
    public let embedding: AnyCodableArray

    public init(index: Int, embedding: [Float]) {
        self.object = "embedding"
        self.index = index
        self.embedding = AnyCodableArray(embedding)
    }
}

public struct AnyCodableArray: Codable, Sendable, Equatable {
    public let values: [Double]

    public init(_ floats: [Float]) {
        self.values = floats.map { Double($0) }
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        var result: [Double] = []
        while !container.isAtEnd {
            result.append(try container.decode(Double.self))
        }
        self.values = result
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        for v in values {
            try container.encode(v)
        }
    }
}

public struct OpenAIEmbeddingUsage: Codable, Sendable {
    public let promptTokens: Int
    public let totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, totalTokens: Int) {
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
    }
}

public struct OpenAICompletionStreamChunk: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAICompletionStreamChoice]
    public let usage: OpenAIUsage?

    public init(id: String, model: String, choices: [OpenAICompletionStreamChoice], usage: OpenAIUsage? = nil) {
        self.id = id
        self.object = "text_completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAICompletionStreamChoice: Codable, Sendable {
    public let index: Int
    public let text: String
    public let finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }

    public init(index: Int, text: String, finishReason: String? = nil) {
        self.index = index
        self.text = text
        self.finishReason = finishReason
    }
}

public struct OpenAIResponseRequest: Codable, Sendable {
    public let model: String
    public let input: ResponseInput?
    public let instructions: String?
    public let temperature: Double?
    public let topP: Double?
    public let maxOutputTokens: Int?
    public let stream: Bool?

    private enum CodingKeys: String, CodingKey {
        case model, input, instructions, temperature, stream
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
    }

    public init(
        model: String,
        input: ResponseInput? = nil,
        instructions: String? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        maxOutputTokens: Int? = nil,
        stream: Bool? = nil
    ) {
        self.model = model
        self.input = input
        self.instructions = instructions
        self.temperature = temperature
        self.topP = topP
        self.maxOutputTokens = maxOutputTokens
        self.stream = stream
    }
}

public enum ResponseInput: Codable, Sendable {
    case text(String)
    case messages([ResponseInputMessage])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else {
            self = .messages(try container.decode([ResponseInputMessage].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .messages(let msgs): try container.encode(msgs)
        }
    }
}

public struct ResponseInputMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct OpenAIResponseObject: Codable, Sendable {
    public let id: String
    public let object: String
    public let createdAt: Int
    public let model: String
    public let status: String
    public let output: [ResponseOutputItem]
    public let usage: OpenAIUsage?

    private enum CodingKeys: String, CodingKey {
        case id, object, model, status, output, usage
        case createdAt = "created_at"
    }

    public init(
        id: String,
        model: String,
        status: String = "completed",
        output: [ResponseOutputItem],
        usage: OpenAIUsage? = nil
    ) {
        self.id = id
        self.object = "response"
        self.createdAt = Int(Date().timeIntervalSince1970)
        self.model = model
        self.status = status
        self.output = output
        self.usage = usage
    }
}

public struct ResponseOutputItem: Codable, Sendable {
    public let type: String
    public let id: String
    public let role: String
    public let content: [ResponseContentItem]

    public init(id: String, role: String = "assistant", content: [ResponseContentItem]) {
        self.type = "message"
        self.id = id
        self.role = role
        self.content = content
    }
}

public struct ResponseContentItem: Codable, Sendable {
    public let type: String
    public let text: String

    public init(text: String) {
        self.type = "output_text"
        self.text = text
    }
}
