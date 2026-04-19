import Foundation

public struct ModelSettings: Codable, Sendable, Equatable {
    public var maxContextWindow: Int?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Float?
    public var repetitionPenalty: Float?
    public var presencePenalty: Float?
    public var frequencyPenalty: Float?
    public var seed: UInt64?
    public var ttlSeconds: Int?
    public var modelTypeOverride: ModelType?
    public var modelAlias: String?
    public var isPinned: Bool
    public var isDefault: Bool
    public var displayName: String?
    public var description: String?
    public var kvBits: Int?
    public var kvGroupSize: Int?
    public var thinkingBudget: Int?
    public var kvMemoryBytesPerTokenOverride: Int?

    public init(
        maxContextWindow: Int? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        seed: UInt64? = nil,
        ttlSeconds: Int? = nil,
        modelTypeOverride: ModelType? = nil,
        modelAlias: String? = nil,
        isPinned: Bool = false,
        isDefault: Bool = false,
        displayName: String? = nil,
        description: String? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int? = nil,
        thinkingBudget: Int? = nil,
        kvMemoryBytesPerTokenOverride: Int? = nil
    ) {
        self.maxContextWindow = maxContextWindow
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.seed = seed
        self.ttlSeconds = ttlSeconds
        self.modelTypeOverride = modelTypeOverride
        self.modelAlias = modelAlias
        self.isPinned = isPinned
        self.isDefault = isDefault
        self.displayName = displayName
        self.description = description
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.thinkingBudget = thinkingBudget
        self.kvMemoryBytesPerTokenOverride = kvMemoryBytesPerTokenOverride
    }

    private enum CodingKeys: String, CodingKey {
        case maxContextWindow = "max_context_window"
        case maxTokens = "max_tokens"
        case temperature, topP = "top_p", topK = "top_k", minP = "min_p", seed
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case ttlSeconds = "ttl_seconds"
        case modelTypeOverride = "model_type_override"
        case modelAlias = "model_alias"
        case isPinned = "is_pinned"
        case isDefault = "is_default"
        case displayName = "display_name"
        case description
        case kvBits = "kv_bits"
        case kvGroupSize = "kv_group_size"
        case thinkingBudget = "thinking_budget"
        case kvMemoryBytesPerTokenOverride = "kv_memory_bytes_per_token_override"
    }

    public func applySamplingOverrides(to request: InferenceRequest) -> InferenceRequest {
        InferenceRequest(
            id: request.id,
            model: request.model,
            messages: request.messages,
            temperature: request.temperature ?? temperature,
            maxTokens: request.maxTokens ?? maxTokens,
            topP: request.topP ?? topP,
            topK: request.topK ?? topK,
            minP: request.minP ?? minP,
            frequencyPenalty: request.frequencyPenalty ?? frequencyPenalty,
            presencePenalty: request.presencePenalty ?? presencePenalty,
            repetitionPenalty: request.repetitionPenalty ?? repetitionPenalty,
            seed: request.seed ?? seed,
            stream: request.stream,
            stop: request.stop,
            sessionId: request.sessionId,
            responseFormat: request.responseFormat,
            jsonSchemaDef: request.jsonSchemaDef,
            regexPattern: request.regexPattern,
            gbnfGrammar: request.gbnfGrammar,
            thinkingBudget: request.thinkingBudget ?? thinkingBudget
        )
    }
}
