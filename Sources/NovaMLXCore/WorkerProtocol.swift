import Foundation

// MARK: - Codable Inference Request (IPC-safe transport)

public struct CodableInferenceRequest: Codable, Sendable {
    public let id: String
    public let model: String
    public let messages: [ChatMessage]
    public let toolsJSON: String?
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
    public let responseFormat: String?
    public let jsonSchemaDefJSON: String?
    public let regexPattern: String?
    public let gbnfGrammar: String?
    public let thinkingBudget: Int?
    public let draftModel: String?
    public let numDraftTokens: Int?

    public init(from request: InferenceRequest) {
        self.id = request.id.uuidString
        self.model = request.model
        self.messages = request.messages
        if let tools = request.tools, !tools.isEmpty,
           let data = try? JSONSerialization.data(withJSONObject: tools),
           let str = String(data: data, encoding: .utf8) {
            self.toolsJSON = str
        } else {
            self.toolsJSON = nil
        }
        self.temperature = request.temperature
        self.maxTokens = request.maxTokens
        self.topP = request.topP
        self.topK = request.topK
        self.minP = request.minP
        self.frequencyPenalty = request.frequencyPenalty
        self.presencePenalty = request.presencePenalty
        self.repetitionPenalty = request.repetitionPenalty
        self.seed = request.seed
        self.stream = request.stream
        self.stop = request.stop
        self.sessionId = request.sessionId
        self.responseFormat = request.responseFormat?.rawValue
        if let schema = request.jsonSchemaDef,
           let data = try? JSONSerialization.data(withJSONObject: schema),
           let str = String(data: data, encoding: .utf8) {
            self.jsonSchemaDefJSON = str
        } else {
            self.jsonSchemaDefJSON = nil
        }
        self.regexPattern = request.regexPattern
        self.gbnfGrammar = request.gbnfGrammar
        self.thinkingBudget = request.thinkingBudget
        self.draftModel = request.draftModel
        self.numDraftTokens = request.numDraftTokens
    }

    public func toInferenceRequest() -> InferenceRequest {
        let format: ResponseFormat? = responseFormat.flatMap { ResponseFormat(rawValue: $0) }
        var schema: [String: Any]?
        if let json = jsonSchemaDefJSON,
           let data = json.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            schema = obj
        }
        var tools: [[String: Any]]?
        if let json = toolsJSON,
           let data = json.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            tools = obj
        }
        return InferenceRequest(
            id: UUID(uuidString: id) ?? UUID(),
            model: model,
            messages: messages,
            tools: tools,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            topK: topK,
            minP: minP,
            frequencyPenalty: frequencyPenalty,
            presencePenalty: presencePenalty,
            repetitionPenalty: repetitionPenalty,
            seed: seed,
            stream: stream,
            stop: stop,
            sessionId: sessionId,
            responseFormat: format,
            jsonSchemaDef: schema,
            regexPattern: regexPattern,
            gbnfGrammar: gbnfGrammar,
            thinkingBudget: thinkingBudget,
            draftModel: draftModel,
            numDraftTokens: numDraftTokens
        )
    }
}

// MARK: - Worker IPC Message Envelope

public struct WorkerMessage: Codable, Sendable {
    public let type: String
    public let modelId: String?
    public let modelPath: String?
    public let modelConfig: ModelConfig?
    public let requestId: String?
    public let request: CodableInferenceRequest?
    public let result: InferenceResult?
    public let token: Token?
    public let finishReason: FinishReason?
    public let promptTokens: Int?
    public let completionTokens: Int?
    public let errorMessage: String?
    public let memoryCurrentBytes: UInt64?
    public let memorySoftLimitBytes: UInt64?
    public let memoryHardLimitBytes: UInt64?
    public let memoryUtilization: Double?
    public let memoryEvictions: UInt64?

    public init(
        type: String,
        modelId: String? = nil,
        modelPath: String? = nil,
        modelConfig: ModelConfig? = nil,
        requestId: String? = nil,
        request: CodableInferenceRequest? = nil,
        result: InferenceResult? = nil,
        token: Token? = nil,
        finishReason: FinishReason? = nil,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        errorMessage: String? = nil,
        memoryCurrentBytes: UInt64? = nil,
        memorySoftLimitBytes: UInt64? = nil,
        memoryHardLimitBytes: UInt64? = nil,
        memoryUtilization: Double? = nil,
        memoryEvictions: UInt64? = nil
    ) {
        self.type = type
        self.modelId = modelId
        self.modelPath = modelPath
        self.modelConfig = modelConfig
        self.requestId = requestId
        self.request = request
        self.result = result
        self.token = token
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.errorMessage = errorMessage
        self.memoryCurrentBytes = memoryCurrentBytes
        self.memorySoftLimitBytes = memorySoftLimitBytes
        self.memoryHardLimitBytes = memoryHardLimitBytes
        self.memoryUtilization = memoryUtilization
        self.memoryEvictions = memoryEvictions
    }
}

// MARK: - Message Type Constants

public enum WorkerMessageType {
    public static let load = "load"
    public static let loaded = "loaded"
    public static let unload = "unload"
    public static let unloaded = "unloaded"
    public static let generate = "generate"
    public static let stream = "stream"
    public static let result = "result"
    public static let token = "token"
    public static let done = "done"
    public static let error = "error"
    public static let abort = "abort"
    public static let ping = "ping"
    public static let pong = "pong"
    public static let memoryStats = "memoryStats"
}
