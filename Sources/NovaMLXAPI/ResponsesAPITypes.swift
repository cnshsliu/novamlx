import Foundation
import NovaMLXCore

// MARK: - Responses API Request

public struct OpenAIResponseRequest: Codable, Sendable {
    public let model: String
    public let input: ResponseInput?
    public let instructions: String?
    public let tools: [ResponsesFunctionTool]?
    public let toolChoice: AnyCodable?
    public let temperature: Double?
    public let topP: Double?
    public let maxOutputTokens: Int?
    public let previousResponseId: String?
    public let text: ResponsesTextConfig?
    public let stream: Bool?
    public let keepAlive: KeepAliveValue?
    public let reasoning: ResponsesReasoningConfig?

    private enum CodingKeys: String, CodingKey {
        case model, input, instructions, tools, temperature, stream, text, reasoning
        case toolChoice = "tool_choice"
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
        case previousResponseId = "previous_response_id"
        case keepAlive = "keep_alive"
    }

    public init(
        model: String,
        input: ResponseInput? = nil,
        instructions: String? = nil,
        tools: [ResponsesFunctionTool]? = nil,
        toolChoice: AnyCodable? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        maxOutputTokens: Int? = nil,
        previousResponseId: String? = nil,
        text: ResponsesTextConfig? = nil,
        stream: Bool? = nil,
        keepAlive: KeepAliveValue? = nil,
        reasoning: ResponsesReasoningConfig? = nil
    ) {
        self.model = model
        self.input = input
        self.instructions = instructions
        self.tools = tools
        self.toolChoice = toolChoice
        self.temperature = temperature
        self.topP = topP
        self.maxOutputTokens = maxOutputTokens
        self.previousResponseId = previousResponseId
        self.text = text
        self.stream = stream
        self.keepAlive = keepAlive
        self.reasoning = reasoning
    }
}

public struct ResponsesReasoningConfig: Codable, Sendable {
    public let effort: String?
    public let summary: String?

    public init(effort: String? = nil, summary: String? = nil) {
        self.effort = effort
        self.summary = summary
    }
}

public struct ResponsesFunctionTool: Codable, Sendable {
    public let type: String
    public let name: String
    public let description: String?
    public let parameters: AnyCodable?
    public let strict: Bool?

    private enum CodingKeys: String, CodingKey {
        case type, name, description, parameters, strict
    }

    public init(type: String = "function", name: String, description: String? = nil, parameters: AnyCodable? = nil, strict: Bool? = nil) {
        self.type = type
        self.name = name
        self.description = description
        self.parameters = parameters
        self.strict = strict
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(String.self, forKey: .type)
        // Some tools (web_search, namespace) don't have a name — use empty string
        name = (try? c.decode(String.self, forKey: .name)) ?? ""
        description = try? c.decode(String.self, forKey: .description)
        parameters = try? c.decode(AnyCodable.self, forKey: .parameters)
        strict = try? c.decode(Bool.self, forKey: .strict)
    }
}

public struct ResponsesTextConfig: Codable, Sendable {
    public let format: ResponsesTextFormat?

    public init(format: ResponsesTextFormat? = nil) {
        self.format = format
    }
}

public struct ResponsesTextFormat: Codable, Sendable {
    public let type: String
    public let schema: AnyCodable?
    public let name: String?
    public let strict: Bool?

    public init(type: String = "text", schema: AnyCodable? = nil, name: String? = nil, strict: Bool? = nil) {
        self.type = type
        self.schema = schema
        self.name = name
        self.strict = strict
    }
}

// MARK: - Input Types

public enum ResponseInput: Codable, Sendable {
    case text(String)
    case items([ResponseInputItem])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else {
            self = .items(try container.decode([ResponseInputItem].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .items(let items): try container.encode(items)
        }
    }
}

/// Polymorphic input item: message or function_call_output
/// Represents a function_call item from Responses API input (assistant's tool call in conversation history)
public struct ResponseInputFunctionCall: Codable, Sendable {
    public let type: String
    public let callId: String
    public let name: String
    public let arguments: String

    private enum CodingKeys: String, CodingKey {
        case type, callId = "call_id", name, arguments
    }
}

/// Reasoning item from Responses API input — summary text from model thinking
public struct ResponseInputReasoning: Codable, Sendable {
    public let type: String
    public let summary: [ResponseReasoningSummary]?

    private enum CodingKeys: String, CodingKey {
        case type, summary
    }
}

public struct ResponseReasoningSummary: Codable, Sendable {
    public let type: String
    public let text: String
}

public enum ResponseInputItem: Codable, Sendable {
    case message(ResponseInputMessage)
    case functionCallOutput(ResponseFunctionCallOutput)
    case functionCall(ResponseInputFunctionCall)
    case reasoning(ResponseInputReasoning)
    case skipped

    private enum Discriminator: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Discriminator.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "message":
            self = .message(try ResponseInputMessage(from: decoder))
        case "function_call_output":
            self = .functionCallOutput(try ResponseFunctionCallOutput(from: decoder))
        case "function_call":
            self = .functionCall(try ResponseInputFunctionCall(from: decoder))
        case "reasoning":
            self = .reasoning(try ResponseInputReasoning(from: decoder))
        default:
            self = .skipped
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .message(let msg): try msg.encode(to: encoder)
        case .functionCallOutput(let out): try out.encode(to: encoder)
        case .functionCall(let fc): try fc.encode(to: encoder)
        case .reasoning(let r): try r.encode(to: encoder)
        case .skipped: break
        }
    }
}

public struct ResponseInputMessage: Codable, Sendable {
    public let type: String
    public let role: String
    public let content: ResponseMessageContent

    private enum CodingKeys: String, CodingKey {
        case type, role, content
    }

    public init(role: String, content: String) {
        self.type = "message"
        self.role = role
        self.content = .text(content)
    }

    public init(role: String, content: ResponseMessageContent) {
        self.type = "message"
        self.role = role
        self.content = content
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = (try? container.decode(String.self, forKey: .type)) ?? "message"
        self.role = try container.decode(String.self, forKey: .role)
        self.content = try container.decode(ResponseMessageContent.self, forKey: .content)
    }
}

/// Message content: plain string or array of content parts
public enum ResponseMessageContent: Codable, Sendable {
    case text(String)
    case parts([ResponseContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else {
            self = .parts(try container.decode([ResponseContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .parts(let parts): try container.encode(parts)
        }
    }

    public var textValue: String {
        switch self {
        case .text(let s): return s
        case .parts(let parts):
            return parts.compactMap { part -> String? in
                if case .inputText(let t) = part { return t }
                return nil
            }.joined()
        }
    }

    public var imageURLs: [String] {
        switch self {
        case .text: return []
        case .parts(let parts):
            return parts.compactMap {
                if case .inputImage(let imgURL) = $0 { return imgURL.url }
                return nil
            }
        }
    }
}

/// Image URL payload: can be a plain URL string or {url, detail} object (Codex sends object form)
public struct ResponseImageURL: Codable, Sendable {
    public let url: String
    public let detail: String?

    public init(url: String, detail: String? = nil) {
        self.url = url
        self.detail = detail
    }
}

/// Content part within a message: input_text or input_image
public enum ResponseContentPart: Codable, Sendable {
    case inputText(String)
    case inputImage(ResponseImageURL)

    private enum PartKeys: String, CodingKey {
        case type, text
        case imageUrl = "image_url"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: PartKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "input_text":
            self = .inputText(try container.decode(String.self, forKey: .text))
        case "input_image":
            // Codex sends {"url": "data:...", "detail": "high"} — object form
            if let obj = try? container.decode(ResponseImageURL.self, forKey: .imageUrl) {
                self = .inputImage(obj)
            } else if let str = try? container.decode(String.self, forKey: .imageUrl) {
                self = .inputImage(ResponseImageURL(url: str))
            } else {
                self = .inputText("")
            }
        default:
            self = .inputText(try container.decode(String.self, forKey: .text))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: PartKeys.self)
        switch self {
        case .inputText(let text):
            try container.encode("input_text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .inputImage(let imgURL):
            try container.encode("input_image", forKey: .type)
            try container.encode(imgURL, forKey: .imageUrl)
        }
    }
}

/// Tool result fed back as input
public struct ResponseFunctionCallOutput: Codable, Sendable {
    public let type: String
    public let callId: String
    public let output: String

    private enum CodingKeys: String, CodingKey {
        case type, output
        case callId = "call_id"
    }

    public init(callId: String, output: String) {
        self.type = "function_call_output"
        self.callId = callId
        self.output = output
    }
}

// MARK: - Response Output Types

public struct OpenAIResponseObject: Codable, Sendable {
    public let id: String
    public let object: String
    public let createdAt: Int
    public let model: String
    public let status: String
    public let output: [ResponseOutputItem]
    public let usage: ResponsesUsage?

    private enum CodingKeys: String, CodingKey {
        case id, object, model, status, output, usage
        case createdAt = "created_at"
    }

    public init(
        id: String,
        model: String,
        status: String = "completed",
        output: [ResponseOutputItem],
        usage: ResponsesUsage? = nil
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

/// Polymorphic output item: message, function_call, or reasoning
public enum ResponseOutputItem: Codable, Sendable {
    case message(ResponseOutputMessage)
    case functionCall(ResponseOutputFunctionCall)
    case reasoning(ResponseOutputReasoning)

    private enum Discriminator: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Discriminator.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "message":
            self = .message(try ResponseOutputMessage(from: decoder))
        case "function_call":
            self = .functionCall(try ResponseOutputFunctionCall(from: decoder))
        case "reasoning":
            self = .reasoning(try ResponseOutputReasoning(from: decoder))
        default:
            self = .message(try ResponseOutputMessage(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .message(let msg): try msg.encode(to: encoder)
        case .functionCall(let fc): try fc.encode(to: encoder)
        case .reasoning(let r): try r.encode(to: encoder)
        }
    }
}

public struct ResponseOutputMessage: Codable, Sendable {
    public let type: String
    public let id: String
    public let status: String
    public let role: String
    public let content: [ResponseContentItem]

    public init(id: String, status: String = "completed", role: String = "assistant", content: [ResponseContentItem]) {
        self.type = "message"
        self.id = id
        self.status = status
        self.role = role
        self.content = content
    }
}

public struct ResponseContentItem: Codable, Sendable {
    public let type: String
    public let text: String
    public let annotations: [String]?

    public init(text: String, annotations: [String]? = nil) {
        self.type = "output_text"
        self.text = text
        self.annotations = annotations ?? []
    }
}

public struct ResponseOutputFunctionCall: Codable, Sendable {
    public let type: String
    public let id: String
    public let status: String
    public let callId: String
    public let name: String
    public let arguments: String

    private enum CodingKeys: String, CodingKey {
        case type, id, status, name, arguments
        case callId = "call_id"
    }

    public init(id: String, callId: String, name: String, arguments: String, status: String = "completed") {
        self.type = "function_call"
        self.id = id
        self.status = status
        self.callId = callId
        self.name = name
        self.arguments = arguments
    }
}

public struct ResponsesUsage: Codable, Sendable {
    public let inputTokens: Int
    public let outputTokens: Int
    public let totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case totalTokens = "total_tokens"
    }

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.totalTokens = inputTokens + outputTokens
    }
}

// MARK: - Streaming SSE Event Types

public struct ResponsesSSEEvent: Codable, Sendable {
    public let type: String

    public init(type: String) {
        self.type = type
    }
}

/// response.created / response.in_progress
public struct ResponsesSSECreated: Codable, Sendable {
    public let type: String
    public let response: ResponsesSSEResponse

    public init(response: ResponsesSSEResponse) {
        self.type = "response.created"
        self.response = response
    }
}

public struct ResponsesSSEResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let status: String
    public let model: String
    public let output: [ResponseOutputItem]

    public init(id: String, status: String, model: String, output: [ResponseOutputItem] = []) {
        self.id = id
        self.object = "response"
        self.status = status
        self.model = model
        self.output = output
    }
}

/// response.output_item.added
public struct ResponsesSSEOutputItemAdded: Codable, Sendable {
    public let type: String
    public let outputIndex: Int
    public let item: ResponseOutputItem

    private enum CodingKeys: String, CodingKey {
        case type, item
        case outputIndex = "output_index"
    }

    public init(outputIndex: Int, item: ResponseOutputItem) {
        self.type = "response.output_item.added"
        self.outputIndex = outputIndex
        self.item = item
    }
}

/// response.content_part.added
public struct ResponsesSSEContentPartAdded: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let contentIndex: Int
    public let part: ResponseContentItem

    private enum CodingKeys: String, CodingKey {
        case type, part
        case itemId = "item_id"
        case outputIndex = "output_index"
        case contentIndex = "content_index"
    }

    public init(itemId: String, outputIndex: Int, contentIndex: Int, part: ResponseContentItem) {
        self.type = "response.content_part.added"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.contentIndex = contentIndex
        self.part = part
    }
}

/// response.output_text.delta
public struct ResponsesSSETextDelta: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let contentIndex: Int
    public let delta: String

    private enum CodingKeys: String, CodingKey {
        case type, delta
        case itemId = "item_id"
        case outputIndex = "output_index"
        case contentIndex = "content_index"
    }

    public init(itemId: String, outputIndex: Int, contentIndex: Int, delta: String) {
        self.type = "response.output_text.delta"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.contentIndex = contentIndex
        self.delta = delta
    }
}

/// response.output_text.done
public struct ResponsesSSETextDone: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let contentIndex: Int
    public let text: String

    private enum CodingKeys: String, CodingKey {
        case type, text
        case itemId = "item_id"
        case outputIndex = "output_index"
        case contentIndex = "content_index"
    }

    public init(itemId: String, outputIndex: Int, contentIndex: Int, text: String) {
        self.type = "response.output_text.done"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.contentIndex = contentIndex
        self.text = text
    }
}

/// response.content_part.done
public struct ResponsesSSEContentPartDone: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let contentIndex: Int
    public let part: ResponseContentItem

    private enum CodingKeys: String, CodingKey {
        case type, part
        case itemId = "item_id"
        case outputIndex = "output_index"
        case contentIndex = "content_index"
    }

    public init(itemId: String, outputIndex: Int, contentIndex: Int, part: ResponseContentItem) {
        self.type = "response.content_part.done"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.contentIndex = contentIndex
        self.part = part
    }
}

/// response.output_item.done
public struct ResponsesSSEOutputItemDone: Codable, Sendable {
    public let type: String
    public let outputIndex: Int
    public let item: ResponseOutputItem

    private enum CodingKeys: String, CodingKey {
        case type, item
        case outputIndex = "output_index"
    }

    public init(outputIndex: Int, item: ResponseOutputItem) {
        self.type = "response.output_item.done"
        self.outputIndex = outputIndex
        self.item = item
    }
}

/// response.completed
public struct ResponsesSSECompleted: Codable, Sendable {
    public let type: String
    public let response: OpenAIResponseObject

    public init(response: OpenAIResponseObject) {
        self.type = "response.completed"
        self.response = response
    }
}

// MARK: - Function Call Streaming Events (Codex CLI compatibility)

/// response.function_call_arguments.delta — incremental JSON argument fragments
public struct ResponsesSSEFunctionCallArgsDelta: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let callId: String
    public let delta: String

    private enum CodingKeys: String, CodingKey {
        case type, delta
        case itemId = "item_id"
        case outputIndex = "output_index"
        case callId = "call_id"
    }

    public init(itemId: String, outputIndex: Int, callId: String, delta: String) {
        self.type = "response.function_call_arguments.delta"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.callId = callId
        self.delta = delta
    }
}

/// response.function_call_arguments.done — full arguments string
public struct ResponsesSSEFunctionCallArgsDone: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let callId: String
    public let arguments: String

    private enum CodingKeys: String, CodingKey {
        case type, arguments
        case itemId = "item_id"
        case outputIndex = "output_index"
        case callId = "call_id"
    }

    public init(itemId: String, outputIndex: Int, callId: String, arguments: String) {
        self.type = "response.function_call_arguments.done"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.callId = callId
        self.arguments = arguments
    }
}

// MARK: - Reasoning Output Types (Codex CLI compatibility)

/// Reasoning output item in response output array
public struct ResponseOutputReasoning: Codable, Sendable {
    public let type: String
    public let id: String
    public let status: String
    public let summary: [ResponsesReasoningSummary]?

    public init(id: String, status: String = "completed", summary: [ResponsesReasoningSummary]? = nil) {
        self.type = "reasoning"
        self.id = id
        self.status = status
        self.summary = summary
    }
}

public struct ResponsesReasoningSummary: Codable, Sendable {
    public let type: String
    public let text: String

    public init(text: String) {
        self.type = "summary_text"
        self.text = text
    }
}

/// response.reasoning.delta — incremental reasoning text
public struct ResponsesSSEReasoningDelta: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let delta: String

    private enum CodingKeys: String, CodingKey {
        case type, delta
        case itemId = "item_id"
        case outputIndex = "output_index"
    }

    public init(itemId: String, outputIndex: Int, delta: String) {
        self.type = "response.reasoning.delta"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.delta = delta
    }
}

/// response.reasoning.done — full reasoning text
public struct ResponsesSSEReasoningDone: Codable, Sendable {
    public let type: String
    public let itemId: String
    public let outputIndex: Int
    public let summary: [ResponsesReasoningSummary]?

    private enum CodingKeys: String, CodingKey {
        case type, summary
        case itemId = "item_id"
        case outputIndex = "output_index"
    }

    public init(itemId: String, outputIndex: Int, summary: [ResponsesReasoningSummary]? = nil) {
        self.type = "response.reasoning.done"
        self.itemId = itemId
        self.outputIndex = outputIndex
        self.summary = summary
    }
}
