import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct AnthropicRequest: Codable, Sendable {
    public let model: String
    public let messages: [AnthropicMessage]
    public let maxTokens: Int
    public let system: String?
    public let temperature: Double?
    public let topP: Double?
    public let stream: Bool?
    public let stopSequences: [String]?

    private enum CodingKeys: String, CodingKey {
        case model, messages, maxTokens = "max_tokens", system, temperature
        case topP = "top_p", stream, stopSequences = "stop_sequences"
    }

    public init(
        model: String,
        messages: [AnthropicMessage],
        maxTokens: Int = 4096,
        system: String? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        stream: Bool? = nil,
        stopSequences: [String]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.system = system
        self.temperature = temperature
        self.topP = topP
        self.stream = stream
        self.stopSequences = stopSequences
    }
}

public struct AnthropicTokenCountRequest: Codable, Sendable {
    public let model: String
    public let messages: [AnthropicMessage]
    public let system: String?

    public init(model: String, messages: [AnthropicMessage], system: String? = nil) {
        self.model = model
        self.messages = messages
        self.system = system
    }
}

public struct AnthropicTokenCountResponse: Codable, Sendable {
    public let inputTokens: Int

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
    }

    public init(inputTokens: Int) {
        self.inputTokens = inputTokens
    }
}

public struct AnthropicMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct AnthropicResponse: Codable, Sendable {
    public let id: String
    public let type: String
    public let role: String
    public let content: [AnthropicContentBlock]
    public let model: String
    public let stopReason: String?
    public let usage: AnthropicUsage

    private enum CodingKeys: String, CodingKey {
        case id, type, role, content, model, usage
        case stopReason = "stop_reason"
    }

    public init(id: String, model: String, content: [AnthropicContentBlock], stopReason: String?, usage: AnthropicUsage) {
        self.id = id
        self.type = "message"
        self.role = "assistant"
        self.content = content
        self.model = model
        self.stopReason = stopReason
        self.usage = usage
    }
}

public struct AnthropicContentBlock: Codable, Sendable {
    public let type: String
    public let text: String

    public init(text: String) {
        self.type = "text"
        self.text = text
    }
}

public struct AnthropicUsage: Codable, Sendable {
    public let inputTokens: Int
    public let outputTokens: Int

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}

public struct AnthropicStreamEvent: Codable, Sendable {
    public let type: String
    public let message: AnthropicMessageStart?
    public let contentBlock: AnthropicContentBlockStart?
    public let delta: AnthropicDelta?
    public let usage: AnthropicUsage?

    private enum CodingKeys: String, CodingKey {
        case type = "type"
        case message, contentBlock, delta, usage
    }

    public static func messageStart(id: String, model: String) -> AnthropicStreamEvent {
        AnthropicStreamEvent(
            type: "message_start",
            message: AnthropicMessageStart(id: id, type: "message", role: "assistant", model: model),
            contentBlock: nil, delta: nil, usage: nil
        )
    }

    public static func contentBlockStart(index: Int) -> AnthropicStreamEvent {
        AnthropicStreamEvent(
            type: "content_block_start",
            message: nil,
            contentBlock: AnthropicContentBlockStart(type: "content_block_start", index: index),
            delta: nil, usage: nil
        )
    }

    public static func textDelta(_ text: String) -> AnthropicStreamEvent {
        AnthropicStreamEvent(
            type: "content_block_delta",
            message: nil,
            contentBlock: nil,
            delta: AnthropicDelta(type: "text_delta", text: text),
            usage: nil
        )
    }

    public static func messageStop() -> AnthropicStreamEvent {
        AnthropicStreamEvent(
            type: "message_stop",
            message: nil, contentBlock: nil, delta: nil, usage: nil
        )
    }
}

public struct AnthropicMessageStart: Codable, Sendable {
    public let id: String
    public let type: String
    public let role: String
    public let model: String
}

public struct AnthropicContentBlockStart: Codable, Sendable {
    public let type: String
    public let index: Int
}

public struct AnthropicDelta: Codable, Sendable {
    public let type: String
    public let text: String?
}
