import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct AnthropicRequest: Codable, Sendable {
    public let model: String
    public let messages: [AnthropicMessage]
    public let maxTokens: Int
    public let system: AnthropicContent?
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
        system: AnthropicContent? = nil,
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
    public let system: AnthropicContent?

    public init(model: String, messages: [AnthropicMessage], system: AnthropicContent? = nil) {
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
    public let content: AnthropicContent

    public init(role: String, content: AnthropicContent) {
        self.role = role
        self.content = content
    }

    /// Convenience init with plain string content
    public init(role: String, content: String) {
        self.role = role
        self.content = .text(content)
    }

    /// Extract plain text from content (for inference)
    public var textContent: String {
        content.textContent
    }
}

/// Anthropic content can be a plain string or an array of content blocks
public enum AnthropicContent: Codable, Sendable, Equatable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public static func == (lhs: AnthropicContent, rhs: AnthropicContent) -> Bool {
        switch (lhs, rhs) {
        case (.text(let a), .text(let b)): return a == b
        case (.blocks(let a), .blocks(let b)):
            guard a.count == b.count else { return false }
            return zip(a, b).allSatisfy { $0.type == $1.type && $0.text == $1.text }
        default: return false
        }
    }

    /// Extract plain text from content
    public var textContent: String {
        switch self {
        case .text(let s): return s
        case .blocks(let blocks):
            return blocks.compactMap { block -> String? in
                if block.type == "text" { return block.text }
                return nil
            }.joined()
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
        } else {
            self = .text("")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .blocks(let b): try container.encode(b)
        }
    }
}

/// Single content block — text, tool_use, tool_result, image, etc.
public struct AnthropicContentBlock: Codable, Sendable {
    public let type: String
    public let text: String?
    public let id: String?
    public let name: String?
    public let input: AnyCodable?
    public let toolUseId: String?
    public let source: AnyCodable?

    private enum CodingKeys: String, CodingKey {
        case type, text, id, name, input, source
        case toolUseId = "tool_use_id"
    }

    public init(
        type: String, text: String? = nil, id: String? = nil,
        name: String? = nil, input: AnyCodable? = nil,
        toolUseId: String? = nil, source: AnyCodable? = nil
    ) {
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input
        self.toolUseId = toolUseId
        self.source = source
    }

    /// Convenience for simple text blocks
    public init(text: String) {
        self.type = "text"
        self.text = text
        self.id = nil
        self.name = nil
        self.input = nil
        self.toolUseId = nil
        self.source = nil
    }
}

/// Type-erased Codable value for flexible JSON handling (tool input/output, image source, etc.)
public enum AnyCodable: Codable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([AnyCodable])
    case dictionary([String: AnyCodable])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([AnyCodable].self) {
            self = .array(a)
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            self = .dictionary(dict)
        } else {
            self = .null
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let b): try container.encode(b)
        case .int(let i): try container.encode(i)
        case .double(let d): try container.encode(d)
        case .string(let s): try container.encode(s)
        case .array(let a): try container.encode(a)
        case .dictionary(let d): try container.encode(d)
        }
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
