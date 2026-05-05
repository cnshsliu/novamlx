import Foundation
import NovaMLXCore

/// Errors from Anthropic → ChatMessage mapping
public enum AnthropicMappingError: Error, Sendable {
    case imageInToolResult(toolUseId: String)
}

/// Map Anthropic-format messages to internal ChatMessage array.
/// Handles: plain text, tool_use (multiple), tool_result (string + array content), thinking.
/// Rejects image blocks in tool_result content per current policy.
public func mapAnthropicMessages(
    _ messages: [AnthropicMessage],
    system: AnthropicContent?
) throws -> [ChatMessage] {
    var result: [ChatMessage] = []

    if let system {
        result.append(ChatMessage(role: .system, content: system.textContent))
    }

    for msg in messages {
        let mapped = try mapSingleAnthropicMessage(msg)
        result.append(contentsOf: mapped)
    }

    return result
}

/// A single AnthropicMessage can expand to multiple ChatMessages:
/// - Plain text user/assistant → 1 message
/// - Assistant with text + tool_use → 1 message (content + toolCalls)
/// - User with tool_result → 1 message per tool_result (.tool role)
/// - User with mixed text + tool_result → 2+ messages
private func mapSingleAnthropicMessage(_ msg: AnthropicMessage) throws -> [ChatMessage] {
    let blocks: [AnthropicContentBlock]
    switch msg.content {
    case .text(let s):
        let role: ChatMessage.Role = msg.role == "assistant" ? .assistant : .user
        return [ChatMessage(role: role, content: s)]
    case .blocks(let b):
        blocks = b
    }

    let toolUseBlocks = blocks.filter { $0.type == "tool_use" }
    let toolResultBlocks = blocks.filter { $0.type == "tool_result" }
    let textBlocks = blocks.filter { $0.type == "text" }
    let thinkingBlocks = blocks.filter { $0.type == "thinking" }

    // Assistant message with tool_use
    if msg.role == "assistant" && !toolUseBlocks.isEmpty {
        let textContent = textBlocks.compactMap { $0.text }.joined()
        let thinkingContent = thinkingBlocks.compactMap { $0.thinking }.joined()
        let content = [textContent, thinkingContent].filter { !$0.isEmpty }.joined(separator: "\n")

        let toolCalls = toolUseBlocks.map { block -> ToolCallResult in
            let argsJSON = encodeAnyCodable(block.input ?? .dictionary([:]))
            return ToolCallResult(
                id: block.id ?? "",
                functionName: block.name ?? "",
                arguments: argsJSON
            )
        }

        return [ChatMessage(role: .assistant, content: content.isEmpty ? nil : content, toolCalls: toolCalls)]
    }

    // User message with tool_result(s)
    if !toolResultBlocks.isEmpty {
        var result: [ChatMessage] = []

        // If there's text before the tool results, emit it as a user message
        let textContent = textBlocks.compactMap { $0.text }.joined()
        if !textContent.isEmpty {
            result.append(ChatMessage(role: .user, content: textContent))
        }

        for block in toolResultBlocks {
            let content = try extractToolResultContent(block)
            result.append(ChatMessage(
                role: .tool,
                content: content,
                toolCallId: block.toolUseId ?? ""
            ))
        }

        return result
    }

    // Plain message (text + thinking blocks, no tools)
    let role: ChatMessage.Role = msg.role == "assistant" ? .assistant : .user
    let textContent = textBlocks.compactMap { $0.text }.joined()
    let thinkingContent = thinkingBlocks.compactMap { $0.thinking }.joined()
    let content = [textContent, thinkingContent].filter { !$0.isEmpty }.joined(separator: "\n")

    return [ChatMessage(role: role, content: content.isEmpty ? nil : content)]
}

/// Extract text content from a tool_result block.
/// Anthropic spec: content can be a string, an array of text blocks, or an array with image blocks.
/// Policy: reject image blocks, concatenate text blocks.
private func extractToolResultContent(_ block: AnthropicContentBlock) throws -> String {
    guard let content = block.content else {
        // Some tool_results have content in the `text` field instead
        return block.text ?? ""
    }

    switch content {
    case .string(let s):
        return s
    case .array(let items):
        var texts: [String] = []
        for item in items {
            if case .dictionary(let dict) = item {
                let type = dict["type"]
                if case .string(let t) = type, t == "text" {
                    if case .string(let txt) = dict["text"] {
                        texts.append(txt)
                    }
                } else if case .string(let t) = type, t == "image" {
                    throw AnthropicMappingError.imageInToolResult(
                        toolUseId: block.toolUseId ?? "unknown"
                    )
                }
            }
        }
        return texts.joined()
    default:
        return block.text ?? ""
    }
}

/// Encode AnyCodable to a JSON string for tool call arguments
private func encodeAnyCodable(_ value: AnyCodable) -> String {
    do {
        let data = try JSONEncoder().encode(value)
        return String(data: data, encoding: .utf8) ?? "{}"
    } catch {
        return "{}"
    }
}
