import Foundation
import NovaMLXCore

/// Map OpenAI-format messages to internal ChatMessage array.
/// Preserves tool_calls (assistant messages with tool invocations),
/// tool_call_id (tool-role messages with tool results), name, and images.
///
/// Replaces the inline mapping at APIServer.swift:375-386 which dropped
/// toolCalls/toolCallId/name fields silently.
public func mapOpenAIMessages(_ messages: [OpenAIChatMessage]) -> [ChatMessage] {
    messages.map { msg in
        let role: ChatMessage.Role = switch msg.role {
        case "system": .system
        case "assistant": .assistant
        case "tool": .tool
        default: .user
        }

        let imageURLs = msg.content?.imageParts.compactMap { part -> String? in
            if case .imageUrl(let img) = part { return img.url } else { return nil }
        }

        let toolCalls: [ToolCallResult]? = msg.toolCalls.map { calls in
            calls.map { tc in
                ToolCallResult(
                    id: tc.id,
                    functionName: tc.function.name,
                    arguments: tc.function.arguments
                )
            }
        }

        return ChatMessage(
            role: role,
            content: msg.content?.textValue,
            images: (imageURLs?.isEmpty ?? true) ? nil : imageURLs,
            name: msg.name,
            toolCallId: msg.toolCallId,
            toolCalls: toolCalls
        )
    }
}
