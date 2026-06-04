import Foundation
import NovaMLXCore

/// Convert Responses API request (input + instructions) to internal ChatMessage array.
public func mapResponsesInput(_ req: OpenAIResponseRequest) -> [ChatMessage] {
    var messages: [ChatMessage] = []

    // instructions → system message
    if let instructions = req.instructions, !instructions.isEmpty {
        messages.append(ChatMessage(role: .system, content: instructions))
    }

    switch req.input {
    case .text(let prompt):
        messages.append(ChatMessage(role: .user, content: prompt))
    case .items(let items):
        for item in items {
            switch item {
            case .message(let msg):
                let role: ChatMessage.Role = switch msg.role {
                case "system", "developer": .system
                case "assistant": .assistant
                case "tool": .tool
                default: .user
                }
                let (text, imageURLs) = (msg.content.textValue, msg.content.imageURLs)
                messages.append(ChatMessage(
                    role: role,
                    content: text,
                    images: imageURLs.isEmpty ? nil : imageURLs
                ))
            case .functionCallOutput(let fcOut):
                messages.append(ChatMessage(
                    role: .tool,
                    content: fcOut.output,
                    toolCallId: fcOut.callId
                ))
            case .functionCall(let fc):
                messages.append(ChatMessage(
                    role: .assistant,
                    content: nil,
                    toolCalls: [ToolCallResult(id: fc.callId, functionName: fc.name, arguments: fc.arguments)]
                ))
            case .reasoning:
                break  // reasoning handled in APIServer for DeepSkip compat
            case .skipped:
                break
            }
        }
    case .none:
        break
    }

    return messages
}
