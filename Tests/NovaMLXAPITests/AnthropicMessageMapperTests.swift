import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXAPI

@Suite("Anthropic Message Mapper")
struct AnthropicMessageMapperTests {

    /// Full Claude Code conversation shape: text → tool_use → tool_result → text
    /// This is the primary use case broken by the current textContent-only mapping.
    @Test("Claude Code 4-turn conversation maps to correct ChatMessage[]")
    func testClaudeCodeConversation() throws {
        let json = """
        {
            "model": "test-model",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Read the file /tmp/test.txt"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "/tmp/test.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "File contents: hello world"}
                ]},
                {"role": "assistant", "content": "The file contains: hello world."}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: req.system)

        // System + 4 messages = 5
        #expect(result.count == 5)

        // [0] System prompt
        #expect(result[0].role == .system)
        #expect(result[0].content == "You are a helpful assistant.")

        // [1] User text
        #expect(result[1].role == .user)
        #expect(result[1].content == "Read the file /tmp/test.txt")

        // [2] Assistant with text content + toolCalls
        #expect(result[2].role == .assistant)
        #expect(result[2].content == "Let me read that file.")
        #expect(result[2].toolCalls?.count == 1)
        let tc = try #require(result[2].toolCalls?.first)
        #expect(tc.id == "toolu_1")
        #expect(tc.functionName == "read_file")
        // Arguments must be valid JSON containing the input
        let args = try JSONSerialization.jsonObject(with: Data(tc.arguments.utf8)) as? [String: String]
        #expect(args?["path"] == "/tmp/test.txt")

        // [3] Tool result — this is the critical bug: content was silently dropped
        #expect(result[3].role == .tool)
        #expect(result[3].toolCallId == "toolu_1")
        #expect(result[3].content == "File contents: hello world")

        // [4] Assistant text
        #expect(result[4].role == .assistant)
        #expect(result[4].content == "The file contains: hello world.")
    }

    /// tool_result.content can be an array of text blocks (Anthropic spec)
    @Test("tool_result with array content maps to concatenated text")
    func testToolResultArrayContent() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Thinking..."},
                    {"type": "tool_use", "id": "toolu_42", "name": "search", "input": {"q": "test"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_42", "content": [
                        {"type": "text", "text": "Result 1. "},
                        {"type": "text", "text": "Result 2."}
                    ]}
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: nil)

        #expect(result.count == 2)

        // Assistant with text + toolCalls
        #expect(result[0].role == .assistant)
        #expect(result[0].content == "Thinking...")
        #expect(result[0].toolCalls?.count == 1)

        // Tool result with concatenated array content
        #expect(result[1].role == .tool)
        #expect(result[1].toolCallId == "toolu_42")
        #expect(result[1].content == "Result 1. Result 2.")
    }

    /// Multiple tool_use blocks in a single assistant message
    @Test("Multiple tool_use blocks in one assistant message")
    func testMultipleToolUse() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Read both files"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "read", "input": {"f": "a.txt"}},
                    {"type": "tool_use", "id": "toolu_b", "name": "read", "input": {"f": "b.txt"}}
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: nil)

        #expect(result.count == 2)
        #expect(result[1].role == .assistant)
        let toolCalls = try #require(result[1].toolCalls)
        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].id == "toolu_a")
        #expect(toolCalls[0].functionName == "read")
        #expect(toolCalls[1].id == "toolu_b")
        #expect(toolCalls[1].functionName == "read")
    }

    /// Image blocks in tool_result must be rejected (policy: HTTP 400 equivalent)
    @Test("tool_result with image content throws error")
    func testToolResultImageRejected() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_img", "name": "screenshot", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_img", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}}
                    ]}
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        #expect(throws: AnthropicMappingError.self) {
            _ = try mapAnthropicMessages(req.messages, system: nil)
        }
    }

    /// Plain text-only conversation (no tools) should still work
    @Test("Plain text conversation maps unchanged")
    func testPlainTextConversation() throws {
        let messages: [AnthropicMessage] = [
            .init(role: "user", content: "Hello"),
            .init(role: "assistant", content: "Hi there!"),
            .init(role: "user", content: "How are you?"),
        ]

        let result = try mapAnthropicMessages(messages, system: .text("Be brief."))

        #expect(result.count == 4)
        #expect(result[0].role == .system)
        #expect(result[0].content == "Be brief.")
        #expect(result[1].role == .user)
        #expect(result[1].content == "Hello")
        #expect(result[2].role == .assistant)
        #expect(result[2].content == "Hi there!")
        #expect(result[3].role == .user)
        #expect(result[3].content == "How are you?")
        #expect(result[1].toolCalls == nil)
        #expect(result[1].toolCallId == nil)
    }

    /// tool_result with no content field at all — maps to empty string
    @Test("tool_result with no content maps to empty string")
    func testToolResultWithNoContent() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_empty", "name": "noop", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_empty"}
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: nil)

        #expect(result.count == 2)
        #expect(result[1].role == .tool)
        #expect(result[1].toolCallId == "toolu_empty")
        #expect(result[1].content == "")
    }

    /// tool_result with empty string content — should map to empty string, not nil
    @Test("tool_result with empty string content maps correctly")
    func testToolResultEmptyStringContent() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_blank", "name": "echo", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_blank", "content": ""}
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: nil)

        #expect(result.count == 2)
        #expect(result[1].role == .tool)
        #expect(result[1].toolCallId == "toolu_blank")
        #expect(result[1].content == "")
    }

    /// User message with text + tool_result blocks — must expand to 2+ ChatMessages
    @Test("Mixed text + tool_result in user message expands to multiple ChatMessages")
    func testMixedTextAndToolResultExpansion() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_x", "name": "read", "input": {"f": "a.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the context you asked for."},
                    {"type": "tool_result", "tool_use_id": "toolu_x", "content": "file contents here"}
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: nil)

        // Expect: assistant(tool_use) + user(text) + tool(result) = 3 messages
        #expect(result.count == 3)

        // [0] Assistant with toolCalls
        #expect(result[0].role == .assistant)
        #expect(result[0].toolCalls?.count == 1)

        // [1] User text extracted from mixed message
        #expect(result[1].role == .user)
        #expect(result[1].content == "Here is the context you asked for.")

        // [2] Tool result
        #expect(result[2].role == .tool)
        #expect(result[2].toolCallId == "toolu_x")
        #expect(result[2].content == "file contents here")
    }

    /// tool_result with is_error=true — documents schema gap: is_error is silently dropped.
    /// See TODO(novamlx-tool-fidelity) in AnthropicTypes.swift and Types.swift.
    @Test("tool_result with is_error=true maps content but drops is_error flag (schema gap)")
    func testToolResultIsErrorFieldDropped() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_err", "name": "run", "input": {}}
                ]},
                {"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_err",
                        "is_error": true,
                        "content": "Command failed with exit code 1"
                    }
                ]}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(AnthropicRequest.self, from: json)
        let result = try mapAnthropicMessages(req.messages, system: nil)

        #expect(result.count == 2)
        #expect(result[1].role == .tool)
        #expect(result[1].toolCallId == "toolu_err")
        // Content is preserved but is_error flag is silently dropped — schema gap documented
        // via TODO(novamlx-tool-fidelity) in AnthropicContentBlock + ChatMessage
        #expect(result[1].content == "Command failed with exit code 1")
        // ChatMessage has no isError field — this information is lost
        // When tool-fidelity work is scheduled, add isError to ChatMessage and wire it through
    }
}
