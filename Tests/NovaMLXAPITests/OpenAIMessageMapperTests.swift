import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXAPI

@Suite("OpenAI Message Mapper")
struct OpenAIMessageMapperTests {

    /// Plain text conversation — system/user/assistant maps unchanged
    @Test("Plain text conversation maps unchanged")
    func testPlainText() throws {
        let messages: [OpenAIChatMessage] = [
            .init(role: "system", content: "Be brief."),
            .init(role: "user", content: "Hello"),
            .init(role: "assistant", content: "Hi there!"),
            .init(role: "user", content: "How are you?"),
        ]

        let result = mapOpenAIMessages(messages)

        #expect(result.count == 4)
        #expect(result[0].role == .system)
        #expect(result[0].content == "Be brief.")
        #expect(result[1].role == .user)
        #expect(result[1].content == "Hello")
        #expect(result[2].role == .assistant)
        #expect(result[2].content == "Hi there!")
        #expect(result[3].role == .user)
        #expect(result[3].content == "How are you?")
        #expect(result[0].toolCalls == nil)
        #expect(result[0].toolCallId == nil)
    }

    /// Assistant message with single tool_call
    @Test("Assistant with single tool_call maps toolCalls")
    func testSingleToolCall() throws {
        let json = """
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Read foo.txt"},
                {
                    "role": "assistant",
                    "content": "Let me read that.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\\"path\\":\\"foo.txt\\"}"}
                    }]
                }
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIRequest.self, from: json)
        let result = mapOpenAIMessages(req.messages)

        #expect(result.count == 2)
        #expect(result[1].role == .assistant)
        #expect(result[1].content == "Let me read that.")

        let toolCalls = try #require(result[1].toolCalls)
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].id == "call_abc")
        #expect(toolCalls[0].functionName == "read_file")
        #expect(toolCalls[0].arguments == #"{"path":"foo.txt"}"#)
    }

    /// Assistant message with multiple tool_calls — preserves order and all fields
    @Test("Assistant with multiple tool_calls preserves order")
    func testMultipleToolCalls() throws {
        let json = """
        {
            "model": "test",
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\\"f\\":\\"a.txt\\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": "{\\"f\\":\\"b.txt\\"}"}}
                ]
            }]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIRequest.self, from: json)
        let result = mapOpenAIMessages(req.messages)

        #expect(result.count == 1)
        let toolCalls = try #require(result[0].toolCalls)
        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].id == "call_1")
        #expect(toolCalls[0].functionName == "read")
        #expect(toolCalls[1].id == "call_2")
        #expect(toolCalls[1].functionName == "read")
    }

    /// Tool result message — role=tool with tool_call_id
    @Test("Tool result message maps role and toolCallId")
    func testToolResult() throws {
        let json = """
        {
            "model": "test",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_99", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_99", "content": "Output: 42"}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIRequest.self, from: json)
        let result = mapOpenAIMessages(req.messages)

        #expect(result.count == 2)
        #expect(result[1].role == .tool)
        #expect(result[1].toolCallId == "call_99")
        #expect(result[1].content == "Output: 42")
    }

    /// Full Claude-Code-style flow: user → assistant+tool_calls → tool → assistant
    @Test("Mixed conversation: full tool round-trip")
    func testMixedConversation() throws {
        let json = """
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Read both files"},
                {
                    "role": "assistant",
                    "content": "Reading now.",
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {"name": "read", "arguments": "{\\"f\\":\\"a.txt\\"}"}},
                        {"id": "call_b", "type": "function", "function": {"name": "read", "arguments": "{\\"f\\":\\"b.txt\\"}"}}
                    ]
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "AAA"},
                {"role": "tool", "tool_call_id": "call_b", "content": "BBB"},
                {"role": "assistant", "content": "File A says AAA, File B says BBB."}
            ]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIRequest.self, from: json)
        let result = mapOpenAIMessages(req.messages)

        #expect(result.count == 5)

        // User
        #expect(result[0].role == .user)
        #expect(result[0].content == "Read both files")

        // Assistant with tool_calls
        #expect(result[1].role == .assistant)
        #expect(result[1].content == "Reading now.")
        let toolCalls = try #require(result[1].toolCalls)
        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].id == "call_a")
        #expect(toolCalls[1].id == "call_b")

        // Tool result A
        #expect(result[2].role == .tool)
        #expect(result[2].toolCallId == "call_a")
        #expect(result[2].content == "AAA")

        // Tool result B
        #expect(result[3].role == .tool)
        #expect(result[3].toolCallId == "call_b")
        #expect(result[3].content == "BBB")

        // Final assistant
        #expect(result[4].role == .assistant)
        #expect(result[4].content == "File A says AAA, File B says BBB.")
    }

    /// Image content preservation — image_url parts make it into ChatMessage.images
    @Test("Image URLs preserved from content parts")
    func testImagePreservation() throws {
        let json = """
        {
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
                ]
            }]
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIRequest.self, from: json)
        let result = mapOpenAIMessages(req.messages)

        #expect(result.count == 1)
        #expect(result[0].content == "What is in this image?")
        let images = try #require(result[0].images)
        #expect(images.count == 1)
        #expect(images[0] == "https://example.com/cat.jpg")
    }
}
