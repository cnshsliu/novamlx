import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXEngine
@testable import NovaMLXAPI

// ────────────────────────────────────────────────────────────
// API Streaming Format Tests
// ────────────────────────────────────────────────────────────

// MARK: - OpenAI Streaming

@Suite("OpenAI Streaming Format Tests")
struct OpenAIStreamingFormatTests {

    @Test("Stream chunk with delta content")
    func streamChunkDeltaContent() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-abc",
            model: "test-model",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(role: "assistant", content: "Hello"))]
        )
        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        #expect(decoded.choices[0].delta.content == "Hello")
        #expect(decoded.choices[0].delta.role == "assistant")
    }

    @Test("Stream chunk with tool call delta")
    func streamChunkToolCallDelta() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-tool",
            model: "test-model",
            choices: [OpenAIStreamChoice(
                index: 0,
                delta: OpenAIDelta(
                    toolCalls: [OpenAIToolCallDelta(
                        index: 0,
                        id: "call_abc123",
                        function: OpenAIFunctionCallDelta(name: "get_weather", arguments: "{\"city\":")
                    )]
                )
            )]
        )
        #expect(chunk.choices[0].delta.toolCalls?.count == 1)
        #expect(chunk.choices[0].delta.toolCalls?[0].id == "call_abc123")
        #expect(chunk.choices[0].delta.toolCalls?[0].function?.name == "get_weather")
    }

    @Test("Stream chunk with finish reason")
    func streamChunkFinishReason() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-fin",
            model: "test-model",
            choices: [OpenAIStreamChoice(
                index: 0,
                delta: OpenAIDelta(content: ""),
                finishReason: "stop"
            )]
        )
        #expect(chunk.choices[0].finishReason == "stop")
    }

    @Test("Multiple stream chunks aggregate to full text")
    func streamChunkAggregation() {
        let chunks = ["Hello", " world", "! How", " can I help?"]
        var fullText = ""
        for text in chunks { fullText += text }
        #expect(fullText == "Hello world! How can I help?")
    }

    @Test("OpenAI stream chunk with null role and toolCalls")
    func streamChunkNullDelta() {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-x",
            model: "test",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "text"))]
        )
        #expect(chunk.choices[0].delta.role == nil)
        #expect(chunk.choices[0].delta.toolCalls == nil)
    }

    @Test("Stream chunk id and object fields")
    func streamChunkId() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-test-id",
            model: "m",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "x"))]
        )
        #expect(chunk.id == "chatcmpl-test-id")
        #expect(chunk.object == "chat.completion.chunk")
    }

    @Test("SSE [DONE] sentinel detection")
    func sseDoneSentinel() {
        let line = "data: [DONE]"
        #expect(line.hasPrefix("data: "))
        let payload = String(line.dropFirst(6))
        #expect(payload == "[DONE]")
    }

    @Test("SSE line parsing for delta content")
    func sseLineParsing() throws {
        let json = """
        {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1717000000,"model":"test","choices":[{"index":0,"delta":{"content":"hi"}}]}
        """
        let data = json.data(using: .utf8)!
        let chunk = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        #expect(chunk.choices[0].delta.content == "hi")
    }
}

// MARK: - Anthropic Streaming

@Suite("Anthropic Streaming Format Tests")
struct AnthropicStreamingFormatTests {

    @Test("message_start event via factory")
    func messageStartEvent() throws {
        let event = AnthropicStreamEvent.messageStart(id: "msg_test", model: "claude-haiku")
        #expect(event.type == "message_start")
        #expect(event.message?.id == "msg_test")
        #expect(event.message?.model == "claude-haiku")
        #expect(event.message?.role == "assistant")
    }

    @Test("content_block_start factory")
    func contentBlockStart() throws {
        let event = AnthropicStreamEvent.contentBlockStart(index: 0)
        #expect(event.type == "content_block_start")
        #expect(event.contentBlock?.index == 0)
    }

    @Test("text_delta factory")
    func textDelta() throws {
        let event = AnthropicStreamEvent.textDelta("Hello world")
        #expect(event.type == "content_block_delta")
        #expect(event.delta?.type == "text_delta")
        #expect(event.delta?.text == "Hello world")
    }

    @Test("message_stop factory")
    func messageStop() throws {
        let event = AnthropicStreamEvent.messageStop()
        #expect(event.type == "message_stop")
    }

    @Test("Full streaming sequence via factories")
    func fullStreamingSequence() {
        let events: [AnthropicStreamEvent] = [
            .messageStart(id: "msg_1", model: "claude"),
            .contentBlockStart(index: 0),
            .textDelta("Hello"),
            .textDelta(" world"),
            .messageStop()
        ]

        #expect(events.count == 5)

        var aggregatedText = ""
        for event in events {
            if event.type == "content_block_delta",
               event.delta?.type == "text_delta",
               let text = event.delta?.text {
                aggregatedText += text
            }
        }
        #expect(aggregatedText == "Hello world")
    }

    @Test("AnthropicStreamEvent JSON round-trip")
    func streamEventRoundTrip() throws {
        // Encode → decode → verify
        let event = AnthropicStreamEvent.textDelta("test")
        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(AnthropicStreamEvent.self, from: data)
        #expect(decoded.type == "content_block_delta")
        #expect(decoded.delta?.text == "test")
    }

    @Test("AnthropicStreamEvent with message and usage")
    func streamEventWithMessageAndUsage() throws {
        let event = AnthropicStreamEvent(
            type: "message_start",
            message: AnthropicMessageStart(id: "m1", type: "message", role: "assistant", model: "claude"),
            contentBlock: nil,
            delta: nil,
            usage: AnthropicUsage(inputTokens: 100, outputTokens: 50)
        )
        #expect(event.message?.id == "m1")
        #expect(event.usage?.inputTokens == 100)
        #expect(event.usage?.outputTokens == 50)
    }

    @Test("AnthropicResponse with tool_use content block")
    func anthropicResponseToolUse() throws {
        let response = AnthropicResponse(
            id: "msg-tool",
            model: "claude-3",
            content: [
                AnthropicContentBlock(type: "text", text: "Let me check."),
                AnthropicContentBlock(type: "tool_use", id: "toolu_1", name: "read", input: .dictionary(["path": .string("/f")])),
            ],
            stopReason: "tool_use",
            usage: AnthropicUsage(inputTokens: 50, outputTokens: 30)
        )
        #expect(response.content.count == 2)
        #expect(response.content[0].type == "text")
        #expect(response.content[1].type == "tool_use")
        #expect(response.content[1].name == "read")
        #expect(response.stopReason == "tool_use")
    }

    @Test("Anthropic token count response")
    func tokenCountResponse() throws {
        let resp = AnthropicTokenCountResponse(inputTokens: 42)
        #expect(resp.inputTokens == 42)

        let data = try JSONEncoder().encode(resp)
        let decoded = try JSONDecoder().decode(AnthropicTokenCountResponse.self, from: data)
        #expect(decoded.inputTokens == 42)
    }

    @Test("Anthropic error response encoding")
    func errorResponse() throws {
        let detail = AnthropicErrorDetail(type: "invalid_request_error", message: "Bad request")
        let response = AnthropicErrorResponse(error: detail)
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(AnthropicErrorResponse.self, from: data)
        #expect(decoded.error.type == "invalid_request_error")
        #expect(decoded.error.message == "Bad request")
    }
}

// MARK: - Tool Call Streaming Format Tests

@Suite("Tool Call Streaming Tests")
struct ToolCallStreamingTests {

    @Test("OpenAI tool call accumulation from streaming deltas")
    func accumulateOpenAIToolCall() {
        let deltas: [(index: Int, id: String?, name: String?, args: String?)] = [
            (0, "call_abc", "get_weather", nil),
            (0, nil, nil, "{\"city\":"),
            (0, nil, nil, " \"Tokyo\"}"),
        ]

        struct State { var id = ""; var name = ""; var args = "" }
        var states: [Int: State] = [:]

        for d in deltas {
            var s = states[d.index] ?? State()
            if let id = d.id { s.id = id }
            if let name = d.name { s.name = name }
            if let a = d.args { s.args += a }
            states[d.index] = s
        }

        let call = states[0]!
        #expect(call.id == "call_abc")
        #expect(call.name == "get_weather")
        #expect(call.args == "{\"city\": \"Tokyo\"}")
    }

    @Test("OpenAI multi-tool call streaming interleaved")
    func multiToolCallStreaming() {
        let deltas: [(index: Int, id: String?, name: String?, args: String?)] = [
            (0, "call_1", "get_weather", nil),
            (1, "call_2", "get_time", nil),
            (0, nil, nil, "{\"city\":\"Tokyo\"}"),
            (1, nil, nil, "{\"tz\":\"UTC\"}"),
        ]
        var states: [Int: (id: String, name: String, args: String)] = [:]

        for d in deltas {
            var s = states[d.index] ?? ("", "", "")
            if let id = d.id { s.0 = id }
            if let name = d.name { s.1 = name }
            if let a = d.args { s.2 += a }
            states[d.index] = s
        }

        #expect(states.count == 2)
        #expect(states[0]?.1 == "get_weather")
        #expect(states[1]?.1 == "get_time")
    }

    @Test("Anthropic tool_use response encoding round-trip")
    func anthropicToolUseEncoding() throws {
        let response = AnthropicResponse(
            id: "msg-001",
            model: "claude",
            content: [
                AnthropicContentBlock(type: "text", text: "I'll search."),
                AnthropicContentBlock(type: "tool_use", id: "toolu_abc", name: "web_search",
                    input: .dictionary(["query": .string("Swift concurrency")]))
            ],
            stopReason: "tool_use",
            usage: AnthropicUsage(inputTokens: 20, outputTokens: 40)
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(AnthropicResponse.self, from: data)
        #expect(decoded.content.count == 2)
        #expect(decoded.content[1].name == "web_search")
        #expect(decoded.stopReason == "tool_use")
    }

    @Test("Tool call responses not corrupted by control token scrubbing")
    func toolCallNotCorruptedByScrubbing() {
        let toolJSON = """
        <tool>{"name": "search", "arguments": {"query": "test<|turn|>"}}</tool>
        """
        let result = MLXEngine.scrubControlTokens(toolJSON)
        #expect(result.contains("\"search\""))
        #expect(result.contains("\"query\""))
        #expect(!result.contains("<|turn|>"))
    }
}

// MARK: - Control Token Integration in API Layer

@Suite("API Control Token Integration Tests")
struct APIControlTokenIntegrationTests {

    @Test("Control tokens scrubbed from text before API serialization")
    func openAIResponseScrubbed() {
        let rawText = "Hello<|turn|>system\nWorld<|im_end|>"
        let trimmed = MLXEngine.trimControlTokens(rawText, patterns: ["<|turn|>", "<|im_end|>"])
        let scrubbed = MLXEngine.scrubControlTokens(trimmed)
        #expect(!scrubbed.contains("<|turn|>"))
        #expect(!scrubbed.contains("<|im_end|>"))
    }

    @Test("Anthropic response content is scrubbed the same way")
    func anthropicResponseScrubbed() {
        // trimControlTokens cuts at first control token; scrub removes regex-matched tokens
        let raw = "Analysis<|turn|>model\nConclusion"
        let scrubbed = MLXEngine.scrubControlTokens(
            MLXEngine.trimControlTokens(raw, patterns: ["<|turn|>", "<|im_end|>"])
        )
        // trim cuts at <|turn|>, keeping only "Analysis"; scrub has nothing left to remove
        #expect(scrubbed == "Analysis")
    }

    @Test("finishReason raw values match API spec")
    func finishReasonValues() {
        #expect(FinishReason.toolCalls.rawValue == "tool_calls")
        #expect(FinishReason.stop.rawValue == "stop")
        #expect(FinishReason.length.rawValue == "length")
    }

    @Test("Combined trim+scrub handles all common control tokens")
    func combinedTrimScrubAllTokens() {
        let raw = "text<|turn|><|im_end|><|mask_start|>hidden<|mask_end|><|tool_call|>"
        let patterns = ["<|turn|>", "<|im_end|>"]
        let trimmed = MLXEngine.trimControlTokens(raw, patterns: patterns)
        let scrubbed = MLXEngine.scrubControlTokens(trimmed)
        #expect(scrubbed == "text")
    }
}
