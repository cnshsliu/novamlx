import Testing
import Foundation
@testable import NovaMLXCore
@testable import NovaMLXUtils

@Suite("Core Types Tests")
struct CoreTypesTests {
    @Test("NovaMLXError descriptions")
    func errorDescriptions() {
        let err = NovaMLXError.modelNotFound("test-model")
        #expect(err.errorDescription?.contains("test-model") == true)

        let err2 = NovaMLXError.inferenceFailed("timeout")
        #expect(err2.errorDescription?.contains("timeout") == true)

        let err3 = NovaMLXError.configurationError("bad config")
        #expect(err3.errorDescription?.contains("bad config") == true)
    }

    @Test("ModelIdentifier creation and display")
    func modelIdentifier() {
        let id = ModelIdentifier(id: "llama-3.1-8b", family: .llama)
        #expect(id.displayName == "llama-3.1-8b")
        #expect(id.family == .llama)
    }

    @Test("ModelFamily decoding unknown")
    func modelFamilyDecoding() {
        let json = "\"unknown_family\"".data(using: .utf8)!
        let family = try? JSONDecoder().decode(ModelFamily.self, from: json)
        #expect(family == .other)
    }

    @Test("ModelConfig defaults")
    func modelConfigDefaults() {
        let id = ModelIdentifier(id: "test", family: .phi)
        let config = ModelConfig(identifier: id)
        #expect(config.contextLength == 4096)
        #expect(config.maxTokens == 2048)
        #expect(config.temperature == 0.7)
        #expect(config.topP == 0.9)
        #expect(config.modelType == .llm)
        #expect(config.seed == nil)
    }

    @Test("ChatMessage creation")
    func chatMessage() {
        let msg = ChatMessage(role: .system, content: "You are helpful.")
        #expect(msg.role == .system)
        #expect(msg.content == "You are helpful.")
        #expect(msg.name == nil)
    }

    @Test("InferenceRequest defaults")
    func inferenceRequest() {
        let req = InferenceRequest(model: "test", messages: [ChatMessage(role: .user, content: "hi")])
        #expect(req.stream == false)
        #expect(req.temperature == nil)
        #expect(req.maxTokens == nil)
        #expect(req.stop == nil)
    }

    @Test("InferenceResult creation")
    func inferenceResult() {
        let result = InferenceResult(
            id: UUID(),
            model: "test",
            text: "Hello!",
            tokensPerSecond: 42.5,
            promptTokens: 10,
            completionTokens: 5,
            finishReason: .stop
        )
        #expect(result.text == "Hello!")
        #expect(result.tokensPerSecond == 42.5)
        #expect(result.finishReason == .stop)
    }

    @Test("ServerConfig defaults")
    func serverConfigDefaults() {
        let config = ServerConfig()
        #expect(config.host == "127.0.0.1")
        #expect(config.port == 6590)
        #expect(config.apiKeys.isEmpty)
        #expect(config.maxConcurrentRequests == 16)
    }

    @Test("SystemStats memory calculation")
    func systemStatsMemory() {
        let stats = SystemStats(memoryUsed: 8_000_000_000, memoryTotal: 16_000_000_000)
        #expect(stats.memoryUsagePercent == 50.0)
    }

    @Test("SystemStats zero total")
    func systemStatsZeroTotal() {
        let stats = SystemStats(memoryUsed: 100, memoryTotal: 0)
        #expect(stats.memoryUsagePercent == 0)
    }

    @Test("FinishReason raw values")
    func finishReasonRawValues() {
        #expect(FinishReason.stop.rawValue == "stop")
        #expect(FinishReason.length.rawValue == "length")
        #expect(FinishReason.toolCalls.rawValue == "tool_calls")
    }

    @Test("ModelFamily all cases")
    func modelFamilyAllCases() {
        #expect(ModelFamily.allCases.count >= 8)
    }

    @Test("Token creation")
    func tokenCreation() {
        let token = Token(id: 42, text: "hello", logprob: -0.5)
        #expect(token.id == 42)
        #expect(token.text == "hello")
        #expect(token.logprob == -0.5)
    }

    @Test("ModelSettings default values")
    func modelSettingsDefaults() {
        let settings = ModelSettings()
        #expect(settings.temperature == nil)
        #expect(settings.topP == nil)
        #expect(settings.isPinned == false)
        #expect(settings.isDefault == false)
        #expect(settings.ttlSeconds == nil)
    }

    @Test("ModelSettings coding round-trip")
    func modelSettingsCoding() throws {
        var settings = ModelSettings()
        settings.temperature = 0.8
        settings.topP = 0.95
        settings.maxTokens = 2048
        settings.isPinned = true
        settings.ttlSeconds = 300
        settings.modelAlias = "my-model"

        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(ModelSettings.self, from: data)
        #expect(decoded.temperature == 0.8)
        #expect(decoded.topP == 0.95)
        #expect(decoded.maxTokens == 2048)
        #expect(decoded.isPinned == true)
        #expect(decoded.ttlSeconds == 300)
        #expect(decoded.modelAlias == "my-model")
    }

    @Test("ModelSettings applySamplingOverrides")
    func modelSettingsApplyOverrides() {
        var settings = ModelSettings()
        settings.temperature = 0.5
        settings.topP = 0.9
        settings.maxTokens = 1024

        let request = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hello")],
            temperature: 0.7,
            stream: false
        )

        let overridden = settings.applySamplingOverrides(to: request)
        #expect(overridden.temperature == 0.7)
        #expect(overridden.topP == 0.9)
        #expect(overridden.maxTokens == 1024)
    }

    @Test("ModelSettings applySamplingOverrides all nil")
    func modelSettingsApplyOverridesNil() {
        let settings = ModelSettings()
        let request = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hello")],
            temperature: 0.8,
            topP: 0.95,
            stream: false
        )
        let overridden = settings.applySamplingOverrides(to: request)
        #expect(overridden.temperature == 0.8)
        #expect(overridden.topP == 0.95)
    }

    @Test("ThinkingParser basic think block")
    func thinkingParserBasic() {
        let parser = ThinkingParser()
        _ = parser.feed("<think")
        _ = parser.feed(">")
        let r1 = parser.feed("reasoning here")
        #expect(r1.type == .thinking)
        _ = parser.feed("</think")
        let r2 = parser.feed(">actual response")
        #expect(r2.type == .content)

        let result = parser.finalize()
        #expect(result.thinking == "reasoning here")
        #expect(result.response == "actual response")
    }

    @Test("ThinkingParser no think block")
    func thinkingParserNoBlock() {
        let parser = ThinkingParser()
        _ = parser.feed("Hello world")
        let result = parser.finalize()
        #expect(result.thinking == "")
        #expect(result.response == "Hello world")
    }

    @Test("ThinkingParser streaming tokens")
    func thinkingParserStreaming() {
        let parser = ThinkingParser()
        let tokens = ["<", "think", ">", "Let me", " think...", "<", "/think", ">", "Answer: 42"]
        var collected = [(String, ParsedTokenType)]()
        for token in tokens {
            let r = parser.feed(token)
            if !r.text.isEmpty {
                collected.append((r.text, r.type))
            }
        }
        let result = parser.finalize()
        #expect(result.thinking.contains("Let me"))
        #expect(result.response.contains("Answer: 42"))
    }

    @Test("ThinkingParser nested thinking in response")
    func thinkingParserOnlyResponse() {
        let parser = ThinkingParser()
        _ = parser.feed("No thinking here, just response.")
        let result = parser.finalize()
        #expect(result.thinking == "")
        #expect(result.response == "No thinking here, just response.")
    }

    @Test("ThinkingParser sticky-close: explicit </think> + empty response keeps reasoning_content")
    func thinkingParserStickyCloseEmptyResponse() {
        // Streaming-mode regression: gpt-oss / Harmony emits <think>analysis</think>
        // and may end (max_tokens or EOS) with no final-channel content. Pre-fix,
        // finalize()'s "promote thinking → response" fallback fired because
        // responseContent was empty, **moving the analysis text into content** and
        // emptying reasoning_content. Then APIServer's streaming finish branch
        // emitted the moved-into-content text as a SECOND `content` chunk after
        // the `reasoning_content` chunks already streamed — duplicating the same
        // text under two delta keys ("WhatWhat is is" when a naive client
        // concatenates). Fix: sticky `sawExplicitCloseEver` flag prevents the
        // promotion when an explicit </think> was observed at any point.
        let parser = ThinkingParser()
        // Streaming feed: <think>...analysis...</think> with no content after
        let tokens = ["<", "think", ">", "Let me reason about this", "</think", ">"]
        for token in tokens { _ = parser.feed(token) }

        let result = parser.finalize()
        // Reasoning content stays in thinking — NOT promoted to response
        #expect(result.thinking == "Let me reason about this",
               "thinking should retain analysis text; got \(result.thinking)")
        #expect(result.response == "",
               "response must remain empty when explicit </think> seen and nothing followed; got \(result.response)")
    }

    @Test("ThinkingParser sticky-close: streaming path through .insideThinking sets the flag")
    func thinkingParserStickyCloseStreamingPath() {
        // Confirms the sticky flag is set even when </think> is processed via the
        // .insideThinking → .normal transition (streaming path), not just the
        // .normal-path detection. Drives the same input via single-char feeds
        // so the parser must traverse the streaming state machine.
        let parser = ThinkingParser()
        for ch in "<think>analysis text</think>" {
            _ = parser.feed(String(ch))
        }
        let result = parser.finalize()
        #expect(result.thinking == "analysis text")
        #expect(result.response == "")
    }

    @Test("ThinkingParser fallback STILL fires for truncated thinking (no close ever)")
    func thinkingParserFallbackStillWorksForTruncated() {
        // Negative test: ensure the sticky-close guard doesn't break the
        // legitimate case where a model truncated mid-thinking and never
        // emitted </think>. In that case the fallback should still fire
        // (promote thinking → response so the user gets *some* answer).
        let parser = ThinkingParser()
        _ = parser.feed("<think>I was reasoning about this when I got cut off")
        let result = parser.finalize()
        // sawExplicitCloseEver was never set → fallback fires
        #expect(result.thinking == "")
        #expect(result.response == "I was reasoning about this when I got cut off")
    }

    @Test("ThinkingParser implicit open tag — close without open")
    func thinkingParserImplicitOpen() {
        let parser = ThinkingParser()
        // Simulates reasoning model where <think> is in chat template prompt,
        // only </think> appears in generated output
        _ = parser.feed("Let me think: 2+2=4\n")
        _ = parser.feed("The answer is straightforward.\n")
        _ = parser.feed("</think")
        _ = parser.feed(">\n\n")
        _ = parser.feed("2+2=4")

        let result = parser.finalize()
        #expect(result.thinking.contains("Let me think"))
        #expect(result.thinking.contains("straightforward"))
        // Newlines after </think> are part of the response content
        #expect(result.response == "\n\n2+2=4")
        // Verify no </think> tag leaked into thinking or response
        #expect(!result.response.contains("</think"))
        #expect(!result.thinking.contains("</think>"))
    }

    @Test("ThinkingParser implicit open tag — close in single token")
    func thinkingParserImplicitOpenSingleToken() {
        let parser = ThinkingParser()
        // </think> arrives as one token (common with special tokens)
        _ = parser.feed("reasoning process here")
        _ = parser.feed("</think>")
        _ = parser.feed("final answer")

        let result = parser.finalize()
        #expect(result.thinking == "reasoning process here")
        #expect(result.response == "final answer")
    }

    @Test("ThinkingParser implicit open tag — streaming char by char")
    func thinkingParserImplicitOpenStreaming() {
        let parser = ThinkingParser()
        // Character-by-character streaming of implicit thinking block
        let tokens = ["I", " am", " thinking", ".", ".", ".", "<", "/think", ">", "Result"]
        var thinkingText = ""
        var responseText = ""

        for token in tokens {
            let r = parser.feed(token)
            if r.type == .thinking {
                thinkingText += r.text
            } else {
                responseText += r.text
            }
        }

        let result = parser.finalize()
        #expect(thinkingText.contains("I am thinking..."))
        #expect(result.thinking.contains("I am thinking..."))
        #expect(result.response == "Result")
    }

    @Test("ThinkingParser no think tags at all — pure response")
    func thinkingParserNoTagsPureResponse() {
        let parser = ThinkingParser()
        _ = parser.feed("Hello! How can I help you today?")
        let result = parser.finalize()
        #expect(result.thinking == "")
        #expect(result.response == "Hello! How can I help you today?")
    }

    @Test("StreamingDetokenizer basic text")
    func streamingDetokenizerBasic() {
        let detok = StreamingDetokenizer(decode: { tokens in
            tokens.map { String(UnicodeScalar($0) ?? "?") }.joined()
        })
        detok.addToken(72) // H
        detok.addToken(105) // i
        let segment = detok.lastSegment
        #expect(segment == "Hi")
    }

    @Test("StreamingDetokenizer withholds incomplete replacement char")
    func streamingDetokenizerWithholdsReplacement() {
        let detok = StreamingDetokenizer(decode: { tokens in
            if tokens == [1] { return "abc\u{FFFD}" }
            if tokens == [1, 2] { return "abc完整" }
            return tokens.map { String($0) }.joined()
        })
        detok.addToken(1)
        let seg1 = detok.lastSegment
        #expect(seg1 == "abc")
        detok.addToken(2)
        let seg2 = detok.lastSegment
        #expect(seg2 == "完整")
    }

    @Test("StreamingDetokenizer finalize flushes remaining")
    func streamingDetokenizerFinalize() {
        let detok = StreamingDetokenizer(decode: { tokens in
            tokens.map { String(UnicodeScalar($0) ?? "?") }.joined()
        })
        detok.addToken(65) // A
        detok.addToken(66) // B
        _ = detok.lastSegment
        detok.addToken(67) // C
        let final = detok.finalize()
        #expect(final == "C")
        #expect(detok.tokenCount == 3)
    }

    @Test("StreamingDetokenizer reset clears state")
    func streamingDetokenizerReset() {
        let detok = StreamingDetokenizer(decode: { tokens in
            tokens.map { String(UnicodeScalar($0) ?? "?") }.joined()
        })
        detok.addToken(88)
        _ = detok.lastSegment
        detok.reset()
        #expect(detok.tokenCount == 0)
        detok.addToken(89)
        let seg = detok.lastSegment
        #expect(seg == "Y")
    }

    @Test("StreamingDetokenizer multi-byte UTF-8 simulation")
    func streamingDetokenizerMultiByte() {
        let fullText = "Hello 世界"
        let _ = ""
        let detok = StreamingDetokenizer(decode: { tokens in
            return fullText
        })
        detok.addToken(0)
        detok.addToken(1)
        detok.addToken(2)
        let seg = detok.lastSegment
        #expect(seg == fullText)
    }

    @Test("ToolCallParser XML JSON format")
    func toolCallParserXMLJSON() {
        let text = "I'll help you with that.\n<tool>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}</tool>"
        let result = ToolCallParser.parse(text)
        #expect(result.content == "I'll help you with that.")
        #expect(result.toolCalls?.count == 1)
        #expect(result.toolCalls?[0].function.name == "get_weather")
        #expect(result.toolCalls?[0].function.arguments.contains("Tokyo") == true)
    }

    @Test("ToolCallParser XML function format")
    func toolCallParserXMLFunction() {
        let text = "<tool><function=get_weather><parameter=city>Tokyo</parameter></function></tool>"
        let result = ToolCallParser.parse(text)
        #expect(result.content.isEmpty)
        #expect(result.toolCalls?.count == 1)
        #expect(result.toolCalls?[0].function.name == "get_weather")
    }

    @Test("ToolCallParser bracket TOOL_CALLS format")
    func toolCallParserBracket() {
        let text = "[TOOL_CALLS] [{\"name\": \"search\", \"arguments\": {\"query\": \"test\"}}]"
        let result = ToolCallParser.parse(text)
        #expect(result.toolCalls?.count == 1)
        #expect(result.toolCalls?[0].function.name == "search")
    }

    @Test("ToolCallParser no tool calls")
    func toolCallParserNone() {
        let text = "Just a regular response with no tool calls."
        let result = ToolCallParser.parse(text)
        #expect(result.toolCalls == nil)
        #expect(result.content == text)
    }

    @Test("ToolCallParser multiple XML tool calls")
    func toolCallParserMultiple() {
        let text = "<tool>{\"name\": \"func1\", \"arguments\": {}}</tool><tool>{\"name\": \"func2\", \"arguments\": {}}</tool>"
        let result = ToolCallParser.parse(text)
        #expect(result.toolCalls?.count == 2)
        #expect(result.toolCalls?[0].function.name == "func1")
        #expect(result.toolCalls?[1].function.name == "func2")
    }

    @Test("ToolCallStreamFilter suppresses tool markup")
    func toolCallStreamFilter() {
        let filter = ToolCallStreamFilter()
        _ = filter.feed("Hello ")
        let suppressed = filter.feed("<tool>")
        #expect(suppressed == "")
    }
}

@Suite("ResponseFormat Tests")
struct ResponseFormatTests {
    @Test("ResponseFormat encoding")
    func responseFormatEncoding() throws {
        let text = ResponseFormat.text
        let json = ResponseFormat.jsonObject
        #expect(text.rawValue == "text")
        #expect(json.rawValue == "json_object")

        let decoded = try JSONDecoder().decode(ResponseFormat.self, from: "\"json_object\"".data(using: .utf8)!)
        #expect(decoded == .jsonObject)
    }

    @Test("InferenceRequest with responseFormat")
    func inferenceRequestResponseFormat() {
        let req = InferenceRequest(model: "test", messages: [], responseFormat: .jsonObject)
        #expect(req.responseFormat == .jsonObject)

        let plain = InferenceRequest(model: "test", messages: [])
        #expect(plain.responseFormat == nil)
    }
}
