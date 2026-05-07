import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXAPI

@Suite("Usage Reporting Tests")
struct UsageReportingTests {

    // MARK: - Token struct carries prompt token count

    @Test("Token with promptTokens round-trips through JSON")
    func tokenWithPromptTokensCodable() throws {
        let token = Token(
            id: 0,
            text: "",
            finishReason: .stop,
            promptTokens: 42
        )
        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(Token.self, from: data)
        #expect(decoded.promptTokens == 42)
        #expect(decoded.finishReason == .stop)
    }

    @Test("Token without promptTokens decodes to nil (backward compat)")
    func tokenWithoutPromptTokensBackwardCompat() throws {
        let json = """
        {"id":0,"text":"hello","logprob":null,"topLogprobs":null,"finishReason":null,"toolCall":null}
        """
        let data = json.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(Token.self, from: data)
        #expect(decoded.promptTokens == nil)
        #expect(decoded.text == "hello")
    }

    @Test("Token promptTokens nil by default")
    func tokenPromptTokensDefaultNil() {
        let token = Token(id: 0, text: "hello")
        #expect(token.promptTokens == nil)
    }

    @Test("Token with promptTokens encodes to JSON with value")
    func tokenPromptTokensEncoding() throws {
        let token = Token(id: 0, text: "", finishReason: .stop, promptTokens: 99)
        let data = try JSONEncoder().encode(token)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["promptTokens"] as? Int == 99)
    }

    // MARK: - Non-streaming usage propagation

    @Test("Non-streaming InferenceResult promptTokens flows to OpenAI usage")
    func nonStreamingPromptTokensInUsage() throws {
        let result = InferenceResult(
            id: UUID(),
            model: "test-model",
            text: "Hello world",
            tokensPerSecond: 50.0,
            promptTokens: 37,
            completionTokens: 5,
            finishReason: .stop
        )
        let ctxWin = 0
        let p = result.promptTokens
        let c = result.completionTokens
        let usage = OpenAIUsage(promptTokens: p, completionTokens: c)

        #expect(usage.promptTokens == 37)
        #expect(usage.completionTokens == 5)
        #expect(usage.totalTokens == 42)
    }

    // MARK: - Streaming usage propagation (the BUG)

    @Test("Streaming: final token promptTokens used for usage, not countTokens fallback")
    func streamingPromptTokensFromFinalToken() async throws {
        // Simulate the token stream from the engine:
        // Regular tokens (no promptTokens), then a final token with finishReason + promptTokens
        let streamTokens: [Token] = [
            Token(id: 0, text: "Hello"),
            Token(id: 1, text: " world"),
            Token(id: 2, text: "", finishReason: .stop, promptTokens: 15),
        ]

        // Simulate what the API server streaming handler does:
        // Extract promptTokens from the final token (the one with finishReason)
        var extractedPromptTokens: Int? = nil
        var completionTokenCount = 0

        for token in streamTokens {
            if let _ = token.finishReason {
                // Final token — extract prompt token count
                extractedPromptTokens = token.promptTokens
            } else {
                completionTokenCount += 1
            }
        }

        // BEFORE FIX: promptTokens would be nil (Token doesn't have the field yet)
        // AFTER FIX: promptTokens should be 15
        #expect(extractedPromptTokens != nil, "Final token must carry promptTokens — got nil")
        #expect(extractedPromptTokens == 15, "promptTokens should be 15, got \(extractedPromptTokens ?? -1)")
        #expect(completionTokenCount == 2)
    }

    @Test("Streaming: Anthropic usage from final token promptTokens")
    func streamingAnthropicUsageFromToken() async throws {
        let finalToken = Token(id: 0, text: "", finishReason: .stop, promptTokens: 28)
        let tokenCount = 7

        // Simulate Anthropic streaming handler usage construction
        let promptCount = finalToken.promptTokens ?? 0
        let usage = AnthropicUsage(inputTokens: promptCount, outputTokens: tokenCount)

        #expect(usage.inputTokens == 28)
        #expect(usage.outputTokens == 7)
    }

    @Test("Streaming: OpenAI usage chunk from final token")
    func streamingOpenAIUsageChunkFromToken() async throws {
        let finalToken = Token(id: 0, text: "", finishReason: .stop, promptTokens: 42)
        let completionTokenCount = 8

        let promptCount = finalToken.promptTokens ?? 0
        let usage = OpenAIUsage(promptTokens: promptCount, completionTokens: completionTokenCount)

        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-test",
            model: "test-model",
            choices: [],
            usage: usage
        )

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        #expect(decoded.usage?.promptTokens == 42, "Streaming usage chunk prompt_tokens should be 42")
        #expect(decoded.usage?.completionTokens == 8)
        #expect(decoded.usage?.totalTokens == 50)
    }

    @Test("Streaming: final token without promptTokens falls back to 0 gracefully")
    func streamingFallbackWhenNoPromptTokens() async throws {
        // When engine doesn't set promptTokens (shouldn't happen after fix, but defensive)
        let finalToken = Token(id: 0, text: "", finishReason: .stop)
        let promptCount = finalToken.promptTokens ?? 0
        #expect(promptCount == 0, "Fallback to 0 when promptTokens not set")
    }

    // MARK: - WorkerMessage backward compat

    @Test("WorkerMessage with promptTokens decodes correctly")
    func workerMessagePromptTokens() throws {
        let msg = WorkerMessage(
            type: "token",
            token: Token(id: 0, text: "hi"),
            promptTokens: 33
        )
        #expect(msg.promptTokens == 33)
        #expect(msg.token?.text == "hi")
    }
}
