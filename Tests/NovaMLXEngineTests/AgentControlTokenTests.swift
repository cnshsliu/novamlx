import Testing
import Foundation
import NovaMLXCore
import NovaMLXUtils
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// Agent × Control Token Integration Tests
// Validates control token filtering and semantic tag preservation
// across all agent formats (Claude Code, Open Code, Open Claw, Hermes)
// ────────────────────────────────────────────────────────────

@Suite("Agent Control Token Integration Tests")
struct AgentControlTokenIntegrationTests {

    // MARK: - Control tokens filtered across agents

    @Test("Control tokens filtered identically across all agent formats")
    func controlTokensFilteredAcrossAgents() {
        let testCases = [
            "Claude Code says <|turn|>hello",
            "Open Code response <|im_end|>done",
            "Open Claw output <|tool_call|>search",
            "Hermes generated <|mask_start|>hidden<|mask_end|> shown",
        ]

        for input in testCases {
            let scrubbed = MLXEngine.scrubControlTokens(input)
            #expect(!scrubbed.contains("<|turn|>"))
            #expect(!scrubbed.contains("<|im_end|>"))
            #expect(!scrubbed.contains("<|tool_call|>"))
            #expect(!scrubbed.contains("<|mask_start|>"))
            #expect(!scrubbed.contains("<|mask_end|>"))
        }
    }

    @Test("Semantic tags preserved across all agent formats")
    func semanticTagsPreserved() {
        let testCases = [
            "Claude Code reasoning <think>analysis here</think> response here",
            "Open Code <think>planning</think> code generation",
            "Hermes <think>step by step</think> final answer",
        ]

        for input in testCases {
            let scrubbed = MLXEngine.scrubControlTokens(input)
            #expect(scrubbed.contains("<think"), "Semantic tag <think> must survive in: \(input)")
            #expect(scrubbed.contains("</think"), "Semantic tag </think> must survive in: \(input)")
        }
    }

    @Test("trimControlTokens with agent output containing <|turn|>")
    func trimAgentOutput() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        let testCases: [(agent: String, input: String, expected: String)] = [
            ("Claude Code", "tool output<|turn|>model\nresponse", "tool output"),
            ("Open Code", "search result<|im_end|>", "search result"),
            ("Hermes", "<|im_start|>assistant\nhello<|im_end|>", "<|im_start|>assistant\nhello"),
        ]

        for tc in testCases {
            let result = MLXEngine.trimControlTokens(tc.input, patterns: patterns)
            #expect(result == tc.expected, "\(tc.agent): expected '\(tc.expected)', got '\(result)'")
        }
    }

    // MARK: - filterControlInChunk with agent-typical outputs

    @Test("filterControlInChunk: Claude Code output with mid-stream control token")
    func claudeCodeStreamingControlToken() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        // Claude Code agent output with tool results followed by turn separator
        let (text, stop) = MLXEngine.filterControlInChunk(
            "File contents:\nimport Foundation\nstruct Config {}\n<|turn|>model\n",
            accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "File contents:\nimport Foundation\nstruct Config {}\n")
        #expect(stop == true)
    }

    @Test("filterControlInChunk: Agent output with tool call JSON then control token")
    func agentToolJSONThenControlToken() {
        let patterns = ["<|turn|>", "<|im_end|>", "<|endoftext|>"]
        var acc = ""
        var yielded = 0

        let (text, stop) = MLXEngine.filterControlInChunk(
            "{\"name\": \"search\", \"arguments\": {\"query\": \"test\"}}<|im_end|>",
            accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "{\"name\": \"search\", \"arguments\": {\"query\": \"test\"}}")
        #expect(stop == true)
    }

    @Test("filterControlInChunk: Multi-step agent conversation with control at end")
    func multiStepAgentConversation() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        // Step 1: User sends question
        let (t1, s1) = MLXEngine.filterControlInChunk(
            "What is the port number?<|turn|>",
            accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(t1 == "What is the port number?")
        #expect(s1 == true)
    }

    // MARK: - Combined scrub + trim (what the engine actually does)

    @Test("Combined trim+scrub: engine's actual post-processing pipeline")
    func combinedTrimScrub() {
        // This is exactly what MLXEngine.generate does:
        // Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: controlTokensForModel))
        let raw = "Hello world<|turn|>assistant\nResponse<|im_end|>trailing"
        let patterns = ["<|turn|>", "<|im_end|>"]

        let trimmed = MLXEngine.trimControlTokens(raw, patterns: patterns)
        let scrubbed = MLXEngine.scrubControlTokens(trimmed)

        #expect(scrubbed == "Hello world")
        #expect(!scrubbed.contains("<|turn|>"))
        #expect(!scrubbed.contains("<|im_end|>"))
    }

    @Test("Combined trim+scrub with regex-only tokens")
    func combinedTrimScrubRegexOnly() {
        // Some tokens like <|mask_start|> are only caught by regex scrub
        let raw = "visible<|mask_start|>hidden<|mask_end|>visible again<|turn|>"
        let patterns = ["<|turn|>", "<|im_end|>"]

        let trimmed = MLXEngine.trimControlTokens(raw, patterns: patterns)
        let scrubbed = MLXEngine.scrubControlTokens(trimmed)

        #expect(scrubbed == "visiblehiddenvisible again")
    }

    @Test("Combined trim+scrub with no control tokens")
    func combinedTrimScrubClean() {
        let raw = "Clean response with no control tokens"
        let patterns = ["<|turn|>", "<|im_end|>"]

        let result = MLXEngine.scrubControlTokens(
            MLXEngine.trimControlTokens(raw, patterns: patterns)
        )
        #expect(result == raw)
    }

    // MARK: - Thinking + control token coexistence

    @Test("Thinking tags survive trim+scrub but control tokens don't")
    func thinkingSurvivesControlDoesNot() {
        // Models like DeepSeek-R1 emit both thinking tags AND control tokens
        let raw = "<think>reasoning</think>response<|turn|>more<|im_end|>"
        let patterns = ["<|turn|>", "<|im_end|>"]

        let trimmed = MLXEngine.trimControlTokens(raw, patterns: patterns)
        let scrubbed = MLXEngine.scrubControlTokens(trimmed)

        #expect(scrubbed.contains("<think>reasoning</think>"))
        #expect(!scrubbed.contains("<|turn|>"))
        #expect(!scrubbed.contains("<|im_end|>"))
    }

    @Test("Thinking tags and control tokens in streaming filter")
    func thinkingAndControlInStreaming() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        // Streaming chunk: thinking content + control token
        let (t1, s1) = MLXEngine.filterControlInChunk(
            "<think>analyzing</think>answer<|turn|>",
            accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(t1 == "<think>analyzing</think>answer")
        #expect(s1 == true)

        // Verify thinking tags weren't in the patterns list
        #expect(!patterns.contains("<think>"))
        #expect(!patterns.contains("</think>"))
    }
}
