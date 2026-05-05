import Testing
import Foundation
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// Harmony (gpt-oss) channel → <think>/</think> normalization.
//
// `MLXEngine.scrubControlTokens` previously stripped
// `<|channel|>analysis<|message|>` and `<|channel|>final<|message|>` as a
// single regex pass, leaving the analysis-channel content concatenated with
// the final-channel content with no boundary marker. Combined with the
// generic `isImplicitThinkingModel` checking only for `<think...>` tags
// (which Harmony templates don't contain), the ThinkingParser had no way
// to tell reasoning_content from content — every gpt-oss test that asked
// about `reasoning_content` returned empty (T4: "0c reasoning_content").
//
// The fix: a Harmony-specific pre-pass converts the channel boundaries to
// explicit `<think>` / `</think>` tags BEFORE the existing channel-message
// strip regex runs. ThinkingParser then operates in normal explicit-tag
// mode and extracts reasoning_content correctly.
//
// **Strict isolation**: the trigger string `<|channel|>analysis<|message|>`
// is unique to Harmony format. Other families (Qwen, Gemma, DeepSeek,
// Bailing, Llama, etc.) never emit it. These tests exercise:
//   1. Pure Harmony — analysis + final → <think>...</think>final
//   2. Final-only Harmony — no analysis → strip boundary, no thinking tag
//   3. Other families' outputs untouched (no false positives)
// ────────────────────────────────────────────────────────────

@Suite("Harmony channel normalization in scrubControlTokens")
struct HarmonyChannelNormalizationTests {

    @Test("Analysis + final channels convert to <think>...</think>")
    func analysisAndFinalConvertToThinkTags() {
        // The exact Harmony output structure for an assistant turn with
        // thinking enabled. The model emits both channels in one stream.
        let raw = "<|channel|>analysis<|message|>I think the answer is 4 because 2+2=4.<|end|>" +
                  "<|start|>assistant<|channel|>final<|message|>The answer is 4.<|return|>"

        let scrubbed = MLXEngine.scrubControlTokens(raw)

        // <think> tag must be present and precede the closing </think>
        #expect(scrubbed.contains("<think>"), "Expected <think> tag in normalized output")
        #expect(scrubbed.contains("</think>"), "Expected </think> tag in normalized output")
        if let openIdx = scrubbed.range(of: "<think>")?.lowerBound,
           let closeIdx = scrubbed.range(of: "</think>")?.lowerBound {
            #expect(openIdx < closeIdx, "<think> must come before </think>")
        }

        // Reasoning content must be inside the tags
        #expect(scrubbed.contains("I think the answer is 4"),
               "Reasoning content preserved between <think>...</think>")
        // Final content must be after </think>
        if let closeRange = scrubbed.range(of: "</think>") {
            let after = String(scrubbed[closeRange.upperBound...])
            #expect(after.contains("The answer is 4"),
                   "Final content must follow </think>; got: \(after)")
        }
    }

    @Test("Final-only (no analysis) strips boundary cleanly without <think> tags")
    func finalOnlyNoThinkTags() {
        // Model skipped analysis and emitted only final channel.
        let raw = "<|start|>assistant<|channel|>final<|message|>Hello world.<|return|>"

        let scrubbed = MLXEngine.scrubControlTokens(raw)

        #expect(!scrubbed.contains("<think>"),
               "No <think> tag should appear when there's no analysis channel")
        #expect(!scrubbed.contains("</think>"),
               "No </think> tag should appear without a matching open")
        #expect(scrubbed.contains("Hello world"), "Final content preserved")
        // No leftover Harmony structural markers
        #expect(!scrubbed.contains("<|channel|>"))
        #expect(!scrubbed.contains("<|message|>"))
    }

    @Test("Analysis-only (no final yet) produces an open <think> with content")
    func analysisOnlyOpensThink() {
        // Streaming partial: analysis channel started but final hasn't arrived yet.
        // ThinkingParser handles open <think> without close gracefully (treats
        // remaining as buffered thinking).
        let raw = "<|channel|>analysis<|message|>Still thinking..."

        let scrubbed = MLXEngine.scrubControlTokens(raw)

        #expect(scrubbed.contains("<think>"))
        #expect(!scrubbed.contains("</think>"), "No close tag yet — analysis incomplete")
        #expect(scrubbed.contains("Still thinking"))
    }

    @Test("Non-Harmony output (Qwen <think> style) untouched by Harmony pre-pass")
    func qwenThinkStyleUntouched() {
        // Qwen emits <think>...</think> directly. Harmony pre-pass triggers
        // on `<|channel|>analysis<|message|>` which never appears in Qwen
        // output, so the result is byte-identical except for the existing
        // control-token scrubbing.
        let raw = "<think>Let me reason about this.</think>The answer is 4."

        let scrubbed = MLXEngine.scrubControlTokens(raw)

        #expect(scrubbed.contains("<think>"))
        #expect(scrubbed.contains("</think>"))
        #expect(scrubbed.contains("Let me reason about this"))
        #expect(scrubbed.contains("The answer is 4"))
    }

    @Test("Gemma channel-thought (<|channel>thought) normalization not regressed")
    func gemmaChannelThoughtNotRegressed() {
        // Pre-existing Gemma-4 normalization (single-pipe, asymmetric markers).
        // Verify Fix C didn't break it.
        let raw = "<|channel>thought\nReasoning here<channel|>The answer is 4."

        let scrubbed = MLXEngine.scrubControlTokens(raw)

        #expect(scrubbed.contains("<think>"), "Gemma channel-thought → <think>")
        #expect(scrubbed.contains("</think>"))
        #expect(scrubbed.contains("Reasoning here"))
        #expect(scrubbed.contains("The answer is 4"))
        // Original Gemma markers must be gone
        #expect(!scrubbed.contains("<|channel>thought"))
        #expect(!scrubbed.contains("<channel|>"))
    }

    @Test("Commentary channel still gets stripped entirely (no special handling)")
    func commentaryChannelStripped() {
        // Harmony defines `commentary` as a non-user-facing channel. Should be
        // dropped, not converted to thinking tags.
        let raw = "<|channel|>commentary<|message|>Internal note about the user.<|end|>" +
                  "<|start|>assistant<|channel|>final<|message|>Hi.<|return|>"

        let scrubbed = MLXEngine.scrubControlTokens(raw)

        // No <think> tag because there was no analysis
        #expect(!scrubbed.contains("<think>"))
        #expect(!scrubbed.contains("</think>"))
        // Commentary marker stripped
        #expect(!scrubbed.contains("<|channel|>commentary<|message|>"))
        // Final content preserved
        #expect(scrubbed.contains("Hi"))
    }

    @Test("Plain text (no markers) unchanged")
    func plainTextUnchanged() {
        let raw = "Just plain text with no special markers at all."
        let scrubbed = MLXEngine.scrubControlTokens(raw)
        #expect(scrubbed == raw)
    }
}
