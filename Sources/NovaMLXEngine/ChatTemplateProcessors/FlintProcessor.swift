import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Recursive Flint format processor — RecursiveSI/flint models.
/// Uses Llama-3-style pipe role headers: `<|system|>` / `<|developer|>` /
/// `<|user|>` / `<|assistant|>` / `<|observation|>` (tool results), with
/// `<think>…</think>` reasoning and GLM4-style `<tool_call>` markup.
///
/// All five role headers are turn boundaries: they stop generation (they are
/// the model's eos ids in config.json) and are scrubbed from output. The
/// tool-call tags themselves are consumed by the GLM4 streaming parser; the
/// thinking tags are semantic and pass through to ThinkingParser.
final class FlintProcessor: ChatTemplateProcessor, @unchecked Sendable {

    private let registryFamily: ModelFamily = .flint

    /// Canonical turn-boundary tokens for the family; unioned with whatever
    /// the tokenizer actually advertises so a trimmed quant stays covered.
    private static let canonicalControlTokens: Set<String> = [
        "<|begin_of_text|>", "<|end_of_text|>",
        "<|system|>", "<|user|>", "<|assistant|>",
        "<|observation|>", "<|developer|>",
    ]

    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var tokens = rawTokens
        tokens.formUnion(templateTokens)
        tokens.formUnion(Self.canonicalControlTokens)
        tokens.formUnion(SharedControlTokenLogic.generateCloseVariants(tokens))
        // The tool-call envelope is markup, not turn boundary: the GLM4 parser
        // consumes it, and string-level copies are suppressed by the stream
        // filter. Keeping them out of the stop set avoids truncating a call.
        tokens.subtract(["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<tools>", "</tools>"])
        tokens.subtract(SharedControlTokenLogic.semanticTags)
        tokens = tokens.filter { !SharedControlTokenLogic.isThinkingStopPattern($0) }
        tokens.remove("")
        return tokens.sorted()
    }

    func isThinkingModel(chatTemplate: String?, addedTokens: [[String: Any]]) -> Bool {
        if let template = chatTemplate, template.contains("<think>") || template.contains("</think>") {
            return true
        }
        for entry in addedTokens {
            if let content = entry["content"] as? String,
                content == "<think>" || content == "</think>"
            {
                return true
            }
        }
        return false
    }

    // Flint's template only emits `<think>` for *history* reasoning content,
    // never opens it for the assistant to fill — no implicit thinking.
    func isImplicitThinkingModel(chatTemplate: String?) -> Bool {
        false
    }

    func hallucinationPatterns() -> [String] {
        ChatTemplateRegistry.shared.familyConfig(for: registryFamily).hallucinationPatterns
    }

    func scrubControlTokens(_ text: String) -> String {
        SharedControlTokenLogic.scrubControlTokens(text)
    }

    func trimControlTokens(_ text: String, patterns: [String]) -> String {
        SharedControlTokenLogic.trimControlTokens(text, patterns: patterns)
    }

    func filterControlInChunk(
        _ text: String,
        accumulated: inout String,
        yieldedCount: inout Int,
        patterns: [String]
    ) -> (String, Bool) {
        SharedControlTokenLogic.filterControlInChunk(text, accumulated: &accumulated, yieldedCount: &yieldedCount, patterns: patterns)
    }

    func shouldStopForHallucination(generatedText: String, completionTokenCount: Int) -> Bool {
        guard completionTokenCount > 20 else { return false }
        let patterns = hallucinationPatterns()
        guard !patterns.isEmpty else { return false }
        for pattern in patterns {
            if generatedText.range(of: pattern, options: .regularExpression) != nil {
                return true
            }
        }
        return false
    }
}
