import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Gemma processor — Gemma3/4 models.
/// Uses <start_of_turn> / <end_of_turn> (NOT bare <|turn>).
/// Richer control token set: <|channel>thought, <|think|>, <|tool_call>, <|"|>.
/// VLM-specific behavior (EOS without images is pre-existing, not our concern).
final class GemmaProcessor: ChatTemplateProcessor, @unchecked Sendable {

    // Gemma-specific tokens to scrub beyond the standard <|xxx|> pattern
    private static let gemmaExtraPatterns: [NSRegularExpression] = {
        let patterns = [
            "<start_of_turn>",
            "<end_of_turn>",
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: $0)) }
    }()

    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var tokens = rawTokens
        tokens.formUnion(templateTokens)

        // Add Gemma-specific turn tokens
        tokens.formUnion(["<start_of_turn>", "<end_of_turn>"])

        tokens.formUnion(SharedControlTokenLogic.generateCloseVariants(tokens))
        tokens.subtract(SharedControlTokenLogic.semanticTags)
        tokens.remove("")
        return tokens.sorted()
    }

    func isThinkingModel(chatTemplate: String?, addedTokens: [[String: Any]]) -> Bool {
        if let template = chatTemplate {
            if template.contains("<think") || template.contains("</think")
                || template.contains("<thinking") || template.contains("</thinking") {
                return true
            }
            // Gemma uses <|channel>thought pattern for thinking
            if template.contains("channel") && template.contains("thought") {
                return true
            }
        }
        // Gemma-4: detect channel-thought capability via added_tokens.
        // Models with both <|channel> (open) and <channel|> (close) tokens emit
        // <|channel>thought\n...<channel|> blocks at runtime even when the chat
        // template doesn't reference them (template injects no prefix).
        var hasChannelOpen = false
        var hasChannelClose = false
        var hasThinkToken = false
        for entry in addedTokens {
            if let content = entry["content"] as? String {
                if content.contains("begin_of_thought") || content.contains("end_of_thought")
                    || content.contains("begin_of_think") || content.contains("end_of_think")
                    || content.contains("<|think|>") {
                    hasThinkToken = true
                }
                if content == "<|channel>" { hasChannelOpen = true }
                if content == "<channel|>" { hasChannelClose = true }
            }
        }
        if hasThinkToken { return true }
        if hasChannelOpen && hasChannelClose { return true }
        return false
    }

    func isImplicitThinkingModel(chatTemplate: String?) -> Bool {
        guard let template = chatTemplate else { return false }
        return template.contains("<think") || template.contains("<thinking")
    }

    func hallucinationPatterns() -> [String] {
        ChatTemplateRegistry.shared.familyConfig(for: .gemma).hallucinationPatterns
    }

    func scrubControlTokens(_ text: String) -> String {
        var result = SharedControlTokenLogic.scrubControlTokens(text)
        // Also scrub Gemma-specific <start_of_turn> / <end_of_turn>
        for regex in Self.gemmaExtraPatterns {
            let nsRange = NSRange(result.startIndex..., in: result)
            let matches = regex.matches(in: result, range: nsRange)
            for match in matches.reversed() {
                if let range = Range(match.range, in: result) {
                    result.removeSubrange(range)
                }
            }
        }
        return result
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
