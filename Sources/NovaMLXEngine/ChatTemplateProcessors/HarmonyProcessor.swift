import Foundation
import NovaMLXCore
import NovaMLXUtils

/// GPT-OSS (OpenAI Harmony) chat template processor.
/// Format: <|start|>role<|channel|>type<|message|>content
///
/// The model outputs through channels: analysis (thinking) and final (response).
/// Only <|return|> is a stop signal — structural tokens like <|channel|>, <|message|>,
/// <|end|>, <|start|> are part of the output format and must not trigger early stopping.
final class HarmonyProcessor: ChatTemplateProcessor, @unchecked Sendable {

    // Structural tokens that are part of the output format, NOT stop signals.
    // The model generates through these during normal response.
    // Note: <|start|> is NOT structural — if the model emits it, it's hallucinating a new turn.
    private let structuralTokens: Set<String> = [
        "<|end|>", "<|channel|>", "<|message|>",
        "<|call|>", "<|constrain|>",
        "<|/end|>", "<|/channel|>", "<|/message|>",
        "<|/call|>", "<|/constrain|>",
    ]

    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var tokens = rawTokens
        tokens.formUnion(templateTokens)
        tokens.formUnion(SharedControlTokenLogic.generateCloseVariants(tokens))

        // Remove structural tokens — they're part of the model's output format.
        // Only <|return|> should be a stop signal.
        tokens.subtract(structuralTokens)

        tokens.subtract(SharedControlTokenLogic.semanticTags)
        tokens.remove("")
        return tokens.sorted()
    }

    func isThinkingModel(chatTemplate: String?, addedTokens: [[String: Any]]) -> Bool {
        if let template = chatTemplate {
            if template.contains("<|channel|>analysis") { return true }
        }
        return false
    }

    func isImplicitThinkingModel(chatTemplate: String?) -> Bool {
        guard let template = chatTemplate else { return false }
        return template.contains("<|channel|>analysis")
    }

    func hallucinationPatterns() -> [String] {
        ChatTemplateRegistry.shared.familyConfig(for: .gptOss).hallucinationPatterns
    }

    // Regex: matches <|channel|>word<|message|> — channel+type+message structure
    private static let channelMessageRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "<\\|channel\\|>\\w+<\\|message\\|>")
    }()

    // Regex: matches <|start|>word<|message|> — start+role+message structure
    private static let startMessageRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "<\\|start\\|>\\w+<\\|message\\|>")
    }()

    func scrubControlTokens(_ text: String) -> String {
        // First pass: remove multi-token structures (channel+type+message, start+role+message)
        // These leave behind role/type words if only the control tokens are removed individually.
        var result = text
        let nsRange = NSRange(result.startIndex..., in: result)
        for regex in [Self.channelMessageRegex, Self.startMessageRegex] {
            let matches = regex.matches(in: result, range: nsRange)
            for match in matches.reversed() {
                if let range = Range(match.range, in: result) {
                    result.replaceSubrange(range, with: "")
                }
            }
        }

        // Second pass: standard control token scrubbing
        result = SharedControlTokenLogic.scrubControlTokens(result)

        // Clean up multiple consecutive spaces
        result = result.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
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
