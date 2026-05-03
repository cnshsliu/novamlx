import Foundation

/// Bailing (Ling) series chat template processor.
/// Uses <|turn>role\n format: <|turn>system, <|turn>user, <|turn>model
///
/// Key behaviors:
/// - Turn refinement: bare <|turn> → <|turn>user (generation stops at new user turn)
/// - Hallucination detection: bare \n\nuser\n patterns after 20+ tokens
/// - Shared scrubbing/filtering via SharedControlTokenLogic
final class BailingProcessor: ChatTemplateProcessor, @unchecked Sendable {

    private let hallucinationPatternsList: [String] = [
        "\n\nuser\n",
        "\n\nmodel\n",
        "\n\nassistant\n"
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

        // Turn refinement: bare <|turn> / <|turn|> are role prefixes, NOT end-of-turn.
        // Replace with <|turn>user so generation stops at a new user turn boundary.
        let turnPrefixes: Set<String> = ["<|turn>", "<|turn|>"]
        if !tokens.intersection(turnPrefixes).isEmpty {
            tokens.subtract(turnPrefixes)
            tokens.subtract(["<|/turn>", "<|/turn|>"])
            tokens.insert("<|turn>user")
        }

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
        }
        for entry in addedTokens {
            if let content = entry["content"] as? String {
                if content.contains("begin_of_thought") || content.contains("end_of_thought") {
                    return true
                }
            }
        }
        return false
    }

    func isImplicitThinkingModel(chatTemplate: String?) -> Bool {
        guard let template = chatTemplate else { return false }
        return template.contains("<think") || template.contains("<thinking")
    }

    func hallucinationPatterns() -> [String] {
        hallucinationPatternsList
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
        for pattern in hallucinationPatternsList {
            if generatedText.contains(pattern) {
                return true
            }
        }
        return false
    }
}
