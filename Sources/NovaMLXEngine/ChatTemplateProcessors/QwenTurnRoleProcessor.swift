import Foundation

/// Qwen turn-role format processor — Qwen3.6+ models.
/// Uses <|turn>role\n turn separators (undocumented divergence from official Qwen ChatML).
///
/// Key behaviors:
/// - Turn refinement: bare <|turn> → <|turn>user (model uses <|turn>model as continuation)
/// - Hallucination detection: bare \n\nuser\n patterns after 20+ tokens
/// - Thinking model detection via tokenizer added_tokens
final class QwenTurnRoleProcessor: ChatTemplateProcessor, @unchecked Sendable {

    // Bare multi-turn hallucination patterns — models sometimes generate
    // `user\nquestion\n\nassistant\nanswer` without any <|turn|> tokens.
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
        // Replace with <|turn>user to only stop at new user turn boundary.
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
        // Check template first
        if let template = chatTemplate {
            if template.contains("<think") || template.contains("</think")
                || template.contains("<thinking") || template.contains("</thinking") {
                return true
            }
        }
        // Qwen3.6 uses <|begin_of_thought|> in tokenizer, not in template
        for entry in addedTokens {
            if let content = entry["content"] as? String {
                if content.contains("begin_of_thought") || content.contains("end_of_thought")
                    || content.contains("begin_of_think") || content.contains("end_of_think") {
                    return true
                }
            }
        }
        return false
    }

    func isImplicitThinkingModel(chatTemplate: String?) -> Bool {
        // Qwen3.6 outputs <|begin_of_thought|> explicitly — NOT implicit
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
