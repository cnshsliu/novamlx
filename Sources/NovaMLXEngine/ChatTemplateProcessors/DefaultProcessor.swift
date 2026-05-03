import Foundation

/// Default processor — fallback for unknown model families.
/// Uses detected template format for format-based behavior.
final class DefaultProcessor: ChatTemplateProcessor, @unchecked Sendable {

    private let format: ChatTemplateFormat

    init(format: ChatTemplateFormat) {
        self.format = format
    }

    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var tokens = rawTokens
        tokens.formUnion(templateTokens)
        tokens.formUnion(SharedControlTokenLogic.generateCloseVariants(tokens))

        // For turnRole format: apply same turn refinement as QwenTurnRole
        if format == .turnRole {
            let turnPrefixes: Set<String> = ["<|turn>", "<|turn|>"]
            if !tokens.intersection(turnPrefixes).isEmpty {
                tokens.subtract(turnPrefixes)
                tokens.subtract(["<|/turn>", "<|/turn|>"])
                tokens.insert("<|turn>user")
            }
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
                if content.contains("begin_of_thought") || content.contains("end_of_thought")
                    || content.contains("begin_of_think") || content.contains("end_of_think") {
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

    func hallucinationPatterns() -> [String] { [] }

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
        false
    }
}
