import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Qwen ChatML format processor — Qwen2/3/3.5 models.
/// Uses <|im_start|>role\n / <|im_end|> turn separators.
/// Stable format under canonical ChatML; hallucination detection runs against
/// registry-supplied patterns so this processor handles malformed cases too
/// without recompiling.
final class QwenChatMLProcessor: ChatTemplateProcessor, @unchecked Sendable {

    private let registryFamily: ModelFamily = .qwen

    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var tokens = rawTokens
        tokens.formUnion(templateTokens)
        tokens.formUnion(SharedControlTokenLogic.generateCloseVariants(tokens))
        // Common hallucinated turn markers — some Qwen models generate these.
        tokens.formUnion(["<|turn|>", "<|turn>", "<|end_turn|>", "<|end_turn>"])
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

    func hallucinationPatterns() -> [String] {
        // Pull from registry so adding a new pattern (e.g. a Chinese turn marker
        // for a regionalized Qwen finetune) doesn't require editing this file.
        // Empty by default for canonical ChatML — registry can extend.
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
        // Apply registry-supplied hallucination patterns after a warmup of 20 tokens
        // (avoids false positives on terse answers like "4" or "ok").
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
