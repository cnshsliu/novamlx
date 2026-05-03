import Foundation
import NovaMLXCore

// MARK: - Chat Template Format

/// Detected chat template format from actual template content.
/// Used for processor selection and fallback behavior for unknown families.
public enum ChatTemplateFormat: Sendable {
    case turnRole      // <|turn>role\n  (Qwen3.6+)
    case imStartEnd    // <|im_start|>role\n / <|im_end|>  (Qwen2/3/3.5, Llama)
    case startOfTurn   // <start_of_turn> / <end_of_turn>  (Gemma)
    case deepSeek      // ｜User｜ / ｜Assistant｜  (DeepSeek V4+)
    case harmony       // <|start|>role<|channel|>type<|message|>  (GPT-OSS / OpenAI Harmony)
    case unknown

    public static func detect(from template: String?) -> ChatTemplateFormat {
        guard let template, !template.isEmpty else { return .unknown }
        if template.contains("<start_of_turn>") { return .startOfTurn }
        if template.contains("<|turn>") { return .turnRole }
        if template.contains("<|im_start>") { return .imStartEnd }
        if template.contains("<|channel|>") { return .harmony }
        if template.contains("｜") { return .deepSeek }
        return .unknown
    }
}

// MARK: - Protocol

/// Per-model-family chat template processor. Handles control token extraction,
/// scrubbing, streaming filtering, thinking model detection, and hallucination patterns.
///
/// Each model family gets its own processor implementation. Within a family,
/// different template formats (e.g. Qwen ChatML vs turnRole) get separate files.
/// Unknown families fall back to DefaultProcessor with format-based behavior.
public protocol ChatTemplateProcessor: Sendable {
    // MARK: Model Load Time

    /// Refine raw extracted control tokens (add close variants, remove semantic tags,
    /// apply family-specific adjustments like Qwen3.6 turn refinement).
    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String]

    /// Whether this model uses thinking/reasoning tags.
    func isThinkingModel(chatTemplate: String?, addedTokens: [[String: Any]]) -> Bool

    /// Whether the model uses implicit open-tag thinking (DeepSeek-R1 style).
    /// Returns true ONLY when the template injects <think or <thinking tags.
    func isImplicitThinkingModel(chatTemplate: String?) -> Bool

    /// Hallucination patterns for TurnStopProcessor (e.g. "\n\nuser\n").
    /// Return empty array to disable hallucination detection.
    func hallucinationPatterns() -> [String]

    // MARK: Generation Time

    /// Regex-based scrub of control tokens from text output.
    func scrubControlTokens(_ text: String) -> String

    /// Truncate text at first control token pattern match.
    func trimControlTokens(_ text: String, patterns: [String]) -> String

    /// Incremental streaming filter. Returns (cleanText, shouldStop).
    func filterControlInChunk(
        _ text: String,
        accumulated: inout String,
        yieldedCount: inout Int,
        patterns: [String]
    ) -> (String, Bool)

    /// Check if generated text contains hallucination patterns (e.g. bare multi-turn).
    func shouldStopForHallucination(generatedText: String, completionTokenCount: Int) -> Bool
}
