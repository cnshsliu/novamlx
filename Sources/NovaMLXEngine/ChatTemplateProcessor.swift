import Foundation
import NovaMLXCore

// MARK: - Chat Template Format

/// Detected chat template format from actual template content.
/// Used for processor selection and fallback behavior for unknown families.
public enum ChatTemplateFormat: Sendable, Equatable {
    case turnRole      // <|turn>role\n  (Qwen3.6+, Bailing/Ling)
    case imStartEnd    // <|im_start|>role\n / <|im_end|>  (Qwen2/3/3.5, Llama)
    case startOfTurn   // <start_of_turn> / <end_of_turn>  (Gemma)
    case deepSeek      // ｜User｜ / ｜Assistant｜  (DeepSeek V4+)
    case harmony       // <|start|>role<|channel|>type<|message|>  (GPT-OSS / OpenAI Harmony)
    case unknown

    /// Human-readable label for logs/diagnostics.
    public var label: String {
        switch self {
        case .turnRole:    return "turnRole"
        case .imStartEnd:  return "imStartEnd (ChatML)"
        case .startOfTurn: return "startOfTurn (Gemma)"
        case .deepSeek:    return "deepSeek"
        case .harmony:     return "harmony (GPT-OSS)"
        case .unknown:     return "unknown"
        }
    }

    /// Single-best-match detection (legacy API kept for callers that just want one format).
    /// Internally delegates to `detectAll` and returns the highest-confidence hit.
    public static func detect(from template: String?) -> ChatTemplateFormat {
        let detected = detectAll(from: template)
        return detected.first?.format ?? .unknown
    }

    /// Multi-marker scan returning all detected formats sorted by confidence (descending).
    /// Confidence = number of distinct evidence markers seen for that format.
    /// Empty array when no marker matches.
    public struct Detection: Sendable {
        public let format: ChatTemplateFormat
        public let confidence: Int
        public let evidence: [String]
    }

    public static func detectAll(from template: String?) -> [Detection] {
        guard let template, !template.isEmpty else { return [] }
        var hits: [(ChatTemplateFormat, [String])] = []

        // Each format is identified by ≥1 evidence marker. We count distinct hits.
        // Matchers are intentionally generous (use canonical AND lowercase variants)
        // so subtle template variations don't slip through.
        let probes: [(ChatTemplateFormat, [String])] = [
            (.startOfTurn, ["<start_of_turn>", "<end_of_turn>"]),
            // Bailing/Ling-style turn role: use markers that include the role marker
            // to disambiguate from semantic words. `<|turn>` followed by role is canonical.
            (.turnRole, ["<|turn>user", "<|turn>model", "<|turn>system", "<|turn>assistant", "<|/turn>"]),
            // Canonical Qwen ChatML uses `<|im_start|>` (both pipes). The previous
            // implementation checked `<|im_start>` (one pipe) which never matched
            // and was masked only by family-first registry routing.
            (.imStartEnd, ["<|im_start|>", "<|im_end|>"]),
            (.harmony, ["<|channel|>", "<|start|>", "<|message|>", "<|return|>"]),
            // DeepSeek uses fullwidth pipe ｜ (U+FF5C); we anchor on its role markers
            // so generic CJK content doesn't false-positive.
            (.deepSeek, ["｜User｜", "｜Assistant｜", "｜begin▁of▁sentence｜", "｜end▁of▁sentence｜"]),
        ]

        for (format, markers) in probes {
            var seen: [String] = []
            for marker in markers where template.contains(marker) {
                seen.append(marker)
            }
            if !seen.isEmpty {
                hits.append((format, seen))
            }
        }

        // Sort by descending evidence count; tie-break by declaration order (already preserved).
        let sorted = hits.enumerated().sorted { lhs, rhs in
            if lhs.element.1.count != rhs.element.1.count {
                return lhs.element.1.count > rhs.element.1.count
            }
            return lhs.offset < rhs.offset
        }

        return sorted.map { Detection(format: $0.element.0, confidence: $0.element.1.count, evidence: $0.element.1) }
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
