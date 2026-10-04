import Foundation
import NovaMLXCore
import NovaMLXUtils

/// HunYuan chat format processor — Hy-MT2 translation models
/// (hunyuan_v1_dense: Hy-MT2-1.8B/7B).
///
/// Turn markers use fullwidth-pipe tokens with the hy_ prefix:
///   <｜hy_begin▁of▁sentence｜>system<｜hy_place▁holder▁no▁3｜>
///   <｜hy_User｜>…<｜hy_Assistant｜>…<｜hy_place▁holder▁no▁2｜>
///
/// These are translation models: never thinking models, no implicit-open
/// think tags. Generation stops on eos_token_id 120020; the processor's
/// job is scrubbing any leaked hy_ markers and stopping on the assistant
/// end placeholder if the model emits it as text.
final class HunyuanProcessor: ChatTemplateProcessor, @unchecked Sendable {

    /// Canonical control markers (as they appear in added_tokens).
    static let markers = [
        "<｜hy_begin▁of▁sentence｜>",
        "<｜hy_end▁of▁sentence｜>",
        "<｜hy_User｜>",
        "<｜hy_Assistant｜>",
        "<｜hy_place▁holder▁no▁2｜>",
        "<｜hy_place▁holder▁no▁3｜>",
        "<｜hy_place▁holder▁no▁8｜>",
    ]

    func refineControlTokens(
        rawTokens: Set<String>,
        templateTokens: Set<String>,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var tokens = rawTokens
        tokens.formUnion(templateTokens)
        tokens.formUnion(Self.markers)
        tokens.subtract(SharedControlTokenLogic.semanticTags)
        tokens = tokens.filter { !SharedControlTokenLogic.isThinkingStopPattern($0) }
        tokens.remove("")
        return tokens.sorted()
    }

    func isThinkingModel(chatTemplate: String?, addedTokens: [[String: Any]]) -> Bool {
        false // fast-thinking translation models — no reasoning tags
    }

    func isImplicitThinkingModel(chatTemplate: String?) -> Bool {
        false
    }

    func hallucinationPatterns() -> [String] {
        // A translation model emitting a new user turn is hallucinating.
        ["<｜hy_User｜>", "｜hy_User｜", "<｜hy_Assistant｜>"]
    }

    func scrubControlTokens(_ text: String) -> String {
        var out = text
        for marker in Self.markers {
            out = out.replacingOccurrences(of: marker, with: "")
        }
        return out
    }

    func trimControlTokens(_ text: String, patterns: [String]) -> String {
        // All hy_ markers are stop patterns — prepend them, shared logic
        // truncates on first regex hit.
        return SharedControlTokenLogic.trimControlTokens(text, patterns: patterns + Self.markers)
    }

    func filterControlInChunk(
        _ text: String,
        accumulated: inout String,
        yieldedCount: inout Int,
        patterns: [String]
    ) -> (String, Bool) {
        // All hy_ markers are stop patterns — stop on them and let the
        // shared filter handle the streaming split.
        var all = patterns
        all.append(contentsOf: Self.markers)
        return SharedControlTokenLogic.filterControlInChunk(
            text, accumulated: &accumulated, yieldedCount: &yieldedCount, patterns: all
        )
    }

    func shouldStopForHallucination(generatedText: String, completionTokenCount: Int) -> Bool {
        guard completionTokenCount > 8 else { return false }
        return generatedText.contains("<｜hy_User｜>") || generatedText.contains("｜hy_User｜")
    }
}
