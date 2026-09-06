import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

/// Production-grade thinking-budget enforcement for reasoning models.
///
/// **Problem solved.** Reasoning models (Qwen3 / Qwen3.5 / Qwen3.6 / DeepSeek-R1
/// / GPT-OSS / Gemma-4 channel-thought) under greedy decoding (`temperature=0`)
/// can get stuck in the thinking phase on complex prompts: they never emit the
/// closing thinking marker (`</think>`, `<|end_of_thought|>`, `<channel|>`) and
/// consume the entire `max_tokens` budget producing only chain-of-thought,
/// leaving `content` empty. This is **expected model behavior** — Qwen team
/// explicitly recommends `temperature ≥ 0.6` for thinking mode — but it breaks
/// any reproducible eval / benchmark / coding-agent pipeline that requires
/// deterministic temp=0 outputs.
///
/// **What this does.** Counts post-prompt generated tokens. When the count
/// exceeds `budget` AND the model has not yet emitted any close-marker token,
/// masks the next-step logits to allow ONLY the close-marker tokens — forcing
/// the model into the response phase regardless of what its raw next-token
/// distribution preferred. Once a close marker is observed (organic OR forced),
/// the processor goes inert for the rest of generation, letting the response
/// phase proceed normally.
///
/// **Why this is not a hack.** This is the same primitive that vLLM, SGLang,
/// and llama.cpp's server expose as `max_thinking_tokens` / `thinking_budget`.
/// It is a documented production feature for serving reasoning models, not a
/// silent parameter rewrite.
///
/// **Compose order.** Place this BEFORE grammar/JSON processors — once the
/// budget fires, all other constraints become subordinate to "must emit close
/// token now". Place AFTER repetition penalty (penalty applies to logits
/// before our mask is). Order: penalty → budget → grammar → turnStop.
public final class ThinkingBudgetProcessor: LogitProcessor, @unchecked Sendable {

    private enum State {
        case active        // counting; haven't hit budget; close marker not seen
        case forceClose    // budget exceeded; force close-marker on next sample
        case responding    // close emitted; suppress EOS until min response tokens
        case closed        // response phase allowed to emit EOS
    }

    private let budget: Int
    private let closeTokenIds: Set<Int>
    private let eosTokenIds: Set<Int>
    private let minResponseTokens: Int
    private let modelId: String

    private var state: State = .active
    private var generated: Int = 0
    private var forcedAt: Int? = nil
    private var responseTokens: Int = 0

    public init(
        budget: Int,
        closeTokenIds: Set<Int>,
        modelId: String = "<unknown>",
        eosTokenIds: Set<Int> = [],
        minResponseTokens: Int = 24
    ) {
        precondition(budget > 0, "ThinkingBudgetProcessor: budget must be > 0")
        precondition(!closeTokenIds.isEmpty, "ThinkingBudgetProcessor: closeTokenIds must be non-empty")
        self.budget = budget
        self.closeTokenIds = closeTokenIds
        self.modelId = modelId
        self.eosTokenIds = eosTokenIds
        self.minResponseTokens = max(0, minResponseTokens)
    }

    public func prompt(_ prompt: MLXArray) {
        // Reset state for each generation. The processor is reused across requests
        // when composed via ComposedLogitProcessor; without reset, prior request
        // state would leak.
        state = .active
        generated = 0
        forcedAt = nil
        responseTokens = 0
    }

    public func process(logits: MLXArray) -> MLXArray {
        if state == .responding, !eosTokenIds.isEmpty, responseTokens < minResponseTokens {
            return Self.maskTokenIds(eosTokenIds, in: logits, allow: false) ?? logits
        }
        guard state == .forceClose else { return logits }

        let result = Self.maskTokenIds(closeTokenIds, in: logits, allow: true)
        if result == nil {
            NovaMLXLog.warning(
                "[ThinkingBudget] \(modelId): no close-marker token id fits within vocab (closeIds=\(closeTokenIds.sorted())). Falling back to no-op; thinking budget will not be enforced this request."
            )
            state = .closed
            return logits
        }

        // After the close token is sampled, hold EOS so the model writes the
        // actual answer instead of immediately stopping.
        state = eosTokenIds.isEmpty || minResponseTokens == 0 ? .closed : .responding
        forcedAt = generated
        NovaMLXLog.info(
            "[ThinkingBudget] \(modelId): budget \(budget) tokens exhausted without organic close — forcing close-marker (allowed ids: \(closeTokenIds.sorted()))"
        )
        return result!
    }

    public func didSample(token: MLXArray) {
        let tokenId = token.item(Int.self)
        generated += 1

        switch state {
        case .closed:
            return
        case .responding:
            if closeTokenIds.contains(tokenId) { return }
            responseTokens += 1
            if responseTokens >= minResponseTokens {
                state = .closed
                NovaMLXLog.info(
                    "[ThinkingBudget] \(modelId): released EOS holdoff after \(responseTokens) response tokens"
                )
            }
            return
        case .forceClose:
            if closeTokenIds.contains(tokenId) {
                enterResponsePhase()
            }
            return
        case .active:
            if closeTokenIds.contains(tokenId) {
                enterResponsePhase()
                return
            }
            if generated >= budget {
                state = .forceClose
            }
        }
    }

    private func enterResponsePhase() {
        if eosTokenIds.isEmpty || minResponseTokens == 0 {
            state = .closed
        } else {
            state = .responding
            responseTokens = 0
        }
    }

    /// `allow: true` keeps only `ids`; `allow: false` suppresses `ids` (EOS holdoff).
    private static func maskTokenIds(_ ids: Set<Int>, in logits: MLXArray, allow: Bool) -> MLXArray? {
        let vocabSize = logits.shape.last ?? 0
        guard vocabSize > 0 else { return nil }
        var maskValues = [Bool](repeating: !allow, count: vocabSize)
        var any = false
        for id in ids where id >= 0 && id < vocabSize {
            maskValues[id] = allow
            any = true
        }
        guard any else { return nil }
        let mask = MLXArray(maskValues)[.newAxis, 0...]
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    /// Diagnostic: returns post-generation summary. Useful for tests and logs.
    public struct Summary: Sendable {
        public let generated: Int
        public let budget: Int
        public let forcedAt: Int?
        public let closed: Bool
    }
    public var summary: Summary {
        Summary(generated: generated, budget: budget, forcedAt: forcedAt, closed: state == .closed)
    }
}

/// Family-aware close-marker token id resolver.
///
/// Returns the set of token ids that, when emitted, terminate the thinking
/// phase for this model. Empty set means "no thinking format detected" — the
/// budget feature should be disabled.
///
/// **Coverage:**
///  • `</think>`         — Qwen3 / Qwen3.5 / Qwen3.6 (mlx-community quants),
///                          DeepSeek-R1
///  • `<|end_of_thought|>` — Qwen3.6+ in tokenizers that ship the explicit-marker
///                          variant (some non-mlx-community quants)
///  • `<|end_of_think|>`   — variant from same family
///
/// **Not yet covered (future work):**
///  • Harmony / GPT-OSS — close marker is the multi-token sequence
///    `<|channel|>final<|message|>`. Requires multi-token state-tracking,
///    not a single-token mask. Tracked as a follow-up; budget feature
///    intentionally returns empty for these.
///  • Gemma-4 channel-thought — similarly multi-token (`<channel|>`).
public enum ThinkingCloseMarkerResolver {

    /// Token strings that, when found in the tokenizer's vocabulary, terminate
    /// the thinking phase. Order is informational only — all matches are added.
    public static let knownCloseTokens: [String] = [
        "</think>",
        "<|end_of_thought|>",
        "<|end_of_think|>",
    ]

    /// Resolve close-marker token ids by encoding each known marker via the
    /// tokenizer and keeping ids that round-trip to a single-token, exact-match
    /// decoding. Multi-token markers are silently dropped (caller treats empty
    /// result as "budget feature unavailable for this model").
    ///
    /// We deliberately encode-then-decode-roundtrip rather than reading
    /// `added_tokens` directly because:
    ///   • `added_tokens` shape varies across families/quants
    ///   • encoding is what the model will actually emit during generation
    ///   • decode round-trip catches BPE-merged multi-token surprises
    public static func resolveCloseTokenIds(
        for tokenizer: NovaMLXEngine.Tokenizer,
        modelId: String = "<unknown>"
    ) -> Set<Int> {
        var ids = Set<Int>()
        for marker in knownCloseTokens {
            let encoded = tokenizer.encode(marker)
            // Single-token marker — direct hit.
            if encoded.count == 1 {
                let id = encoded[0]
                let decoded = tokenizer.decode([id])
                if decoded == marker {
                    ids.insert(id)
                    continue
                }
                // Non-exact decode (e.g., extra whitespace from sentencepiece)
                // — log and skip; we won't reliably recognize it on the
                // streaming side either.
                NovaMLXLog.info(
                    "[ThinkingBudget] \(modelId): marker \(marker.debugDescription) encoded to single id \(id) but decoded to \(decoded.debugDescription) — skipping"
                )
                continue
            }
            // Multi-token markers (Harmony channel boundaries, etc.) are not
            // supported by this single-token mask processor. Skip silently —
            // the caller's smart-default selection won't enable budget if no
            // single-token close exists.
        }
        return ids
    }
}
