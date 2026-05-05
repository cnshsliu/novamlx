import Foundation
import NovaMLXCore

/// Process-wide cache of `TokenMaskBuilder` instances keyed by model id.
///
/// **Why this exists.** Constructing a `TokenMaskBuilder` requires decoding
/// every token in the vocabulary (250k+ tokens for Qwen3.6 / Gemma-4). That
/// work is invariant for the model's lifetime — we only need to do it once,
/// not on every request that asks for structured output.
///
/// **Lifecycle.** An entry is created (lazily) the first time a request asks
/// for any grammar-constrained processor (JSON / regex / schema / GBNF) for
/// that model. The entry is removed when the model is unloaded so its memory
/// (decoded-token tables + per-state mask cache) is released.
///
/// **Thread safety.** Implemented as an actor so concurrent requests for the
/// same model serialize cache lookups without rebuilding.
public actor TokenMaskBuilderCache {
    private var entries: [String: TokenMaskBuilder] = [:]

    public init() {}

    /// Get-or-create. The returned builder may not be materialized yet — the
    /// caller (a logit processor) will materialize it on its first
    /// `process(logits:)` call using the authoritative logits dim.
    public func builder(for modelId: String, tokenizer: NovaMLXEngine.Tokenizer) -> TokenMaskBuilder {
        if let existing = entries[modelId] {
            return existing
        }
        let fresh = TokenMaskBuilder(tokenizer: tokenizer)
        entries[modelId] = fresh
        return fresh
    }

    /// Drop the entry for `modelId`. Called by the engine when a model is
    /// evicted from the pool.
    public func invalidate(modelId: String) {
        entries.removeValue(forKey: modelId)
    }

    /// Drop everything (e.g. on full pool flush).
    public func invalidateAll() {
        entries.removeAll()
    }
}
