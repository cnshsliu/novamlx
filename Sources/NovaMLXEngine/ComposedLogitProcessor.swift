import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore

/// Composes up to four LogitProcessors into a single chain.
///
/// **Process order:** `penalty → grammar → turnStop → thinkingBudget`
///
///  1. **penalty** modifies logits in place (repetition/freq/presence) — it
///     never masks, so its result is a strictly-monotonic perturbation of the
///     input distribution.
///  2. **grammar** (JSON / Schema / GBNF / Regex) masks logits to its allowed
///     vocabulary subset. May force EOS via mask.
///  3. **turnStop** masks to EOS-only when a turn separator string is observed
///     in the decoded output (works regardless of whether the model emits its
///     configured EOS token id).
///  4. **thinkingBudget** masks to close-marker-only when the configured
///     thinking budget is exhausted and the model has not yet emitted a
///     close-marker token. Placed LAST so its mask overrides any prior
///     constraint — when the budget fires, "emit close marker now" wins
///     unconditionally. In production we ensure budget is only built when
///     grammar is absent, so the override is benign.
///
/// All four slots are optional. Caller must ensure at least one is non-nil
/// (otherwise compose() returns nil — see the convenience builder below).
public final class ComposedLogitProcessor: LogitProcessor, @unchecked Sendable {
    private var grammarProcessor: (any LogitProcessor)?
    private var penaltyProcessor: (any LogitProcessor)?
    private var turnStopProcessor: (any LogitProcessor)?
    private var thinkingBudgetProcessor: (any LogitProcessor)?

    public init(
        grammarProcessor: any LogitProcessor,
        penaltyProcessor: (any LogitProcessor)?,
        turnStopProcessor: (any LogitProcessor)? = nil,
        thinkingBudgetProcessor: (any LogitProcessor)? = nil
    ) {
        self.grammarProcessor = grammarProcessor
        self.penaltyProcessor = penaltyProcessor
        self.turnStopProcessor = turnStopProcessor
        self.thinkingBudgetProcessor = thinkingBudgetProcessor
    }

    /// Designated initializer accepting an optional grammar slot. Used when
    /// the chain consists of `penalty + turnStop + thinkingBudget` only
    /// (no grammar / JSON / schema constraint).
    public init(
        grammarProcessor: (any LogitProcessor)?,
        penaltyProcessor: (any LogitProcessor)?,
        turnStopProcessor: (any LogitProcessor)?,
        thinkingBudgetProcessor: (any LogitProcessor)?,
        _ allowAllNil: Bool = false
    ) {
        precondition(
            allowAllNil
                || grammarProcessor != nil
                || penaltyProcessor != nil
                || turnStopProcessor != nil
                || thinkingBudgetProcessor != nil,
            "ComposedLogitProcessor: at least one slot must be non-nil"
        )
        self.grammarProcessor = grammarProcessor
        self.penaltyProcessor = penaltyProcessor
        self.turnStopProcessor = turnStopProcessor
        self.thinkingBudgetProcessor = thinkingBudgetProcessor
    }

    /// Convenience builder — flattens trivially-empty chains.
    /// - If exactly one slot is non-nil, returns that slot directly (avoids
    ///   the per-call allocation overhead of the composed wrapper).
    /// - If all slots are nil, returns nil.
    /// - Otherwise returns a `ComposedLogitProcessor` with all non-nil slots.
    public static func compose(
        grammar: (any LogitProcessor)? = nil,
        penalty: (any LogitProcessor)? = nil,
        turnStop: (any LogitProcessor)? = nil,
        thinkingBudget: (any LogitProcessor)? = nil
    ) -> (any LogitProcessor)? {
        let nonNil = [grammar, penalty, turnStop, thinkingBudget].compactMap { $0 }
        if nonNil.isEmpty { return nil }
        if nonNil.count == 1 { return nonNil[0] }
        return ComposedLogitProcessor(
            grammarProcessor: grammar,
            penaltyProcessor: penalty,
            turnStopProcessor: turnStop,
            thinkingBudgetProcessor: thinkingBudget
        )
    }

    public func prompt(_ prompt: MLXArray) {
        grammarProcessor?.prompt(prompt)
        penaltyProcessor?.prompt(prompt)
        turnStopProcessor?.prompt(prompt)
        thinkingBudgetProcessor?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        var result = penaltyProcessor?.process(logits: logits) ?? logits
        result = grammarProcessor?.process(logits: result) ?? result
        result = turnStopProcessor?.process(logits: result) ?? result
        result = thinkingBudgetProcessor?.process(logits: result) ?? result
        return result
    }

    public func didSample(token: MLXArray) {
        grammarProcessor?.didSample(token: token)
        penaltyProcessor?.didSample(token: token)
        turnStopProcessor?.didSample(token: token)
        thinkingBudgetProcessor?.didSample(token: token)
    }
}
