import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore

/// Per-token vocabulary metadata + bool-mask construction utilities.
///
/// **Lifecycle:**
/// - Construct via `init(tokenizer:)` → returns an *unmaterialized* builder
///   (`vocabSize == 0`).
/// - First `process(logits:)` of any consumer (JSONLogitProcessor etc.) calls
///   `materialize(vocabSize:)` with the authoritative vocab from
///   `logits.dim(-1)`.  After that, `vocabSize` is non-zero and `buildMask*`
///   methods are usable.
///
/// **Why lazy:** brand-new model families (Qwen3.6 = 248k vocab, Gemma-4 =
/// 262k) exceed the historical 128k probe-cap.  Reading the dim from the first
/// forward pass guarantees ground truth without trusting `config.json` (which
/// can lie in repacked community quants).
///
/// **Caching:** see `TokenMaskBuilderCache` — once materialized for a model id,
/// the same instance is reused across requests.
public final class TokenMaskBuilder: @unchecked Sendable {
    public private(set) var vocabSize: Int
    private let eosTokenId: Int?
    private let tokenizer: NovaMLXEngine.Tokenizer

    private var tokenDecodedTexts: [String] = []
    private var tokenFirstChars: [Character?] = []
    private var tokenIsWhitespaceOrEmpty: [Bool] = []

    private let lock = NSLock()
    private var materialized: Bool = false

    public init(tokenizer: NovaMLXEngine.Tokenizer) {
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.vocabSize = 0
    }

    /// Eager constructor (used by tests and callers that already know the
    /// vocab size).  Equivalent to `init(tokenizer:) → materialize(vocabSize:)`.
    public convenience init(tokenizer: NovaMLXEngine.Tokenizer, vocabSize: Int) {
        self.init(tokenizer: tokenizer)
        self.materialize(vocabSize: vocabSize)
    }

    /// Idempotent: populates per-token tables for the given vocab size.
    /// Subsequent calls are no-ops unless `vocabSize` differs (in which case
    /// the larger of the two wins to keep tables monotonically growing —
    /// guards against logits dim mismatches across requests).
    public func materialize(vocabSize newSize: Int) {
        lock.lock()
        defer { lock.unlock() }

        guard newSize > 0 else { return }
        if materialized && newSize <= vocabSize { return }

        // (Re)build per-token tables. We decode every id from 0..<newSize.
        // Tokens that decode to empty (reserved/special) are kept with empty
        // text — their `isWhitespaceOrEmpty` flag is true so they pass mask
        // checks where whitespace is allowed.
        var decodedTexts = [String](repeating: "", count: newSize)
        var firstChars = [Character?](repeating: nil, count: newSize)
        var wsOrEmpty = [Bool](repeating: true, count: newSize)

        for i in 0..<newSize {
            let text = tokenizer.decode([i])
            decodedTexts[i] = text
            let fc = text.first(where: { !$0.isWhitespace && $0 != "▁" && $0 != " " })
            firstChars[i] = fc
            wsOrEmpty[i] = text.isEmpty || text.allSatisfy({ $0.isWhitespace })
        }

        self.tokenDecodedTexts = decodedTexts
        self.tokenFirstChars = firstChars
        self.tokenIsWhitespaceOrEmpty = wsOrEmpty
        self.vocabSize = newSize
        self.materialized = true
    }

    public func decodedText(for tokenId: Int) -> String {
        guard tokenId >= 0, tokenId < tokenDecodedTexts.count else { return "" }
        return tokenDecodedTexts[tokenId]
    }

    public func buildMask(allowedChars: Set<Character>, eosAllowed: Bool = true) -> MLXArray {
        precondition(materialized, "TokenMaskBuilder.buildMask called before materialize")
        var maskValues = [Bool](repeating: false, count: vocabSize)
        for i in 0..<vocabSize {
            if i == eosTokenId {
                maskValues[i] = eosAllowed
                continue
            }
            if tokenIsWhitespaceOrEmpty[i] {
                maskValues[i] = true
                continue
            }
            if let fc = tokenFirstChars[i], allowedChars.contains(fc) {
                maskValues[i] = true
            }
        }
        return MLXArray(maskValues)[.newAxis, 0...]
    }

    public func buildMaskWithWhitespaceCheck(
        allowedChars: Set<Character>,
        eosAllowed: Bool = true,
        canAcceptWhitespace: () -> Bool
    ) -> MLXArray {
        precondition(materialized, "TokenMaskBuilder.buildMaskWithWhitespaceCheck called before materialize")
        let wsAllowed = canAcceptWhitespace()
        var maskValues = [Bool](repeating: false, count: vocabSize)
        for i in 0..<vocabSize {
            if i == eosTokenId {
                maskValues[i] = eosAllowed
                continue
            }
            if tokenIsWhitespaceOrEmpty[i] {
                maskValues[i] = wsAllowed
                continue
            }
            if let fc = tokenFirstChars[i], allowedChars.contains(fc) {
                maskValues[i] = true
            }
        }
        return MLXArray(maskValues)[.newAxis, 0...]
    }

    public func buildEOSMask() -> MLXArray {
        precondition(materialized, "TokenMaskBuilder.buildEOSMask called before materialize")
        var maskValues = [Bool](repeating: false, count: vocabSize)
        if let eosId = eosTokenId, eosId >= 0, eosId < vocabSize {
            maskValues[eosId] = true
        }
        return MLXArray(maskValues)[.newAxis, 0...]
    }

    /// Apply a `[1, M]` bool mask to `[..., V]` logits, surviving any
    /// `M != V` mismatch defensively.
    ///
    /// - If `M == V`: standard `where(mask, logits, -inf)`.
    /// - If `M  > V`: slice mask to `[..., 0..<V]`.
    /// - If `M  < V`: pad mask with `false` up to V (positions beyond M
    ///   are forbidden — safer than `true` because no consumer expects to
    ///   sample tokens it did not enumerate).
    public static func applyMask(_ mask: MLXArray, to logits: MLXArray) -> MLXArray {
        let lastDim = logits.shape[logits.shape.count - 1]
        let maskDim = mask.shape[mask.shape.count - 1]

        let resolvedMask: MLXArray
        if maskDim == lastDim {
            resolvedMask = mask
        } else if maskDim > lastDim {
            resolvedMask = mask[0..., 0..<lastDim]
        } else {
            // Pad with `false` so untracked tail tokens cannot be sampled.
            // mask is always rank ≥ 2 with a leading [1, ...] batch axis (see
            // buildMask*); pad shape mirrors that.
            let padCount = lastDim - maskDim
            let padValues = [Bool](repeating: false, count: padCount)
            let padFlat = MLXArray(padValues)
            // Reshape pad to match mask rank: prepend leading 1-dims as needed.
            let leadingDims = Array(repeating: 1, count: max(mask.shape.count - 1, 0))
            let pad = padFlat.reshaped(leadingDims + [padCount])
            resolvedMask = MLX.concatenated([mask, pad], axis: -1)
        }
        return MLX.where(resolvedMask, logits, MLXArray(-Float.infinity))
    }
}
