import Testing
import Foundation
import MLX
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// TokenMaskBuilder — vocab discovery + defensive applyMask
//
// Regression coverage for the T13 crash on Qwen3.6-27B / Qwen3.6-35B /
// Gemma-4-26b / Ling-2.6-flash, all of which have vocab > 128k. The
// historical implementation hard-capped probing at 128 000 then sliced
// `mask[..., 0..<lastDim]` against the model's true logits dim, triggering
// SIGTRAP in `applyMask`. The replacement materializes lazily from the
// authoritative `logits.dim(-1)` and never crashes on a size mismatch.
// ────────────────────────────────────────────────────────────

@Suite("TokenMaskBuilder")
struct TokenMaskBuilderTests {

    /// Build a synthetic tokenizer that decodes id -> "x{id}" for the first
    /// `populated` tokens and "" for the rest. EOS is configurable.
    private func makeTokenizer(populated: Int, eos: Int?) -> NovaMLXEngine.Tokenizer {
        return NovaMLXEngine.Tokenizer(
            encode: { _ in [] },
            decode: { ids in
                guard let id = ids.first else { return "" }
                if id < populated {
                    // Mix of punctuation, letters, digits — exercises the
                    // first-char/whitespace classification.
                    let pool: [Character] = ["a", "1", "{", "}", "[", "]", "\"", " ", "\n"]
                    return String(pool[id % pool.count])
                }
                return ""
            },
            eosToken: nil,
            eosTokenId: eos
        )
    }

    @Test("Lazy builder reports vocabSize=0 before materialize")
    func lazyBuilderReportsZeroBeforeMaterialize() {
        let tokenizer = makeTokenizer(populated: 100, eos: 0)
        let builder = TokenMaskBuilder(tokenizer: tokenizer)
        #expect(builder.vocabSize == 0)
    }

    @Test("materialize populates vocab and survives 300k vocab without crashing")
    func materializeLargeVocab() {
        let tokenizer = makeTokenizer(populated: 300_000, eos: 0)
        let builder = TokenMaskBuilder(tokenizer: tokenizer)
        builder.materialize(vocabSize: 300_000)
        #expect(builder.vocabSize == 300_000)
    }

    @Test("materialize is idempotent at the same size")
    func materializeIdempotent() {
        let tokenizer = makeTokenizer(populated: 1024, eos: 0)
        let builder = TokenMaskBuilder(tokenizer: tokenizer)
        builder.materialize(vocabSize: 1024)
        builder.materialize(vocabSize: 1024)
        #expect(builder.vocabSize == 1024)
    }

    @Test("Eager constructor materializes immediately")
    func eagerConstructor() {
        let tokenizer = makeTokenizer(populated: 512, eos: 0)
        let builder = TokenMaskBuilder(tokenizer: tokenizer, vocabSize: 512)
        #expect(builder.vocabSize == 512)
    }

    // MARK: - applyMask defensive behavior

    @Test("applyMask: equal sizes — standard where()")
    func applyMaskEqualSizes() {
        // mask: [1, 4] = [true, false, true, false]
        let mask = MLXArray([true, false, true, false] as [Bool])[.newAxis, 0...]
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float], [1, 4])

        let out = TokenMaskBuilder.applyMask(mask, to: logits)
        let outFlat = out[0].asArray(Float.self)

        #expect(outFlat.count == 4)
        #expect(outFlat[0] == 1.0)
        #expect(outFlat[1] == -.infinity)
        #expect(outFlat[2] == 3.0)
        #expect(outFlat[3] == -.infinity)
    }

    @Test("applyMask: mask larger than logits — slices to logits dim")
    func applyMaskLargerMask() {
        // mask: [1, 6], logits: [1, 4]
        let mask = MLXArray([true, false, true, false, true, true] as [Bool])[.newAxis, 0...]
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float], [1, 4])

        let out = TokenMaskBuilder.applyMask(mask, to: logits)
        let outFlat = out[0].asArray(Float.self)

        #expect(outFlat.count == 4)
        #expect(outFlat[0] == 1.0)
        #expect(outFlat[1] == -.infinity)
        #expect(outFlat[2] == 3.0)
        #expect(outFlat[3] == -.infinity)
    }

    @Test("applyMask: mask smaller than logits — pads tail with false (forbidden)")
    func applyMaskSmallerMaskPadsForbidden() {
        // mask: [1, 2], logits: [1, 5] — this is the historical crash:
        // builder cap=128k against 248k+ logits. After fix: tail tokens
        // (positions 2..4) become -inf, never sampled.
        let mask = MLXArray([true, true] as [Bool])[.newAxis, 0...]
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float], [1, 5])

        let out = TokenMaskBuilder.applyMask(mask, to: logits)
        let outFlat = out[0].asArray(Float.self)

        #expect(outFlat.count == 5)
        #expect(outFlat[0] == 1.0)
        #expect(outFlat[1] == 2.0)
        #expect(outFlat[2] == -.infinity)
        #expect(outFlat[3] == -.infinity)
        #expect(outFlat[4] == -.infinity)
    }

    @Test("applyMask: 248k logits vs 128k mask — no crash, tail forbidden")
    func applyMaskRealisticVocabMismatch() {
        // Synthesizes the exact pre-fix crash condition. We build a 128k
        // bool mask of `true` values and apply it against [1, 248_320]
        // logits. Pre-fix: assertion / SIGTRAP inside MLX `where`. Post-fix:
        // pads with `false`, returns a [1, 248_320] result where the 128k
        // head matches input and the 120k tail is -inf.
        let maskSize = 128_000
        let logitsSize = 248_320
        let maskValues = [Bool](repeating: true, count: maskSize)
        let mask = MLXArray(maskValues)[.newAxis, 0...]
        let logits = MLXArray.ones([1, logitsSize])

        let out = TokenMaskBuilder.applyMask(mask, to: logits)
        #expect(out.shape == [1, logitsSize])

        // Head: value preserved.
        let headSample = out[0, 0].item(Float.self)
        #expect(headSample == 1.0)

        // Tail: forbidden.
        let tailSample = out[0, logitsSize - 1].item(Float.self)
        #expect(tailSample == -.infinity)
    }
}
