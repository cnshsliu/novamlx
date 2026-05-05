import Testing
import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils
@testable import NovaMLXEngine

/// Regression tests for `ThinkingBudgetProcessor` — production thinking-budget
/// enforcement for reasoning models under greedy decoding (temp=0).
///
/// **Background**: Verified against Python `mlx_lm` 0.31.3 on
/// `mlx-community/Qwen3.6-27B-4bit` that at temp=0 + complex prompts the model
/// gets stuck in chain-of-thought and never emits `</think>` (T7 of
/// `todo.markdown` §1, also reproducible with `mlx_lm.generate`). Qwen team
/// explicitly recommends `temperature ≥ 0.6` for thinking mode. The budget
/// processor enforces a fall-back close marker so production temp=0 pipelines
/// (eval, benchmark, coding agents) still get response content.
@Suite("ThinkingBudgetProcessor Tests")
struct ThinkingBudgetProcessorTests {

    /// Build a logits tensor where token `winnerId` has the highest score
    /// and every other token has a lower score. Useful for asserting that
    /// argmax of the post-processing logits matches the budget mask.
    private func logits(vocabSize: Int, winnerId: Int, winnerScore: Float = 10, baseline: Float = 0) -> MLXArray {
        var values = [Float](repeating: baseline, count: vocabSize)
        if winnerId >= 0 && winnerId < vocabSize {
            values[winnerId] = winnerScore
        }
        return MLXArray(values, [1, vocabSize])
    }

    /// Compute argmax over a `[1, V]` MLXArray and return the integer id.
    private func argmaxId(_ array: MLXArray) -> Int {
        let last = array.shape.last ?? 0
        var bestId = 0
        var bestVal = -Float.infinity
        let flat = array.asArray(Float.self)
        for i in 0..<last {
            if flat[i] > bestVal {
                bestVal = flat[i]
                bestId = i
            }
        }
        return bestId
    }

    // MARK: - prompt() resets state

    @Test("prompt() resets generated counter and state")
    func promptResetsState() {
        let proc = ThinkingBudgetProcessor(budget: 5, closeTokenIds: [42])
        proc.prompt(MLXArray(0))
        // Simulate 5 sampled tokens → trips budget
        for _ in 0..<5 { proc.didSample(token: MLXArray(7)) }
        #expect(proc.summary.generated == 5)

        proc.prompt(MLXArray(0))
        #expect(proc.summary.generated == 0, "prompt() must zero the counter for the next request")
        #expect(proc.summary.closed == false)
        #expect(proc.summary.forcedAt == nil)
    }

    // MARK: - Within budget = passthrough

    @Test("Within budget: process() returns logits unchanged shape")
    func withinBudgetPassthrough() {
        let proc = ThinkingBudgetProcessor(budget: 100, closeTokenIds: [42])
        proc.prompt(MLXArray(0))
        // Sample a few non-close tokens
        for _ in 0..<10 { proc.didSample(token: MLXArray(7)) }

        let l = logits(vocabSize: 50, winnerId: 7)
        let out = proc.process(logits: l)
        // Shape preserved
        #expect(out.shape == l.shape)
        // Argmax unchanged (no mask applied)
        #expect(argmaxId(out) == 7, "below budget = passthrough; argmax should stay at the original winner")
    }

    // MARK: - Budget exceeded → mask to close-only

    @Test("Budget exceeded: process() masks to close-token-only; argmax becomes close id")
    func exceededBudgetMasksToClose() {
        let proc = ThinkingBudgetProcessor(budget: 3, closeTokenIds: [42])
        proc.prompt(MLXArray(0))

        // Sample 3 tokens — budget exhausted (state transitions to .forceClose)
        for _ in 0..<3 { proc.didSample(token: MLXArray(7)) }
        #expect(proc.summary.generated == 3)

        // The next process() should mask everything except the close token.
        let l = logits(vocabSize: 50, winnerId: 7) // raw winner is 7
        let out = proc.process(logits: l)
        let chosen = argmaxId(out)
        #expect(chosen == 42, "after budget exhausted, argmax must be the forced close-token id (got \(chosen))")
    }

    @Test("Multiple close-token ids: argmax picks one of them, not any other token")
    func multipleCloseIdsValidArgmax() {
        let closeIds: Set<Int> = [42, 43, 44]
        let proc = ThinkingBudgetProcessor(budget: 2, closeTokenIds: closeIds)
        proc.prompt(MLXArray(0))
        for _ in 0..<2 { proc.didSample(token: MLXArray(7)) }

        let l = logits(vocabSize: 50, winnerId: 7)
        let out = proc.process(logits: l)
        let chosen = argmaxId(out)
        #expect(closeIds.contains(chosen),
            "argmax of masked logits must be one of the close-token ids \(closeIds.sorted()), got \(chosen)")
    }

    // MARK: - Organic close: processor goes inert

    @Test("Organic close emission deactivates the processor")
    func organicCloseDeactivates() {
        let proc = ThinkingBudgetProcessor(budget: 100, closeTokenIds: [42])
        proc.prompt(MLXArray(0))

        // Model emits close on its own at token 5
        for _ in 0..<5 { proc.didSample(token: MLXArray(7)) }
        proc.didSample(token: MLXArray(42)) // close
        #expect(proc.summary.closed == true)

        // Subsequent samples don't trigger budget. Vocab has to be >= the
        // winner id we're checking — pick winner=30 within vocab=50.
        for _ in 0..<200 { proc.didSample(token: MLXArray(30)) }
        let l = logits(vocabSize: 50, winnerId: 30)
        let out = proc.process(logits: l)
        #expect(argmaxId(out) == 30,
            "post-close samples must pass through unchanged (response phase, no masking)")
    }

    // MARK: - Forced close: subsequent samples don't re-trigger

    @Test("Forced close: only one mask application, then inert")
    func forcedCloseOnlyOnce() {
        let proc = ThinkingBudgetProcessor(budget: 2, closeTokenIds: [42])
        proc.prompt(MLXArray(0))

        for _ in 0..<2 { proc.didSample(token: MLXArray(7)) }
        // First post-budget process → masks
        let masked = proc.process(logits: logits(vocabSize: 50, winnerId: 7))
        #expect(argmaxId(masked) == 42)
        // Simulate sampler picking the forced token
        proc.didSample(token: MLXArray(42))
        #expect(proc.summary.closed == true)

        // Subsequent process() calls must be passthrough
        let after = proc.process(logits: logits(vocabSize: 50, winnerId: 7))
        #expect(argmaxId(after) == 7,
            "after forced close, response phase should not be re-masked")
    }

    // MARK: - Vocab-size mismatches

    @Test("Close token id outside vocab: graceful no-op + warning")
    func closeIdOutOfVocab() {
        // Close ids [9999] but vocab size 50 → no allowed id fits.
        let proc = ThinkingBudgetProcessor(budget: 1, closeTokenIds: [9999])
        proc.prompt(MLXArray(0))
        proc.didSample(token: MLXArray(7))

        let l = logits(vocabSize: 50, winnerId: 7)
        let out = proc.process(logits: l)
        // Defensive fallback: log + treat as closed; passthrough unchanged
        #expect(argmaxId(out) == 7,
            "when no close id fits in vocab, processor must NOT trap the sampler — falls back to passthrough")
        #expect(proc.summary.closed == true,
            "after fallback, processor should be inert for the rest of the request")
    }

    // MARK: - State leakage prevention

    @Test("State doesn't leak between successive requests on the same instance")
    func stateNoLeak() {
        let proc = ThinkingBudgetProcessor(budget: 3, closeTokenIds: [42])

        // Request 1: trigger budget, force close
        proc.prompt(MLXArray(0))
        for _ in 0..<3 { proc.didSample(token: MLXArray(7)) }
        _ = proc.process(logits: logits(vocabSize: 50, winnerId: 7))
        proc.didSample(token: MLXArray(42))
        #expect(proc.summary.closed == true)

        // Request 2: fresh prompt, should NOT be in closed state
        proc.prompt(MLXArray(0))
        #expect(proc.summary.closed == false)
        #expect(proc.summary.generated == 0)

        // Verify budget still enforced on request 2
        for _ in 0..<3 { proc.didSample(token: MLXArray(7)) }
        let masked = proc.process(logits: logits(vocabSize: 50, winnerId: 7))
        #expect(argmaxId(masked) == 42, "request 2 should re-enforce budget")
    }
}

/// Tests for `ThinkingCloseMarkerResolver.resolveCloseTokenIds(for:modelId:)`
@Suite("ThinkingCloseMarkerResolver Tests")
struct ThinkingCloseMarkerResolverTests {

    /// Build a synthetic tokenizer where specific strings encode to specific
    /// single-token ids. Other strings encode to multi-token (char codes).
    private func makeTokenizer(singleTokenMap: [String: Int]) -> NovaMLXEngine.Tokenizer {
        // Reverse map for decode (immutable, captured by value into Sendable closures)
        let encodeMap: [String: Int] = singleTokenMap
        var rev: [Int: String] = [:]
        for (text, id) in singleTokenMap { rev[id] = text }
        let decodeMap: [Int: String] = rev
        return NovaMLXEngine.Tokenizer(
            encode: { @Sendable text in
                if let id = encodeMap[text] { return [id] }
                return Array(text.unicodeScalars).map { Int($0.value) }
            },
            decode: { @Sendable ids in
                ids.map { decodeMap[$0] ?? UnicodeScalar($0).map(String.init) ?? "" }.joined()
            },
            eosToken: nil,
            eosTokenId: nil
        )
    }

    @Test("Qwen3.6 mlx-community shape: </think> is single token id 248069 → resolved")
    func qwen36SingleTokenClose() {
        let tokenizer = makeTokenizer(singleTokenMap: [
            "</think>": 248069,
        ])
        let ids = ThinkingCloseMarkerResolver.resolveCloseTokenIds(for: tokenizer, modelId: "test")
        #expect(ids == [248069],
            "single-token </think> must round-trip to a single id and be included")
    }

    @Test("Multi-marker model: both </think> and <|end_of_thought|> resolve")
    func multiMarkerResolves() {
        let tokenizer = makeTokenizer(singleTokenMap: [
            "</think>": 248069,
            "<|end_of_thought|>": 248044,
        ])
        let ids = ThinkingCloseMarkerResolver.resolveCloseTokenIds(for: tokenizer, modelId: "test")
        #expect(ids == [248044, 248069],
            "both registered close markers must be resolved")
    }

    @Test("No close markers in tokenizer: returns empty set")
    func noCloseMarkers() {
        // Tokenizer doesn't have any of the known markers as single tokens.
        let tokenizer = makeTokenizer(singleTokenMap: [:])
        let ids = ThinkingCloseMarkerResolver.resolveCloseTokenIds(for: tokenizer, modelId: "test")
        #expect(ids.isEmpty,
            "if no marker is single-token, resolver must return empty (caller treats as 'budget unsupported')")
    }

    @Test("Multi-token marker: silently dropped")
    func multiTokenMarkerDropped() {
        // </think> NOT registered → encoded as multi-token via fallback
        let tokenizer = makeTokenizer(singleTokenMap: [:])
        let encoded = tokenizer.encode("</think>")
        #expect(encoded.count > 1, "sanity: synthetic tokenizer encodes </think> as multi-token")

        let ids = ThinkingCloseMarkerResolver.resolveCloseTokenIds(for: tokenizer, modelId: "test")
        #expect(ids.isEmpty, "multi-token markers cannot be enforced by single-token mask — must be dropped")
    }
}
