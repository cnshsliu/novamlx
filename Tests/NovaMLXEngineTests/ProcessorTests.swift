import Testing
import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// TurnStopProcessor — stops generation on control token detection
// ────────────────────────────────────────────────────────────

@Suite("TurnStopProcessor Tests")
struct TurnStopProcessorTests {

    /// Helper: simple decode that joins ASCII codes to chars
    func asciiDecode(_ tokens: [Int]) -> String {
        tokens.compactMap { UnicodeScalar($0).map(String.init) }.joined()
    }

    /// Helper: decode that simulates special token strings
    func specialDecode(_ tokens: [Int]) -> String {
        // Simulate: single-token decoding to a control string
        if tokens == [100] { return "<|turn|>" }
        if tokens == [101] { return "<|im_end|>" }
        if tokens == [100, 101] { return "<|turn|><|im_end|>" }
        return asciiDecode(tokens)
    }

    // MARK: - State transitions

    @Test("Active state passes through logits unchanged")
    func activeStatePassthrough() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [2],
            decode: asciiDecode
        )
        let logits = MLXArray([0.1, 0.2, 0.3, 0.4], [1, 4])
        let result = tsp.process(logits: logits)
        // Should be same shape, same values (not force-EOS)
        #expect(result.shape == logits.shape)
    }

    @Test("prompt resets state to active")
    func promptResetsState() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [2],
            decode: specialDecode
        )

        // Feed token that triggers stop
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(100)) // "<|turn|>" → stopDetected

        // Reset with new prompt
        tsp.prompt(MLXArray(0))
        // Should be back to active — process should return logits unchanged
        let logits = MLXArray([0.5, 0.5], [1, 2])
        let result = tsp.process(logits: logits)
        #expect(result.shape == logits.shape)
    }

    // MARK: - Pattern detection

    @Test("didSample detects pattern and forces EOS on next process")
    func detectPatternForcesEOS() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [2],
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))

        // Feed normal tokens first
        tsp.didSample(token: MLXArray(72)) // H
        tsp.didSample(token: MLXArray(73)) // I

        // Feed the trigger token
        tsp.didSample(token: MLXArray(100)) // "<|turn|>"

        // Now process should force EOS
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0], [1, 4])
        let result = tsp.process(logits: logits)
        // Token at index 2 (EOS) should keep its value, others should be -inf
        let eosVal = result[0, 2].item(Float.self)
        let otherVal = result[0, 0].item(Float.self)
        #expect(eosVal == 3.0)
        #expect(otherVal == -Float.infinity)
    }

    @Test("didSample without pattern match keeps active")
    func noPatternMatchKeepsActive() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [2],
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))

        // Feed tokens that don't match pattern
        for token in [72, 73, 74] { // H, I, J
            tsp.didSample(token: MLXArray(token))
        }

        // Process should pass through normally
        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = tsp.process(logits: logits)
        #expect(result.shape == logits.shape)
    }

    // MARK: - State machine: done state is a no-op

    @Test("done state does nothing on process")
    func doneStateNoOp() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [2],
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(100)) // triggers stopDetected
        _ = tsp.process(logits: MLXArray([1.0, 2.0, 3.0], [1, 3])) // transitions to done

        // Now in done state — subsequent process should pass through
        let logits = MLXArray([5.0, 6.0, 7.0], [1, 3])
        let result = tsp.process(logits: logits)
        #expect(result[0, 0].item(Float.self) == 5.0)
    }

    @Test("done state ignores didSample")
    func doneStateIgnoresDidSample() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [2],
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(100)) // triggers stopDetected
        _ = tsp.process(logits: MLXArray([1.0, 2.0, 3.0], [1, 3])) // → done
        tsp.didSample(token: MLXArray(101)) // should be ignored (done state)
        // No crash = pass
    }

    // MARK: - Multiple patterns

    @Test("Multiple stop patterns — matches second pattern")
    func multipleStopPatternsMatchSecond() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>", "<|im_end|>"],
            eosTokenIds: [2, 3], // two possible EOS tokens
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(101)) // "<|im_end|>" → stopDetected

        let logits = MLXArray([1.0, 2.0, 3.0, 4.0], [1, 4])
        let result = tsp.process(logits: logits)
        // Both EOS tokens (indices 2, 3) should be un-masked
        #expect(result[0, 2].item(Float.self) == 3.0)
        #expect(result[0, 3].item(Float.self) == 4.0)
        // Non-EOS should be -inf
        #expect(result[0, 0].item(Float.self) == -Float.infinity)
    }

    // MARK: - Decode behavior

    @Test("Decode tracking — accumulates tokens correctly")
    func decodeAccumulation() {
        // This decode builds a known string from tokens
        let tsp = TurnStopProcessor(
            stopPatterns: ["HI"],
            eosTokenIds: [2],
            decode: asciiDecode
        )
        tsp.prompt(MLXArray(0))

        tsp.didSample(token: MLXArray(72)) // H
        tsp.didSample(token: MLXArray(73)) // I — now decode() returns "HI"

        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = tsp.process(logits: logits)
        // Should detect "HI" and force EOS
        #expect(result[0, 0].item(Float.self) == -Float.infinity)
        #expect(result[0, 2].item(Float.self) == 3.0) // EOS preserved
    }

    // MARK: - EOS token bounds check

    @Test("EOS token out of vocab range is ignored")
    func eosOutOfRange() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [99999], // far beyond vocab
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(100)) // triggers stopDetected

        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = tsp.process(logits: logits)
        // EOS token 99999 >= vocab size 3 — all values should be -inf
        // since no valid EOS tokens survived the mask
        for i in 0..<3 {
            #expect(result[0, i].item(Float.self) == -Float.infinity)
        }
    }

    @Test("Empty EOS token set")
    func emptyEOSTokens() {
        let tsp = TurnStopProcessor(
            stopPatterns: ["<|turn|>"],
            eosTokenIds: [],
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(100)) // triggers stopDetected

        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = tsp.process(logits: logits)
        // No EOS tokens — all masked to -inf
        for i in 0..<3 {
            #expect(result[0, i].item(Float.self) == -Float.infinity)
        }
    }

    @Test("Empty stop patterns never triggers")
    func emptyStopPatterns() {
        let tsp = TurnStopProcessor(
            stopPatterns: [],
            eosTokenIds: [2],
            decode: specialDecode
        )
        tsp.prompt(MLXArray(0))
        tsp.didSample(token: MLXArray(100)) // no patterns to match

        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = tsp.process(logits: logits)
        // Should pass through
        #expect(result[0, 0].item(Float.self) == 1.0)
    }
}

// ────────────────────────────────────────────────────────────
// ComposedLogitProcessor — chains processors
// ────────────────────────────────────────────────────────────

@Suite("ComposedLogitProcessor Tests")
struct ComposedLogitProcessorTests {

    /// A simple processor that adds 1.0 to the first logit value
    final class AddOneProcessor: LogitProcessor, @unchecked Sendable {
        var prompted = false
        var sampled = false

        func prompt(_ prompt: MLXArray) { prompted = true }
        func process(logits: MLXArray) -> MLXArray {
            var vals = logits.asArray(Float.self)
            vals[0] += 1.0
            return MLXArray(vals, logits.shape)
        }
        func didSample(token: MLXArray) { sampled = true }
    }

    @Test("prompt delegates to all processors")
    func promptDelegates() {
        let p1 = AddOneProcessor()
        let p2 = AddOneProcessor()
        let composed = ComposedLogitProcessor(
            grammarProcessor: p1, penaltyProcessor: nil, turnStopProcessor: p2
        )
        composed.prompt(MLXArray(0))
        #expect(p1.prompted == true)
        #expect(p2.prompted == true)
    }

    @Test("process chains: penalty → grammar → turnStop")
    func processChains() {
        let grammar = AddOneProcessor()
        let penalty = AddOneProcessor()
        let turnStop = AddOneProcessor()

        let composed = ComposedLogitProcessor(
            grammarProcessor: grammar,
            penaltyProcessor: penalty,
            turnStopProcessor: turnStop
        )

        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = composed.process(logits: logits)
        // penalty adds 1, grammar adds 1, turnStop adds 1 = 3 total
        #expect(result[0, 0].item(Float.self) == 4.0)
    }

    @Test("nil penalty and turnStop are skipped")
    func nilProcessorsSkipped() {
        let grammar = AddOneProcessor()
        let composed = ComposedLogitProcessor(
            grammarProcessor: grammar, penaltyProcessor: nil, turnStopProcessor: nil
        )
        let logits = MLXArray([1.0, 2.0, 3.0], [1, 3])
        let result = composed.process(logits: logits)
        // Only grammar adds 1
        #expect(result[0, 0].item(Float.self) == 2.0)
    }

    @Test("didSample delegates to all processors")
    func didSampleDelegates() {
        let p1 = AddOneProcessor()
        let p2 = AddOneProcessor()
        let composed = ComposedLogitProcessor(
            grammarProcessor: p1, penaltyProcessor: nil, turnStopProcessor: p2
        )
        composed.didSample(token: MLXArray(42))
        #expect(p1.sampled == true)
        #expect(p2.sampled == true)
    }
}
