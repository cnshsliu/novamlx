import Testing
import Foundation
import MLX
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// JSONLogitProcessor — strict-FSM, parent-frame-aware mask precomputation.
//
// These tests target the Tier-3 fix: tokens like `,`, `}`, `]` must be
// admitted from `.inNumber` (and other in-flight states) exactly when the
// current `parentFrame` makes them grammatically valid. The Tier-2
// implementation conservatively forbade ALL of them in any depth-ambiguous
// state, which on Qwen3.6-27B caused a runaway `30.000000…` because the
// model could only escape `.inNumber` via a whitespace-leading token.
//
// We exercise `simulateTokenIsValid` and `makeInitialSimContext` directly
// (no real model) so the assertions are deterministic and don't depend on
// MLX kernel state. Mask precomputation is exercised end-to-end via a
// synthetic tokenizer.
// ────────────────────────────────────────────────────────────

@Suite("JSONLogitProcessor (strict-FSM, parent-frame-aware)")
struct JSONLogitProcessorTests {

    // MARK: - Synthetic tokenizer helper

    /// Build a tokenizer where each id maps to a fixed string. Used to drive
    /// mask precomputation for known-token vocabularies.
    private func makeTokenizer(vocab: [String], eosId: Int? = nil) -> NovaMLXEngine.Tokenizer {
        NovaMLXEngine.Tokenizer(
            encode: { _ in [] },
            decode: { ids in
                guard let id = ids.first, id >= 0, id < vocab.count else { return "" }
                return vocab[id]
            },
            eosToken: nil,
            eosTokenId: eosId
        )
    }

    /// Build a processor with an explicit vocab, materialized eagerly so
    /// `precomputedMasks` is populated and we can probe specific token ids.
    private func makeProcessor(vocab: [String], eosId: Int? = nil) -> JSONLogitProcessor {
        let tokenizer = makeTokenizer(vocab: vocab, eosId: eosId)
        return JSONLogitProcessor(tokenizer: tokenizer, vocabSize: vocab.count)
    }

    // MARK: - Mask key invariants

    @Test("Object-only states only build masks for parentFrame=.object")
    func objectOnlyStatesRestrictedToObjectParent() {
        // Vocab containing tokens that exercise object-context terminators.
        let vocab = ["{", "}", ",", ":", "\"", " "]
        let proc = makeProcessor(vocab: vocab, eosId: 100)

        // `.expectObjectKeyOrEnd` must have a mask for parent=.object and NOT
        // for parent=.topLevel or parent=.array (those combinations are
        // structurally unreachable at runtime).
        #expect(proc.testMask(state: .expectObjectKeyOrEnd, parent: .object) != nil)
        #expect(proc.testMask(state: .expectObjectKeyOrEnd, parent: .topLevel) == nil)
        #expect(proc.testMask(state: .expectObjectKeyOrEnd, parent: .array) == nil)

        #expect(proc.testMask(state: .expectObjectCommaOrEnd, parent: .object) != nil)
        #expect(proc.testMask(state: .expectObjectCommaOrEnd, parent: .array) == nil)

        #expect(proc.testMask(state: .afterKeyQuote, parent: .object) != nil)
        #expect(proc.testMask(state: .afterKeyQuote, parent: .array) == nil)

        #expect(proc.testMask(state: .expectColon, parent: .object) != nil)
        #expect(proc.testMask(state: .expectColon, parent: .topLevel) == nil)
    }

    @Test("Array-only states only build masks for parentFrame=.array")
    func arrayOnlyStatesRestrictedToArrayParent() {
        let vocab = ["[", "]", ",", "1", "2", "3"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        #expect(proc.testMask(state: .expectArrayValueOrEnd, parent: .array) != nil)
        #expect(proc.testMask(state: .expectArrayValueOrEnd, parent: .object) == nil)
        #expect(proc.testMask(state: .expectArrayCommaOrEnd, parent: .array) != nil)
        #expect(proc.testMask(state: .expectArrayCommaOrEnd, parent: .topLevel) == nil)
    }

    @Test("expectValueAfterColon is object-only; expectValue and done are top-level only")
    func valueStartParentRestrictions() {
        let vocab = ["1", "{", "[", "\""]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        #expect(proc.testMask(state: .expectValueAfterColon, parent: .object) != nil)
        #expect(proc.testMask(state: .expectValueAfterColon, parent: .topLevel) == nil)

        #expect(proc.testMask(state: .expectValue, parent: .topLevel) != nil)
        #expect(proc.testMask(state: .expectValue, parent: .object) == nil)
        #expect(proc.testMask(state: .expectValue, parent: .array) == nil)

        #expect(proc.testMask(state: .done, parent: .topLevel) != nil)
        #expect(proc.testMask(state: .done, parent: .object) == nil)
    }

    @Test("Multi-context states (inNumber, inString, inLiteral*) build masks for ALL parents")
    func multiContextStatesBuildAllParentMasks() {
        let vocab = ["1", "2", ",", "}", "]", " "]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        for parent: JSONLogitProcessor.ParentFrame in [.topLevel, .object, .array] {
            #expect(proc.testMask(state: .inNumber, parent: parent) != nil)
            #expect(proc.testMask(state: .inString, parent: parent) != nil)
            #expect(proc.testMask(state: .inLiteralTru, parent: parent) != nil)
            #expect(proc.testMask(state: .inLiteralFals, parent: parent) != nil)
            #expect(proc.testMask(state: .inLiteralNul, parent: parent) != nil)
        }
    }

    // MARK: - .inNumber termination — the core Qwen3.6-27B bug

    @Test("inNumber w/ object parent admits ',' and '}' but NOT ']'")
    func inNumberObjectParentAdmitsObjectTerminators() {
        // The repro: inside `{"value": 30…}`, FSM is at .inNumber with
        // parentFrame=.object. Pre-fix, only whitespace was allowed — model
        // got stuck producing more digits. Post-fix, `,` and `}` admitted.
        let vocab = [",", "}", "]", " ", "0", "1"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        let mask = proc.testMaskBoolValues(state: .inNumber, parent: .object)
        #expect(mask?[0] == true,  "',' must be admitted from .inNumber w/ object parent")
        #expect(mask?[1] == true,  "'}' must be admitted from .inNumber w/ object parent")
        #expect(mask?[2] == false, "']' must NOT be admitted from .inNumber w/ object parent")
        #expect(mask?[3] == true,  "' ' (whitespace) admitted")
        #expect(mask?[4] == true,  "'0' digit admitted")
        #expect(mask?[5] == true,  "'1' digit admitted")
    }

    @Test("inNumber w/ array parent admits ',' and ']' but NOT '}'")
    func inNumberArrayParentAdmitsArrayTerminators() {
        let vocab = [",", "}", "]", " ", "0"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        let mask = proc.testMaskBoolValues(state: .inNumber, parent: .array)
        #expect(mask?[0] == true,  "',' admitted (next array element)")
        #expect(mask?[1] == false, "'}' NOT admitted from array context")
        #expect(mask?[2] == true,  "']' admitted (close array)")
        #expect(mask?[3] == true,  "whitespace admitted")
        #expect(mask?[4] == true,  "digit admitted")
    }

    @Test("inNumber w/ topLevel parent admits whitespace and digits, NOT structural")
    func inNumberTopLevelParentRestrictsTerminators() {
        let vocab = [",", "}", "]", " ", "0"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        let mask = proc.testMaskBoolValues(state: .inNumber, parent: .topLevel)
        #expect(mask?[0] == false, "',' invalid at top level (would imply 2nd value)")
        #expect(mask?[1] == false, "'}' invalid — no enclosing object")
        #expect(mask?[2] == false, "']' invalid — no enclosing array")
        #expect(mask?[3] == true,  "whitespace allowed (transitions to .done)")
        #expect(mask?[4] == true,  "digit allowed")
    }

    // MARK: - Multi-char tokens crossing finishValue mid-token

    @Test("Multi-char tokens that finish-then-continue admitted only in matching parent")
    func multiCharTerminatorTokens() {
        // "1," — single token ending with object/array terminator. Must be
        // admitted from .inNumber with object OR array parent, and from
        // .expectValue with array parent (where 1 enters .inNumber and , is a
        // valid array separator).
        let vocab = ["1,", "1}", "1]"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)

        let inNumObject = proc.testMaskBoolValues(state: .inNumber, parent: .object)
        // "1," : digit, then ',' — finishValue → .expectObjectCommaOrEnd → .expectObjectKeyOrEnd. OK.
        #expect(inNumObject?[0] == true)
        // "1}" : digit, then '}' — finishValue → pop → .done. OK.
        #expect(inNumObject?[1] == true)
        // "1]" : digit, then ']' — finishValue → .expectObjectCommaOrEnd → ']' rejected. Forbidden.
        #expect(inNumObject?[2] == false)

        let inNumArray = proc.testMaskBoolValues(state: .inNumber, parent: .array)
        #expect(inNumArray?[0] == true,  "'1,' valid in array (next element)")
        #expect(inNumArray?[1] == false, "'1}' invalid in array")
        #expect(inNumArray?[2] == true,  "'1]' valid (close array)")
    }

    @Test("Multi-frame-pop tokens conservatively rejected (we don't know the outer frame)")
    func multiFramePopTokensRejected() {
        // "]}" requires the stack to be [..., true, false] (array inside
        // object). We deliberately don't claim to know the outer frame at
        // precompute time — the mask says "rejected" and the runtime FSM
        // will validate against actual depth.
        let vocab = ["]}", "}]", "}}", "]]"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)

        for parent: JSONLogitProcessor.ParentFrame in [.topLevel, .object, .array] {
            let mask = proc.testMaskBoolValues(state: .inNumber, parent: parent) ?? []
            for (i, _) in vocab.enumerated() where i < mask.count {
                #expect(mask[i] == false,
                       "Multi-pop token '\(vocab[i])' must be rejected from .inNumber, parent=\(parent)")
            }
        }
    }

    // MARK: - Literal sub-state termination (true/false/null)

    @Test("inLiteralTru w/ object parent admits closing 'e' followed by ',' or '}'")
    func literalTrueClosingInObjectContext() {
        // Tokens that end the literal `true` and then continue: "e," "e}"
        let vocab = ["e", "e,", "e}", "e]", "e ", "x"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)
        let mask = proc.testMaskBoolValues(state: .inLiteralTru, parent: .object)
        #expect(mask?[0] == true,  "'e' closes literal; finishValue → .expectObjectCommaOrEnd")
        #expect(mask?[1] == true,  "'e,' valid in object context")
        #expect(mask?[2] == true,  "'e}' valid (close object)")
        #expect(mask?[3] == false, "'e]' invalid in object context")
        #expect(mask?[4] == true,  "'e ' valid (trailing whitespace)")
        #expect(mask?[5] == false, "'x' invalid — not the terminating letter of 'true'")
    }

    // MARK: - EOS gating

    @Test("EOS only admitted at .done with topLevel parent")
    func eosOnlyAtDone() {
        let vocab = ["a"]  // index 0 will be EOS for this test
        let proc = makeProcessor(vocab: vocab, eosId: 0)

        // EOS forbidden mid-structure
        let inNum = proc.testMaskBoolValues(state: .inNumber, parent: .object) ?? []
        #expect(inNum.first == false, "EOS forbidden inside .inNumber")

        let inObjKey = proc.testMaskBoolValues(state: .expectObjectKeyOrEnd, parent: .object) ?? []
        #expect(inObjKey.first == false, "EOS forbidden inside .expectObjectKeyOrEnd")

        // EOS allowed only at done/topLevel
        let done = proc.testMaskBoolValues(state: .done, parent: .topLevel) ?? []
        #expect(done.first == true, "EOS allowed at .done")
    }

    // MARK: - End-to-end FSM trace (sanity)

    @Test("E2E: number in object terminates correctly via ','")
    func endToEndNumberTerminationViaComma() {
        // Walk the runtime FSM through `{"a":1,"b":2}` to verify the
        // parent-frame-aware masks line up with actual transitions.
        let vocab = ["{", "}", "\"", "a", "b", ":", ",", "1", "2"]
        let proc = makeProcessor(vocab: vocab, eosId: 100)

        // Drive didSample manually with each char.
        let chars: [Character] = ["{", "\"", "a", "\"", ":", "1", ",", "\"", "b", "\"", ":", "2", "}"]
        var state = JSONLogitProcessor.JSONState.expectValue
        var depth: [Bool] = []
        var escape = false
        var isKey = false
        for c in chars {
            let ctx = JSONLogitProcessor.stepForTesting(
                state: state, depthStack: depth, escapeNext: escape, stringIsKey: isKey, char: c
            )
            if let next = ctx {
                state = next.state
                depth = next.depthStack
                escape = next.escapeNext
                isKey = next.stringIsKey
            } else {
                Issue.record("FSM rejected valid character '\(c)' at state \(state)")
                return
            }
        }
        #expect(state == .done, "Object should be closed; got \(state)")
        #expect(depth.isEmpty, "Stack should be empty at end")
        _ = proc  // silence unused warning
    }
}

// MARK: - Test helpers (driving via the JSONLogitProcessor's internal hooks)

extension JSONLogitProcessor {
    /// Convenience: probe the precomputed mask for a (state, parent) pair.
    func testMask(state: JSONState, parent: ParentFrame) -> MLXArray? {
        return precomputedMasksForTesting[MaskKey(state: state, parentFrame: parent)]
    }

    /// Convenience: read mask as [Bool] for assertions on individual token ids.
    func testMaskBoolValues(state: JSONState, parent: ParentFrame) -> [Bool]? {
        guard let mask = testMask(state: state, parent: parent) else { return nil }
        // Mask shape is [1, vocabSize]
        return mask[0].asArray(Bool.self)
    }
}
