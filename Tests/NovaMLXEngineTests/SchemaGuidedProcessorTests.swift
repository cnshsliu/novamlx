import Testing
import Foundation
import MLX
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// SchemaGuidedProcessor — strict pure stepper + full-token simulation.
//
// Regression coverage for the historical Gemma-4-26B garbage-output bug
// where ` thoughtful` slipped past the first-char `.expectValue` mask
// (boolean schema starts with `t`) because the legacy `.inLiteral` state
// advanced its index on ANY char without validating against `lit[idx]`.
// We exercise the new strict pure stepper directly so assertions don't
// depend on MLX kernel state.
//
// Object-key + parent-context coverage: after the FSM was rewired to
// distinguish "before key opens" (`.objectKey`) from "inside the quoted
// key string" (`.inObjectKey`), and `.inString`/`.inNumber`/`.inLiteral`
// learned to carry a `returnTo` post-value state, we added E2E walks for
// object schemas — previously blocked by the legacy conflation.
// ────────────────────────────────────────────────────────────

@Suite("SchemaGuidedProcessor (strict-FSM, full-token simulation)")
struct SchemaGuidedProcessorTests {

    // MARK: - Pure stepper invariants

    @Test(".inLiteral validates char against lit[idx] — REJECTS divergent chars")
    func literalValidatesChar() {
        // The Gemma-4 regression: ` thoughtful` from .expectValue(boolean)
        // advanced through .inLiteral("true", 1) → .inLiteral("true", 2)…
        // accepting `h`, `o`, `u`, `g`, `h`, `t`, `f`, `u`, `l` without
        // checking them. New stepper: `h` ≠ "true"[1] ('r') → nil.
        var escape = false
        let result = SchemaGuidedProcessor.stepStrict(
            state: .inLiteral("true", 1, returnTo: .done),
            escapeNext: &escape,
            char: "h"  // should be 'r'
        )
        #expect(result == nil, "Strict stepper must reject 'h' from .inLiteral(\"true\", 1)")
    }

    @Test(".inLiteral accepts the correct next char")
    func literalAcceptsCorrectChar() {
        var escape = false
        // .inLiteral("true", 1) expects 'r'
        if let next = SchemaGuidedProcessor.stepStrict(
            state: .inLiteral("true", 1, returnTo: .done), escapeNext: &escape, char: "r"
        ) {
            if case .inLiteral(let lit, let idx, _) = next {
                #expect(lit == "true")
                #expect(idx == 2)
            } else {
                Issue.record("Expected .inLiteral, got \(next)")
            }
        } else {
            Issue.record("Strict stepper rejected valid char 'r'")
        }
    }

    @Test(".inLiteral terminates correctly — 'e' from idx=3 finishes 'true'")
    func literalCompletesCleanly() {
        var escape = false
        let result = SchemaGuidedProcessor.stepStrict(
            state: .inLiteral("true", 3, returnTo: .done), escapeNext: &escape, char: "e"
        )
        if case .done = result {
            // ok
        } else {
            Issue.record("Expected .done after closing 'e' of true; got \(String(describing: result))")
        }
    }

    @Test(".inLiteral 'false' rejects wrong letter at any position")
    func falseLiteralRejectsWrongChars() {
        var escape = false
        // false[1] = 'a'
        #expect(SchemaGuidedProcessor.stepStrict(
            state: .inLiteral("false", 1, returnTo: .done), escapeNext: &escape, char: "x"
        ) == nil)
        // false[2] = 'l'
        #expect(SchemaGuidedProcessor.stepStrict(
            state: .inLiteral("false", 2, returnTo: .done), escapeNext: &escape, char: "z"
        ) == nil)
    }

    // MARK: - Permissive states made strict

    @Test(".objectColon rejects non-':' / non-ws chars")
    func objectColonRejectsInvalid() {
        var escape = false
        let parent = SchemaNode(type: .object(properties: ["k": SchemaNode(type: .string)], required: ["k"]))
        let state = SchemaGuidedProcessor.SchemaState.objectColon(
            parent, ["k": SchemaNode(type: .string)], ["k"], "k", returnTo: .done
        )
        #expect(SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "x") == nil)
        #expect(SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: " ") != nil)
        if case .objectValue = SchemaGuidedProcessor.stepStrict(
            state: state, escapeNext: &escape, char: ":"
        ) {} else {
            Issue.record("':' should transition .objectColon → .objectValue")
        }
    }

    @Test(".objectComma rejects digits / letters")
    func objectCommaRejectsInvalid() {
        var escape = false
        let parent = SchemaNode(type: .object(properties: [:], required: []))
        let state = SchemaGuidedProcessor.SchemaState.objectComma(parent, [:], [], returnTo: .done)
        #expect(SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "1") == nil)
        #expect(SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "a") == nil)
        // Valid transitions
        if case .objectKey = SchemaGuidedProcessor.stepStrict(
            state: state, escapeNext: &escape, char: ","
        ) {} else {
            Issue.record("',' should transition .objectComma → .objectKey")
        }
    }

    @Test(".inString rejects ASCII control chars per RFC 8259")
    func inStringRejectsControlChars() {
        var escape = false
        let bell: Character = "\u{0007}"
        #expect(SchemaGuidedProcessor.stepStrict(
            state: .inString(returnTo: .done), escapeNext: &escape, char: bell
        ) == nil, "ASCII bell (0x07) must be rejected in string")
        // \n, \r, \t are also < 0x20 — must also be rejected as raw chars
        let newline: Character = "\n"
        #expect(SchemaGuidedProcessor.stepStrict(
            state: .inString(returnTo: .done), escapeNext: &escape, char: newline
        ) == nil)
    }

    @Test(".inString accepts printable ASCII and most Unicode")
    func inStringAcceptsPrintable() {
        var escape = false
        for c in "abcXYZ123 -+!?" {
            let r = SchemaGuidedProcessor.stepStrict(
                state: .inString(returnTo: .done), escapeNext: &escape, char: c
            )
            #expect(r != nil, "Printable char '\(c)' rejected in string")
        }
    }

    @Test(".inString backslash sets escape; next char is consumed regardless")
    func inStringEscape() {
        var escape = false
        guard let s1 = SchemaGuidedProcessor.stepStrict(
            state: .inString(returnTo: .done), escapeNext: &escape, char: "\\"
        ) else {
            Issue.record("Backslash should be accepted")
            return
        }
        #expect(escape == true)
        if case .inString = s1 {} else { Issue.record("Should remain in string after backslash") }
        // Next char (any) consumed as part of escape
        guard let s2 = SchemaGuidedProcessor.stepStrict(
            state: s1, escapeNext: &escape, char: "n"
        ) else {
            Issue.record("Char after backslash should be accepted")
            return
        }
        #expect(escape == false)
        if case .inString = s2 {} else { Issue.record("Should remain in string after escape pair") }
    }

    // MARK: - Object key string handling (post-refactor)
    //
    // Legacy `.objectKey` conflated "before the opening quote" with "inside
    // the key body", treating the first `"` as a *close* quote. These tests
    // pin down the corrected behavior where `"` from `.objectKey` enters the
    // string body (`.inObjectKey`) and only the matching close `"` advances
    // to `.objectColon`.

    @Test("'.objectKey + \"' opens the key string (does NOT close an empty key)")
    func objectKeyQuoteOpensString() {
        var escape = false
        let parent = SchemaNode(type: .object(properties: ["x": SchemaNode(type: .number)], required: ["x"]))
        let s0: SchemaGuidedProcessor.SchemaState = .objectKey(parent, ["x": SchemaNode(type: .number)], ["x"], returnTo: .done)
        guard let next = SchemaGuidedProcessor.stepStrict(state: s0, escapeNext: &escape, char: "\"") else {
            Issue.record("'\"' should open the key string, not be rejected")
            return
        }
        if case .inObjectKey(_, _, _, let accumulated, _) = next {
            #expect(accumulated == "", "Accumulated key must start empty after open quote")
        } else {
            Issue.record("Expected .inObjectKey after opening quote; got \(next)")
        }
    }

    @Test(".inObjectKey accumulates chars and closes on matching quote")
    func inObjectKeyWalksKeyBody() {
        var escape = false
        let parent = SchemaNode(type: .object(properties: ["name": SchemaNode(type: .string)], required: ["name"]))
        var state: SchemaGuidedProcessor.SchemaState = .inObjectKey(parent, ["name": SchemaNode(type: .string)], ["name"], "", returnTo: .done)
        for c in "name" {
            guard let next = SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: c) else {
                Issue.record("FSM rejected valid key char '\(c)'")
                return
            }
            state = next
        }
        if case .inObjectKey(_, _, _, let acc, _) = state {
            #expect(acc == "name")
        } else {
            Issue.record("Should still be in .inObjectKey after key body; got \(state)")
        }
        // Closing quote transitions to .objectColon with accumulated key
        guard let closed = SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "\"") else {
            Issue.record("Closing quote should be accepted")
            return
        }
        if case .objectColon(_, _, _, let keyName, _) = closed {
            #expect(keyName == "name")
        } else {
            Issue.record("Expected .objectColon after close quote; got \(closed)")
        }
    }

    @Test(".objectKey '}' closes an empty object")
    func objectKeyBraceClosesEmptyObject() {
        var escape = false
        let parent = SchemaNode(type: .object(properties: [:], required: []))
        let state: SchemaGuidedProcessor.SchemaState = .objectKey(parent, [:], [], returnTo: .done)
        guard let next = SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "}") else {
            Issue.record("'}' should close empty object from .objectKey")
            return
        }
        if case .done = next {} else {
            Issue.record("Expected .done after '}'; got \(next)")
        }
    }

    // MARK: - Value completion carries parent context

    @Test(".inNumber + ws transitions to returnTo")
    func inNumberWhitespaceEndsValue() {
        var escape = false
        // Number inside an object value: returnTo = .objectComma
        let parent = SchemaNode(type: .object(properties: [:], required: []))
        let returnTo: SchemaGuidedProcessor.SchemaState = .objectComma(parent, [:], [], returnTo: .done)
        let state: SchemaGuidedProcessor.SchemaState = .inNumber(returnTo: returnTo)
        guard let next = SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: " ") else {
            Issue.record("Whitespace should end the number")
            return
        }
        if case .objectComma = next {} else {
            Issue.record("Expected .objectComma after ws-ending number; got \(next)")
        }
    }

    @Test(".inNumber + structural char re-processes in returnTo")
    func inNumberStructuralCharReprocesses() {
        var escape = false
        // Number in object value, model emits `1}` as separate tokens.
        // After `1` is consumed (.inNumber), `}` should close the object.
        let parent = SchemaNode(type: .object(properties: [:], required: []))
        let returnTo: SchemaGuidedProcessor.SchemaState = .objectComma(parent, [:], [], returnTo: .done)
        let state: SchemaGuidedProcessor.SchemaState = .inNumber(returnTo: returnTo)
        guard let next = SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "}") else {
            Issue.record("Structural '}' should complete number + close object")
            return
        }
        if case .done = next {} else {
            Issue.record("Expected .done after '}'; got \(next)")
        }
    }

    @Test(".inString close-quote transitions to returnTo (not .done for non-top-level)")
    func inStringReturnsToParentOnClose() {
        var escape = false
        let parent = SchemaNode(type: .object(properties: [:], required: []))
        let returnTo: SchemaGuidedProcessor.SchemaState = .objectComma(parent, [:], [], returnTo: .done)
        let state: SchemaGuidedProcessor.SchemaState = .inString(returnTo: returnTo)
        guard let next = SchemaGuidedProcessor.stepStrict(state: state, escapeNext: &escape, char: "\"") else {
            Issue.record("Close quote should be accepted")
            return
        }
        if case .objectComma = next {} else {
            Issue.record("Expected .objectComma after value-string close; got \(next)")
        }
    }

    // MARK: - End-to-end simulation

    @Test("E2E: garbage like ' thoughtful' from boolean is rejected at first divergence")
    func e2eGarbageRejected() {
        // The exact Gemma-4 regression: model sampled ' thoughtful' as one
        // token from .expectValue(boolean). The first non-ws char is 't' so
        // first-char masking admitted it. Walking the chars through the
        // strict FSM:
        //   ' ' → ws, no transition (state unchanged)
        //   't' → handleValueStartPure for boolean → .inLiteral("true", 1)
        //   'h' → .inLiteral("true", 1) expects 'r' → REJECT
        let schema: [String: Any] = ["type": "boolean"]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        let chars: [Character] = Array(" thoughtful")
        var sawReject = false
        for c in chars {
            if let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) {
                state = next
            } else {
                sawReject = true
                break
            }
        }
        #expect(sawReject, "Strict FSM must reject garbage token ' thoughtful' from boolean schema")
    }

    @Test("E2E: top-level boolean 'true' walks cleanly")
    func e2eTopLevelTrue() {
        let schema: [String: Any] = ["type": "boolean"]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        for c in "true" {
            guard let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) else {
                Issue.record("FSM rejected valid char '\(c)' at \(state)")
                return
            }
            state = next
        }
        if case .done = state {} else { Issue.record("Expected .done; got \(state)") }
    }

    @Test("E2E: top-level boolean 'false' walks cleanly")
    func e2eTopLevelFalse() {
        let schema: [String: Any] = ["type": "boolean"]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        for c in "false" {
            guard let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) else {
                Issue.record("FSM rejected valid char '\(c)' at \(state)")
                return
            }
            state = next
        }
        if case .done = state {} else { Issue.record("Expected .done; got \(state)") }
    }

    @Test("E2E: top-level number '42' walks cleanly")
    func e2eTopLevelNumber() {
        let schema: [String: Any] = ["type": "number"]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        for c in "42" {
            guard let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) else {
                Issue.record("FSM rejected valid char '\(c)' at \(state)")
                return
            }
            state = next
        }
        // .inNumber is the terminal state for top-level numbers (no explicit
        // close char). The runtime would emit EOS here.
        if case .inNumber = state {} else {
            Issue.record("Expected .inNumber after digits; got \(state)")
        }
    }

    @Test("Wrong literal start char from boolean is rejected immediately")
    func wrongLiteralStartRejected() {
        // From .expectValue(boolean), only 't' or 'f' should be admitted at
        // the value-start position. 'n' (would start `null`) is rejected.
        let schema: [String: Any] = ["type": "boolean"]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        let r = SchemaGuidedProcessor.stepStrict(
            state: state, escapeNext: &escape, char: "n"
        )
        #expect(r == nil, "'n' must be rejected when schema expects boolean")
        _ = state  // silence unused warning
    }

    // MARK: - Object E2E (previously blocked by the legacy `.objectKey` bug)

    @Test("E2E: object {\"x\":1} walks cleanly through the key/string/value/close states")
    func e2eObjectWithNumberValue() {
        // User-reported reproduction: json_schema with
        // {type:object, properties:{x:{type:number}}, required:[x]}.
        // Legacy FSM got stuck after `{"` — open quote was treated as the
        // close of an empty key, then `.objectColon` admitted only `:` /
        // whitespace and the model saturated on whitespace.
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["x": ["type": "number"]],
            "required": ["x"],
        ]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        for c in "{\"x\":1}" {
            guard let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) else {
                Issue.record("FSM rejected valid char '\(c)' at state \(state)")
                return
            }
            state = next
        }
        if case .done = state {} else {
            Issue.record("Expected .done after `{\"x\":1}`; got \(state)")
        }
    }

    @Test("E2E: object {\"name\":\"alice\",\"age\":30} walks cleanly")
    func e2eObjectWithMixedValueTypes() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"],
            ],
            "required": ["name", "age"],
        ]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        for c in "{\"name\":\"alice\",\"age\":30}" {
            guard let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) else {
                Issue.record("FSM rejected valid char '\(c)' at state \(state)")
                return
            }
            state = next
        }
        if case .done = state {} else {
            Issue.record("Expected .done after object walk; got \(state)")
        }
    }

    @Test("E2E: object with array value {\"items\":[1,2]} walks cleanly")
    func e2eObjectWithArrayValue() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "items": [
                    "type": "array",
                    "items": ["type": "integer"],
                ],
            ],
            "required": ["items"],
        ]
        let root = SchemaNode.parse(schema)
        var state: SchemaGuidedProcessor.SchemaState = .expectValue(root)
        var escape = false
        for c in "{\"items\":[1,2]}" {
            guard let next = SchemaGuidedProcessor.stepStrict(
                state: state, escapeNext: &escape, char: c
            ) else {
                Issue.record("FSM rejected valid char '\(c)' at state \(state)")
                return
            }
            state = next
        }
        if case .done = state {} else {
            Issue.record("Expected .done after object+array walk; got \(state)")
        }
    }

    // MARK: - Token-level admission (mask precompute path)
    //
    // Drives a real `SchemaGuidedProcessor` instance token-by-token through
    // the mask precompute path (`simulateTokenIsValid` — the exact function
    // the runtime `process(logits:)` uses to admit/reject tokens). This is
    // the layer the Qwen3.6-35B curl Case C exercises; the FSM-only tests
    // above are a layer below. If the FSM is correct AND the token-level
    // simulation agrees, then the masks emitted at runtime are correct.

    /// Synthetic tokenizer that decodes each id to a fixed string. Drives
    /// `simulateTokenIsValid` without requiring a real HF tokenizer.
    private func makeTokenizer(vocab: [String]) -> NovaMLXEngine.Tokenizer {
        NovaMLXEngine.Tokenizer(
            encode: { _ in [] },
            decode: { ids in
                guard let id = ids.first, id >= 0, id < vocab.count else { return "" }
                return vocab[id]
            },
            eosToken: nil,
            eosTokenId: nil
        )
    }

    @Test("Mask admission across {\"x\":1} — key/string/value/close at every step")
    func maskAdmissionForObjectWithNumber() {
        // Schema: {type:object, properties:{x:{type:number}}, required:[x]}
        // Walk the exact token sequence a compliant model would emit, and at
        // every step assert that the *next* token the model needs is
        // admitted while obviously-wrong tokens are rejected.
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["x": ["type": "number"]],
            "required": ["x"],
        ]
        let tokenizer = makeTokenizer(vocab: [
            "{", "\"", "x", ":", "1", "}", ",", " ", "true", "false",
        ])
        let proc = SchemaGuidedProcessor(schema: schema, tokenizer: tokenizer)

        // .expectValue — only `{` (and ws / EOS) admitted
        #expect(proc.simulateTokenIsValid(text: "{"), "Open object `{` must be admitted")
        #expect(!proc.simulateTokenIsValid(text: ":"), "`:` must be rejected before `{`")
        #expect(!proc.simulateTokenIsValid(text: "true"), "`true` must be rejected — schema expects object")

        // Consume `{` → state should now be .objectKey
        proc.didSample(token: MLXArray(0))  // token id 0 = "{"
        if case .objectKey = proc.stateForTesting {} else {
            Issue.record("After `{`, expected .objectKey; got \(proc.stateForTesting)")
        }

        // .objectKey — only `"` (open key string) and `}` (empty object)
        #expect(proc.simulateTokenIsValid(text: "\""), "Open-quote `\"` must be admitted from .objectKey")
        #expect(proc.simulateTokenIsValid(text: "}"), "Close object `}` must be admitted (empty object)")
        #expect(!proc.simulateTokenIsValid(text: "x"), "Bare key char must be rejected — JSON requires quoted keys")
        #expect(!proc.simulateTokenIsValid(text: ":"))

        // Consume `"` → state should now be .inObjectKey
        proc.didSample(token: MLXArray(1))  // token id 1 = "\""
        if case .inObjectKey = proc.stateForTesting {} else {
            Issue.record("After `\"`, expected .inObjectKey; got \(proc.stateForTesting)")
        }

        // .inObjectKey — accumulate key chars, close-quote transitions out.
        // Note: JSON strings can contain any printable char, so `:` is a
        // legal key-body character (e.g. `{"x:y": 1}`); it's only treated
        // as the colon terminator once the close-quote fires.
        #expect(proc.simulateTokenIsValid(text: "x"), "Key char `x` must be admitted in string body")
        #expect(proc.simulateTokenIsValid(text: ":"))
        #expect(!proc.simulateTokenIsValid(text: "\n"), "Raw newline rejected in string per RFC 8259")

        // Consume `x` → still .inObjectKey with accumulated key "x"
        proc.didSample(token: MLXArray(2))  // token id 2 = "x"
        if case .inObjectKey(_, _, _, let acc, _) = proc.stateForTesting {
            #expect(acc == "x", "Accumulated key must be 'x'; got '\(acc)'")
        } else {
            Issue.record("After `x`, expected .inObjectKey(_,_,_,\"x\",_); got \(proc.stateForTesting)")
        }

        // Close-quote → .objectColon with key="x"
        proc.didSample(token: MLXArray(1))  // "\""
        if case .objectColon(_, _, _, let keyName, _) = proc.stateForTesting {
            #expect(keyName == "x")
        } else {
            Issue.record("After close quote, expected .objectColon; got \(proc.stateForTesting)")
        }

        // .objectColon — only `:` (and ws) admitted
        #expect(proc.simulateTokenIsValid(text: ":"))
        #expect(!proc.simulateTokenIsValid(text: "{"))
        #expect(!proc.simulateTokenIsValid(text: "1"))

        // Consume `:` → .objectValue with vnode=.number
        proc.didSample(token: MLXArray(3))  // ":"
        if case .objectValue(_, _, _, let vnode, _) = proc.stateForTesting {
            if case .number = vnode.type {} else {
                Issue.record("Value schema must be .number; got \(vnode.type)")
            }
        } else {
            Issue.record("After `:`, expected .objectValue; got \(proc.stateForTesting)")
        }

        // .objectValue — start the number via digit
        #expect(proc.simulateTokenIsValid(text: "1"))
        #expect(!proc.simulateTokenIsValid(text: "true"))

        // Consume `1` → .inNumber with returnTo = .objectComma
        proc.didSample(token: MLXArray(4))  // "1"
        if case .inNumber(let returnTo) = proc.stateForTesting {
            if case .objectComma = returnTo {} else {
                Issue.record("Number's returnTo must be .objectComma; got \(returnTo)")
            }
        } else {
            Issue.record("After `1`, expected .inNumber; got \(proc.stateForTesting)")
        }

        // .inNumber — digit, ws, AND structural terminator `}` (via re-process)
        #expect(proc.simulateTokenIsValid(text: "1"), "More digits must be admitted")
        #expect(proc.simulateTokenIsValid(text: " "), "Whitespace must end the number")
        #expect(proc.simulateTokenIsValid(text: "}"), "`}` must complete number AND close object")

        // Consume `}` → should close both the number and the object → .done
        proc.didSample(token: MLXArray(5))  // "}"
        if case .done = proc.stateForTesting {} else {
            Issue.record("After `}`, expected .done; got \(proc.stateForTesting)")
        }
    }
}
