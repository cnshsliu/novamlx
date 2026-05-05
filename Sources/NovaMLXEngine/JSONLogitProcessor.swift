import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

struct SendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

final class MutableSendableBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

/// Strict-FSM JSON logit processor.
///
/// **Why strict:** The historical implementation precomputed per-state masks
/// keyed on the *first non-whitespace character* of each candidate token (via
/// `TokenMaskBuilder.firstChars`). That approximation accepted any token whose
/// first letter happened to be a valid value-start (`t`/`f`/`n` for literals,
/// digits for numbers) without validating the rest of the token. The
/// pathological failure on Gemma-4-26B was a 2-token output ` thoughtful` →
/// ` diesel` followed by EOS:
///
///   1. ` thoughtful` passes the `.expectValue` mask because its first non-ws
///      char `t` is in the value-start set (looks like the start of `true`).
///   2. The runtime FSM walks each char of ` thoughtful`: ` ` is silently
///      ignored in `.expectValue`; `t` transitions to `.inLiteral`; `houghtful`
///      stays in `.inLiteral` (any letter accepted — the old FSM didn't
///      validate that the accumulated text spelled a real literal).
///   3. ` diesel` then passes the `.inLiteral` mask because its first non-ws
///      char `d` is a letter. Walking it: ` ` is not a letter → `finishValue()`
///      → `.done`; `diesel` no-ops in `.done`.
///   4. Mask at `.done` allows only EOS → generation stops.
///
/// **The fix (Tier 2):**
///   - Replace the single `.inLiteral` state with explicit literal sub-states
///     (`.inLiteralT`, `.inLiteralTr`, …) so partial-literal progress is
///     tracked and only the next valid letter is accepted.
///   - Replace first-char masking with **full-token FSM simulation** at
///     materialization: for each `(state, tokenId)` pair we walk every char
///     of the token's decoded text through the strict FSM and accept the
///     token only if no transition violates the grammar. Rejected: tokens
///     with embedded whitespace that would prematurely call `finishValue()`,
///     tokens whose letters don't continue an in-progress literal, etc.
///
/// **Tier 3 — parent-frame-aware masks:**
///   - The Tier-2 implementation collapsed every "depth-could-be-anything"
///     state (`.inNumber`, `.inLiteral*`, `.expectValue`,
///     `.expectValueAfterColon`, `.inString`) into a single mask whose
///     simulation context had `depthAmbiguous = true`. That conservatism
///     forbade *every* structural terminator (`,`/`}`/`]`) from those
///     states' masks, so the only way out of `.inNumber` was a
///     whitespace-leading token. On models whose BPE rarely emits a bare
///     space token (e.g. Qwen3.6-27B), the model was forced to keep
///     extending the number, producing pathologies like
///     `30.000000000000004440892...`.
///   - We now precompute per-`(state, parentFrame)` masks where
///     `parentFrame ∈ {topLevel, object, array}` is `depthStack.last`.
///     At runtime we look up the correct mask for the current depth top.
///     `,` is admitted from `.inNumber` exactly when the parent is an
///     object or array; `}` only when parent is object; `]` only when
///     parent is array. Multi-frame-pop tokens (e.g. `]}`) remain
///     conservatively rejected at the precompute layer because they require
///     knowing two stack frames; the runtime FSM still validates them via
///     `step()` on actual depth, and a violation clamps to `.done`.
///   - Memory cost: ~3× the previous mask table. For a 250k-vocab model
///     this is ~16 MB total — negligible relative to model weights.
public final class JSONLogitProcessor: LogitProcessor, @unchecked Sendable {
    enum JSONState: Hashable {
        case expectValue
        case expectObjectKeyOrEnd
        case inString
        case afterKeyQuote
        case expectColon
        case expectValueAfterColon
        case expectObjectCommaOrEnd
        case expectArrayValueOrEnd
        case expectArrayCommaOrEnd
        case inNumber
        // Literal sub-states — explicit prefix tracking replaces the old single
        // `.inLiteral` catch-all. Each sub-state's mask only admits the next
        // valid letter (and at the terminal sub-state, the closing letter
        // followed by `finishValue`).
        case inLiteralT, inLiteralTr, inLiteralTru
        case inLiteralF, inLiteralFa, inLiteralFal, inLiteralFals
        case inLiteralN, inLiteralNu, inLiteralNul
        case done
    }

    /// Snapshot used by both runtime stepping and materialization-time
    /// simulation. With Tier-3 parent-frame-aware masks, depth is always
    /// concrete (the simulation seeds the stack with the assumed parent
    /// frame), so there is no longer an "ambiguous-depth" flag.
    private struct SimContext {
        var state: JSONState
        var depthStack: [Bool]   // true = object frame, false = array frame
        var escapeNext: Bool
        var stringIsKey: Bool
    }

    /// Immediate parent frame on the depth stack — used as a precompute key so
    /// that termination tokens (`,`/`}`/`]`) can be admitted exactly when they
    /// are grammatically valid for the current context.
    enum ParentFrame: Hashable {
        case topLevel
        case object
        case array
    }

    /// Precomputed mask key. We materialize one mask per `(state, parentFrame)`
    /// combination. Most combinations are reachable; the unreachable ones
    /// (e.g. `.expectObjectKeyOrEnd` with `parentFrame == .topLevel`) are still
    /// computed but never looked up at runtime.
    struct MaskKey: Hashable {
        let state: JSONState
        let parentFrame: ParentFrame
    }

    private var state: JSONState = .expectValue
    private var depthStack: [Bool] = []
    private var escapeNext = false
    private var stringIsKey = false
    private let tokenizer: NovaMLXEngine.Tokenizer
    private let eosTokenIds: Set<Int>
    private var vocabSize: Int
    private var precomputedMasks: [MaskKey: MLXArray]
    private var firstToken = true
    private let maskBuilder: TokenMaskBuilder

    public init(tokenizer: NovaMLXEngine.Tokenizer) {
        self.tokenizer = tokenizer
        self.eosTokenIds = tokenizer.eosTokenId.map { [$0] } ?? []
        self.precomputedMasks = [:]
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
        self.vocabSize = 0
    }

    /// Eager constructor (mainly for tests / callers with a known vocab size).
    init(tokenizer: NovaMLXEngine.Tokenizer, vocabSize: Int) {
        self.tokenizer = tokenizer
        self.eosTokenIds = tokenizer.eosTokenId.map { [$0] } ?? []
        self.vocabSize = vocabSize
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer, vocabSize: vocabSize)
        self.precomputedMasks = [:]
        self.precomputeMasks()
    }

    /// Caller-supplied builder (used by the engine's TokenMaskBuilder cache so
    /// per-token vocabulary tables aren't rebuilt for every request).
    init(tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder) {
        self.tokenizer = tokenizer
        self.eosTokenIds = tokenizer.eosTokenId.map { [$0] } ?? []
        self.maskBuilder = sharedBuilder
        self.vocabSize = sharedBuilder.vocabSize
        self.precomputedMasks = [:]
        if sharedBuilder.vocabSize > 0 {
            self.precomputeMasks()
        }
    }

    /// Caller-supplied builder + full EOS set (from model config, which may list
    /// multiple EOS IDs — e.g. Qwen3.6 has [248046, 248044]).
    init(tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder, allEosTokenIds: Set<Int>) {
        self.tokenizer = tokenizer
        self.eosTokenIds = allEosTokenIds
        self.maskBuilder = sharedBuilder
        self.vocabSize = sharedBuilder.vocabSize
        self.precomputedMasks = [:]
        if sharedBuilder.vocabSize > 0 {
            self.precomputeMasks()
        }
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        // Lazy materialization: the first time we see logits we know the
        // authoritative vocab dim (ground truth, regardless of what
        // tokenizer/config claim).
        if vocabSize == 0 {
            let dim = logits.shape[logits.shape.count - 1]
            maskBuilder.materialize(vocabSize: dim)
            vocabSize = maskBuilder.vocabSize
            precomputeMasks()
        }
        let currentState = firstToken ? JSONState.expectValue : state
        let key = MaskKey(state: currentState, parentFrame: currentParentFrame())
        if let mask = precomputedMasks[key] {
            return TokenMaskBuilder.applyMask(mask, to: logits)
        }
        // Fall back to the topLevel mask if the (state, parent) combination is
        // unreachable in practice — preserves prior behavior rather than
        // returning unmasked logits, which would let arbitrary tokens through.
        if let mask = precomputedMasks[MaskKey(state: currentState, parentFrame: .topLevel)] {
            return TokenMaskBuilder.applyMask(mask, to: logits)
        }
        return logits
    }

    /// `parentFrame` derived from the top of `depthStack`. At init or after
    /// the value tree fully closes the stack is empty → `.topLevel`.
    private func currentParentFrame() -> ParentFrame {
        guard let top = depthStack.last else { return .topLevel }
        return top ? .object : .array
    }

    public func didSample(token: MLXArray) {
        firstToken = false
        let tokenId = token.item(Int.self)
        if eosTokenIds.contains(tokenId) { return }

        let text = tokenizer.decode([tokenId])

        // Walk the runtime FSM through every char of the sampled token. With
        // strict-mask precomputation, every step should succeed — but defend
        // against vocab-dim mismatches (mask short of logits → unconstrained
        // tail tokens leak through) by clamping to `.done` on any violation,
        // which forces EOS on the next `process(logits:)` call.
        var ctx = SimContext(
            state: state,
            depthStack: depthStack,
            escapeNext: escapeNext,
            stringIsKey: stringIsKey
        )
        for c in text {
            guard let next = Self.step(ctx, c) else {
                ctx.state = .done
                ctx.depthStack.removeAll()
                break
            }
            ctx = next
        }
        state = ctx.state
        depthStack = ctx.depthStack
        escapeNext = ctx.escapeNext
        stringIsKey = ctx.stringIsKey
    }

    // MARK: - Strict FSM core (used by both runtime stepping and materialization)

    /// Process a single character. Returns `nil` if the char is a grammar
    /// violation in the given context.
    private static func step(_ ctx: SimContext, _ c: Character) -> SimContext? {
        var ctx = ctx

        // Whitespace handling — most non-content states tolerate whitespace
        // without state change. Inside strings whitespace is content. Inside
        // numbers it ends the number. Inside literal sub-states it's a hard
        // violation (`tr ue` is not valid JSON).
        if isWhitespace(c) {
            switch ctx.state {
            case .inString:
                return processInStringChar(&ctx, c)
            case .inNumber:
                finishValue(&ctx)
                return ctx
            case .inLiteralT, .inLiteralTr, .inLiteralTru,
                 .inLiteralF, .inLiteralFa, .inLiteralFal, .inLiteralFals,
                 .inLiteralN, .inLiteralNu, .inLiteralNul:
                return nil
            case .done:
                return ctx
            default:
                return ctx
            }
        }

        switch ctx.state {
        case .expectValue, .expectValueAfterColon:
            return processValueStartChar(&ctx, c)

        case .expectArrayValueOrEnd:
            if c == "]" {
                guard !ctx.depthStack.isEmpty else { return nil }
                ctx.depthStack.removeLast()
                finishValue(&ctx)
                return ctx
            }
            return processValueStartChar(&ctx, c)

        case .expectObjectKeyOrEnd:
            if c == "\"" {
                ctx.stringIsKey = true
                ctx.state = .inString
                return ctx
            }
            if c == "}" {
                guard !ctx.depthStack.isEmpty else { return nil }
                ctx.depthStack.removeLast()
                finishValue(&ctx)
                return ctx
            }
            return nil

        case .inString:
            return processInStringChar(&ctx, c)

        case .afterKeyQuote:
            if c == ":" { ctx.state = .expectValueAfterColon; return ctx }
            return nil

        case .expectColon:
            if c == ":" { ctx.state = .expectValueAfterColon; return ctx }
            return nil

        case .expectObjectCommaOrEnd:
            if c == "," { ctx.state = .expectObjectKeyOrEnd; return ctx }
            if c == "}" {
                guard !ctx.depthStack.isEmpty else { return nil }
                ctx.depthStack.removeLast()
                finishValue(&ctx)
                return ctx
            }
            return nil

        case .expectArrayCommaOrEnd:
            if c == "," { ctx.state = .expectArrayValueOrEnd; return ctx }
            if c == "]" {
                guard !ctx.depthStack.isEmpty else { return nil }
                ctx.depthStack.removeLast()
                finishValue(&ctx)
                return ctx
            }
            return nil

        case .inNumber:
            if c.isNumber || c == "." || c == "e" || c == "E" || c == "+" || c == "-" {
                return ctx
            }
            finishValue(&ctx)
            return step(ctx, c)  // re-process in new state

        // Literal sub-states — strict prefix validation. Only the exact next
        // letter of "true" / "false" / "null" is admitted; anything else is a
        // violation. Closing letter triggers `finishValue` immediately.
        case .inLiteralT:
            if c == "r" { ctx.state = .inLiteralTr; return ctx }
            return nil
        case .inLiteralTr:
            if c == "u" { ctx.state = .inLiteralTru; return ctx }
            return nil
        case .inLiteralTru:
            if c == "e" { finishValue(&ctx); return ctx }
            return nil
        case .inLiteralF:
            if c == "a" { ctx.state = .inLiteralFa; return ctx }
            return nil
        case .inLiteralFa:
            if c == "l" { ctx.state = .inLiteralFal; return ctx }
            return nil
        case .inLiteralFal:
            if c == "s" { ctx.state = .inLiteralFals; return ctx }
            return nil
        case .inLiteralFals:
            if c == "e" { finishValue(&ctx); return ctx }
            return nil
        case .inLiteralN:
            if c == "u" { ctx.state = .inLiteralNu; return ctx }
            return nil
        case .inLiteralNu:
            if c == "l" { ctx.state = .inLiteralNul; return ctx }
            return nil
        case .inLiteralNul:
            if c == "l" { finishValue(&ctx); return ctx }
            return nil

        case .done:
            // Already balanced — anything except whitespace is a violation.
            return nil
        }
    }

    private static func processValueStartChar(_ ctx: inout SimContext, _ c: Character) -> SimContext? {
        switch c {
        case "{":
            ctx.depthStack.append(true)
            ctx.state = .expectObjectKeyOrEnd
            return ctx
        case "[":
            ctx.depthStack.append(false)
            ctx.state = .expectArrayValueOrEnd
            return ctx
        case "\"":
            ctx.stringIsKey = false
            ctx.state = .inString
            return ctx
        case "-":
            ctx.state = .inNumber
            return ctx
        case "t":
            ctx.state = .inLiteralT
            return ctx
        case "f":
            ctx.state = .inLiteralF
            return ctx
        case "n":
            ctx.state = .inLiteralN
            return ctx
        default:
            if c.isNumber {
                ctx.state = .inNumber
                return ctx
            }
            return nil
        }
    }

    private static func processInStringChar(_ ctx: inout SimContext, _ c: Character) -> SimContext? {
        if ctx.escapeNext {
            ctx.escapeNext = false
            return ctx
        }
        if c == "\\" {
            ctx.escapeNext = true
            return ctx
        }
        if c == "\"" {
            if ctx.stringIsKey && ctx.depthStack.last == true {
                ctx.state = .afterKeyQuote
            } else {
                finishValue(&ctx)
            }
            return ctx
        }
        // Reject ASCII control chars (< 0x20) per RFC 8259 §7.
        if let ascii = c.asciiValue, ascii < 0x20 {
            return nil
        }
        return ctx
    }

    private static func finishValue(_ ctx: inout SimContext) {
        if ctx.depthStack.isEmpty {
            ctx.state = .done
        } else {
            ctx.state = ctx.depthStack.last! ? .expectObjectCommaOrEnd : .expectArrayCommaOrEnd
        }
    }

    private static func isWhitespace(_ c: Character) -> Bool {
        // Match the FSM's char-level whitespace set: ASCII WS only. Unicode
        // whitespace is treated as content (string-internal) or violation
        // (structural states), matching JSON spec.
        return c == " " || c == "\n" || c == "\r" || c == "\t"
    }

    // MARK: - Materialization

    private static let allStates: [JSONState] = [
        .expectValue, .expectObjectKeyOrEnd, .inString, .afterKeyQuote,
        .expectColon, .expectValueAfterColon, .expectObjectCommaOrEnd,
        .expectArrayValueOrEnd, .expectArrayCommaOrEnd, .inNumber,
        .inLiteralT, .inLiteralTr, .inLiteralTru,
        .inLiteralF, .inLiteralFa, .inLiteralFal, .inLiteralFals,
        .inLiteralN, .inLiteralNu, .inLiteralNul,
        .done
    ]

    private static let allParentFrames: [ParentFrame] = [.topLevel, .object, .array]

    private func precomputeMasks() {
        for s in Self.allStates {
            for pf in Self.allParentFrames {
                guard let initialCtx = makeInitialSimContext(forState: s, parentFrame: pf) else {
                    // (state, parentFrame) combination is structurally
                    // impossible (e.g. `.expectObjectKeyOrEnd` with no
                    // enclosing object). Skip — runtime will never query it.
                    continue
                }
                var maskValues = [Bool](repeating: false, count: vocabSize)

                for i in 0..<vocabSize {
                    if eosTokenIds.contains(i) {
                        // EOS may only be sampled once the JSON tree is fully
                        // balanced. Mid-structure EOS produces truncated
                        // output. `.done` is reached by popping all frames,
                        // so `parentFrame == .topLevel` is the only case
                        // where EOS can fire.
                        maskValues[i] = (s == .done && pf == .topLevel)
                        continue
                    }
                    let text = maskBuilder.decodedText(for: i)
                    if text.isEmpty {
                        // Padding/reserved/special tokens whose decoded form
                        // is empty (e.g. ids beyond the tokenizer's actual
                        // vocab in models that pad to a power-of-two
                        // embedding dim like Qwen3.6: vocab_size=248320 but
                        // only 248077 real tokens). Sampling them has
                        // undefined behavior — never allow.
                        continue
                    }
                    if simulateTokenIsValid(text: text, from: initialCtx) {
                        maskValues[i] = true
                    }
                }
                precomputedMasks[MaskKey(state: s, parentFrame: pf)] = MLXArray(maskValues)[.newAxis, 0...]
            }
        }
    }

    /// Seed a simulation context for `(state, parentFrame)`. The depth stack
    /// is populated to match the assumed parent frame so that termination
    /// tokens like `,`/`}`/`]` resolve correctly during materialization.
    /// Returns `nil` for combinations that cannot occur at runtime — those
    /// masks are never built. The runtime never queries them either, because
    /// each state's reachable parent frame is fully determined by the FSM
    /// transitions (e.g. `.expectObjectKeyOrEnd` only follows `{` which
    /// pushes a `[true]` frame).
    private func makeInitialSimContext(forState s: JSONState, parentFrame: ParentFrame) -> SimContext? {
        // Build the assumed depth stack for `parentFrame`.
        let baseStack: [Bool]
        switch parentFrame {
        case .topLevel: baseStack = []
        case .object:   baseStack = [true]
        case .array:    baseStack = [false]
        }

        switch s {
        // Object-only states. Reached only after `{` pushed a [true] frame.
        case .expectObjectKeyOrEnd, .expectObjectCommaOrEnd, .afterKeyQuote, .expectColon:
            guard parentFrame == .object else { return nil }
            let isKey = (s == .afterKeyQuote || s == .expectColon)
            return SimContext(state: s, depthStack: [true],
                              escapeNext: false, stringIsKey: isKey)

        // Array-only states. Reached only after `[` pushed a [false] frame.
        case .expectArrayValueOrEnd, .expectArrayCommaOrEnd:
            guard parentFrame == .array else { return nil }
            return SimContext(state: s, depthStack: [false],
                              escapeNext: false, stringIsKey: false)

        // Value-after-colon only occurs inside an object key/value pair.
        case .expectValueAfterColon:
            guard parentFrame == .object else { return nil }
            return SimContext(state: s, depthStack: [true],
                              escapeNext: false, stringIsKey: false)

        // Initial state — only reachable at top level.
        case .expectValue:
            guard parentFrame == .topLevel else { return nil }
            return SimContext(state: s, depthStack: [],
                              escapeNext: false, stringIsKey: false)

        // `.done` is reached only after the outermost value closes — the stack
        // is always empty there, so parent is `.topLevel`.
        case .done:
            guard parentFrame == .topLevel else { return nil }
            return SimContext(state: s, depthStack: [],
                              escapeNext: false, stringIsKey: false)

        // Genuinely multi-context states: in-flight numeric / literal /
        // string content can occur as a top-level value, an object value, or
        // an array element. We build a mask for each.
        case .inNumber,
             .inLiteralT, .inLiteralTr, .inLiteralTru,
             .inLiteralF, .inLiteralFa, .inLiteralFal, .inLiteralFals,
             .inLiteralN, .inLiteralNu, .inLiteralNul,
             .inString:
            return SimContext(state: s, depthStack: baseStack,
                              escapeNext: false, stringIsKey: false)
        }
    }

    /// Walk every char of `text` through the strict FSM. Returns `false` if
    /// any char produces a violation. With parent-frame-aware seeding, depth
    /// is concrete throughout simulation — but tokens that pop *more* frames
    /// than the seed contains are still rejected (they require knowing an
    /// outer parent frame we deliberately did not supply).
    private func simulateTokenIsValid(text: String, from initial: SimContext) -> Bool {
        var ctx = initial
        for c in text {
            guard let next = Self.step(ctx, c) else { return false }
            ctx = next
        }
        return true
    }

    // MARK: - Test hooks (internal — for unit tests via @testable import)

    /// Read-only view of the precomputed mask table. Kept internal so the
    /// public surface area is unchanged; consumers should never mutate the
    /// map.
    var precomputedMasksForTesting: [MaskKey: MLXArray] { precomputedMasks }

    /// Drive a single FSM transition. Returns the next-state tuple or nil if
    /// the character is a grammar violation.
    static func stepForTesting(
        state: JSONState, depthStack: [Bool], escapeNext: Bool, stringIsKey: Bool,
        char: Character
    ) -> (state: JSONState, depthStack: [Bool], escapeNext: Bool, stringIsKey: Bool)? {
        let ctx = SimContext(state: state, depthStack: depthStack,
                             escapeNext: escapeNext, stringIsKey: stringIsKey)
        guard let next = step(ctx, char) else { return nil }
        return (next.state, next.depthStack, next.escapeNext, next.stringIsKey)
    }
}
