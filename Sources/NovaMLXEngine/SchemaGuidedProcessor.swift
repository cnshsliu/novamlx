import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public indirect enum SchemaNodeType: Sendable {
    case object(properties: [String: SchemaNode], required: Set<String>)
    case array(items: SchemaNode?)
    case stringEnum(values: [String])
    case string
    case integer
    case number
    case boolean
    case null
    case anyOf([SchemaNode])
    case anything
}

public struct SchemaNode: Sendable {
    public let type: SchemaNodeType

    public init(type: SchemaNodeType) {
        self.type = type
    }

    public static func parse(_ schema: [String: Any]) -> SchemaNode {
        if let anyOf = schema["anyOf"] as? [[String: Any]] {
            return SchemaNode(type: .anyOf(anyOf.map { .parse($0) }))
        }
        if let oneOf = schema["oneOf"] as? [[String: Any]] {
            return SchemaNode(type: .anyOf(oneOf.map { .parse($0) }))
        }
        if let allOf = schema["allOf"] as? [[String: Any]], let first = allOf.first {
            return .parse(first)
        }
        if let enumVals = schema["enum"] as? [String] {
            return SchemaNode(type: .stringEnum(values: enumVals))
        }

        switch schema["type"] as? String {
        case "object":
            let propsRaw = schema["properties"] as? [String: [String: Any]] ?? [:]
            var props: [String: SchemaNode] = [:]
            for (k, v) in propsRaw { props[k] = .parse(v) }
            let req = schema["required"] as? [String] ?? []
            return SchemaNode(type: .object(properties: props, required: Set(req)))
        case "array":
            let itemsSchema = schema["items"] as? [String: Any]
            return SchemaNode(type: .array(items: itemsSchema.map { .parse($0) }))
        case "string":
            return SchemaNode(type: .string)
        case "integer":
            return SchemaNode(type: .integer)
        case "number":
            return SchemaNode(type: .number)
        case "boolean":
            return SchemaNode(type: .boolean)
        case "null":
            return SchemaNode(type: .null)
        default:
            return SchemaNode(type: .anything)
        }
    }
}

public final class SchemaGuidedProcessor: LogitProcessor, @unchecked Sendable {
    private var state: SchemaState
    private let rootSchema: SchemaNode
    private let tokenizer: NovaMLXEngine.Tokenizer
    private let eosTokenIds: Set<Int>
    private let maskBuilder: TokenMaskBuilder
    private var escapeNext = false

    /// Parent-aware FSM states.
    ///
    /// Every object/array state, and every value-internal state
    /// (`.inString`, `.inNumber`, `.inLiteral`), carries a `returnTo`
    /// state — the state the FSM transitions to when the current value
    /// completes or the current structure closes. Without that outer
    /// context, value/structure completion collapsed to `.done`
    /// regardless of whether the value was an object field, an array
    /// element, or the top-level value, which made it impossible for the
    /// model to ever emit a structural terminator (``,``, `}`, `]`)
    /// after a value. The Qwen3.6-35B `json_schema` failure — `{"`
    /// followed by infinite whitespace — was the visible symptom:
    /// `.objectKey` treated the opening `"` as a *close* quote
    /// (transitioning to `.objectColon` with an empty key) and then
    /// `.objectColon` only admitted `:` or whitespace, so the model
    /// saturated on whitespace.
    ///
    /// `.inObjectKey` is dedicated to "inside the quoted key string", so
    /// the opening `"` from `.objectKey` properly enters the string body,
    /// accumulates characters, and only the matching close `"` transitions
    /// to `.objectColon` with the accumulated key.
    indirect enum SchemaState {
        case expectValue(SchemaNode)
        /// Object opened; expecting either `"` (to open a key string) or
        /// `}` (to close an empty object). `returnTo` is the outer state
        /// to resume when `}` closes this object.
        case objectKey(SchemaNode, [String: SchemaNode], Set<String>, returnTo: SchemaState)
        /// Inside a quoted key string — accumulating chars until close `"`
        /// transitions to `.objectColon`.
        case inObjectKey(SchemaNode, [String: SchemaNode], Set<String>, String, returnTo: SchemaState)
        case objectColon(SchemaNode, [String: SchemaNode], Set<String>, String, returnTo: SchemaState)
        case objectValue(SchemaNode, [String: SchemaNode], Set<String>, SchemaNode, returnTo: SchemaState)
        case objectComma(SchemaNode, [String: SchemaNode], Set<String>, returnTo: SchemaState)
        case arrayValue(SchemaNode, SchemaNode?, returnTo: SchemaState)
        case arrayComma(SchemaNode, SchemaNode?, returnTo: SchemaState)
        /// Inside a value string; on close `"` transitions to `returnTo`.
        case inString(returnTo: SchemaState)
        /// Inside a number; on termination (ws or structural char) transitions
        /// to `returnTo`.
        case inNumber(returnTo: SchemaState)
        /// Inside a multi-char literal (`true`/`false`/`null`); on completion
        /// transitions to `returnTo`.
        case inLiteral(String, Int, returnTo: SchemaState)
        case done
    }

    private static let ws: Set<Character> = [" ", "\n", "\r", "\t"]
    private static let digits: Set<Character> = Set("0123456789")
    private static func isWhitespace(_ c: Character) -> Bool { ws.contains(c) }

    public init(schema: [String: Any], tokenizer: NovaMLXEngine.Tokenizer) {
        self.rootSchema = SchemaNode.parse(schema)
        self.state = .expectValue(rootSchema)
        self.tokenizer = tokenizer
        self.eosTokenIds = tokenizer.eosTokenId.map { [$0] } ?? []
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    /// Caller-supplied builder for cache reuse.
    init(schema: [String: Any], tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder) {
        self.rootSchema = SchemaNode.parse(schema)
        self.state = .expectValue(rootSchema)
        self.tokenizer = tokenizer
        self.eosTokenIds = tokenizer.eosTokenId.map { [$0] } ?? []
        self.maskBuilder = sharedBuilder
    }

    /// Caller-supplied builder + full EOS set from model config.
    init(schema: [String: Any], tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder, allEosTokenIds: Set<Int>) {
        self.rootSchema = SchemaNode.parse(schema)
        self.state = .expectValue(rootSchema)
        self.tokenizer = tokenizer
        self.eosTokenIds = allEosTokenIds
        self.maskBuilder = sharedBuilder
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        if maskBuilder.vocabSize == 0 {
            maskBuilder.materialize(vocabSize: logits.shape[logits.shape.count - 1])
        }
        // Two-stage filter:
        //   1. First-char (cheap): drops most of vocab using the existing
        //      `allowedChars()` set against `tokenFirstChars`.
        //   2. Full-token simulation (strict): walks every char of each
        //      first-char-passing token through the strict pure stepper.
        //      This prevents the historical regression where multi-char
        //      tokens like ` thoughtful` would slip past first-char masking
        //      from `.expectValue` (Boolean schema) — char `t` looks like
        //      the start of `true`, but the rest of the token is garbage.
        //      `Self.stepStrict` validates each char against the FSM and
        //      rejects the token outright.
        let allowed = allowedChars()
        let eosAllowed: Bool
        if case .done = state {
            eosAllowed = true
        } else {
            eosAllowed = false
        }

        let vocabSize = maskBuilder.vocabSize
        var maskValues = [Bool](repeating: false, count: vocabSize)

        for i in 0..<vocabSize {
            if eosTokenIds.contains(i) {
                maskValues[i] = eosAllowed
                continue
            }
            let text = maskBuilder.decodedText(for: i)
            if text.isEmpty {
                // Empty/padding tokens — never sample them. Distinct from
                // whitespace-only tokens (those have a real decoded text).
                continue
            }
            // First-char fast filter
            let firstNonWS = text.first(where: { !$0.isWhitespace && $0 != "▁" && $0 != " " })
            if let fc = firstNonWS, !allowed.contains(fc) {
                continue
            }
            // Strict full-token simulation
            if simulateTokenIsValid(text: text) {
                maskValues[i] = true
            }
        }

        let mask = MLXArray(maskValues)[.newAxis, 0...]
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    public func didSample(token: MLXArray) {
        let tokenId = token.item(Int.self)
        if eosTokenIds.contains(tokenId) { return }
        let text = tokenizer.decode([tokenId])

        // Walk the runtime FSM through every char. We use the strict stepper
        // here too (not the legacy `advance`) so runtime and mask precompute
        // stay in sync. On any violation — which should be impossible if the
        // mask was applied correctly — clamp to `.done` to force EOS on the
        // next process call. This is the same defensive pattern used by
        // `JSONLogitProcessor.didSample`.
        var simState = state
        var simEscape = escapeNext
        for char in text {
            guard let next = Self.stepStrict(state: simState, escapeNext: &simEscape, char: char) else {
                state = .done
                escapeNext = false
                return
            }
            simState = next
        }
        state = simState
        escapeNext = simEscape
    }

    /// Walk every char of `text` through the strict stepper from the current
    /// `state` / `escapeNext`. Returns `false` on any grammar violation.
    internal func simulateTokenIsValid(text: String) -> Bool {
        var simState = state
        var simEscape = escapeNext
        for c in text {
            guard let next = Self.stepStrict(state: simState, escapeNext: &simEscape, char: c) else {
                return false
            }
            simState = next
        }
        return true
    }

    /// Test hook — read-only access to the current FSM state.
    internal var stateForTesting: SchemaState { state }

    private func allowedChars() -> Set<Character> {
        let ws = Self.ws
        switch state {
        case .expectValue(let node): return startChars(for: node).union(ws)
        case .objectKey: return Set<Character>(["\"", "}"]).union(ws)
        case .inObjectKey: return printableChars().union(Set<Character>(["\""]))
        case .objectColon: return Set<Character>([":"]).union(ws)
        case .objectValue(_, _, _, let vnode, _): return startChars(for: vnode).union(ws)
        case .objectComma: return Set<Character>([",", "}"]).union(ws)
        case .arrayValue(_, let item, _): return startChars(for: item ?? SchemaNode(type: .anything)).union(ws).union(Set<Character>(["]"]))
        case .arrayComma: return Set<Character>([",", "]"]).union(ws)
        case .inString: return printableChars().union(Set<Character>(["\""]))
        case .inNumber(let returnTo):
            // Admit digit-shaped chars plus anything the `returnTo` state
            // would accept (so a single token like `1}` or `1,` can complete
            // the number AND transition through the post-value state).
            return Self.digits.union(Set<Character>([".", "e", "E", "+", "-"]))
                .union(ws)
                .union(allowedCharsForTerminators(of: returnTo))
        case .inLiteral(let lit, let idx, let returnTo):
            let i = lit.index(lit.startIndex, offsetBy: min(idx, lit.count))
            if i < lit.endIndex { return Set<Character>([lit[i]]) }
            return Set<Character>([",", "}", "]"]).union(ws)
                .union(allowedCharsForTerminators(of: returnTo))
        case .done: return ws
        }
    }

    /// Compute the set of structural terminator chars (``,``, `}`, `]`)
    /// admissible from `state`. Used by `.inNumber`/`.inLiteral` masks so a
    /// structural terminator token can complete the value AND close the
    /// surrounding structure in one step.
    private func allowedCharsForTerminators(of state: SchemaState) -> Set<Character> {
        switch state {
        case .objectComma: return [",", "}"]
        case .arrayComma: return [",", "]"]
        case .done: return []
        default: return []
        }
    }

    private func startChars(for node: SchemaNode) -> Set<Character> {
        let d = Self.digits
        switch node.type {
        case .object: return Set<Character>(["{"])
        case .array: return Set<Character>(["["])
        case .string, .stringEnum: return Set<Character>(["\""])
        case .integer, .number: return Set<Character>(["-"]).union(d)
        case .boolean: return Set<Character>(["t", "f"])
        case .null: return Set<Character>(["n"])
        case .anything: return Set<Character>(["{", "[", "\"", "-", "t", "f", "n"]).union(d)
        case .anyOf(let nodes): return nodes.reduce(into: Set<Character>()) { $0.formUnion(startChars(for: $1)) }
        }
    }

    private func printableChars() -> Set<Character> {
        (32...126).reduce(into: Set<Character>()) { $0.insert(Character(UnicodeScalar($1)!)) }
    }

    // MARK: - Strict pure stepper
    //
    // Replaces the historical `advance(_:)` instance method which mutated
    // `self.state` in place AND was permissive (invalid chars silently
    // produced no transition, relying on the first-char mask to keep them
    // out). With multi-char tokens, that permissiveness was the source of
    // the Gemma-4 ` thoughtful` regression — `.inLiteral` advanced its index
    // on ANY char without validating against `lit[idx]`.
    //
    // `stepStrict` is pure (returns next state or `nil`) and validates every
    // transition. It is used by:
    //   1. Mask precompute (`simulateTokenIsValid`) — rejects multi-char
    //      tokens whose mid-token chars violate the FSM.
    //   2. `didSample` — walks the sampled token char-by-char, clamping to
    //      `.done` on any violation as a defensive safety net.
    //
    // Returning `nil` means "this character is a grammar violation in the
    // current state". Callers either reject the token (precompute) or clamp
    // the runtime state (didSample).
    static func stepStrict(
        state: SchemaState, escapeNext: inout Bool, char c: Character
    ) -> SchemaState? {
        switch state {

        case .expectValue(let node):
            if isWhitespace(c) { return state }
            return handleValueStartPure(c, node: node, returnTo: .done)

        case .objectKey(let parent, let props, let req, let returnTo):
            if isWhitespace(c) { return state }
            if c == "\"" {
                // Opening quote of the key string. Enter the string body,
                // preserving the outer returnTo through every inner state.
                return .inObjectKey(parent, props, req, "", returnTo: returnTo)
            }
            if c == "}" {
                // Empty object — close and resume the outer context.
                return returnTo
            }
            return nil

        case .inObjectKey(let parent, let props, let req, let accumulatedKey, let returnTo):
            if escapeNext {
                escapeNext = false
                return .inObjectKey(parent, props, req, accumulatedKey + String(c), returnTo: returnTo)
            }
            if c == "\\" {
                escapeNext = true
                return state
            }
            if c == "\"" {
                // Close quote — transition to colon with the accumulated key.
                return .objectColon(parent, props, req, accumulatedKey, returnTo: returnTo)
            }
            // RFC 8259 §7: unescaped control chars (< 0x20) are rejected
            // inside strings.
            if let ascii = c.asciiValue, ascii < 0x20 { return nil }
            return .inObjectKey(parent, props, req, accumulatedKey + String(c), returnTo: returnTo)

        case .objectColon(let parent, let props, let req, let keyName, let returnTo):
            if isWhitespace(c) { return state }
            if c == ":" {
                let valueSchema = props[keyName] ?? SchemaNode(type: .anything)
                return .objectValue(parent, props, req, valueSchema, returnTo: returnTo)
            }
            return nil

        case .objectValue(let parent, let props, let req, let vnode, let returnTo):
            if isWhitespace(c) { return state }
            // When the value completes, we transition to .objectComma which
            // itself returns to `returnTo` once `}` closes the object.
            let innerReturnTo = SchemaState.objectComma(parent, props, req, returnTo: returnTo)
            return handleValueStartPure(c, node: vnode, returnTo: innerReturnTo)

        case .objectComma(let parent, let props, let req, let returnTo):
            if isWhitespace(c) { return state }
            if c == "," { return .objectKey(parent, props, req, returnTo: returnTo) }
            if c == "}" { return returnTo }
            return nil

        case .arrayValue(let parent, let item, let returnTo):
            if isWhitespace(c) { return state }
            if c == "]" { return returnTo }
            let innerReturnTo = SchemaState.arrayComma(parent, item, returnTo: returnTo)
            return handleValueStartPure(c, node: item ?? SchemaNode(type: .anything), returnTo: innerReturnTo)

        case .arrayComma(let parent, let item, let returnTo):
            if isWhitespace(c) { return state }
            if c == "," { return .arrayValue(parent, item, returnTo: returnTo) }
            if c == "]" { return returnTo }
            return nil

        case .inString(let returnTo):
            if escapeNext {
                escapeNext = false
                // RFC 8259 §7: escapes are limited to "\/, \\, \", \b, \f, \n, \r, \t, \uXXXX
                // For pragmatic compatibility with model outputs we accept any
                // char after `\` here (including \uXXXX hex which would
                // otherwise need a 4-char counter).
                return state
            }
            if c == "\\" {
                escapeNext = true
                return state
            }
            if c == "\"" {
                // Close quote — return to the post-value state.
                return returnTo
            }
            if let ascii = c.asciiValue, ascii < 0x20 { return nil }
            return state

        case .inNumber(let returnTo):
            if Self.digits.contains(c) || c == "." || c == "e" || c == "E" || c == "+" || c == "-" {
                return state
            }
            if isWhitespace(c) { return returnTo }
            // Structuralchar (`,`, `}`, `]`): finish the number AND
            // re-process the char in the post-value state, exactly mirroring
            // `JSONLogitProcessor.step`'s re-process trick. This lets a
            // single token like `1}` or `1,` complete the number AND close
            // the object / advance to the next key.
            return stepStrict(state: returnTo, escapeNext: &escapeNext, char: c)

        case .inLiteral(let lit, let idx, let returnTo):
            // STRICT validation — original `advance` advanced the index for
            // ANY char, which let ` thoughtful` (after first-char `t`)
            // breeze through 8 unrelated chars. Now we require the char to
            // match the literal at the current position.
            let i = lit.index(lit.startIndex, offsetBy: min(idx, lit.count))
            guard i < lit.endIndex else { return nil }
            guard c == lit[i] else { return nil }
            if idx + 1 >= lit.count {
                // Literal completed. Hand control to `returnTo`.
                return returnTo
            }
            return .inLiteral(lit, idx + 1, returnTo: returnTo)

        case .done:
            // Trailing whitespace tolerated, anything else is a violation.
            if isWhitespace(c) { return state }
            return nil
        }
    }

    /// Pure version of `handleValueStart`. Returns the entered state on a
    /// valid value-start char, or `nil` if no transition matches. `returnTo`
    /// is the state the FSM should transition to once the value completes —
    /// plumbed through `.inString`/`.inNumber`/`.inLiteral` so they can find
    /// their way back to the correct post-value state, and through
    /// `.objectKey`/`.arrayValue` so nested structures resume the right
    /// outer context when they close.
    private static func handleValueStartPure(
        _ c: Character, node: SchemaNode, returnTo: SchemaState
    ) -> SchemaState? {
        switch node.type {
        case .object(let props, let req):
            if c == "{" { return .objectKey(node, props, req, returnTo: returnTo) }
            return nil
        case .array(let items):
            if c == "[" { return .arrayValue(node, items, returnTo: returnTo) }
            return nil
        case .string, .stringEnum:
            if c == "\"" { return .inString(returnTo: returnTo) }
            return nil
        case .integer, .number:
            if c == "-" || Self.digits.contains(c) { return .inNumber(returnTo: returnTo) }
            return nil
        case .boolean:
            if c == "t" { return .inLiteral("true", 1, returnTo: returnTo) }
            if c == "f" { return .inLiteral("false", 1, returnTo: returnTo) }
            return nil
        case .null:
            if c == "n" { return .inLiteral("null", 1, returnTo: returnTo) }
            return nil
        case .anything:
            if c == "{" { return .objectKey(node, [:], [], returnTo: returnTo) }
            if c == "[" { return .arrayValue(node, nil, returnTo: returnTo) }
            if c == "\"" { return .inString(returnTo: returnTo) }
            if c == "-" || Self.digits.contains(c) { return .inNumber(returnTo: returnTo) }
            if c == "t" { return .inLiteral("true", 1, returnTo: returnTo) }
            if c == "f" { return .inLiteral("false", 1, returnTo: returnTo) }
            if c == "n" { return .inLiteral("null", 1, returnTo: returnTo) }
            return nil
        case .anyOf(let nodes):
            // Try each branch; first that admits the char wins.
            for n in nodes {
                if let entered = handleValueStartPure(c, node: n, returnTo: returnTo) {
                    return entered
                }
            }
            return nil
        }
    }
}
