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

    enum SchemaState {
        case expectValue(SchemaNode)
        case objectKey(SchemaNode, [String: SchemaNode], Set<String>, String)
        case objectColon(SchemaNode, [String: SchemaNode], Set<String>, String)
        case objectValue(SchemaNode, [String: SchemaNode], Set<String>, SchemaNode)
        case objectComma(SchemaNode, [String: SchemaNode], Set<String>)
        case arrayValue(SchemaNode, SchemaNode?)
        case arrayComma(SchemaNode, SchemaNode?)
        case inString(Bool)
        case inNumber
        case inLiteral(String, Int)
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
    private func simulateTokenIsValid(text: String) -> Bool {
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

    private func allowedChars() -> Set<Character> {
        let ws = Self.ws
        switch state {
        case .expectValue(let node): return startChars(for: node).union(ws)
        case .objectKey: return Set<Character>(["\"", "}"]).union(ws)
        case .objectColon: return Set<Character>([":"]).union(ws)
        case .objectValue(_, _, _, let vnode): return startChars(for: vnode).union(ws)
        case .objectComma: return Set<Character>([",", "}"]).union(ws)
        case .arrayValue(_, let item): return startChars(for: item ?? SchemaNode(type: .anything)).union(ws).union(Set<Character>(["]"]))
        case .arrayComma: return Set<Character>([",", "]"]).union(ws)
        case .inString: return printableChars()
        case .inNumber: return Self.digits.union(Set<Character>([".", "e", "E", "+", "-"]))
        case .inLiteral(let lit, let idx):
            let i = lit.index(lit.startIndex, offsetBy: min(idx, lit.count))
            if i < lit.endIndex { return Set<Character>([lit[i]]) }
            return Set<Character>([",", "}", "]"]).union(ws)
        case .done: return ws
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
            return handleValueStartPure(c, node: node)

        case .objectKey(let parent, let props, let req, let accumulatedKey):
            if escapeNext {
                escapeNext = false
                return .objectKey(parent, props, req, accumulatedKey + String(c))
            }
            if c == "\\" {
                escapeNext = true
                return state
            }
            if c == "\"" {
                return .objectColon(parent, props, req, accumulatedKey)
            }
            if c == "}" {
                // Match legacy `advance` behavior: any `}` in `.objectKey`
                // closes the object. (The legacy state machine doesn't
                // distinguish between "before the key opens" and "in the
                // key body" — both share `.objectKey`.)
                return .done
            }
            // RFC 8259 §7: unescaped control chars (< 0x20) are rejected
            // inside strings.
            if let ascii = c.asciiValue, ascii < 0x20 { return nil }
            return .objectKey(parent, props, req, accumulatedKey + String(c))

        case .objectColon(let parent, let props, let req, let keyName):
            if isWhitespace(c) { return state }
            if c == ":" {
                let valueSchema = props[keyName] ?? SchemaNode(type: .anything)
                return .objectValue(parent, props, req, valueSchema)
            }
            return nil

        case .objectValue(let parent, let props, var req, let vnode):
            if isWhitespace(c) { return state }
            guard let inner = handleValueStartPure(c, node: vnode) else { return nil }
            if case .done = inner {
                if let keyName = req.first { req.remove(keyName) }
                return .objectComma(parent, props, req)
            }
            return inner

        case .objectComma(let parent, let props, let req):
            if isWhitespace(c) { return state }
            if c == "," { return .objectKey(parent, props, req, "") }
            if c == "}" { return .done }
            return nil

        case .arrayValue(let parent, let item):
            if isWhitespace(c) { return state }
            if c == "]" { return .done }
            guard let inner = handleValueStartPure(c, node: item ?? SchemaNode(type: .anything)) else {
                return nil
            }
            if case .done = inner { return .arrayComma(parent, item) }
            return inner

        case .arrayComma(let parent, let item):
            if isWhitespace(c) { return state }
            if c == "," { return .arrayValue(parent, item) }
            if c == "]" { return .done }
            return nil

        case .inString(let isKey):
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
                // Distinguish key-string from value-string (tracked via
                // the associated bool). Matches the original `advance`
                // which transitions to `.done` for both — schema-aware
                // continuation (`.objectColon` for keys, `.objectComma`/
                // `.arrayComma` for values) is handled by the surrounding
                // states (.objectKey explicitly produces .objectColon when
                // the close-quote arrives without going through `.inString`).
                _ = isKey  // intentionally unused — both branches finish the string
                return .done
            }
            if let ascii = c.asciiValue, ascii < 0x20 { return nil }
            return state

        case .inNumber:
            if Self.digits.contains(c) || c == "." || c == "e" || c == "E" || c == "+" || c == "-" {
                return state
            }
            // Number ends. The original advance transitioned to .done here
            // unconditionally (and SchemaGuided's .inNumber state doesn't
            // carry parent context). Whitespace cleanly ends the number;
            // any other char would require knowing the parent frame to
            // re-process correctly. For now we accept whitespace only at
            // the end of a number — structural terminators (`,`/`}`/`]`)
            // are rejected here at the precompute level. The runtime
            // safety net in `didSample` clamps to `.done` if a structural
            // char is encountered, which forces EOS.
            //
            // Practical note: at runtime the model usually emits a separate
            // token for the structural terminator after the number, so
            // this restriction rarely matters. The Qwen3.6-27B "30.0000…"
            // runaway is fixed in `JSONLogitProcessor` (parent-frame-aware
            // masks); SchemaGuidedProcessor is mainly used for
            // `json_schema` mode where the schema constrains structure
            // more tightly anyway.
            if isWhitespace(c) { return .done }
            return nil

        case .inLiteral(let lit, let idx):
            // STRICT validation — original `advance` advanced the index for
            // ANY char, which let ` thoughtful` (after first-char `t`)
            // breeze through 8 unrelated chars. Now we require the char to
            // match the literal at the current position.
            let i = lit.index(lit.startIndex, offsetBy: min(idx, lit.count))
            guard i < lit.endIndex else { return nil }
            guard c == lit[i] else { return nil }
            if idx + 1 >= lit.count { return .done }
            return .inLiteral(lit, idx + 1)

        case .done:
            // Trailing whitespace tolerated, anything else is a violation.
            if isWhitespace(c) { return state }
            return nil
        }
    }

    /// Pure version of `handleValueStart`. Returns the entered state on a
    /// valid value-start char, or `nil` if no transition matches.
    private static func handleValueStartPure(_ c: Character, node: SchemaNode) -> SchemaState? {
        switch node.type {
        case .object(let props, let req):
            if c == "{" { return .objectKey(node, props, req, "") }
            return nil
        case .array(let items):
            if c == "[" { return .arrayValue(node, items) }
            return nil
        case .string, .stringEnum:
            if c == "\"" { return .inString(false) }
            return nil
        case .integer, .number:
            if c == "-" || Self.digits.contains(c) { return .inNumber }
            return nil
        case .boolean:
            if c == "t" { return .inLiteral("true", 1) }
            if c == "f" { return .inLiteral("false", 1) }
            return nil
        case .null:
            if c == "n" { return .inLiteral("null", 1) }
            return nil
        case .anything:
            if c == "{" { return .objectKey(node, [:], [], "") }
            if c == "[" { return .arrayValue(node, nil) }
            if c == "\"" { return .inString(false) }
            if c == "-" || Self.digits.contains(c) { return .inNumber }
            if c == "t" { return .inLiteral("true", 1) }
            if c == "f" { return .inLiteral("false", 1) }
            if c == "n" { return .inLiteral("null", 1) }
            return nil
        case .anyOf(let nodes):
            // Try each branch; first that admits the char wins.
            for n in nodes {
                if let entered = handleValueStartPure(c, node: n) {
                    return entered
                }
            }
            return nil
        }
    }
}
