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
    private let eosTokenId: Int?
    private let vocabSize: Int
    private let maskBuilder: TokenMaskBuilder
    private var escapeNext = false
    private var changed = false

    private enum SchemaState {
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

    public init(schema: [String: Any], tokenizer: NovaMLXEngine.Tokenizer) {
        self.rootSchema = SchemaNode.parse(schema)
        self.state = .expectValue(rootSchema)
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
        self.vocabSize = maskBuilder.vocabSize
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        let allowed = allowedChars()
        let mask = maskBuilder.buildMask(allowedChars: allowed)
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    public func didSample(token: MLXArray) {
        let tokenId = token.item(Int.self)
        if tokenId == eosTokenId { return }
        let text = tokenizer.decode([tokenId])
        for char in text { advance(char) }
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

    private func advance(_ char: Character) {
        switch state {
        case .expectValue(let node):
            handleValueStart(char, node: node)
        case .objectKey(let parent, let props, let req, var accumulatedKey):
            if escapeNext {
                escapeNext = false
                accumulatedKey.append(char)
                state = .objectKey(parent, props, req, accumulatedKey)
            } else if char == "\\" {
                escapeNext = true
            } else if char == "\"" {
                state = .objectColon(parent, props, req, accumulatedKey)
            } else if char == "}" {
                state = .done
            } else {
                accumulatedKey.append(char)
                state = .objectKey(parent, props, req, accumulatedKey)
            }
        case .objectColon(let parent, let props, let req, let keyName):
            if char == ":" {
                let valueSchema = props[keyName] ?? SchemaNode(type: .anything)
                state = .objectValue(parent, props, req, valueSchema)
            }
        case .objectValue(let parent, let props, var req, let vnode):
            handleValueStart(char, node: vnode)
            if case .done = state {
                if let keyName = req.first {
                    req.remove(keyName)
                }
                state = .objectComma(parent, props, req)
            }
        case .objectComma(let parent, let props, let req):
            if char == "," { state = .objectKey(parent, props, req, "") }
            else if char == "}" { state = .done }
        case .arrayValue(let parent, let item):
            if char == "]" { state = .done }
            else {
                handleValueStart(char, node: item ?? SchemaNode(type: .anything))
                if case .done = state { state = .arrayComma(parent, item) }
            }
        case .arrayComma(let parent, let item):
            if char == "," { state = .arrayValue(parent, item) }
            else if char == "]" { state = .done }
        case .inString(let isKey):
            if escapeNext { escapeNext = false }
            else if char == "\\" { escapeNext = true }
            else if char == "\"" {
                if isKey { state = .done }
                else { state = .done }
            }
        case .inNumber:
            if !Self.digits.contains(char) && char != "." && char != "e" && char != "E" && char != "+" && char != "-" {
                state = .done
            }
        case .inLiteral(let lit, let idx):
            if idx + 1 >= lit.count { state = .done }
            else { state = .inLiteral(lit, idx + 1) }
        case .done:
            break
        }
    }

    private func handleValueStart(_ char: Character, node: SchemaNode) {
        switch node.type {
        case .object(let props, let req):
            if char == "{" { state = .objectKey(node, props, req, "") }
        case .array(let items):
            if char == "[" { state = .arrayValue(node, items) }
        case .string, .stringEnum:
            if char == "\"" { state = .inString(false) }
        case .integer, .number:
            if char == "-" || Self.digits.contains(char) { state = .inNumber }
        case .boolean:
            if char == "t" { state = .inLiteral("true", 1) }
            else if char == "f" { state = .inLiteral("false", 1) }
        case .null:
            if char == "n" { state = .inLiteral("null", 1) }
        case .anything:
            if char == "{" { state = .objectKey(node, [:], [], "") }
            else if char == "[" { state = .arrayValue(node, nil) }
            else if char == "\"" { state = .inString(false) }
            else if char == "-" || Self.digits.contains(char) { state = .inNumber }
            else if char == "t" { state = .inLiteral("true", 1) }
            else if char == "f" { state = .inLiteral("false", 1) }
            else if char == "n" { state = .inLiteral("null", 1) }
        case .anyOf(let nodes):
            for n in nodes {
                let prev = state
                handleValueStart(char, node: n)
                if case .done = state { return }
                if case .expectValue = prev, case .expectValue = state { continue }
                return
            }
        }
    }
}
