import CoreFoundation
import Foundation

/// Decoded token text for one Flint request. Struct copies of the grammar share
/// this cache so a filtered packet and the host replay do not decode the
/// vocabulary twice. Text is filled on first use, and only while a schema
/// field is actually masking free text.
public final class FlintVocabTextCache: @unchecked Sendable {
    private let decode: @Sendable (Int) -> String
    private var values: [Int: String]
    private let lock = NSLock()

    public init(decode: @escaping @Sendable (Int) -> String) {
        self.decode = decode
        self.values = [:]
    }

    /// Test double. Ids missing from `preset` decode as empty text.
    public init(preset: [Int: String]) {
        self.decode = { _ in "" }
        self.values = preset
    }

    public func text(_ id: Int) -> String {
        lock.lock()
        if let cached = values[id] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let decoded = decode(id)
        lock.lock()
        if let cached = values[id] {
            lock.unlock()
            return cached
        }
        values[id] = decoded
        lock.unlock()
        return decoded
    }
}

/// Host stand-in for `rec` `build_flint_glm_grammar(..., enforce_tool_schemas=)`.
///
/// xgrammar is not linked here. This session is the constraint that runs
/// inside `FlintStructuralGrammar` once a tool call has started. It forces
/// tool names, argument keys, and value shapes. It does not compile regex
/// `pattern` / `format`, string length, numeric bounds, or nested object
/// properties. Property order is not fixed: the JSON bridge does not keep it.
/// Raw string values are not vocabulary-scanned; only the dedicated marker
/// ids are masked, matching the loose tool body.
struct FlintToolSchemaSession: Equatable, Sendable {
    static let objectSchemaMessage = "Flint grammar requires object parameter schemas"
    static let functionNameMessage = "Flint grammar requires a function name"

    enum Compile: Equatable, Sendable {
        case inactive
        case ready(FlintToolSchemaSession)
        case rejected(String)
    }

    enum ValueKind: Equatable, Sendable {
        case rawString
        case enumeration([String])
        case integer
        case number
        case boolean
        case null
        case jsonObject
        case jsonArray
        indirect case union([ValueKind])

        var constrainsFreeText: Bool {
            switch self {
            case .rawString: return false
            case .union(let kinds): return kinds.contains { $0.constrainsFreeText }
            default: return true
            }
        }
    }

    struct ToolSpec: Equatable, Sendable {
        var name: String
        var enforced: Bool
        var properties: [String: ValueKind]
        var required: Set<String>
        var allowsAdditional: Bool
    }

    private enum Step: Equatable, Sendable {
        case refused
        case loose
        case name(buffer: String, trailing: Bool)
        case gap(name: String, seen: Set<String>)
        case expectValue(name: String, seen: Set<String>, key: String, kind: ValueKind)
        case key(name: String, seen: Set<String>, buffer: String, held: String)
        case value(name: String, seen: Set<String>, key: String, kind: ValueKind, buffer: String, held: String)
    }

    private var tools: [ToolSpec]
    private var step: Step

    static func refusing() -> FlintToolSchemaSession {
        FlintToolSchemaSession(tools: [], step: .refused)
    }

    /// `nil` when every tool that would be enforced has an object schema and
    /// a function name. Loose tools are not checked.
    static func rejection(tools: [[String: Any]]?, enforceToolSchemas: Bool) -> String? {
        if case .rejected(let message) = compile(tools: tools, enforceToolSchemas: enforceToolSchemas) {
            return message
        }
        return nil
    }

    /// `.inactive` keeps today's loose tool body, including unknown names.
    /// A tool is enforced when `enforceToolSchemas` is on or `strict` is
    /// `true`. Omitted `strict` stays loose unless the flag is on.
    static func compile(tools: [[String: Any]]?, enforceToolSchemas: Bool) -> Compile {
        guard let tools, !tools.isEmpty else { return .inactive }
        var specs: [ToolSpec] = []
        var anyEnforced = false
        for tool in tools {
            let function = dictionary(tool["function"]) ?? tool
            let name = (function["name"] as? String) ?? (tool["name"] as? String) ?? ""
            let strict = boolValue(function["strict"]) ?? boolValue(tool["strict"])
            let enforced = enforceToolSchemas || strict == true
            if !enforced {
                if !name.isEmpty {
                    specs.append(ToolSpec(
                        name: name, enforced: false, properties: [:], required: [], allowsAdditional: true))
                }
                continue
            }
            anyEnforced = true
            if name.isEmpty || name.contains(where: { $0.isWhitespace || $0 == "<" }) {
                return .rejected(functionNameMessage)
            }
            guard let parameters = dictionary(function["parameters"]) ?? dictionary(tool["parameters"]),
                  (parameters["type"] as? String) == "object"
            else { return .rejected(objectSchemaMessage) }
            let properties = dictionary(parameters["properties"]) ?? [:]
            var kinds: [String: ValueKind] = [:]
            for (key, schema) in properties {
                kinds[key] = valueKind(dictionary(schema) ?? [:])
            }
            let required = Set(stringList(parameters["required"]) ?? [])
            let allowsAdditional = additionalAllowed(parameters["additionalProperties"])
            specs.append(ToolSpec(
                name: name,
                enforced: true,
                properties: kinds,
                required: required,
                allowsAdditional: allowsAdditional))
        }
        guard anyEnforced else { return .inactive }
        let names = specs.map(\.name)
        if Set(names).count != names.count {
            return .rejected("Flint grammar requires unique function names")
        }
        return .ready(FlintToolSchemaSession(tools: specs, step: .name(buffer: "", trailing: false)))
    }

    /// False after a `strict: false` tool was selected inside a mixed request.
    /// The grammar then uses the loose tool body.
    var constrainsThisCall: Bool {
        if case .loose = step { return false }
        return true
    }

    /// Vocabulary scan is only useful where a token can be rejected from its
    /// text. Raw strings and additional-property keys stay marker-masked.
    var constrainsFreeText: Bool {
        switch step {
        case .refused, .name, .gap, .expectValue:
            return true
        case .loose:
            return false
        case .key(let name, _, _, _):
            return spec(named: name)?.allowsAdditional != true
        case .value(_, _, _, let kind, _, _):
            return kind.constrainsFreeText
        }
    }

    var held: String {
        switch step {
        case .key(_, _, _, let held), .value(_, _, _, _, _, let held):
            return held
        default:
            return ""
        }
    }

    mutating func begin() {
        if case .refused = step { return }
        step = .name(buffer: "", trailing: false)
    }

    func allows(tokenId: Int, text: String) -> Bool {
        advanced(tokenId: tokenId, text: text) != nil
    }

    /// Leaves `self` unchanged when the token is illegal.
    mutating func take(tokenId: Int, text: String) -> Bool {
        guard let next = advanced(tokenId: tokenId, text: text) else { return false }
        step = next
        return true
    }

    // MARK: - Transitions

    private func advanced(tokenId: Int, text: String) -> Step? {
        if case .refused = step { return nil }
        if Self.markerText(tokenId) != nil {
            return takeMarker(tokenId: tokenId)
        }
        return takeText(text)
    }

    private func takeMarker(tokenId: Int) -> Step? {
        switch step {
        case .name(let buffer, _):
            guard let spec = exactSpec(buffer) else { return nil }
            if tokenId == FlintStructuralGrammar.argKeyOpenId {
                return spec.enforced
                    ? .key(name: spec.name, seen: [], buffer: "", held: "")
                    : .loose
            }
            if tokenId == FlintStructuralGrammar.toolCloseId {
                if spec.enforced, !spec.required.isEmpty { return nil }
                return .gap(name: spec.name, seen: [])
            }
            return nil
        case .gap(let name, let seen):
            guard let spec = spec(named: name) else { return nil }
            if tokenId == FlintStructuralGrammar.argKeyOpenId {
                return .key(name: name, seen: seen, buffer: "", held: "")
            }
            if tokenId == FlintStructuralGrammar.toolCloseId {
                if !spec.required.isSubset(of: seen) { return nil }
                return .gap(name: name, seen: seen)
            }
            return nil
        case .key(let name, let seen, let buffer, _):
            guard tokenId == FlintStructuralGrammar.argKeyCloseId,
                  let spec = spec(named: name),
                  let kind = kindForKey(buffer, spec: spec, seen: seen)
            else { return nil }
            return .expectValue(name: name, seen: seen, key: buffer, kind: kind)
        case .expectValue(let name, let seen, let key, let kind):
            guard tokenId == FlintStructuralGrammar.argValueOpenId else { return nil }
            return .value(name: name, seen: seen, key: key, kind: kind, buffer: "", held: "")
        case .value(let name, let seen, let key, let kind, let buffer, _):
            guard tokenId == FlintStructuralGrammar.argValueCloseId else { return nil }
            guard Self.valueComplete(buffer, kind: kind) else { return nil }
            var nextSeen = seen
            nextSeen.insert(key)
            return .gap(name: name, seen: nextSeen)
        case .loose, .refused:
            return nil
        }
    }

    private func takeText(_ text: String) -> Step? {
        switch step {
        case .name(var buffer, var trailing):
            guard appendName(&buffer, trailing: &trailing, text) else { return nil }
            return .name(buffer: buffer, trailing: trailing)
        case .gap:
            guard text.allSatisfy(Self.isSpace) else { return nil }
            return step
        case .expectValue:
            guard text.allSatisfy(Self.isSpace) else { return nil }
            return step
        case .key(let name, let seen, let buffer, let held):
            guard let spec = spec(named: name) else { return nil }
            if spec.allowsAdditional {
                guard let nextHeld = appendFree(held, text: text) else { return nil }
                return .key(name: name, seen: seen, buffer: buffer + text, held: nextHeld)
            }
            let next = buffer + text
            guard Self.isKeyPrefix(next, spec: spec, seen: seen) else { return nil }
            return .key(name: name, seen: seen, buffer: next, held: "")
        case .value(let name, let seen, let key, let kind, let buffer, let held):
            if !kind.constrainsFreeText {
                guard let nextHeld = appendFree(held, text: text) else { return nil }
                return .value(
                    name: name, seen: seen, key: key, kind: kind, buffer: buffer + text, held: nextHeld)
            }
            let next = buffer + text
            guard Self.valuePrefix(next, kind: kind) else { return nil }
            return .value(name: name, seen: seen, key: key, kind: kind, buffer: next, held: "")
        case .loose, .refused:
            return nil
        }
    }

    private func appendName(_ buffer: inout String, trailing: inout Bool, _ text: String) -> Bool {
        for char in text {
            if Self.isSpace(char) {
                if buffer.isEmpty { continue }
                guard exactSpec(buffer) != nil else { return false }
                trailing = true
                continue
            }
            if trailing || char == "<" { return false }
            let next = buffer + String(char)
            guard tools.contains(where: { $0.name.hasPrefix(next) }) else { return false }
            buffer = next
        }
        return true
    }

    /// Proper prefix of a protocol marker the loose body already excludes.
    /// Structure tags stay legal as text; the dedicated id closes the field.
    private func appendFree(_ held: String, text: String) -> String? {
        let combined = held + text
        for marker in Self.looseForbidden where combined.contains(marker) { return nil }
        return FlintStructuralGrammar.retainedPrefix(combined, markers: Self.looseForbidden)
    }

    private func exactSpec(_ name: String) -> ToolSpec? {
        guard !name.isEmpty else { return nil }
        return tools.first { $0.name == name }
    }

    private func spec(named name: String) -> ToolSpec? {
        tools.first { $0.name == name }
    }

    private func kindForKey(_ key: String, spec: ToolSpec, seen: Set<String>) -> ValueKind? {
        if key.isEmpty || seen.contains(key) { return nil }
        if let kind = spec.properties[key] { return kind }
        if spec.allowsAdditional { return .rawString }
        return nil
    }

    private static func isKeyPrefix(_ buffer: String, spec: ToolSpec, seen: Set<String>) -> Bool {
        if buffer.isEmpty { return true }
        return spec.properties.keys.contains { key in
            !seen.contains(key) && key.hasPrefix(buffer)
        }
    }

    private static func markerText(_ tokenId: Int) -> String? {
        FlintStructuralGrammar.protocolMarkers.first { $0.tokenId == tokenId }?.text
    }

    private static let looseForbidden: [String] = {
        let structure: Set<Int> = [
            FlintStructuralGrammar.toolCloseId,
            FlintStructuralGrammar.argKeyOpenId,
            FlintStructuralGrammar.argKeyCloseId,
            FlintStructuralGrammar.argValueOpenId,
            FlintStructuralGrammar.argValueCloseId,
        ]
        return FlintStructuralGrammar.protocolMarkers
            .filter { !structure.contains($0.tokenId) }
            .map(\.text)
    }()

    // MARK: - Value shapes

    private static func valuePrefix(_ text: String, kind: ValueKind) -> Bool {
        switch kind {
        case .rawString:
            return true
        case .enumeration(let values):
            return values.contains { $0.hasPrefix(text) }
        case .integer:
            return integerStatus(text).prefix
        case .number:
            return numberStatus(text).prefix
        case .boolean:
            return literalStatus(text, words: ["true", "false"]).prefix
        case .null:
            return literalStatus(text, words: ["null"]).prefix
        case .jsonObject:
            return JSONPrefix.root(text, object: true).prefix
        case .jsonArray:
            return JSONPrefix.root(text, object: false).prefix
        case .union(let kinds):
            return kinds.contains { valuePrefix(text, kind: $0) }
        }
    }

    private static func valueComplete(_ text: String, kind: ValueKind) -> Bool {
        switch kind {
        case .rawString:
            return true
        case .enumeration(let values):
            return values.contains(text)
        case .integer:
            return integerStatus(text).complete
        case .number:
            return numberStatus(text).complete
        case .boolean:
            return literalStatus(text, words: ["true", "false"]).complete
        case .null:
            return literalStatus(text, words: ["null"]).complete
        case .jsonObject:
            return JSONPrefix.root(text, object: true).complete
        case .jsonArray:
            return JSONPrefix.root(text, object: false).complete
        case .union(let kinds):
            return kinds.contains { valueComplete(text, kind: $0) }
        }
    }

    private static func integerStatus(_ text: String) -> (prefix: Bool, complete: Bool) {
        var index = text.startIndex
        while index < text.endIndex, isSpace(text[index]) { index = text.index(after: index) }
        if index == text.endIndex { return (true, false) }
        if text[index] == "-" {
            index = text.index(after: index)
            if index == text.endIndex { return (true, false) }
        }
        guard index < text.endIndex, isDigit(text[index]) else { return (false, false) }
        let leadZero = text[index] == "0"
        index = text.index(after: index)
        if !leadZero {
            while index < text.endIndex, isDigit(text[index]) { index = text.index(after: index) }
        } else if index < text.endIndex, isDigit(text[index]) {
            return (false, false)
        }
        while index < text.endIndex {
            if !isSpace(text[index]) { return (false, false) }
            index = text.index(after: index)
        }
        return (true, true)
    }

    private static func numberStatus(_ text: String) -> (prefix: Bool, complete: Bool) {
        var index = text.startIndex
        while index < text.endIndex, isSpace(text[index]) { index = text.index(after: index) }
        if index == text.endIndex { return (true, false) }
        if text[index] == "-" {
            index = text.index(after: index)
            if index == text.endIndex { return (true, false) }
        }
        guard index < text.endIndex, isDigit(text[index]) else { return (false, false) }
        if text[index] == "0" {
            index = text.index(after: index)
        } else {
            while index < text.endIndex, isDigit(text[index]) { index = text.index(after: index) }
        }
        var complete = true
        if index < text.endIndex, text[index] == "." {
            index = text.index(after: index)
            var fraction = 0
            while index < text.endIndex, isDigit(text[index]) {
                fraction += 1
                index = text.index(after: index)
            }
            if fraction == 0 {
                return index == text.endIndex ? (true, false) : (false, false)
            }
            complete = true
        }
        if index < text.endIndex, text[index] == "e" || text[index] == "E" {
            index = text.index(after: index)
            if index < text.endIndex, text[index] == "+" || text[index] == "-" {
                index = text.index(after: index)
            }
            var exponent = 0
            while index < text.endIndex, isDigit(text[index]) {
                exponent += 1
                index = text.index(after: index)
            }
            if exponent == 0 {
                return index == text.endIndex ? (true, false) : (false, false)
            }
            complete = true
        } else if index < text.endIndex, isDigit(text[index]) {
            return (false, false)
        }
        while index < text.endIndex {
            if !isSpace(text[index]) { return (false, false) }
            index = text.index(after: index)
        }
        return (true, complete)
    }

    private static func literalStatus(_ text: String, words: [String]) -> (prefix: Bool, complete: Bool) {
        var index = text.startIndex
        while index < text.endIndex, isSpace(text[index]) { index = text.index(after: index) }
        let body = text[index...]
        if body.isEmpty { return (true, false) }
        let matched = words.filter { $0.hasPrefix(body) || body.hasPrefix($0) }
        guard let word = matched.first(where: { body.hasPrefix($0) }) ?? matched.first else {
            return (false, false)
        }
        if body.count < word.count {
            return (word.hasPrefix(body), false)
        }
        let rest = body.dropFirst(word.count)
        if rest.allSatisfy(isSpace) { return (true, true) }
        return (false, false)
    }

    private static func isSpace(_ char: Character) -> Bool {
        char == " " || char == "\n" || char == "\t" || char == "\r"
    }

    private static func isDigit(_ char: Character) -> Bool {
        char >= "0" && char <= "9"
    }

    // MARK: - Schema read

    private static func valueKind(_ schema: [String: Any]) -> ValueKind {
        if let branches = unionBranches(schema) {
            if branches.isEmpty { return .rawString }
            if branches.contains(where: { FlintAlignment.schemaAccepts($0, kind: "string") }) {
                return .rawString
            }
            return .union(branches.map(valueKind))
        }
        if FlintAlignment.schemaAccepts(schema, kind: "string") {
            if let values = stringList(schema["enum"]), !values.isEmpty {
                return .enumeration(values)
            }
            return .rawString
        }
        var kinds: [ValueKind] = []
        if FlintAlignment.schemaAccepts(schema, kind: "number") {
            kinds.append(.number)
        } else if FlintAlignment.schemaAccepts(schema, kind: "integer") {
            kinds.append(.integer)
        }
        if FlintAlignment.schemaAccepts(schema, kind: "boolean") { kinds.append(.boolean) }
        if FlintAlignment.schemaAccepts(schema, kind: "null") { kinds.append(.null) }
        if FlintAlignment.schemaAccepts(schema, kind: "object") { kinds.append(.jsonObject) }
        if FlintAlignment.schemaAccepts(schema, kind: "array") { kinds.append(.jsonArray) }
        if kinds.count == 1, let only = kinds.first { return only }
        if kinds.count > 1 { return .union(kinds) }
        return .rawString
    }

    private static func unionBranches(_ schema: [String: Any]) -> [[String: Any]]? {
        for key in ["anyOf", "oneOf"] where schema[key] != nil {
            return schemaList(schema[key])
        }
        return nil
    }

    private static func additionalAllowed(_ value: Any?) -> Bool {
        if value == nil { return true }
        if let flag = boolValue(value) { return flag }
        return false
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        if let flag = value as? Bool { return flag }
        if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() {
            return number.boolValue
        }
        return nil
    }

    private static func dictionary(_ value: Any?) -> [String: Any]? {
        if let dict = value as? [String: Any] { return dict }
        guard let object = value as? NSDictionary else { return nil }
        var out: [String: Any] = [:]
        for (key, item) in object {
            guard let key = key as? String else { continue }
            out[key] = item
        }
        return out
    }

    private static func schemaList(_ value: Any?) -> [[String: Any]] {
        if let list = value as? [Any] { return list.compactMap(dictionary) }
        guard let list = value as? NSArray else { return [] }
        return list.compactMap { dictionary($0) }
    }

    private static func stringList(_ value: Any?) -> [String]? {
        if let list = value as? [String] { return list }
        if let list = value as? [Any] {
            let strings = list.compactMap { $0 as? String }
            return strings.count == list.count ? strings : nil
        }
        guard let list = value as? NSArray else { return nil }
        var strings: [String] = []
        for item in list {
            guard let text = item as? String else { return nil }
            strings.append(text)
        }
        return strings
    }
}

/// JSON object/array prefix check. Inner properties are not constrained.
private enum JSONPrefix {
    static func root(_ text: String, object: Bool) -> (prefix: Bool, complete: Bool) {
        var parser = Parser(chars: Array(text))
        return parser.root(object: object)
    }

    struct Parser {
        let chars: [Character]
        var index = 0

        enum Outcome { case invalid, partial, done }

        mutating func root(object: Bool) -> (prefix: Bool, complete: Bool) {
            skipWS()
            if index >= chars.count { return (true, false) }
            if chars[index] != (object ? "{" : "[") { return (false, false) }
            switch parseValue() {
            case .invalid: return (false, false)
            case .partial: return (true, false)
            case .done:
                skipWS()
                return index < chars.count ? (false, false) : (true, true)
            }
        }

        mutating func parseValue() -> Outcome {
            skipWS()
            guard index < chars.count else { return .partial }
            switch chars[index] {
            case "{": return parseObject()
            case "[": return parseArray()
            case "\"": return parseString()
            case "t": return parseWord("true")
            case "f": return parseWord("false")
            case "n": return parseWord("null")
            case "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9":
                return parseNumber()
            default:
                return .invalid
            }
        }

        mutating func parseObject() -> Outcome {
            index += 1
            skipWS()
            if index >= chars.count { return .partial }
            if chars[index] == "}" {
                index += 1
                return .done
            }
            while true {
                switch parseString() {
                case .invalid: return .invalid
                case .partial: return .partial
                case .done: break
                }
                skipWS()
                if index >= chars.count { return .partial }
                if chars[index] != ":" { return .invalid }
                index += 1
                switch parseValue() {
                case .invalid: return .invalid
                case .partial: return .partial
                case .done: break
                }
                skipWS()
                if index >= chars.count { return .partial }
                if chars[index] == "}" {
                    index += 1
                    return .done
                }
                if chars[index] != "," { return .invalid }
                index += 1
                skipWS()
                if index >= chars.count { return .partial }
            }
        }

        mutating func parseArray() -> Outcome {
            index += 1
            skipWS()
            if index >= chars.count { return .partial }
            if chars[index] == "]" {
                index += 1
                return .done
            }
            while true {
                switch parseValue() {
                case .invalid: return .invalid
                case .partial: return .partial
                case .done: break
                }
                skipWS()
                if index >= chars.count { return .partial }
                if chars[index] == "]" {
                    index += 1
                    return .done
                }
                if chars[index] != "," { return .invalid }
                index += 1
                skipWS()
                if index >= chars.count { return .partial }
            }
        }

        mutating func parseString() -> Outcome {
            guard index < chars.count, chars[index] == "\"" else { return .invalid }
            index += 1
            while index < chars.count {
                let char = chars[index]
                index += 1
                if char == "\\" {
                    if index >= chars.count { return .partial }
                    index += 1
                    continue
                }
                if char == "\"" { return .done }
                if char == "\n" || char == "\r" { return .invalid }
            }
            return .partial
        }

        mutating func parseWord(_ word: String) -> Outcome {
            for char in word {
                if index >= chars.count { return .partial }
                if chars[index] != char { return .invalid }
                index += 1
            }
            return .done
        }

        mutating func parseNumber() -> Outcome {
            if index < chars.count, chars[index] == "-" {
                index += 1
                if index >= chars.count { return .partial }
            }
            guard index < chars.count, isDigit(chars[index]) else { return .invalid }
            if chars[index] == "0" {
                index += 1
            } else {
                while index < chars.count, isDigit(chars[index]) { index += 1 }
            }
            if index < chars.count, chars[index] == "." {
                index += 1
                guard index < chars.count else { return .partial }
                guard isDigit(chars[index]) else { return .invalid }
                while index < chars.count, isDigit(chars[index]) { index += 1 }
            }
            if index < chars.count, chars[index] == "e" || chars[index] == "E" {
                index += 1
                if index < chars.count, chars[index] == "+" || chars[index] == "-" { index += 1 }
                guard index < chars.count else { return .partial }
                guard isDigit(chars[index]) else { return .invalid }
                while index < chars.count, isDigit(chars[index]) { index += 1 }
            }
            return .done
        }

        mutating func skipWS() {
            while index < chars.count, chars[index] == " " || chars[index] == "\n"
                || chars[index] == "\t" || chars[index] == "\r"
            {
                index += 1
            }
        }

        func isDigit(_ char: Character) -> Bool { char >= "0" && char <= "9" }
    }
}
