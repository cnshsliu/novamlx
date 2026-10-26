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

/// Host stand-in for `rec` `build_flint_glm_grammar` / xgrammar `glm_4_7`
/// (`any_order=False`).
///
/// Tool names are always constrained when tools are present. Argument schemas
/// are enforced when `enforce_tool_schemas` is on or `strict` is not `false`
/// (omitted `strict` enforces). `strict: false` stays on the client tool and
/// is still enforced when the flag is on. Patterns, formats, string length,
/// numeric bounds, nested objects, and schema key order are masked here.
struct FlintToolSchemaSession: Equatable, Sendable {
    static let objectSchemaMessage = "Flint grammar requires object parameter schemas"
    static let functionNameMessage = "Flint grammar requires a function name"

    enum Compile: Equatable, Sendable {
        case inactive
        case ready(FlintToolSchemaSession)
        case rejected(String)
    }

    struct TextRule: Equatable, Sendable {
        var pattern: String?
        var format: String?
        var minLength: Int?
        var maxLength: Int?
    }

    struct NumberRule: Equatable, Sendable {
        var minimum: Double?
        var maximum: Double?
        var exclusiveMinimum: Double?
        var exclusiveMaximum: Double?

        func contains(_ value: Double) -> Bool {
            if let minimum, value < minimum { return false }
            if let maximum, value > maximum { return false }
            if let exclusiveMinimum, value <= exclusiveMinimum { return false }
            if let exclusiveMaximum, value >= exclusiveMaximum { return false }
            return true
        }
    }

    struct ObjectShape: Equatable, Sendable {
        var fields: [(name: String, kind: ValueKind, required: Bool)]
        var allowsAdditional: Bool

        static func == (lhs: ObjectShape, rhs: ObjectShape) -> Bool {
            guard lhs.allowsAdditional == rhs.allowsAdditional,
                  lhs.fields.count == rhs.fields.count
            else { return false }
            return zip(lhs.fields, rhs.fields).allSatisfy { a, b in
                a.name == b.name && a.kind == b.kind && a.required == b.required
            }
        }
    }

    enum ValueKind: Equatable, Sendable {
        case rawString
        case text(TextRule)
        case enumeration([String])
        case integer
        case number
        case boundedInt(NumberRule)
        case boundedNumber(NumberRule)
        case boolean
        case null
        case jsonObject
        case jsonArray
        indirect case objectValue(ObjectShape)
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
        /// Schema order. xgrammar `glm_xml` emits keys in this order.
        var fieldOrder: [String]
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

    /// `nil` when every tool has a function name and an object parameter schema.
    static func rejection(
        tools: [[String: Any]]?,
        enforceToolSchemas: Bool,
        toolsJSON: String? = nil
    ) -> String? {
        if case .rejected(let message) = compile(
            tools: tools, enforceToolSchemas: enforceToolSchemas, toolsJSON: toolsJSON)
        {
            return message
        }
        return nil
    }

    /// Names are constrained whenever tools exist. A tool is schema-enforced
    /// when `enforceToolSchemas` is on or `strict` is not `false`. Omitted
    /// `strict` enforces. `toolsJSON` preserves property key order; without it,
    /// object keys are sorted so the order is deterministic.
    static func compile(
        tools: [[String: Any]]?,
        enforceToolSchemas: Bool,
        toolsJSON: String? = nil
    ) -> Compile {
        if let toolsJSON,
           let parsed = FlintOrderedJSON.parse(toolsJSON),
           case .array(let items) = parsed,
           !items.isEmpty
        {
            return compile(orderedTools: items, enforceToolSchemas: enforceToolSchemas)
        }
        guard let tools, !tools.isEmpty else { return .inactive }
        var specs: [ToolSpec] = []
        for tool in tools {
            let function = dictionary(tool["function"]) ?? tool
            let name = (function["name"] as? String) ?? (tool["name"] as? String) ?? ""
            let strict = boolValue(function["strict"]) ?? boolValue(tool["strict"])
            let enforced = enforceToolSchemas || strict != false
            if let rejected = validateToolShape(name: name, parameters: dictionary(function["parameters"]) ?? dictionary(tool["parameters"])) {
                return .rejected(rejected)
            }
            if !enforced {
                specs.append(ToolSpec(
                    name: name, enforced: false, properties: [:], fieldOrder: [],
                    required: [], allowsAdditional: true))
                continue
            }
            let parameters = dictionary(function["parameters"]) ?? dictionary(tool["parameters"]) ?? [:]
            let properties = dictionary(parameters["properties"]) ?? [:]
            let order = properties.keys.sorted()
            var kinds: [String: ValueKind] = [:]
            for key in order {
                kinds[key] = valueKind(dictionary(properties[key]) ?? [:])
            }
            specs.append(spec(
                name: name, enforced: true, order: order, kinds: kinds,
                required: stringList(parameters["required"]) ?? [],
                allowsAdditional: additionalAllowed(parameters["additionalProperties"])))
        }
        return ready(specs)
    }

    private static func compile(orderedTools: [FlintOrderedJSON], enforceToolSchemas: Bool) -> Compile {
        var specs: [ToolSpec] = []
        for tool in orderedTools {
            let function = tool.field("function") ?? tool
            let name = function.field("name")?.stringValue ?? tool.field("name")?.stringValue ?? ""
            let strict = function.field("strict")?.boolValue ?? tool.field("strict")?.boolValue
            let enforced = enforceToolSchemas || strict != false
            let parameters = function.field("parameters") ?? tool.field("parameters")
            if let rejected = validateToolShape(name: name, ordered: parameters) {
                return .rejected(rejected)
            }
            if !enforced {
                specs.append(ToolSpec(
                    name: name, enforced: false, properties: [:], fieldOrder: [],
                    required: [], allowsAdditional: true))
                continue
            }
            let properties = parameters?.field("properties")
            let order = properties?.objectFields.map(\.0) ?? []
            var kinds: [String: ValueKind] = [:]
            for (key, schema) in properties?.objectFields ?? [] {
                kinds[key] = valueKind(ordered: schema)
            }
            let required = parameters?.field("required")?.arrayItems.compactMap(\.stringValue) ?? []
            specs.append(spec(
                name: name, enforced: true, order: order, kinds: kinds,
                required: required,
                allowsAdditional: additionalAllowed(parameters?.field("additionalProperties")?.asAny())))
        }
        return ready(specs)
    }

    private static func validateToolShape(name: String, parameters: [String: Any]?) -> String? {
        if name.isEmpty || name.contains(where: { $0.isWhitespace || $0 == "<" }) {
            return functionNameMessage
        }
        guard let parameters, (parameters["type"] as? String) == "object" else {
            return objectSchemaMessage
        }
        return nil
    }

    private static func validateToolShape(name: String, ordered: FlintOrderedJSON?) -> String? {
        if name.isEmpty || name.contains(where: { $0.isWhitespace || $0 == "<" }) {
            return functionNameMessage
        }
        guard let ordered, ordered.field("type")?.stringValue == "object" else {
            return objectSchemaMessage
        }
        return nil
    }

    private static func spec(
        name: String,
        enforced: Bool,
        order: [String],
        kinds: [String: ValueKind],
        required: [String],
        allowsAdditional: Bool
    ) -> ToolSpec {
        ToolSpec(
            name: name,
            enforced: enforced,
            properties: kinds,
            fieldOrder: order,
            required: Set(required),
            allowsAdditional: allowsAdditional)
    }

    private static func ready(_ specs: [ToolSpec]) -> Compile {
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
        let cursor = Self.cursor(seen: seen, spec: spec)
        if let index = spec.fieldOrder.firstIndex(of: key) {
            guard Self.keyIsLegal(index: index, cursor: cursor, spec: spec, seen: seen) else { return nil }
            return spec.properties[key]
        }
        if spec.allowsAdditional, !Self.hasUnmetRequired(cursor: cursor, spec: spec, seen: seen) {
            return .rawString
        }
        return nil
    }

    private static func isKeyPrefix(_ buffer: String, spec: ToolSpec, seen: Set<String>) -> Bool {
        if buffer.isEmpty { return true }
        if buffer.contains("<") || buffer.contains(where: isSpace) { return false }
        let cursor = cursor(seen: seen, spec: spec)
        let matchesSchema = spec.fieldOrder.enumerated().contains { index, name in
            keyIsLegal(index: index, cursor: cursor, spec: spec, seen: seen) && name.hasPrefix(buffer)
        }
        if matchesSchema { return true }
        if spec.allowsAdditional, !hasUnmetRequired(cursor: cursor, spec: spec, seen: seen) {
            if spec.fieldOrder.contains(where: { $0.hasPrefix(buffer) }) { return false }
            return true
        }
        return false
    }

    private static func cursor(seen: Set<String>, spec: ToolSpec) -> Int {
        var cursor = -1
        for (index, name) in spec.fieldOrder.enumerated() where seen.contains(name) {
            cursor = index
        }
        return cursor
    }

    private static func keyIsLegal(index: Int, cursor: Int, spec: ToolSpec, seen: Set<String>) -> Bool {
        guard index > cursor, spec.fieldOrder.indices.contains(index) else { return false }
        let name = spec.fieldOrder[index]
        if seen.contains(name) { return false }
        if cursor + 1 < index {
            for earlier in (cursor + 1)..<index {
                let field = spec.fieldOrder[earlier]
                if spec.required.contains(field), !seen.contains(field) { return false }
            }
        }
        return true
    }

    private static func hasUnmetRequired(cursor: Int, spec: ToolSpec, seen: Set<String>) -> Bool {
        for (index, name) in spec.fieldOrder.enumerated() where index > cursor {
            if spec.required.contains(name), !seen.contains(name) { return true }
        }
        return false
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
        case .text(let rule):
            return textStatus(text, rule: rule).prefix
        case .boundedInt(let rule):
            return boundedStatus(text, rule: rule, integer: true).prefix
        case .boundedNumber(let rule):
            return boundedStatus(text, rule: rule, integer: false).prefix
        case .jsonObject:
            return JSONPrefix.root(text, object: true).prefix
        case .jsonArray:
            return JSONPrefix.root(text, object: false).prefix
        case .objectValue(let shape):
            return objectStatus(text, shape: shape).prefix
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
        case .text(let rule):
            return textStatus(text, rule: rule).complete
        case .boundedInt(let rule):
            return boundedStatus(text, rule: rule, integer: true).complete
        case .boundedNumber(let rule):
            return boundedStatus(text, rule: rule, integer: false).complete
        case .jsonObject:
            return JSONPrefix.root(text, object: true).complete
        case .jsonArray:
            return JSONPrefix.root(text, object: false).complete
        case .objectValue(let shape):
            return objectStatus(text, shape: shape).complete
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
        if FlintAlignment.schemaAccepts(schema, kind: "object"),
           dictionary(schema["properties"]) != nil
        {
            return .objectValue(objectShape(schema))
        }
        if FlintAlignment.schemaAccepts(schema, kind: "string") {
            if let values = stringList(schema["enum"]), !values.isEmpty {
                return .enumeration(values)
            }
            if let rule = textRule(schema) { return .text(rule) }
            return .rawString
        }
        var kinds: [ValueKind] = []
        if FlintAlignment.schemaAccepts(schema, kind: "number") {
            if let rule = numberRule(schema) { kinds.append(.boundedNumber(rule)) }
            else { kinds.append(.number) }
        } else if FlintAlignment.schemaAccepts(schema, kind: "integer") {
            if let rule = numberRule(schema) { kinds.append(.boundedInt(rule)) }
            else { kinds.append(.integer) }
        }
        if FlintAlignment.schemaAccepts(schema, kind: "boolean") { kinds.append(.boolean) }
        if FlintAlignment.schemaAccepts(schema, kind: "null") { kinds.append(.null) }
        if FlintAlignment.schemaAccepts(schema, kind: "object") { kinds.append(.jsonObject) }
        if FlintAlignment.schemaAccepts(schema, kind: "array") { kinds.append(.jsonArray) }
        if kinds.count == 1, let only = kinds.first { return only }
        if kinds.count > 1 { return .union(kinds) }
        return .rawString
    }

    private static func valueKind(ordered schema: FlintOrderedJSON) -> ValueKind {
        let object = schema.asAny()
        let dict = dictionary(object) ?? [:]
        if let properties = schema.field("properties") {
            let required = Set(schema.field("required")?.arrayItems.compactMap(\.stringValue) ?? [])
            let fields = properties.objectFields.map { key, child in
                (name: key, kind: valueKind(ordered: child), required: required.contains(key))
            }
            let allows = additionalAllowed(schema.field("additionalProperties")?.asAny())
            if schema.field("type")?.stringValue == "object" || !fields.isEmpty {
                return .objectValue(ObjectShape(fields: fields, allowsAdditional: allows))
            }
        }
        return valueKind(dict)
    }

    private static func objectShape(_ schema: [String: Any]) -> ObjectShape {
        let properties = dictionary(schema["properties"]) ?? [:]
        let required = Set(stringList(schema["required"]) ?? [])
        let fields = properties.keys.sorted().map { key in
            (name: key, kind: valueKind(dictionary(properties[key]) ?? [:]), required: required.contains(key))
        }
        return ObjectShape(
            fields: fields,
            allowsAdditional: additionalAllowed(schema["additionalProperties"]))
    }

    private static func textRule(_ schema: [String: Any]) -> TextRule? {
        let pattern = schema["pattern"] as? String
        let format = schema["format"] as? String
        let minLength = intValue(schema["minLength"])
        let maxLength = intValue(schema["maxLength"])
        if pattern == nil && format == nil && minLength == nil && maxLength == nil { return nil }
        return TextRule(pattern: pattern, format: format, minLength: minLength, maxLength: maxLength)
    }

    private static func numberRule(_ schema: [String: Any]) -> NumberRule? {
        let minimum = doubleValue(schema["minimum"])
        let maximum = doubleValue(schema["maximum"])
        var exclusiveMinimum = doubleValue(schema["exclusiveMinimum"])
        var exclusiveMaximum = doubleValue(schema["exclusiveMaximum"])
        if exclusiveMinimum == nil, boolValue(schema["exclusiveMinimum"]) == true {
            exclusiveMinimum = minimum
        }
        if exclusiveMaximum == nil, boolValue(schema["exclusiveMaximum"]) == true {
            exclusiveMaximum = maximum
        }
        if minimum == nil && maximum == nil && exclusiveMinimum == nil && exclusiveMaximum == nil {
            return nil
        }
        return NumberRule(
            minimum: minimum, maximum: maximum,
            exclusiveMinimum: exclusiveMinimum, exclusiveMaximum: exclusiveMaximum)
    }

    private static func textStatus(_ text: String, rule: TextRule) -> (prefix: Bool, complete: Bool) {
        let count = text.unicodeScalars.count
        if let maxLength = rule.maxLength, count > maxLength { return (false, false) }
        let pattern = rule.pattern ?? formatPattern(rule.format)
        var patternPrefix = true
        var patternComplete = true
        if let pattern {
            guard let compiled = FlintPattern(pattern) else { return (false, false) }
            patternPrefix = compiled.isPrefix(text)
            patternComplete = compiled.matches(text)
        }
        if !patternPrefix { return (false, false) }
        let minLength = rule.minLength ?? 0
        let complete = patternComplete && count >= minLength && (rule.maxLength.map { count <= $0 } ?? true)
        if count < minLength {
            let room = rule.maxLength.map { count < $0 } ?? true
            return (room, false)
        }
        return (true, complete)
    }

    private static func formatPattern(_ format: String?) -> String? {
        switch format {
        case "date":
            return #"^\d{4}-\d{2}-\d{2}$"#
        case "time":
            return #"^\d{2}:\d{2}:\d{2}$"#
        case "date-time":
            return #"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?$"#
        case "uuid":
            return #"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"#
        case "email":
            return #"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"#
        case "uri":
            return #"^[A-Za-z][A-Za-z0-9+.\-]*:.+$"#
        default:
            return nil
        }
    }

    private static func boundedStatus(
        _ text: String, rule: NumberRule, integer: Bool
    ) -> (prefix: Bool, complete: Bool) {
        let base = integer ? integerStatus(text) : numberStatus(text)
        if !base.prefix { return (false, false) }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "-" { return (true, false) }
        if !base.complete {
            return (numberCanExtend(trimmed, rule: rule, integer: integer), false)
        }
        guard let value = Double(trimmed), value.isFinite else { return (false, false) }
        if rule.contains(value) { return (true, true) }
        if numberCanExtend(trimmed, rule: rule, integer: integer) { return (true, false) }
        return (false, false)
    }

    /// Digit and fraction extensions only. An exponent is not introduced to
    /// pull an out-of-range prefix such as `11` with maximum 10 back into range.
    private static func numberCanExtend(_ text: String, rule: NumberRule, integer: Bool) -> Bool {
        if text.hasSuffix(".") || text.hasSuffix("e") || text.hasSuffix("E")
            || text.hasSuffix("e+") || text.hasSuffix("e-")
            || text.hasSuffix("E+") || text.hasSuffix("E-")
        {
            return true
        }
        guard let value = Double(text), value.isFinite else { return false }
        if rule.contains(value) { return true }
        if text.contains("e") || text.contains("E") || text.contains(".") { return false }
        let digits = text.hasPrefix("-") ? String(text.dropFirst()) : text
        if digits.isEmpty { return true }
        if digits.first == "0" {
            if integer { return false }
            let sign = text.hasPrefix("-") ? -1.0 : 1.0
            return rule.contains(sign * 0.5) || rule.contains(sign * 0.1)
        }
        guard let base = Int(digits) else { return false }
        let signed = text.hasPrefix("-") ? -base : base
        var current = signed
        for _ in 0..<6 {
            let start = current * 10
            for digit in 0...9 {
                let next = start + (current < 0 ? -digit : digit)
                if rule.contains(Double(next)) { return true }
            }
            current = start
        }
        return false
    }

    private static func objectStatus(_ text: String, shape: ObjectShape) -> (prefix: Bool, complete: Bool) {
        var parser = ObjectParser(chars: Array(text), shape: shape)
        return parser.parse()
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let int = value as? Int { return int }
        if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() {
            return number.intValue
        }
        return nil
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        if let double = value as? Double { return double }
        if let int = value as? Int { return Double(int) }
        if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() {
            return number.doubleValue
        }
        return nil
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

/// Schema-ordered JSON object inside one Flint `<arg_value>`.
private struct ObjectParser {
    var chars: [Character]
    var index = 0
    var shape: FlintToolSchemaSession.ObjectShape

    mutating func parse() -> (prefix: Bool, complete: Bool) {
        skip()
        if index >= chars.count { return (true, false) }
        guard chars[index] == "{" else { return (false, false) }
        index += 1
        return parseBody(shape)
    }

    mutating func parseBody(_ shape: FlintToolSchemaSession.ObjectShape) -> (prefix: Bool, complete: Bool) {
        skip()
        if index >= chars.count { return (true, false) }
        if chars[index] == "}" {
            index += 1
            let missing = shape.fields.contains { $0.required }
            return missing ? (false, false) : (true, true)
        }
        var cursor = -1
        var seen = Set<String>()
        while index < chars.count {
            let keyStart = index
            guard chars[index] == "\"" else { return (false, false) }
            guard let key = readString() else {
                return index >= chars.count ? (true, false) : (false, false)
            }
            _ = keyStart
            skip()
            if index >= chars.count { return (true, false) }
            guard chars[index] == ":" else { return (false, false) }
            index += 1
            skip()
            if index >= chars.count { return (true, false) }
            guard let kind = kind(for: key, shape: shape, cursor: cursor, seen: seen) else {
                return (false, false)
            }
            let valueStart = index
            guard consumeValue() else {
                let partial = String(chars[valueStart..<min(index, chars.count)])
                if index >= chars.count {
                    return (FlintToolSchemaSession.valueAllowsPrefix(partial, kind: kind), false)
                }
                return (false, false)
            }
            let raw = String(chars[valueStart..<index])
            let probed = FlintToolSchemaSession.schemaProbe(raw, kind: kind)
            if !FlintToolSchemaSession.valueIsComplete(probed, kind: kind) {
                return (false, false)
            }
            if let fieldIndex = shape.fields.firstIndex(where: { $0.name == key }) {
                cursor = fieldIndex
            }
            seen.insert(key)
            skip()
            if index >= chars.count { return (true, false) }
            if chars[index] == "}" {
                index += 1
                let missing = shape.fields.contains { $0.required && !seen.contains($0.name) }
                skip()
                if index < chars.count { return (false, false) }
                return missing ? (false, false) : (true, true)
            }
            guard chars[index] == "," else { return (false, false) }
            index += 1
            skip()
            if index >= chars.count { return (true, false) }
        }
        return (true, false)
    }

    func kind(
        for key: String,
        shape: FlintToolSchemaSession.ObjectShape,
        cursor: Int,
        seen: Set<String>
    ) -> FlintToolSchemaSession.ValueKind? {
        if seen.contains(key) { return nil }
        if let index = shape.fields.firstIndex(where: { $0.name == key }) {
            if index <= cursor { return nil }
            for earlier in shape.fields[(cursor + 1)..<index] where earlier.required {
                return nil
            }
            return shape.fields[index].kind
        }
        let unmet = shape.fields.enumerated().contains { offset, field in
            offset > cursor && field.required && !seen.contains(field.name)
        }
        if shape.allowsAdditional && !unmet { return .rawString }
        return nil
    }

    mutating func consumeValue() -> Bool {
        skip()
        guard index < chars.count else { return false }
        switch chars[index] {
        case "{":
            return consumeObject()
        case "[":
            return consumeArray()
        case "\"":
            return readString() != nil
        case "t":
            return consumeWord("true")
        case "f":
            return consumeWord("false")
        case "n":
            return consumeWord("null")
        default:
            return consumeNumber()
        }
    }

    mutating func consumeObject() -> Bool {
        guard index < chars.count, chars[index] == "{" else { return false }
        index += 1
        skip()
        if index < chars.count, chars[index] == "}" {
            index += 1
            return true
        }
        while index < chars.count {
            guard readString() != nil else { return false }
            skip()
            guard index < chars.count, chars[index] == ":" else { return false }
            index += 1
            guard consumeValue() else { return false }
            skip()
            guard index < chars.count else { return false }
            if chars[index] == "}" {
                index += 1
                return true
            }
            guard chars[index] == "," else { return false }
            index += 1
            skip()
        }
        return false
    }

    mutating func consumeArray() -> Bool {
        guard index < chars.count, chars[index] == "[" else { return false }
        index += 1
        skip()
        if index < chars.count, chars[index] == "]" {
            index += 1
            return true
        }
        while index < chars.count {
            guard consumeValue() else { return false }
            skip()
            guard index < chars.count else { return false }
            if chars[index] == "]" {
                index += 1
                return true
            }
            guard chars[index] == "," else { return false }
            index += 1
            skip()
        }
        return false
    }

    mutating func readString() -> String? {
        guard index < chars.count, chars[index] == "\"" else { return nil }
        index += 1
        var out = ""
        while index < chars.count {
            let char = chars[index]
            index += 1
            if char == "\"" { return out }
            if char == "\\" {
                guard index < chars.count else { return nil }
                out.append(chars[index])
                index += 1
                continue
            }
            out.append(char)
        }
        return nil
    }

    mutating func consumeWord(_ word: String) -> Bool {
        for char in word {
            guard index < chars.count, chars[index] == char else { return false }
            index += 1
        }
        return true
    }

    mutating func consumeNumber() -> Bool {
        let start = index
        if index < chars.count, chars[index] == "-" { index += 1 }
        guard index < chars.count, chars[index] >= "0" && chars[index] <= "9" else { return false }
        while index < chars.count, chars[index] >= "0" && chars[index] <= "9" { index += 1 }
        if index < chars.count, chars[index] == "." {
            index += 1
            guard index < chars.count, chars[index] >= "0" && chars[index] <= "9" else { return false }
            while index < chars.count, chars[index] >= "0" && chars[index] <= "9" { index += 1 }
        }
        if index < chars.count, chars[index] == "e" || chars[index] == "E" {
            index += 1
            if index < chars.count, chars[index] == "+" || chars[index] == "-" { index += 1 }
            guard index < chars.count, chars[index] >= "0" && chars[index] <= "9" else { return false }
            while index < chars.count, chars[index] >= "0" && chars[index] <= "9" { index += 1 }
        }
        return index > start
    }

    mutating func skip() {
        while index < chars.count {
            let char = chars[index]
            if char == " " || char == "\n" || char == "\t" || char == "\r" {
                index += 1
            } else {
                break
            }
        }
    }
}

extension FlintToolSchemaSession {
    fileprivate static func valueAllowsPrefix(_ text: String, kind: ValueKind) -> Bool {
        valuePrefix(schemaProbe(text, kind: kind), kind: kind)
    }

    fileprivate static func valueIsComplete(_ text: String, kind: ValueKind) -> Bool {
        valueComplete(text, kind: kind)
    }

    fileprivate static func schemaProbe(_ raw: String, kind: ValueKind) -> String {
        switch kind {
        case .text, .rawString, .enumeration:
            guard raw.count >= 2, raw.first == "\"", raw.last == "\"" else { return raw }
            let body = raw.dropFirst().dropLast()
            return body.replacingOccurrences(of: "\\\"", with: "\"").replacingOccurrences(of: "\\\\", with: "\\")
        default:
            return raw
        }
    }
}
