import CoreFoundation
import CryptoKit
import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Official Recursive `rec` request rules that sit beside the decode kernel.
///
/// `server.py` fills an omitted `max_tokens` to the remaining context.
/// `chat.py` `normalize_messages` strips injected role and think markers
/// before the template. `parse_tool_call` types arguments from the tool
/// schema, and `seeded_tool_parser` derives call ids from the seed.
package enum FlintAlignment {

    // MARK: - Omitted max_tokens

    /// When the client omits `max_tokens`, Flint uses the remaining context
    /// window (`contextLength - promptTokens`), matching `rec` `server.py`
    /// `handle_completion`. An explicit value is unchanged. Other families
    /// keep `fallback` (ModelConfig, normally 2048).
    ///
    /// There is no second token cap under the context window. 2048 is not
    /// the official fill. A prompt that already fills the window throws
    /// `contextWindowExceeded`, the same gate as `ContextLimitError` when
    /// `remaining < 1`.
    static func outputTokenBudget(
        family: ModelFamily,
        requested: Int?,
        promptTokens: Int,
        contextLength: Int,
        fallback: Int
    ) throws -> Int {
        if let requested { return requested }
        guard family == .flint else { return fallback }
        let remaining = contextLength - promptTokens
        if remaining < 1 {
            throw NovaMLXError.contextWindowExceeded(
                promptTokens: promptTokens, maxTokens: 0, contextLength: contextLength)
        }
        return remaining
    }

    // MARK: - Prompt sanitize

    /// Flint messages for the chat template. Rejects a custom `stop` while
    /// structural grammar would be on, then sanitizes like `normalize_messages`.
    static func prepareMessages(request: InferenceRequest) throws -> [ChatMessage] {
        let grammarOn = flintGrammarEnabled(request)
        if let message = FlintChatDecode.customStopConflict(
            stop: request.stop,
            hasRequestConstraint: !grammarOn)
        {
            throw NovaMLXError.apiError(message)
        }
        if grammarOn,
           let message = FlintToolSchemaSession.rejection(
            tools: effectiveTools(request),
            enforceToolSchemas: request.enforceToolSchemas,
            toolsJSON: request.toolChoice == "none" ? nil : request.toolsJSON)
        {
            throw NovaMLXError.apiError(message)
        }
        try validateRoles(request.messages)
        return try sanitizeMessages(request.messages)
    }

    /// Omitted Flint seed is 0, matching `Sampling.seed`. An explicit request
    /// seed wins over the model config.
    static func resolvedSeed(requestSeed: UInt64?, modelSeed: UInt64?) -> UInt64 {
        requestSeed ?? modelSeed ?? 0
    }

    /// `validate_request` sampling and API constraints. Call only for `.flint`.
    static func validate(request: InferenceRequest, config: ModelConfig) throws {
        let frequency = request.frequencyPenalty ?? config.frequencyPenalty
        if frequency != 0 {
            throw NovaMLXError.apiError("Nonzero frequency_penalty is not supported by this decoder")
        }
        let minP = request.minP ?? config.minP
        if minP != 0 {
            throw NovaMLXError.apiError("Nonzero min_p is not supported by this decoder")
        }
        let repetition = request.repetitionPenalty ?? config.repeatPenalty
        if repetition != 0 && repetition != 1 {
            throw NovaMLXError.apiError("Only neutral repetition_penalty=1 is supported")
        }
        if request.stream && request.includeLogprobs {
            throw NovaMLXError.apiError("Token logprobs require stream=false")
        }
        if let store = request.storeCompletions, store != false {
            throw NovaMLXError.apiError("Stored-completion retrieval is not supported; use store=false")
        }
        let choice = request.toolChoice ?? "auto"
        if choice != "auto" && choice != "none" {
            throw NovaMLXError.apiError(
                "tool_choice currently supports auto and none; forced calls are not implemented")
        }
        if request.parallelToolCalls == false {
            throw NovaMLXError.apiError("parallel_tool_calls=false is not supported")
        }
        let grammar = request.flintGrammar ?? "flint"
        if grammar != "none" && grammar != "flint" {
            throw NovaMLXError.apiError("grammar must be none or flint")
        }
        try validateRoles(request.messages)
    }

    static func validateRoles(_ messages: [ChatMessage]) throws {
        let allowed: Set<String> = ["system", "developer", "user", "assistant", "tool"]
        for message in messages {
            let role = message.wireRole ?? message.role.rawValue
            if !allowed.contains(role) {
                throw NovaMLXError.apiError("Unsupported message role")
            }
        }
    }

    /// `tool_choice=none` drops tools before the template and the grammar.
    static func effectiveTools(_ request: InferenceRequest) -> [[String: Any]]? {
        if request.toolChoice == "none" { return nil }
        return request.tools
    }

    /// Structural grammar is on unless the client set `grammar` to `none`
    /// or another constraint (JSON, regex, GBNF) already replaced it.
    static func flintGrammarEnabled(_ request: InferenceRequest) -> Bool {
        let mode = request.flintGrammar ?? "flint"
        if mode != "flint" { return false }
        return !FlintChatDecode.requestHasOtherConstraint(request)
    }

    static func anthropicToolChoice(_ value: Any?) -> String? {
        if let text = value as? String { return text }
        guard let object = value as? [String: Any] ?? (value as? NSDictionary).map({ dict in
            var out: [String: Any] = [:]
            for (key, item) in dict {
                if let key = key as? String { out[key] = item }
            }
            return out
        }) else { return nil }
        guard let type = object["type"] as? String else { return "forced" }
        if type == "auto" || type == "none" { return type }
        return type
    }

    /// `rec` `normalize_messages`. Non-assistant string content drops NUL and
    /// think tags, then escapes role-header tokens. Assistant text is left
    /// alone so history reasoning survives. Tool-call argument strings must
    /// be JSON objects (blank becomes `{}`).
    static func sanitizeMessages(_ messages: [ChatMessage]) throws -> [ChatMessage] {
        try messages.map(sanitize)
    }

    static func sanitize(_ message: ChatMessage) throws -> ChatMessage {
        var content = message.content
        if message.role != .assistant, let text = content {
            content = sanitizeNonAssistantContent(text)
        }
        let toolCalls = try message.toolCalls?.map(normalizeHistoryArguments)
        return ChatMessage(
            role: message.role,
            content: content,
            images: message.images,
            name: message.name,
            toolCallId: message.toolCallId,
            toolCalls: toolCalls,
            reasoningContent: message.reasoningContent,
            wireRole: message.wireRole
        )
    }

    static func sanitizeNonAssistantContent(_ text: String) -> String {
        var text = text.replacingOccurrences(of: "\u{0000}", with: "")
        text = text.replacingOccurrences(of: "<think>", with: "")
        text = text.replacingOccurrences(of: "</think>", with: "")
        return escapeRoleMarkers(text)
    }

    private static let roleMarkers: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: "<\\|(?:begin_of_text|end_of_text|system|developer|user|assistant|observation)\\|>")
    }()

    static func escapeRoleMarkers(_ text: String) -> String {
        let matches = roleMarkers.matches(
            in: text, range: NSRange(text.startIndex..., in: text))
        guard !matches.isEmpty else { return text }
        var result = text
        for match in matches.reversed() {
            guard let range = Range(match.range, in: result) else { continue }
            let token = String(result[range])
            let escaped = token
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
            result.replaceSubrange(range, with: escaped)
        }
        return result
    }

    /// History tool calls: `json.loads` the argument string, blank → `{}`.
    /// A non-object or invalid JSON is the same failure `normalize_messages` raises.
    static func normalizeHistoryArguments(_ call: ToolCallResult) throws -> ToolCallResult {
        let trimmed = call.arguments.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return ToolCallResult(id: call.id, functionName: call.functionName, arguments: "{}")
        }
        guard let data = trimmed.data(using: .utf8) else {
            throw NovaMLXError.apiError("Malformed tool call arguments")
        }
        do {
            let object = try JSONSerialization.jsonObject(with: data)
            guard object is [String: Any] else {
                throw NovaMLXError.apiError("Malformed tool call arguments")
            }
        } catch let error as NovaMLXError {
            throw error
        } catch {
            throw NovaMLXError.apiError("Malformed tool call arguments")
        }
        return call
    }

    // MARK: - Schema-aware tool parse

    /// One Flint tool call after `parse_tool_call`. `argumentsJSON` is the
    /// OpenAI arguments string (object, sorted keys).
    struct ParsedCall: Equatable {
        var name: String
        var argumentsJSON: String
    }

    /// Invert the template's tool-call body (inside `<tool_call>`…`</tool_call>`).
    /// String schema values stay raw. Other values are JSON. `nil` when the
    /// body is not one call (`ValueError` in `rec`).
    static func parseToolCall(body: String, tools: [[String: Any]]?) -> ParsedCall? {
        guard let matched = callPattern.firstMatch(
            in: body, range: NSRange(body.startIndex..., in: body)),
            matched.range.location != NSNotFound,
            let nameRange = Range(matched.range(at: 1), in: body),
            let argsRange = Range(matched.range(at: 2), in: body)
        else { return nil }
        let name = String(body[nameRange])
        guard !name.isEmpty else { return nil }
        let argsText = String(body[argsRange])
        let properties = properties(named: name, tools: tools)
        var arguments: [(String, FlintJSON)] = []
        let pairs = argumentPattern.matches(
            in: argsText, range: NSRange(argsText.startIndex..., in: argsText))
        var cursor = argsText.startIndex
        for pair in pairs {
            guard let keyRange = Range(pair.range(at: 1), in: argsText),
                  let valueRange = Range(pair.range(at: 2), in: argsText),
                  let whole = Range(pair.range, in: argsText)
            else { return nil }
            let gap = argsText[cursor..<whole.lowerBound]
            if gap.contains(where: { !$0.isWhitespace }) { return nil }
            let key = String(argsText[keyRange])
            let raw = String(argsText[valueRange])
            let schema = dictionary(properties[key]) ?? [:]
            let value = typedValue(raw, schema: schema)
            if let index = arguments.firstIndex(where: { $0.0 == key }) {
                arguments[index] = (key, value)
            } else {
                arguments.append((key, value))
            }
            cursor = whole.upperBound
        }
        if argsText[cursor...].contains(where: { !$0.isWhitespace }) { return nil }
        return ParsedCall(
            name: name,
            argumentsJSON: FlintJSON.object(arguments).encodedObject())
    }

    private static let callPattern: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: #"\A\s*([^<\s]+)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)\z"#,
            options: [.dotMatchesLineSeparators])
    }()

    private static let argumentPattern: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: #"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"#,
            options: [.dotMatchesLineSeparators])
    }()

    /// Request flag wins. An omitted flag uses `NOVA_ENFORCE_TOOL_SCHEMAS`
    /// (`1` / `true` / `yes`) and otherwise the server config, which defaults off.
    package static func resolvedEnforceToolSchemas(requestFlag: Bool?, serverConfig: Bool) -> Bool {
        resolvedEnforceToolSchemas(
            requestFlag: requestFlag,
            serverConfig: serverConfig,
            environment: ProcessInfo.processInfo.environment["NOVA_ENFORCE_TOOL_SCHEMAS"])
    }

    package static func resolvedEnforceToolSchemas(
        requestFlag: Bool?,
        serverConfig: Bool,
        environment: String?
    ) -> Bool {
        if let requestFlag { return requestFlag }
        let raw = environment?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        if raw == "1" || raw == "true" || raw == "yes" { return true }
        return serverConfig
    }

    /// `rec` `_accepts`: type, type list, string enums, and anyOf/oneOf.
    static func schemaAccepts(_ schema: [String: Any], kind: String) -> Bool {
        if let type = schema["type"] as? String, type == kind { return true }
        if let types = schema["type"] as? [String], types.contains(kind) { return true }
        if let types = schema["type"] as? [Any],
           types.contains(where: { ($0 as? String) == kind })
        {
            return true
        }
        if kind == "string" {
            if let enums = schema["enum"] as? [String], !enums.isEmpty { return true }
            if let enums = schema["enum"] as? [Any], enums.contains(where: { $0 is String }) {
                return true
            }
        }
        for key in ["anyOf", "oneOf"] {
            for option in schemaList(schema[key]) where schemaAccepts(option, kind: kind) {
                return true
            }
        }
        return false
    }

    private static func typedValue(_ raw: String, schema: [String: Any]) -> FlintJSON {
        if raw == "null", schemaAccepts(schema, kind: "null") { return .null }
        if schemaAccepts(schema, kind: "string") { return .string(raw) }
        if let parsed = parseJSONFragment(raw) { return parsed }
        return .string(raw)
    }

    static func propertySchema(toolName: String, key: String, tools: [[String: Any]]?) -> [String: Any] {
        dictionary(properties(named: toolName, tools: tools)[key]) ?? [:]
    }

    /// JSON text of one argument value. String schemas keep the raw text.
    static func encodedArgument(raw: String, schema: [String: Any]) -> String {
        typedValue(raw, schema: schema).encodedFragment()
    }

    private static func properties(named name: String, tools: [[String: Any]]?) -> [String: Any] {
        var properties: [String: Any] = [:]
        for tool in tools ?? [] {
            let function = dictionary(tool["function"]) ?? tool
            guard function["name"] as? String == name else { continue }
            let parameters = dictionary(function["parameters"]) ?? [:]
            properties = dictionary(parameters["properties"]) ?? [:]
        }
        return properties
    }

    /// `[String: String]` does not cast to `[String: Any]`. HTTP JSON and the
    /// test literals both bridge through `NSDictionary`.
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

    private static func parseJSONFragment(_ raw: String) -> FlintJSON? {
        guard let data = raw.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return nil }
        return FlintJSON.from(object)
    }

    // MARK: - Seeded call ids

    /// `seeded_tool_parser`. `nil` seed keeps Nova's random `call_` ids.
    /// A seed (request or model config) hashes seed ‖ sampling JSON ‖ prompt
    /// token ids ‖ call index into a UUID version 4, so a replay is identical.
    struct CallIdentity {
        private let base: SHA256?
        private var nextIndex: UInt32 = 0

        init(
            seed: UInt64?,
            temperature: Double,
            topP: Double,
            topK: Int,
            presencePenalty: Double,
            maxTokens: Int,
            promptTokens: [Int]
        ) {
            guard let seed else {
                base = nil
                return
            }
            var hasher = SHA256()
            var seedLE = seed.littleEndian
            withUnsafeBytes(of: &seedLE) { hasher.update(bufferPointer: $0) }
            let config = Self.samplingJSON(
                temperature: temperature,
                topP: topP,
                topK: topK,
                presencePenalty: presencePenalty,
                maxTokens: maxTokens
            )
            hasher.update(data: Data(config.utf8))
            var packed = Data()
            packed.reserveCapacity(promptTokens.count * 4)
            for token in promptTokens {
                var le = UInt32(truncatingIfNeeded: token).littleEndian
                withUnsafeBytes(of: &le) { packed.append(contentsOf: $0) }
            }
            hasher.update(data: packed)
            base = hasher
        }

        mutating func nextID() -> String {
            let index = nextIndex
            nextIndex &+= 1
            guard let base else {
                return "call_\(UUID().uuidString.prefix(8).lowercased())"
            }
            var hasher = base
            var indexLE = index.littleEndian
            withUnsafeBytes(of: &indexLE) { hasher.update(bufferPointer: $0) }
            return Self.uuidVersion4(Array(hasher.finalize()))
        }

        /// Python `json.dumps(..., separators=(",", ":"), sort_keys=True)` for
        /// the five sampling fields `seeded_tool_parser` hashes.
        static func samplingJSON(
            temperature: Double,
            topP: Double,
            topK: Int,
            presencePenalty: Double,
            maxTokens: Int
        ) -> String {
            let temperature = pythonJSONFloat(temperature)
            let topP = pythonJSONFloat(topP)
            let presence = pythonJSONFloat(presencePenalty)
            return "{\"max_tokens\":\(maxTokens),\"presence_penalty\":\(presence),\"temperature\":\(temperature),\"top_k\":\(topK),\"top_p\":\(topP)}"
        }

        /// `json.dumps` of a finite float: shortest round trip, with `.0` when
        /// the spelling has no decimal point and no exponent.
        static func pythonJSONFloat(_ value: Double) -> String {
            if value == 0 { return value.sign == .minus ? "-0.0" : "0.0" }
            let text = String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), value)
            if text.contains(".") || text.contains("e") || text.contains("E") {
                return text.replacingOccurrences(of: "E", with: "e")
            }
            return text + ".0"
        }

        /// `uuid.UUID(bytes=digest[:16], version=4)`.
        static func uuidVersion4(_ digest: [UInt8]) -> String {
            var bytes = Array(digest.prefix(16))
            if bytes.count < 16 { bytes.append(contentsOf: repeatElement(0, count: 16 - bytes.count)) }
            bytes[6] = (bytes[6] & 0x0F) | 0x40
            bytes[8] = (bytes[8] & 0x3F) | 0x80
            let hex = bytes.map { String(format: "%02x", $0) }.joined()
            let chars = Array(hex)
            return "\(String(chars[0..<8]))-\(String(chars[8..<12]))-\(String(chars[12..<16]))-\(String(chars[16..<20]))-\(String(chars[20..<32]))"
        }
    }
}

/// JSON value produced by `parse_tool_call`. Object key order follows the
/// call. Wire encoding sorts keys so the arguments string is stable.
enum FlintJSON: Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([FlintJSON])
    /// Key order follows the call; tuples are not Equatable, so `==` is manual.
    case object([(String, FlintJSON)])

    static func == (lhs: FlintJSON, rhs: FlintJSON) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.int(let a), .int(let b)): return a == b
        case (.double(let a), .double(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)):
            guard a.count == b.count else { return false }
            return zip(a, b).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default: return false
        }
    }

    static func from(_ object: Any) -> FlintJSON {
        if object is NSNull { return .null }
        if let number = object as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return .bool(number.boolValue)
            }
            if CFNumberIsFloatType(number) {
                return .double(number.doubleValue)
            }
            return .int(number.intValue)
        }
        if let text = object as? String { return .string(text) }
        if let array = object as? [Any] { return .array(array.map(from)) }
        if let dict = object as? [String: Any] {
            return .object(dict.map { ($0.key, from($0.value)) })
        }
        return .string(String(describing: object))
    }

    var jsonObject: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let items): return items.map(\.jsonObject)
        case .object(let pairs):
            var dict: [String: Any] = [:]
            for (key, value) in pairs { dict[key] = value.jsonObject }
            return dict
        }
    }

    func encodedObject() -> String {
        let object = jsonObject
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }

    /// One value, not a sorted object. Nested objects still sort their keys.
    func encodedFragment() -> String {
        switch self {
        case .null: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .string(let value):
            return "\"" + JSONStringEscape.body(value) + "\""
        case .double, .array, .object:
            let wrapped: [Any] = [jsonObject]
            guard JSONSerialization.isValidJSONObject(wrapped),
                  let data = try? JSONSerialization.data(withJSONObject: wrapped, options: [.sortedKeys]),
                  let text = String(data: data, encoding: .utf8),
                  text.count >= 2
            else { return "null" }
            return String(text.dropFirst().dropLast())
        }
    }
}

/// Streaming split of response text and `<tool_call>` bodies for the fused
/// Flint path. Replaces the stock GLM4 parser so argument types and call ids
/// follow `rec` `chat.py` rather than a trim-and-guess parse.
final class FlintToolCallExtractor: @unchecked Sendable {
    enum Event: Equatable {
        case response(String)
        /// Name is known. `id` is the same id the finished `.tool` will use.
        case toolName(name: String, id: String)
        /// One piece of the arguments JSON. Pieces concatenate to `.tool`'s arguments.
        case toolArguments(id: String, delta: String)
        case tool(name: String, arguments: String, id: String)
    }

    private var buffer = ""
    private var inCall = false
    private let tools: [[String: Any]]?
    private var identity: FlintAlignment.CallIdentity
    private var scanOffset = 0
    private var nameCommitted = false
    private var callName = ""
    private var callID = ""
    private var json = ""
    private var openedObject = false
    private var stringOpen = false
    private var pendingKey: String?
    private var valuePhase: ValuePhase = .between

    private enum ValuePhase {
        case between
        case string
        case raw
    }

    init(tools: [[String: Any]]?, identity: FlintAlignment.CallIdentity) {
        self.tools = tools
        self.identity = identity
    }

    func feed(_ chunk: String) -> [Event] {
        guard !chunk.isEmpty else { return [] }
        buffer += chunk
        var events: [Event] = []
        while true {
            if !inCall {
                if let range = buffer.range(of: "<tool_call>") {
                    let before = String(buffer[..<range.lowerBound])
                    if !before.isEmpty { events.append(.response(before)) }
                    buffer = String(buffer[range.upperBound...])
                    beginCall()
                    continue
                }
                let hold = Self.prefixHold(buffer, marker: "<tool_call>")
                let emitCount = buffer.count - hold
                if emitCount > 0 {
                    let end = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    let text = String(buffer[..<end])
                    buffer = String(buffer[end...])
                    if !text.isEmpty { events.append(.response(text)) }
                }
                break
            }
            if let range = buffer.range(of: "</tool_call>") {
                let body = String(buffer[..<range.lowerBound])
                let tail = String(buffer[range.upperBound...])
                buffer = body
                closeCall(events: &events)
                buffer = tail
                inCall = false
                continue
            }
            streamOpen(events: &events)
            break
        }
        return events
    }

    func finish() -> [Event] {
        if inCall {
            if nameCommitted {
                var events: [Event] = []
                drainArguments(events: &events)
                closeJSON(events: &events)
                events.append(.tool(name: callName, arguments: json, id: callID))
                buffer = ""
                inCall = false
                resetCall()
                return events
            }
            let text = "<tool_call>" + buffer
            buffer = ""
            inCall = false
            resetCall()
            return text.isEmpty ? [] : [.response(text)]
        }
        guard !buffer.isEmpty else { return [] }
        let text = buffer
        buffer = ""
        return [.response(text)]
    }

    private func beginCall() {
        inCall = true
        resetCall()
    }

    private func resetCall() {
        scanOffset = 0
        nameCommitted = false
        callName = ""
        callID = ""
        json = ""
        openedObject = false
        stringOpen = false
        pendingKey = nil
        valuePhase = .between
    }

    private func closeCall(events: inout [Event]) {
        if !nameCommitted { _ = commitNameIfPossible(events: &events) }
        if nameCommitted {
            drainArguments(events: &events)
            closeJSON(events: &events)
            events.append(.tool(name: callName, arguments: json, id: callID))
        } else {
            events.append(.response("<tool_call>" + buffer + "</tool_call>"))
        }
        resetCall()
    }

    /// Stream a call that has not seen `</tool_call>` yet.
    /// A name is committed only at `<arg_key>`, so `write trailing` stays text.
    private func streamOpen(events: inout [Event]) {
        if !nameCommitted {
            if !commitNameIfPossible(events: &events) { return }
        }
        drainArguments(events: &events)
    }

    @discardableResult
    private func commitNameIfPossible(events: inout [Event]) -> Bool {
        let rest = tailText()
        guard let regex = Self.nameCommit,
              let match = regex.firstMatch(in: rest, range: NSRange(rest.startIndex..., in: rest)),
              match.range.location != NSNotFound,
              let nameRange = Range(match.range(at: 1), in: rest)
        else { return false }
        let name = String(rest[nameRange])
        guard !name.isEmpty else { return false }
        callName = name
        callID = identity.nextID()
        nameCommitted = true
        scanOffset += match.range.length
        events.append(.toolName(name: callName, id: callID))
        return true
    }

    private func drainArguments(events: inout [Event]) {
        while scanOffset < buffer.count {
            switch valuePhase {
            case .string:
                if !drainString(events: &events) { return }
            case .raw:
                if !drainRaw(events: &events) { return }
            case .between:
                if !drainBetween(events: &events) { return }
            }
        }
    }

    private func drainBetween(events: inout [Event]) -> Bool {
        let rest = tailText()
        let ws = rest.prefix(while: { $0.isWhitespace })
        if ws.count == rest.count { return false }
        if !ws.isEmpty {
            scanOffset += ws.count
        }
        let trimmed = tailText()
        if trimmed.hasPrefix("<arg_key>") {
            guard let end = trimmed.range(of: "</arg_key>") else { return false }
            let keyStart = trimmed.index(trimmed.startIndex, offsetBy: "<arg_key>".count)
            pendingKey = String(trimmed[keyStart..<end.lowerBound])
            scanOffset += trimmed.distance(from: trimmed.startIndex, to: end.upperBound)
            return true
        }
        if trimmed.hasPrefix("<arg_value>"), let key = pendingKey {
            let schema = FlintAlignment.propertySchema(toolName: callName, key: key, tools: tools)
            let live = FlintAlignment.schemaAccepts(schema, kind: "string")
                && !FlintAlignment.schemaAccepts(schema, kind: "null")
            scanOffset += "<arg_value>".count
            if live {
                emitArg(keyHeader(key, quote: true), events: &events)
                stringOpen = true
                valuePhase = .string
            } else {
                valuePhase = .raw
            }
            return true
        }
        if "<arg_key>".hasPrefix(trimmed) || "<arg_value>".hasPrefix(trimmed) { return false }
        return false
    }

    private func drainString(events: inout [Event]) -> Bool {
        let rest = tailText()
        if let close = rest.range(of: "</arg_value>") {
            emitArg(JSONStringEscape.body(String(rest[..<close.lowerBound])), events: &events)
            emitArg("\"", events: &events)
            stringOpen = false
            pendingKey = nil
            valuePhase = .between
            scanOffset += rest.distance(from: rest.startIndex, to: close.upperBound)
            return true
        }
        let hold = Self.prefixHold(rest, marker: "</arg_value>")
        let take = rest.count - hold
        if take > 0 {
            let end = rest.index(rest.startIndex, offsetBy: take)
            emitArg(JSONStringEscape.body(String(rest[..<end])), events: &events)
            scanOffset += take
        }
        return false
    }

    private func drainRaw(events: inout [Event]) -> Bool {
        let rest = tailText()
        guard let close = rest.range(of: "</arg_value>"), let key = pendingKey else { return false }
        let raw = String(rest[..<close.lowerBound])
        let schema = FlintAlignment.propertySchema(toolName: callName, key: key, tools: tools)
        emitArg(keyHeader(key, quote: false) + FlintAlignment.encodedArgument(raw: raw, schema: schema), events: &events)
        pendingKey = nil
        valuePhase = .between
        scanOffset += rest.distance(from: rest.startIndex, to: close.upperBound)
        return true
    }

    private func keyHeader(_ key: String, quote: Bool) -> String {
        var delta = ""
        if !openedObject {
            delta += "{"
            openedObject = true
        } else {
            delta += ","
        }
        delta += "\"" + JSONStringEscape.body(key) + "\":"
        if quote { delta += "\"" }
        return delta
    }

    private func emitArg(_ delta: String, events: inout [Event]) {
        guard !delta.isEmpty, nameCommitted else { return }
        json += delta
        events.append(.toolArguments(id: callID, delta: delta))
    }

    private func closeJSON(events: inout [Event]) {
        if stringOpen {
            emitArg("\"", events: &events)
            stringOpen = false
        }
        if openedObject {
            emitArg("}", events: &events)
        } else if json.isEmpty {
            emitArg("{}", events: &events)
        }
    }

    private func tailText() -> String {
        guard scanOffset < buffer.count else { return "" }
        let start = buffer.index(buffer.startIndex, offsetBy: scanOffset)
        return String(buffer[start...])
    }

    private static let nameCommit: NSRegularExpression? = {
        try? NSRegularExpression(pattern: #"\A\s*([^<\s]+)\s*(?=<arg_key>)"#)
    }()

    /// True when more text cannot produce `name<arg_key>`.
    static func nameIsImpossible(_ rest: String) -> Bool {
        var index = rest.startIndex
        while index < rest.endIndex, rest[index].isWhitespace {
            index = rest.index(after: index)
        }
        if index == rest.endIndex { return false }
        if rest[index] == "<" {
            let tail = rest[index...]
            if tail.hasPrefix("<arg_key>") { return false }
            return !"<arg_key>".hasPrefix(tail)
        }
        while index < rest.endIndex, rest[index] != "<", !rest[index].isWhitespace {
            index = rest.index(after: index)
        }
        if index == rest.endIndex { return false }
        while index < rest.endIndex, rest[index].isWhitespace {
            index = rest.index(after: index)
        }
        if index == rest.endIndex { return false }
        let tail = rest[index...]
        if tail.hasPrefix("<arg_key>") { return false }
        return !"<arg_key>".hasPrefix(tail)
    }

    /// Longest suffix of `text` that is a proper prefix of `marker`.
    static func prefixHold(_ text: String, marker: String) -> Int {
        let chars = Array(text)
        let mark = Array(marker)
        let maxCount = min(chars.count, max(0, mark.count - 1))
        guard maxCount > 0 else { return 0 }
        for count in stride(from: maxCount, through: 1, by: -1) {
            if Array(mark.prefix(count)) == Array(chars.suffix(count)) { return count }
        }
        return 0
    }
}
