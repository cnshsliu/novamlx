import CoreFoundation
import CryptoKit
import Foundation
import NovaMLXCore

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
        if let message = FlintChatDecode.customStopConflict(
            stop: request.stop,
            hasRequestConstraint: FlintChatDecode.requestHasOtherConstraint(request))
        {
            throw NovaMLXError.apiError(message)
        }
        if !FlintChatDecode.requestHasOtherConstraint(request),
           let message = FlintToolSchemaSession.rejection(
            tools: request.tools, enforceToolSchemas: request.enforceToolSchemas)
        {
            throw NovaMLXError.apiError(message)
        }
        return try sanitizeMessages(request.messages)
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
            toolCalls: toolCalls
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
}

/// Streaming split of response text and `<tool_call>` bodies for the fused
/// Flint path. Replaces the stock GLM4 parser so argument types and call ids
/// follow `rec` `chat.py` rather than a trim-and-guess parse.
final class FlintToolCallExtractor: @unchecked Sendable {
    enum Event: Equatable {
        case response(String)
        case tool(name: String, arguments: String, id: String)
    }

    private var buffer = ""
    private var inCall = false
    private let tools: [[String: Any]]?
    private var identity: FlintAlignment.CallIdentity

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
                    inCall = true
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
                buffer = String(buffer[range.upperBound...])
                inCall = false
                if let call = FlintAlignment.parseToolCall(body: body, tools: tools) {
                    events.append(.tool(
                        name: call.name,
                        arguments: call.argumentsJSON,
                        id: identity.nextID()))
                } else {
                    events.append(.response("<tool_call>" + body + "</tool_call>"))
                }
                continue
            }
            break
        }
        return events
    }

    func finish() -> [Event] {
        if inCall {
            let text = "<tool_call>" + buffer
            buffer = ""
            inCall = false
            return text.isEmpty ? [] : [.response(text)]
        }
        guard !buffer.isEmpty else { return [] }
        let text = buffer
        buffer = ""
        return [.response(text)]
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
