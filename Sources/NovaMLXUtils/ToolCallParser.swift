import Foundation

public struct FunctionCall: Codable, Sendable, Equatable {
    public let name: String
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolCall: Codable, Sendable, Equatable {
    public let id: String
    public let type: String
    public let function: FunctionCall

    private enum CodingKeys: String, CodingKey {
        case id, type, function
    }

    public init(id: String, function: FunctionCall) {
        self.id = id
        self.type = "function"
        self.function = function
    }
}

public struct ToolDefinition: Codable, Sendable {
    public let type: String
    public let function: ToolFunction

    public init(function: ToolFunction) {
        self.type = "function"
        self.function = function
    }
}

public struct ToolFunction: Codable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: [String: AnyCodable]?

    public init(name: String, description: String? = nil, parameters: [String: AnyCodable]? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public struct AnyCodable: Codable, @unchecked Sendable, Equatable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(String.self) { value = v }
        else if let v = try? container.decode(Int.self) { value = v }
        else if let v = try? container.decode(Double.self) { value = v }
        else if let v = try? container.decode(Bool.self) { value = v }
        else if let v = try? container.decode([AnyCodable].self) { value = v.map(\.value) }
        else if let v = try? container.decode([String: AnyCodable].self) { value = v.mapValues(\.value) }
        else { value = NSNull() }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let v as String: try container.encode(v)
        case let v as Int: try container.encode(v)
        case let v as Double: try container.encode(v)
        case let v as Bool: try container.encode(v)
        case let v as [Any]: try container.encode(v.map { AnyCodable($0) })
        case let v as [String: Any]: try container.encode(v.mapValues { AnyCodable($0) })
        default: try container.encodeNil()
        }
    }

    public static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        String(describing: lhs.value) == String(describing: rhs.value)
    }
}

public enum ToolCallParser {

    public struct ParseResult: Sendable {
        public let content: String
        public let toolCalls: [ToolCall]?

        public init(content: String, toolCalls: [ToolCall]?) {
            self.content = content
            self.toolCalls = toolCalls
        }
    }

    public static func parse(_ text: String) -> ParseResult {
        var cleaned = text

        if let result = parseXMLToolCalls(cleaned) {
            return result
        }

        if let result = parseBracketToolCalls(cleaned) {
            return result
        }

        if let result = parseMarkerToolCalls(cleaned) {
            return result
        }

        if let result = parseNamespacedToolCalls(cleaned) {
            return result
        }

        if let result = parseGLMToolCalls(cleaned) {
            return result
        }

        if let result = parseGemmaToolCalls(cleaned) {
            return result
        }

        if let result = parseThinkingToolCalls(cleaned) {
            return result
        }

        cleaned = stripToolCallMarkup(cleaned)
        return ParseResult(content: cleaned, toolCalls: nil)
    }

    private static func parseXMLToolCalls(_ text: String) -> ParseResult? {
        // Match both <tool>...</tool> (Claude Code, Open Code, Open Claw)
        // and <tool_call>...</tool_call> (Hermes)
        let pattern = "<tool(?:_call)?>(.*?)</tool(?:_call)?>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { return nil }

        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return nil }

        var toolCalls: [ToolCall] = []
        for (i, match) in matches.enumerated() {
            guard let contentRange = Range(match.range(at: 1), in: text) else { continue }
            let toolContent = String(text[contentRange]).trimmingCharacters(in: .whitespacesAndNewlines)

            if let tc = parseJSONToolCall(toolContent, index: i) {
                toolCalls.append(tc)
                continue
            }

            if let tc = parseXMLFunctionCall(toolContent, index: i) {
                toolCalls.append(tc)
                continue
            }
        }

        guard !toolCalls.isEmpty else { return nil }

        let cleaned = regex.stringByReplacingMatches(
            in: text, options: [], range: range,
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return ParseResult(content: cleaned, toolCalls: toolCalls)
    }

    private static func parseJSONToolCall(_ content: String, index: Int) -> ToolCall? {
        guard let data = content.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }

        let name = json["name"] as? String ?? ""
        let args: String
        if let argumentsDict = json["arguments"] as? [String: Any],
           let argsData = try? JSONSerialization.data(withJSONObject: argumentsDict),
           let argsStr = String(data: argsData, encoding: .utf8) {
            args = argsStr
        } else if let argumentsStr = json["arguments"] as? String {
            args = argumentsStr
        } else {
            args = "{}"
        }

        guard !name.isEmpty else { return nil }
        return ToolCall(id: "call_\(String(format: "%04x", index))", function: FunctionCall(name: name, arguments: args))
    }

    private static func parseXMLFunctionCall(_ content: String, index: Int) -> ToolCall? {
        let funcPattern = "<function=(.*?)>(.*?)</function>"
        guard let regex = try? NSRegularExpression(pattern: funcPattern, options: [.dotMatchesLineSeparators]) else { return nil }

        let range = NSRange(content.startIndex..., in: content)
        let matches = regex.matches(in: content, options: [], range: range)
        guard let match = matches.first else { return nil }

        guard let nameRange = Range(match.range(at: 1), in: content) else { return nil }
        let name = String(content[nameRange])

        var args: [String: String] = [:]
        let paramPattern = "<parameter=(.*?)>(.*?)</parameter>"
        if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
            let innerRange = match.range(at: 2)
            let paramMatches = paramRegex.matches(in: content, options: [], range: innerRange)
            for pMatch in paramMatches {
                guard let keyRange = Range(pMatch.range(at: 1), in: content),
                      let valRange = Range(pMatch.range(at: 2), in: content) else { continue }
                args[String(content[keyRange])] = String(content[valRange])
            }
        }

        let argsStr = (try? JSONSerialization.data(withJSONObject: args)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return ToolCall(id: "call_\(String(format: "%04x", index))", function: FunctionCall(name: name, arguments: argsStr))
    }

    private static func parseBracketToolCalls(_ text: String) -> ParseResult? {
        let patterns = [
            "\\[TOOL_CALLS\\]\\s*(\\[.*?\\])",
            "\\[Calling tool:\\s*(.*?)\\]",
            "\\[Tool call:\\s*(.*?)\\]",
        ]

        for (pi, pattern) in patterns.enumerated() {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { continue }
            let range = NSRange(text.startIndex..., in: text)
            let matches = regex.matches(in: text, options: [], range: range)
            guard !matches.isEmpty else { continue }

            var toolCalls: [ToolCall] = []

            for (i, match) in matches.enumerated() {
                if pi == 0 {
                    guard let contentRange = Range(match.range(at: 1), in: text) else { continue }
                    let jsonStr = String(text[contentRange])
                    if let data = jsonStr.data(using: .utf8),
                       let jsonArray = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        for (j, item) in jsonArray.enumerated() {
                            let name = item["name"] as? String ?? ""
                            let args = item["arguments"] as? [String: Any]
                            let argsStr = args.flatMap { try? JSONSerialization.data(withJSONObject: $0) }.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                            if !name.isEmpty {
                                toolCalls.append(ToolCall(id: "call_\(i)_\(j)", function: FunctionCall(name: name, arguments: argsStr)))
                            }
                        }
                    }
                } else {
                    guard let contentRange = Range(match.range(at: 1), in: text) else { continue }
                    let callText = String(text[contentRange])
                    let parts = callText.split(separator: "(", maxSplits: 1)
                    let name = String(parts[0]).trimmingCharacters(in: .whitespaces)
                    let argsStr = parts.count > 1 ? String(parts[1].dropLast()) : "{}"
                    toolCalls.append(ToolCall(id: "call_\(i)", function: FunctionCall(name: name, arguments: argsStr)))
                }
            }

            guard !toolCalls.isEmpty else { continue }

            let cleaned = regex.stringByReplacingMatches(
                in: text, options: [], range: range,
                withTemplate: ""
            ).trimmingCharacters(in: .whitespacesAndNewlines)

            return ParseResult(content: cleaned, toolCalls: toolCalls)
        }

        return nil
    }

    private static func parseMarkerToolCalls(_ text: String) -> ParseResult? {
        let patterns = ["<\\|tool_call\\|>(.*?)<\\|/tool_call\\|>", "<\\|tool_call\\|>(.*?)$"]

        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { continue }
            let range = NSRange(text.startIndex..., in: text)
            let matches = regex.matches(in: text, options: [], range: range)
            guard !matches.isEmpty else { continue }

            var toolCalls: [ToolCall] = []
            for (i, match) in matches.enumerated() {
                guard let contentRange = Range(match.range(at: 1), in: text) else { continue }
                let content = String(text[contentRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                if let tc = parseJSONToolCall(content, index: i) {
                    toolCalls.append(tc)
                }
            }

            guard !toolCalls.isEmpty else { continue }

            let cleaned = regex.stringByReplacingMatches(
                in: text, options: [], range: range,
                withTemplate: ""
            ).trimmingCharacters(in: .whitespacesAndNewlines)

            return ParseResult(content: cleaned, toolCalls: toolCalls)
        }

        return nil
    }

    private static func parseNamespacedToolCalls(_ text: String) -> ParseResult? {
        let nsPattern = "<(\\w+):tool_call>(.*?)</\\1:tool_call>"
        guard let regex = try? NSRegularExpression(pattern: nsPattern, options: [.dotMatchesLineSeparators]) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return nil }

        var toolCalls: [ToolCall] = []
        for (i, match) in matches.enumerated() {
            guard let contentRange = Range(match.range(at: 2), in: text) else { continue }
            let content = String(text[contentRange]).trimmingCharacters(in: .whitespacesAndNewlines)

            let invokePattern = "<invoke\\s+name=\"(.*?)\">(.*?)</invoke>"
            if let invokeRegex = try? NSRegularExpression(pattern: invokePattern, options: [.dotMatchesLineSeparators]) {
                let innerRange = NSRange(content.startIndex..., in: content)
                if let invokeMatch = invokeRegex.firstMatch(in: content, options: [], range: innerRange) {
                    guard let nameRange = Range(invokeMatch.range(at: 1), in: content) else { continue }
                    let name = String(content[nameRange])

                    var args: [String: String] = [:]
                    let paramPattern = "<parameter\\s+name=\"(.*?)\">(.*?)</parameter>"
                    if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
                        let innerContent = NSRange(location: invokeMatch.range(at: 2).location, length: invokeMatch.range(at: 2).length)
                        let paramMatches = paramRegex.matches(in: content, options: [], range: innerContent)
                        for pMatch in paramMatches {
                            guard let keyRange = Range(pMatch.range(at: 1), in: content),
                                  let valRange = Range(pMatch.range(at: 2), in: content) else { continue }
                            args[String(content[keyRange])] = String(content[valRange])
                        }
                    }
                    let argsStr = (try? JSONSerialization.data(withJSONObject: args)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                    toolCalls.append(ToolCall(id: "call_\(String(format: "%04x", i))", function: FunctionCall(name: name, arguments: argsStr)))
                    continue
                }
            }

            if let tc = parseJSONToolCall(content, index: i) {
                toolCalls.append(tc)
            }
        }

        guard !toolCalls.isEmpty else { return nil }

        let cleaned = regex.stringByReplacingMatches(
            in: text, options: [], range: range,
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return ParseResult(content: cleaned, toolCalls: toolCalls)
    }

    private static func parseGLMToolCalls(_ text: String) -> ParseResult? {
        let glmPattern = "☐(.*?)■"
        guard let regex = try? NSRegularExpression(pattern: glmPattern, options: [.dotMatchesLineSeparators]) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return nil }

        var toolCalls: [ToolCall] = []
        for (i, match) in matches.enumerated() {
            guard let contentRange = Range(match.range(at: 1), in: text) else { continue }
            let content = String(text[contentRange]).trimmingCharacters(in: .whitespacesAndNewlines)

            if let tc = parseJSONToolCall(content, index: i) {
                toolCalls.append(tc)
                continue
            }

            if let tc = parseXMLFunctionCall(content, index: i) {
                toolCalls.append(tc)
                continue
            }

            let parts = content.components(separatedBy: "☐")
            if parts.count >= 2 {
                let name = String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty else { continue }
                var args: [String: String] = [:]
                var j = 1
                while j + 1 < parts.count {
                    let key = String(parts[j]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let valPart = String(parts[j + 1])
                    let val: String
                    if let firstSep = valPart.components(separatedBy: "■").first {
                        val = firstSep
                    } else {
                        val = valPart
                    }
                    args[key] = val.trimmingCharacters(in: .whitespacesAndNewlines)
                    j += 2
                }
                let argsStr = (try? JSONSerialization.data(withJSONObject: args)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                toolCalls.append(ToolCall(id: "call_\(String(format: "%04x", i))", function: FunctionCall(name: name, arguments: argsStr)))
            }
        }

        guard !toolCalls.isEmpty else { return nil }

        let cleaned = regex.stringByReplacingMatches(
            in: text, options: [], range: range,
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return ParseResult(content: cleaned, toolCalls: toolCalls)
    }

    private static func parseGemmaToolCalls(_ text: String) -> ParseResult? {
        let gemmaPattern = "call:(\\w+)\\s*\\{(.+?)\\}"
        guard let regex = try? NSRegularExpression(pattern: gemmaPattern, options: [.dotMatchesLineSeparators]) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return nil }

        var toolCalls: [ToolCall] = []
        for (i, match) in matches.enumerated() {
            guard let nameRange = Range(match.range(at: 1), in: text),
                  let argsRange = Range(match.range(at: 2), in: text) else { continue }
            let name = String(text[nameRange])
            let argsStr = String(text[argsRange]).trimmingCharacters(in: .whitespacesAndNewlines)

            var jsonArgs: [String: Any] = [:]
            let keyValuePattern = "(\\w+)\\s*:\\s*(\"[^\"]*\"|\\[[^\\]]*\\]|\\{[^}]*\\}|[^,}]+)"
            if let kvRegex = try? NSRegularExpression(pattern: keyValuePattern, options: []) {
                let argsNSRange = NSRange(argsStr.startIndex..., in: argsStr)
                let kvMatches = kvRegex.matches(in: argsStr, options: [], range: argsNSRange)
                for kvMatch in kvMatches {
                    guard let kRange = Range(kvMatch.range(at: 1), in: argsStr),
                          let vRange = Range(kvMatch.range(at: 2), in: argsStr) else { continue }
                    let key = String(argsStr[kRange])
                    var value: Any = String(argsStr[vRange]).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                    if let intVal = Int(value as? String ?? "") { value = intVal }
                    else if let doubleVal = Double(value as? String ?? "") { value = doubleVal }
                    jsonArgs[key] = value
                }
            }

            let finalArgs = (try? JSONSerialization.data(withJSONObject: jsonArgs)).flatMap { String(data: $0, encoding: .utf8) } ?? argsStr
            toolCalls.append(ToolCall(id: "call_\(String(format: "%04x", i))", function: FunctionCall(name: name, arguments: finalArgs)))
        }

        guard !toolCalls.isEmpty else { return nil }

        let cleaned = regex.stringByReplacingMatches(
            in: text, options: [], range: range,
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return ParseResult(content: cleaned, toolCalls: toolCalls)
    }

    private static func parseThinkingToolCalls(_ text: String) -> ParseResult? {
        let thinkPattern = "<think[^>]*>([\\s\\S]*?)</think"
        guard let regex = try? NSRegularExpression(pattern: thinkPattern, options: []) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return nil }

        for match in matches {
            guard let contentRange = Range(match.range(at: 1), in: text) else { continue }
            let thinkingContent = String(text[contentRange])

            if let result = parseXMLToolCalls(thinkingContent) {
                if let calls = result.toolCalls, !calls.isEmpty {
                    let cleaned = regex.stringByReplacingMatches(
                        in: text, options: [], range: range,
                        withTemplate: ""
                    ).trimmingCharacters(in: .whitespacesAndNewlines)
                    return ParseResult(content: cleaned, toolCalls: calls)
                }
            }

            if let result = parseMarkerToolCalls(thinkingContent) {
                if let calls = result.toolCalls, !calls.isEmpty {
                    let cleaned = regex.stringByReplacingMatches(
                        in: text, options: [], range: range,
                        withTemplate: ""
                    ).trimmingCharacters(in: .whitespacesAndNewlines)
                    return ParseResult(content: cleaned, toolCalls: calls)
                }
            }
        }

        return nil
    }

    private static func stripToolCallMarkup(_ text: String) -> String {
        var result = text
        let patterns = [
            "<tool(?:_call)?>.*?</tool(?:_call)?>",
            "<\\|tool_call\\|>.*?(<\\|/tool_call\\|>|$)",
            "\\[TOOL_CALLS\\].*?\\]",
        ]
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let range = NSRange(result.startIndex..., in: result)
                result = regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
            }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public final class ToolCallStreamFilter: @unchecked Sendable {
    private var buffer: String
    private var suppressPatterns: [NSRegularExpression]

    public init() {
        self.buffer = ""
        self.suppressPatterns = []
        for pattern in [
            "<tool>",
            "</tool>",
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
            "[TOOL_CALLS]",
            "[Calling tool:",
            "[Tool call:",
            "<|tool_call|>",
            "</|tool_call|>",
            "☐",
            "■",
            "call:",
        ] {
            if let regex = try? NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: pattern), options: []) {
                suppressPatterns.append(regex)
            }
        }
    }

    public func feed(_ token: String) -> String {
        buffer += token

        for pattern in suppressPatterns {
            let range = NSRange(buffer.startIndex..., in: buffer)
            if pattern.firstMatch(in: buffer, options: [], range: range) != nil {
                return ""
            }
        }

        let result = buffer
        buffer = ""
        return result
    }

    public func finalize() -> String {
        let result = buffer
        buffer = ""
        return result
    }

    public func reset() {
        buffer = ""
    }
}
