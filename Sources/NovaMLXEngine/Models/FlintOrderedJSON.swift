import Foundation

/// JSON parser that keeps object key order. `JSONSerialization` dictionaries do not.
/// Flint schema compilation uses this so property order matches the request body
/// and xgrammar `any_order=False`.
package enum FlintOrderedJSON: Equatable, Sendable {
    /// Tuples do not synthesize `Equatable`, so object equality is manual.
    case object([(String, FlintOrderedJSON)])
    case array([FlintOrderedJSON])
    case string(String)
    case number(String)
    case bool(Bool)
    case null

    package static func == (lhs: FlintOrderedJSON, rhs: FlintOrderedJSON) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null):
            return true
        case (.bool(let left), .bool(let right)):
            return left == right
        case (.number(let left), .number(let right)):
            return left == right
        case (.string(let left), .string(let right)):
            return left == right
        case (.array(let left), .array(let right)):
            return left == right
        case (.object(let left), .object(let right)):
            guard left.count == right.count else { return false }
            return zip(left, right).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default:
            return false
        }
    }

    static func parse(_ data: Data) -> FlintOrderedJSON? {
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        return parse(text)
    }

    static func parse(_ text: String) -> FlintOrderedJSON? {
        var parser = Parser(chars: Array(text))
        guard let value = parser.parseValue() else { return nil }
        parser.skipWS()
        guard parser.index == parser.chars.count else { return nil }
        return value
    }

    /// The `tools` array of an OpenAI or Anthropic JSON body, re-encoded with
    /// stable spacing and the original key order.
    package static func toolsJSON(in body: Data) -> String? {
        guard let root = parse(body), case .object(let fields) = root,
              let tools = fields.first(where: { $0.0 == "tools" })?.1
        else { return nil }
        return tools.encoded()
    }

    var objectFields: [(String, FlintOrderedJSON)] {
        if case .object(let fields) = self { return fields }
        return []
    }

    var arrayItems: [FlintOrderedJSON] {
        if case .array(let items) = self { return items }
        return []
    }

    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    func field(_ name: String) -> FlintOrderedJSON? {
        objectFields.first { $0.0 == name }?.1
    }

    func encoded() -> String {
        switch self {
        case .null:
            return "null"
        case .bool(let value):
            return value ? "true" : "false"
        case .number(let value):
            return value
        case .string(let value):
            return FlintJSONText.string(value)
        case .array(let items):
            return "[" + items.map { $0.encoded() }.joined(separator: ",") + "]"
        case .object(let fields):
            let body = fields.map { FlintJSONText.string($0.0) + ":" + $0.1.encoded() }.joined(separator: ",")
            return "{" + body + "}"
        }
    }

    /// Loose `[String: Any]` bridge. Key order is not retained past this call.
    func asAny() -> Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .number(let value):
            if let int = Int(value), String(int) == value { return int }
            return Double(value) ?? value
        case .string(let value): return value
        case .array(let items): return items.map { $0.asAny() }
        case .object(let fields):
            var dict: [String: Any] = [:]
            for (key, value) in fields { dict[key] = value.asAny() }
            return dict
        }
    }
}

enum FlintJSONText {
    /// `json.dumps` string spelling with `ensure_ascii=False`.
    static func string(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{0008}": out += "\\b"
            case "\u{000C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
        return out
    }

    /// Python `json.dumps` default separators `(', ', ': ')` for a JSON value
    /// whose object keys are already ordered.
    static func python(_ value: FlintOrderedJSON) -> String {
        switch value {
        case .object(let fields):
            let body = fields.map { string($0.0) + ": " + python($0.1) }.joined(separator: ", ")
            return "{" + body + "}"
        case .array(let items):
            return "[" + items.map { python($0) }.joined(separator: ", ") + "]"
        default:
            return value.encoded()
        }
    }
}

private struct Parser {
    var chars: [Character]
    var index = 0

    mutating func parseValue() -> FlintOrderedJSON? {
        skipWS()
        guard index < chars.count else { return nil }
        switch chars[index] {
        case "{": return parseObject()
        case "[": return parseArray()
        case "\"": return parseString().map { .string($0) }
        case "t": return consume("true") ? .bool(true) : nil
        case "f": return consume("false") ? .bool(false) : nil
        case "n": return consume("null") ? .null : nil
        default: return parseNumber()
        }
    }

    mutating func parseObject() -> FlintOrderedJSON? {
        guard chars[index] == "{" else { return nil }
        index += 1
        skipWS()
        var fields: [(String, FlintOrderedJSON)] = []
        if index < chars.count, chars[index] == "}" {
            index += 1
            return .object(fields)
        }
        while index < chars.count {
            guard let key = parseString() else { return nil }
            skipWS()
            guard index < chars.count, chars[index] == ":" else { return nil }
            index += 1
            guard let value = parseValue() else { return nil }
            fields.append((key, value))
            skipWS()
            guard index < chars.count else { return nil }
            if chars[index] == "}" {
                index += 1
                return .object(fields)
            }
            guard chars[index] == "," else { return nil }
            index += 1
            skipWS()
        }
        return nil
    }

    mutating func parseArray() -> FlintOrderedJSON? {
        guard chars[index] == "[" else { return nil }
        index += 1
        skipWS()
        var items: [FlintOrderedJSON] = []
        if index < chars.count, chars[index] == "]" {
            index += 1
            return .array(items)
        }
        while index < chars.count {
            guard let value = parseValue() else { return nil }
            items.append(value)
            skipWS()
            guard index < chars.count else { return nil }
            if chars[index] == "]" {
                index += 1
                return .array(items)
            }
            guard chars[index] == "," else { return nil }
            index += 1
            skipWS()
        }
        return nil
    }

    mutating func parseString() -> String? {
        guard index < chars.count, chars[index] == "\"" else { return nil }
        index += 1
        var out = ""
        while index < chars.count {
            let char = chars[index]
            index += 1
            if char == "\"" { return out }
            if char == "\\" {
                guard index < chars.count else { return nil }
                let escaped = chars[index]
                index += 1
                switch escaped {
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                case "/": out.append("/")
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "b": out.append("\u{0008}")
                case "f": out.append("\u{000C}")
                case "u":
                    guard let scalar = readHexScalar() else { return nil }
                    out.unicodeScalars.append(scalar)
                default:
                    return nil
                }
            } else if char == "\n" || char == "\r" {
                return nil
            } else {
                out.append(char)
            }
        }
        return nil
    }

    mutating func readHexScalar() -> Unicode.Scalar? {
        guard index + 4 <= chars.count else { return nil }
        let hex = String(chars[index..<(index + 4)])
        index += 4
        guard let value = UInt32(hex, radix: 16), let scalar = Unicode.Scalar(value) else { return nil }
        return scalar
    }

    mutating func parseNumber() -> FlintOrderedJSON? {
        let begin = index
        if index < chars.count, chars[index] == "-" { index += 1 }
        guard index < chars.count, isDigit(chars[index]) else { return nil }
        if chars[index] == "0" {
            index += 1
        } else {
            while index < chars.count, isDigit(chars[index]) { index += 1 }
        }
        if index < chars.count, chars[index] == "." {
            index += 1
            guard index < chars.count, isDigit(chars[index]) else { return nil }
            while index < chars.count, isDigit(chars[index]) { index += 1 }
        }
        if index < chars.count, chars[index] == "e" || chars[index] == "E" {
            index += 1
            if index < chars.count, chars[index] == "+" || chars[index] == "-" { index += 1 }
            guard index < chars.count, isDigit(chars[index]) else { return nil }
            while index < chars.count, isDigit(chars[index]) { index += 1 }
        }
        return .number(String(chars[begin..<index]))
    }

    mutating func consume(_ word: String) -> Bool {
        for char in word {
            guard index < chars.count, chars[index] == char else { return false }
            index += 1
        }
        return true
    }

    mutating func skipWS() {
        while index < chars.count {
            let char = chars[index]
            if char == " " || char == "\n" || char == "\t" || char == "\r" {
                index += 1
            } else {
                break
            }
        }
    }

    func isDigit(_ char: Character) -> Bool { char >= "0" && char <= "9" }
}
