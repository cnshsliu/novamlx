import Foundation

/// One piece of a prompt whose specials are known at render time.
/// Plain regions are encoded on their own, so a special spelling inside
/// user text is not turned into a special id.
public enum PromptSegment: Equatable, Sendable {
    case plain(String)
    case special(String)

    public var text: String {
        switch self {
        case .plain(let text), .special(let text):
            return text
        }
    }
}

/// Encode a prompt in segments. Specials become one id from `specialID`.
/// Plain regions go through `plainEncode` and are not scanned for specials.
public enum SegmentedPromptEncoder {
    public struct EncodeError: Error, Equatable, Sendable {
        public let message: String

        public init(_ message: String) {
            self.message = message
        }
    }

    public static func encode(
        _ segments: [PromptSegment],
        plainEncode: (String) -> [Int],
        specialID: (String) -> Int?
    ) -> Result<[Int], EncodeError> {
        var ids: [Int] = []
        for segment in segments {
            switch segment {
            case .plain(let text):
                if text.isEmpty { continue }
                ids.append(contentsOf: plainEncode(text))
            case .special(let token):
                guard let id = specialID(token) else {
                    return .failure(EncodeError("missing special token id for \(token)"))
                }
                ids.append(id)
            }
        }
        return .success(ids)
    }

    /// Split `text` on UTF-8 byte ranges. Ranges must be ascending,
    /// non-overlapping, and land on Unicode scalar boundaries.
    /// Bytes outside the ranges stay plain, including text that spells a special.
    public static func segments(
        text: String,
        specialUTF8Ranges: [Range<Int>]
    ) -> Result<[PromptSegment], EncodeError> {
        let utf8 = Array(text.utf8)
        var out: [PromptSegment] = []
        var cursor = 0
        for range in specialUTF8Ranges {
            if range.lowerBound < cursor || range.upperBound > utf8.count || range.lowerBound > range.upperBound {
                return .failure(EncodeError("special range is out of order or past the end"))
            }
            if !isScalarBoundary(utf8, index: range.lowerBound)
                || !isScalarBoundary(utf8, index: range.upperBound)
            {
                return .failure(EncodeError("special range splits a Unicode scalar"))
            }
            if range.lowerBound > cursor {
                guard let plain = string(utf8, cursor..<range.lowerBound) else {
                    return .failure(EncodeError("plain region is not valid UTF-8"))
                }
                if !plain.isEmpty { out.append(.plain(plain)) }
            }
            if range.lowerBound < range.upperBound {
                guard let special = string(utf8, range) else {
                    return .failure(EncodeError("special region is not valid UTF-8"))
                }
                out.append(.special(special))
            }
            cursor = range.upperBound
        }
        if cursor < utf8.count {
            guard let plain = string(utf8, cursor..<utf8.count) else {
                return .failure(EncodeError("trailing region is not valid UTF-8"))
            }
            if !plain.isEmpty { out.append(.plain(plain)) }
        }
        return .success(out)
    }

    private static func isScalarBoundary(_ utf8: [UInt8], index: Int) -> Bool {
        if index == 0 || index == utf8.count { return true }
        return utf8[index] & 0b1100_0000 != 0b1000_0000
    }

    private static func string(_ utf8: [UInt8], _ range: Range<Int>) -> String? {
        String(decoding: utf8[range], as: UTF8.self)
    }
}

/// Role markers for an untemplated prompt. Returns nil unless every
/// non-empty message has its role special in the vocabulary. The caller
/// then keeps the one-shot join, so a missing marker does not change ids.
public enum RecipePrompt {
    public static func roleToken(_ role: String) -> String? {
        switch role {
        case "system": return "<|system|>"
        case "developer": return "<|developer|>"
        case "user": return "<|user|>"
        case "assistant": return "<|assistant|>"
        case "tool": return "<|observation|>"
        default: return nil
        }
    }

    public static func segments(
        messages: [(role: String, content: String)],
        specialID: (String) -> Int?
    ) -> [PromptSegment]? {
        var out: [PromptSegment] = []
        for message in messages {
            if message.content.isEmpty { continue }
            guard let token = roleToken(message.role), specialID(token) != nil else {
                return nil
            }
            out.append(.special(token))
            out.append(.plain(message.content))
        }
        return out
    }
}

/// JSON string body (no quotes) matching `JSONSerialization`: escape
/// `\`, `"`, and the C0 controls it writes as short escapes. Non-ASCII
/// scalars are left as themselves, so per-scalar pieces concatenate to
/// the escape of the whole string.
public enum JSONStringEscape {
    public static func body(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.utf8.count)
        for unit in text.unicodeScalars {
            out += scalar(unit)
        }
        return out
    }

    public static func scalar(_ scalar: Unicode.Scalar) -> String {
        switch scalar {
        case "\\": return "\\\\"
        case "\"": return "\\\""
        case "\n": return "\\n"
        case "\r": return "\\r"
        case "\t": return "\\t"
        case "/": return "/"
        default:
            if scalar.value < 0x20 {
                return String(format: "\\u%04x", scalar.value)
            }
            return String(scalar)
        }
    }
}
