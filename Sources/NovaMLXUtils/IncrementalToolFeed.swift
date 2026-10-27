import Foundation
import NovaMLXCore

/// Streaming tool-call events. `started` fires once the name is known.
/// `arguments` pieces concatenate to the `finished` arguments string.
/// OpenAI clients that append `delta.tool_calls[].function.arguments`
/// therefore see the same JSON a non-stream response would return.
public final class IncrementalToolFeed: @unchecked Sendable {
    public enum Event: Equatable, Sendable {
        case text(String)
        case started(id: String, name: String)
        case arguments(id: String, delta: String)
        case finished(id: String, name: String, arguments: String)
    }

    public struct Output: Equatable, Sendable {
        public var events: [Event]
        /// The stock whole-call processor must ignore this chunk.
        /// Text before the call is already in `events`.
        public var ownsChunk: Bool
        /// Markup this feed is not parsing. Empty when `ownsChunk` is true.
        public var processorText: String

        public init(events: [Event] = [], ownsChunk: Bool = false, processorText: String = "") {
            self.events = events
            self.ownsChunk = ownsChunk
            self.processorText = processorText
        }
    }

    private final class Counter: @unchecked Sendable {
        var n = 0
    }

    private let nextID: @Sendable () -> String
    private var scan = ""
    private var held = ""
    private var phase: Phase = .scan
    private var callID: String?
    private var callName = ""
    private var emittedArguments = ""
    private var gaveUp = false

    private enum Phase {
        case scan
        case watch
        case json
        case xml
        case flint
    }

    public init(nextID: (@Sendable () -> String)? = nil) {
        if let nextID {
            self.nextID = nextID
        } else {
            let counter = Counter()
            self.nextID = {
                counter.n += 1
                return String(format: "call_%04x", counter.n)
            }
        }
    }

    public func feed(_ chunk: String) -> Output {
        guard !chunk.isEmpty else { return Output() }
        switch phase {
        case .scan:
            scan += chunk
            return drainScan()
        case .watch, .json, .xml, .flint:
            held += chunk
            return drainHeld()
        }
    }

    public func finish() -> Output {
        switch phase {
        case .scan:
            let text = scan
            scan = ""
            return Output(processorText: text)
        case .watch:
            let raw = "<tool_call>" + held
            held = ""
            phase = .scan
            return Output(processorText: raw)
        case .json, .xml, .flint:
            let body = held
            held = ""
            let events = finishBody(body)
            resetCall()
            phase = .scan
            return Output(events: events, ownsChunk: true)
        }
    }

    private func drainScan() -> Output {
        if let range = scan.range(of: "<tool_call>") {
            let before = String(scan[..<range.lowerBound])
            held = String(scan[range.upperBound...])
            scan = ""
            phase = .watch
            let heldOut = drainHeld()
            if heldOut.ownsChunk {
                var events: [Event] = []
                if !before.isEmpty { events.append(.text(before)) }
                events.append(contentsOf: heldOut.events)
                return Output(events: events, ownsChunk: true)
            }
            return Output(processorText: before + heldOut.processorText)
        }
        let hold = Self.prefixHold(scan, marker: "<tool_call>")
        let emitCount = scan.count - hold
        let emit = String(scan.prefix(emitCount))
        scan = String(scan.suffix(hold))
        return Output(processorText: emit)
    }

    private func drainHeld() -> Output {
        if phase == .watch {
            if let decided = classify(held) {
                switch decided {
                case .abort:
                    let raw = "<tool_call>" + held
                    held = ""
                    phase = .scan
                    return Output(processorText: raw)
                case .json:
                    phase = .json
                case .xml:
                    phase = .xml
                case .flint:
                    phase = .flint
                }
            } else if held.contains("</tool_call>") {
                let raw = "<tool_call>" + held
                held = ""
                phase = .scan
                return Output(processorText: raw)
            } else {
                return Output()
            }
        }
        return parseOwned()
    }

    private enum Decision {
        case abort
        case json
        case xml
        case flint
    }

    private func classify(_ body: String) -> Decision? {
        let trimmed = body.drop(while: { $0.isWhitespace })
        if trimmed.isEmpty { return nil }
        if trimmed.hasPrefix("{") { return .json }
        if trimmed.hasPrefix("<function=") { return .xml }
        if trimmed.hasPrefix("<") {
            if "<function=".hasPrefix(trimmed) { return nil }
            return .abort
        }
        if Self.flintReady(String(trimmed)) { return .flint }
        if Self.flintPossible(String(trimmed)) { return nil }
        return .abort
    }

    private func parseOwned() -> Output {
        guard let range = held.range(of: "</tool_call>") else {
            let events = progressBody(held, closing: false)
            return Output(events: events, ownsChunk: true)
        }
        let body = String(held[..<range.lowerBound])
        let tail = String(held[range.upperBound...])
        held = ""
        var events = finishBody(body)
        resetCall()
        phase = .scan
        if !tail.isEmpty {
            let rest = feed(tail)
            if rest.ownsChunk {
                events.append(contentsOf: rest.events)
            } else if !rest.processorText.isEmpty {
                events.append(.text(rest.processorText))
            }
        }
        return Output(events: events, ownsChunk: true)
    }

    private func finishBody(_ body: String) -> [Event] {
        let events = progressBody(body, closing: true)
        if let id = callID {
            if events.contains(where: {
                if case .finished = $0 { return true }
                return false
            }) {
                return events
            }
            let final = emittedArguments.isEmpty ? "{}" : emittedArguments
            return events + [.finished(id: id, name: callName, arguments: final)]
        }
        return [.text("<tool_call>" + body + "</tool_call>")]
    }

    private func progressBody(_ body: String, closing: Bool) -> [Event] {
        if gaveUp { return [] }
        let scan: FormatScan
        switch phase {
        case .json:
            scan = JSONToolScanner.read(body)
        case .xml:
            scan = XMLToolScanner.read(body, closing: closing)
        case .flint:
            scan = FlintShapedScanner.read(body, closing: closing)
        default:
            return []
        }
        if scan.failed && callID == nil {
            gaveUp = true
            return []
        }
        return diff(scan, closing: closing)
    }

    private func diff(_ scan: FormatScan, closing: Bool) -> [Event] {
        var events: [Event] = []
        if let name = scan.name, callID == nil {
            let id = nextID()
            callID = id
            callName = name
            events.append(.started(id: id, name: name))
        }
        guard let id = callID else { return events }
        let arguments = scan.arguments
        if arguments.isEmpty || arguments.hasPrefix(emittedArguments) {
            if arguments.count > emittedArguments.count {
                let delta = String(arguments.dropFirst(emittedArguments.count))
                events.append(.arguments(id: id, delta: delta))
                emittedArguments = arguments
            }
        } else {
            gaveUp = true
            return events
        }
        if closing {
            let final = emittedArguments.isEmpty ? "{}" : emittedArguments
            events.append(.finished(id: id, name: callName, arguments: final))
        }
        return events
    }

    private func resetCall() {
        callID = nil
        callName = ""
        emittedArguments = ""
        gaveUp = false
    }

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

    private static func flintReady(_ text: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: #"\A([^<\s]+)\s*<arg_key>"#) else { return false }
        let range = NSRange(text.startIndex..., in: text)
        return regex.firstMatch(in: text, range: range) != nil
    }

    private static func flintPossible(_ text: String) -> Bool {
        var index = text.startIndex
        while index < text.endIndex, !text[index].isWhitespace, text[index] != "<" {
            index = text.index(after: index)
        }
        if index == text.startIndex { return false }
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        if index == text.endIndex { return true }
        let tail = text[index...]
        if tail.hasPrefix("<arg_key>") { return true }
        return "<arg_key>".hasPrefix(tail)
    }
}

struct FormatScan: Equatable {
    var name: String?
    var arguments: String = ""
    var failed = false
}

enum JSONToolScanner {
    static func read(_ raw: String) -> FormatScan {
        var scan = FormatScan()
        var sawArguments = false
        var i = raw.startIndex
        func skip() {
            while i < raw.endIndex, raw[i].isWhitespace { i = raw.index(after: i) }
        }
        skip()
        guard i < raw.endIndex else { return scan }
        guard raw[i] == "{" else {
            scan.failed = true
            return scan
        }
        i = raw.index(after: i)
        skip()
        if i >= raw.endIndex { return scan }
        if raw[i] == "}" {
            if scan.name != nil && !sawArguments { scan.arguments = "{}" }
            return scan
        }
        while i < raw.endIndex {
            skip()
            if i >= raw.endIndex { return scan }
            if raw[i] == "}" {
                if scan.name != nil && !sawArguments { scan.arguments = "{}" }
                return scan
            }
            guard raw[i] == "\"" else {
                scan.failed = true
                return scan
            }
            let key = readString(raw, &i)
            if key.failed {
                scan.failed = true
                return scan
            }
            if !key.complete { return scan }
            skip()
            if i >= raw.endIndex { return scan }
            guard raw[i] == ":" else {
                scan.failed = true
                return scan
            }
            i = raw.index(after: i)
            skip()
            if i >= raw.endIndex { return scan }
            if key.text == "name" {
                guard raw[i] == "\"" else {
                    scan.failed = true
                    return scan
                }
                let name = readString(raw, &i)
                if name.failed {
                    scan.failed = true
                    return scan
                }
                if !name.complete { return scan }
                scan.name = name.text
            } else if key.text == "arguments" {
                sawArguments = true
                if !readArguments(raw, &i, into: &scan) { return scan }
            } else if !skipValue(raw, &i) {
                return scan
            }
            skip()
            if i >= raw.endIndex { return scan }
            if raw[i] == "," {
                i = raw.index(after: i)
                continue
            }
            if raw[i] == "}" {
                if scan.name != nil && !sawArguments { scan.arguments = "{}" }
                return scan
            }
            scan.failed = true
            return scan
        }
        return scan
    }

    private static func readArguments(
        _ raw: String,
        _ i: inout String.Index,
        into scan: inout FormatScan
    ) -> Bool {
        guard i < raw.endIndex else { return false }
        let c = raw[i]
        if c == "\"" {
            let value = readString(raw, &i)
            if value.failed {
                scan.failed = true
                return false
            }
            scan.arguments = value.text
            return value.complete
        }
        if c == "{" || c == "[" {
            let start = i
            if consumeContainer(raw, &i) {
                scan.arguments = String(raw[start..<i])
                return true
            }
            scan.arguments = String(raw[start..<raw.endIndex])
            return false
        }
        let start = i
        while i < raw.endIndex {
            let ch = raw[i]
            if ch == "," || ch == "}" || ch == "]" || ch.isWhitespace { break }
            i = raw.index(after: i)
        }
        if i == start {
            scan.failed = true
            return false
        }
        scan.arguments = String(raw[start..<i])
        return i < raw.endIndex
    }

    private static func skipValue(_ raw: String, _ i: inout String.Index) -> Bool {
        guard i < raw.endIndex else { return false }
        let c = raw[i]
        if c == "\"" {
            let value = readString(raw, &i)
            return value.complete && !value.failed
        }
        if c == "{" || c == "[" {
            return consumeContainer(raw, &i)
        }
        let start = i
        while i < raw.endIndex {
            let ch = raw[i]
            if ch == "," || ch == "}" || ch == "]" || ch.isWhitespace { break }
            i = raw.index(after: i)
        }
        return i > start && i < raw.endIndex
    }

    private static func consumeContainer(_ raw: String, _ i: inout String.Index) -> Bool {
        guard i < raw.endIndex else { return false }
        var depth = 0
        var inString = false
        var escape = false
        var j = i
        while j < raw.endIndex {
            let c = raw[j]
            if inString {
                if escape {
                    escape = false
                } else if c == "\\" {
                    escape = true
                } else if c == "\"" {
                    inString = false
                }
                j = raw.index(after: j)
                continue
            }
            if c == "\"" {
                inString = true
            } else if c == "{" || c == "[" {
                depth += 1
            } else if c == "}" || c == "]" {
                depth -= 1
                if depth == 0 {
                    i = raw.index(after: j)
                    return true
                }
            }
            j = raw.index(after: j)
        }
        return false
    }

    private struct StringRead {
        var text: String
        var complete: Bool
        var failed: Bool
    }

    private static func readString(_ raw: String, _ i: inout String.Index) -> StringRead {
        guard i < raw.endIndex, raw[i] == "\"" else {
            return StringRead(text: "", complete: false, failed: i < raw.endIndex)
        }
        var j = raw.index(after: i)
        var out = ""
        var escape = false
        var hex = ""
        var readingHex = false
        while j < raw.endIndex {
            let c = raw[j]
            if readingHex {
                hex.append(c)
                if hex.count == 4 {
                    if let value = UInt32(hex, radix: 16), let scalar = Unicode.Scalar(value) {
                        out.append(Character(scalar))
                    }
                    readingHex = false
                    hex = ""
                }
                j = raw.index(after: j)
                continue
            }
            if escape {
                switch c {
                case "\"", "\\", "/": out.append(c)
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "u":
                    readingHex = true
                    hex = ""
                default:
                    out.append(c)
                }
                escape = false
                j = raw.index(after: j)
                continue
            }
            if c == "\\" {
                escape = true
                j = raw.index(after: j)
                continue
            }
            if c == "\"" {
                i = raw.index(after: j)
                return StringRead(text: out, complete: true, failed: false)
            }
            out.append(c)
            j = raw.index(after: j)
        }
        return StringRead(text: out, complete: false, failed: false)
    }
}

enum XMLToolScanner {
    static func read(_ raw: String, closing: Bool) -> FormatScan {
        shaped(
            raw,
            closing: closing,
            namePattern: #"\A\s*<function=([^>]*)>"#,
            keyTag: "<parameter=",
            keyClose: "</parameter>"
        )
    }
}

enum FlintShapedScanner {
    static func read(_ raw: String, closing: Bool) -> FormatScan {
        var scan = FormatScan()
        guard let regex = try? NSRegularExpression(pattern: #"\A\s*([^<\s]+)\s*"#) else {
            scan.failed = true
            return scan
        }
        let range = NSRange(raw.startIndex..., in: raw)
        guard let match = regex.firstMatch(in: raw, range: range),
              let nameRange = Range(match.range(at: 1), in: raw)
        else { return scan }
        let name = String(raw[nameRange])
        guard !name.isEmpty else { return scan }
        let restStart = raw.index(raw.startIndex, offsetBy: match.range.length)
        let rest = String(raw[restStart...])
        if !rest.hasPrefix("<arg_key>") && !"<arg_key>".hasPrefix(rest) && !rest.isEmpty {
            if rest.contains("<") && !rest.hasPrefix("<arg_key>") {
                scan.failed = true
            }
            return scan
        }
        if !rest.hasPrefix("<arg_key>") { return scan }
        scan.name = name
        scan.arguments = parameterJSON(rest, keyTag: "<arg_key>", keyClose: "</arg_key>", valueTag: "<arg_value>", valueClose: "</arg_value>", closing: closing)
        return scan
    }
}

private func shaped(
    _ raw: String,
    closing: Bool,
    namePattern: String,
    keyTag: String,
    keyClose: String
) -> FormatScan {
    var scan = FormatScan()
    guard let regex = try? NSRegularExpression(pattern: namePattern) else {
        scan.failed = true
        return scan
    }
    let range = NSRange(raw.startIndex..., in: raw)
    guard let match = regex.firstMatch(in: raw, range: range),
          let nameRange = Range(match.range(at: 1), in: raw)
    else {
        let trimmed = raw.drop(while: { $0.isWhitespace })
        if !trimmed.isEmpty && !"<function=".hasPrefix(trimmed) && !trimmed.hasPrefix("<function=") {
            scan.failed = true
        }
        return scan
    }
    let name = String(raw[nameRange])
    guard !name.isEmpty else {
        scan.failed = true
        return scan
    }
    scan.name = name
    let restStart = raw.index(raw.startIndex, offsetBy: match.range.length)
    let rest = String(raw[restStart...])
    scan.arguments = parameterJSON(
        rest,
        keyTag: keyTag,
        keyClose: keyClose,
        valueTag: ">",
        valueClose: keyClose,
        closing: closing,
        xmlValue: true
    )
    return scan
}

/// Encounter-order JSON object. String values stream inside the quotes.
/// `closing` appends `}` once the call ends.
private func parameterJSON(
    _ raw: String,
    keyTag: String,
    keyClose: String,
    valueTag: String,
    valueClose: String,
    closing: Bool,
    xmlValue: Bool = false
) -> String {
    var json = ""
    var opened = false
    var cursor = raw.startIndex
    func append(_ piece: String) { json += piece }
    func openKey(_ key: String) {
        if !opened {
            append("{")
            opened = true
        } else {
            append(",")
        }
        append("\"" + JSONStringEscape.body(key) + "\":")
    }
    while cursor < raw.endIndex {
        let rest = raw[cursor...]
        let lead = rest.drop(while: { $0.isWhitespace })
        if lead.isEmpty { break }
        let skipped = raw.distance(from: cursor, to: lead.startIndex)
        if xmlValue {
            if lead.hasPrefix("</function>") || lead.hasPrefix("</tool_call>") { break }
            guard lead.hasPrefix(keyTag) else { break }
            cursor = raw.index(cursor, offsetBy: skipped)
            let afterTag = raw.index(cursor, offsetBy: keyTag.count)
            guard afterTag <= raw.endIndex else { break }
            let tail = raw[afterTag...]
            guard let gt = tail.firstIndex(of: ">") else { break }
            let key = String(tail[..<gt])
            let valueStart = raw.index(after: gt)
            let valueTail = raw[valueStart...]
            if let close = valueTail.range(of: valueClose) {
                openKey(key)
                append("\"" + JSONStringEscape.body(String(valueTail[..<close.lowerBound])) + "\"")
                cursor = close.upperBound
                continue
            }
            let hold = IncrementalToolFeed.prefixHold(String(valueTail), marker: valueClose)
            let take = valueTail.count - hold
            openKey(key)
            append("\"")
            if take > 0 {
                let end = valueTail.index(valueTail.startIndex, offsetBy: take)
                append(JSONStringEscape.body(String(valueTail[..<end])))
            }
            return json
        }
        guard lead.hasPrefix(keyTag) else { break }
        cursor = raw.index(cursor, offsetBy: skipped)
        let afterKey = raw.index(cursor, offsetBy: keyTag.count)
        guard let keyEnd = raw[afterKey...].range(of: keyClose) else { break }
        let key = String(raw[afterKey..<keyEnd.lowerBound])
        cursor = keyEnd.upperBound
        let afterKeyClose = raw[cursor...].drop(while: { $0.isWhitespace })
        guard afterKeyClose.hasPrefix(valueTag) else { break }
        cursor = raw.index(afterKeyClose.startIndex, offsetBy: valueTag.count)
        let valueTail = raw[cursor...]
        if let close = valueTail.range(of: valueClose) {
            openKey(key)
            append("\"" + JSONStringEscape.body(String(valueTail[..<close.lowerBound])) + "\"")
            cursor = close.upperBound
            continue
        }
        let hold = IncrementalToolFeed.prefixHold(String(valueTail), marker: valueClose)
        let take = valueTail.count - hold
        openKey(key)
        append("\"")
        if take > 0 {
            let end = valueTail.index(valueTail.startIndex, offsetBy: take)
            append(JSONStringEscape.body(String(valueTail[..<end])))
        }
        return json
    }
    if closing {
        if opened { append("}") }
        else { json = "{}" }
    }
    return json
}

/// OpenAI tool-delta shape. Nil fields are omitted by the API encoder.
public enum ToolCallWire {
    public struct Delta: Equatable, Sendable {
        public var index: Int
        public var id: String?
        public var type: String?
        public var name: String?
        public var arguments: String?

        public init(
            index: Int,
            id: String? = nil,
            type: String? = nil,
            name: String? = nil,
            arguments: String? = nil
        ) {
            self.index = index
            self.id = id
            self.type = type
            self.name = name
            self.arguments = arguments
        }
    }

    /// First chunk: index, id, type, name. Arguments stay absent.
    public static func openAIStarted(index: Int, id: String, name: String) -> Delta {
        Delta(index: index, id: id, type: "function", name: name, arguments: nil)
    }

    /// Later chunks: index and one arguments fragment.
    public static func openAIArguments(index: Int, id: String, arguments: String) -> Delta {
        Delta(index: index, id: nil, type: nil, name: nil, arguments: arguments)
    }

    /// One chunk carrying the finished call. Legacy whole-call streams use this.
    public static func openAIWholeCall(index: Int, id: String, name: String, arguments: String) -> Delta {
        Delta(index: index, id: id, type: "function", name: name, arguments: arguments)
    }
}

/// Responses function-call plan. Incremental ids emit `added`, argument
/// deltas, then `done` whose arguments equal the concatenation. A call that
/// arrives already finished emits one arguments delta of the full JSON.
public struct ResponsesToolPlan: Sendable {
    public enum Step: Equatable, Sendable {
        case added(callID: String, name: String)
        case argumentDelta(callID: String, delta: String)
        case done(callID: String, name: String, arguments: String)
    }

    private var names: [String: String] = [:]
    private var started: Set<String> = []
    private var finished: Set<String> = []

    public init() {}

    public mutating func observeDelta(
        id: String,
        kind: ToolCallDelta.Kind,
        name: String?,
        arguments: String?
    ) -> [Step] {
        switch kind {
        case .started:
            let resolved = name ?? ""
            names[id] = resolved
            started.insert(id)
            return [.added(callID: id, name: resolved)]
        case .arguments:
            return [.argumentDelta(callID: id, delta: arguments ?? "")]
        }
    }

    public mutating func observeFinished(id: String, name: String, arguments: String) -> [Step] {
        if finished.contains(id) { return [] }
        finished.insert(id)
        if started.contains(id) {
            return [.done(callID: id, name: names[id] ?? name, arguments: arguments)]
        }
        return [
            .added(callID: id, name: name),
            .argumentDelta(callID: id, delta: arguments),
            .done(callID: id, name: name, arguments: arguments),
        ]
    }
}
