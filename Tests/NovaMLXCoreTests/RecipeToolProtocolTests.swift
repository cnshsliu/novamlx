import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXUtils

@Suite("Recipe tool protocol")
struct RecipeToolProtocolTests {
    @Test("segments encode specials as one id and leave plain text unscanned")
    func testSegmentedEncode() throws {
        let text = "hi<|user|>!"
        let upper = 2 + "<|user|>".utf8.count
        let segments = try SegmentedPromptEncoder.segments(
            text: text, specialUTF8Ranges: [2..<upper]
        ).get()
        #expect(segments == [.plain("hi"), .special("<|user|>"), .plain("!")])
        let ids = try SegmentedPromptEncoder.encode(
            segments,
            plainEncode: { text in text.unicodeScalars.map { 100 + Int($0.value) } },
            specialID: { $0 == "<|user|>" ? 7 : nil }
        ).get()
        #expect(ids == [204, 205, 7, 133])
        let missing = SegmentedPromptEncoder.encode(
            [.special("<|nope|>")],
            plainEncode: { _ in [] },
            specialID: { _ in nil }
        )
        #expect(missing == .failure(.init("missing special token id for <|nope|>")))
    }

    @Test("special ranges must be ordered and on scalar boundaries")
    func testSegmentRangeChecks() throws {
        #expect(throws: SegmentedPromptEncoder.EncodeError.self) {
            try SegmentedPromptEncoder.segments(text: "abcd", specialUTF8Ranges: [0..<2, 1..<3]).get()
        }
        let accent = "é"
        #expect(throws: SegmentedPromptEncoder.EncodeError.self) {
            try SegmentedPromptEncoder.segments(text: accent, specialUTF8Ranges: [0..<1]).get()
        }
        let whole = try SegmentedPromptEncoder.segments(text: accent, specialUTF8Ranges: [0..<2]).get()
        #expect(whole == [.special("é")])
    }

    @Test("untemplated segments require every role special")
    func testRecipePromptSegments() throws {
        let messages = [(role: "user", content: "hi"), (role: "assistant", content: "yo")]
        #expect(RecipePrompt.segments(messages: messages, specialID: { _ in nil }) == nil)
        let known = try #require(RecipePrompt.segments(messages: messages, specialID: { _ in 1 }))
        #expect(known == [
            .special("<|user|>"), .plain("hi"),
            .special("<|assistant|>"), .plain("yo"),
        ])
        let skipped = RecipePrompt.segments(
            messages: [(role: "user", content: ""), (role: "system", content: "s")],
            specialID: { $0 == "<|system|>" ? 3 : nil }
        )
        #expect(skipped == [.special("<|system|>"), .plain("s")])
        let injected = try SegmentedPromptEncoder.encode(
            [.special("<|user|>"), .plain("x<|assistant|>y")],
            plainEncode: { $0.unicodeScalars.map { 100 + Int($0.value) } },
            specialID: { $0 == "<|user|>" ? 5 : $0 == "<|assistant|>" ? 6 : nil }
        ).get()
        #expect(!injected.contains(6))
    }

    @Test("json escapes concatenate across scalar splits")
    func testJSONStringEscape() {
        let text = "a\"b\n\u{0001}é"
        let whole = JSONStringEscape.body(text)
        let parts = text.unicodeScalars.map { JSONStringEscape.scalar($0) }.joined()
        #expect(parts == whole)
        #expect(whole.contains("\\\""))
        #expect(whole.contains("\\n"))
        #expect(whole.contains("\\u0001"))
        #expect(whole.contains("é"))
    }

    @Test("json tool arguments stream before the call closes")
    func testJSONToolStream() {
        let feed = IncrementalToolFeed { "call_json" }
        let idle = feed.feed("hello")
        #expect(idle == .init(processorText: "hello"))
        let held = feed.feed("<tool_")
        #expect(held == .init())
        let name = feed.feed("call>{\"name\":\"wr")
        #expect(name.ownsChunk)
        #expect(name.events.isEmpty)
        let mid = feed.feed("ite\",\"arguments\":{\"pa")
        #expect(mid.events.contains(.started(id: "call_json", name: "write")))
        let end = feed.feed("th\":\"a.py\"}}</tool_call>bye")
        let events = mid.events + end.events
        let deltas = events.compactMap { event -> String? in
            if case .arguments(_, let delta) = event { return delta }
            return nil
        }.joined()
        #expect(deltas == "{\"path\":\"a.py\"}")
        #expect(events.contains(.finished(id: "call_json", name: "write", arguments: deltas)))
        #expect(end.events.contains(.text("bye")))
    }

    @Test("a split json escape stays one arguments string")
    func testJSONEscapeStream() {
        let feed = IncrementalToolFeed { "call_esc" }
        let first = feed.feed("<tool_call>{\"name\":\"write\",\"arguments\":\"a\\")
        let done = feed.feed("nb\"}</tool_call>")
        let joined = (first.events + done.events).compactMap { event -> String? in
            if case .arguments(_, let delta) = event { return delta }
            return nil
        }.joined()
        #expect(joined == "a\nb")
        #expect(done.events.contains(.finished(id: "call_esc", name: "write", arguments: joined)))
    }

    @Test("qwen xml and flint-shaped calls stream name then arguments")
    func testXMLAndFlintShaped() {
        let xml = IncrementalToolFeed { "call_xml" }
        let head = xml.feed("<tool_call><function=write><parameter=path>a")
        #expect(head.ownsChunk)
        #expect(head.events.contains(.started(id: "call_xml", name: "write")))
        let tail = xml.feed(".py</parameter></function></tool_call>")
        let xmlArgs = (head.events + tail.events).compactMap { event -> String? in
            if case .arguments(_, let delta) = event { return delta }
            return nil
        }.joined()
        #expect(xmlArgs == "{\"path\":\"a.py\"}")
        #expect(tail.events.contains(.finished(id: "call_xml", name: "write", arguments: xmlArgs)))

        let flint = IncrementalToolFeed { "call_flint" }
        let call = flint.feed("<tool_call>write<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call>")
        let flintArgs = call.events.compactMap { event -> String? in
            if case .arguments(_, let delta) = event { return delta }
            return nil
        }.joined()
        #expect(flintArgs == "{\"path\":\"a.py\"}")
        #expect(call.events.contains(.started(id: "call_flint", name: "write")))
        #expect(call.events.contains(.finished(id: "call_flint", name: "write", arguments: flintArgs)))
    }

    @Test("an unrecognized tool tag stays on the whole-call processor")
    func testUnrecognizedReleases() {
        let feed = IncrementalToolFeed()
        let released = feed.feed("see <tool_call>hello</tool_call>")
        #expect(!released.ownsChunk)
        #expect(released.processorText == "see <tool_call>hello</tool_call>")
        #expect(released.events.isEmpty)
    }

    @Test("openai deltas keep a whole call in one chunk and stream the rest")
    func testOpenAIWire() {
        let started = ToolCallWire.openAIStarted(index: 0, id: "call_1", name: "write")
        #expect(started.id == "call_1")
        #expect(started.type == "function")
        #expect(started.name == "write")
        #expect(started.arguments == nil)
        let piece = ToolCallWire.openAIArguments(index: 0, id: "call_1", arguments: "{\"a\":")
        #expect(piece.id == nil)
        #expect(piece.name == nil)
        #expect(piece.arguments == "{\"a\":")
        let whole = ToolCallWire.openAIWholeCall(index: 1, id: "call_2", name: "read", arguments: "{}")
        #expect(whole.name == "read")
        #expect(whole.arguments == "{}")
    }

    @Test("responses plan emits deltas then one done, or one delta for a finished call")
    func testResponsesPlan() {
        var plan = ResponsesToolPlan()
        let started = plan.observeDelta(id: "c1", kind: .started, name: "write", arguments: nil)
        #expect(started == [.added(callID: "c1", name: "write")])
        let part = plan.observeDelta(id: "c1", kind: .arguments, name: nil, arguments: "{\"path\":")
        #expect(part == [.argumentDelta(callID: "c1", delta: "{\"path\":")])
        let done = plan.observeFinished(id: "c1", name: "write", arguments: "{\"path\":\"a.py\"}")
        #expect(done == [.done(callID: "c1", name: "write", arguments: "{\"path\":\"a.py\"}")])
        let legacy = plan.observeFinished(id: "c2", name: "read", arguments: "{}")
        #expect(legacy == [
            .added(callID: "c2", name: "read"),
            .argumentDelta(callID: "c2", delta: "{}"),
            .done(callID: "c2", name: "read", arguments: "{}"),
        ])
    }
}
