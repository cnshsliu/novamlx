import Testing
import Foundation
@testable import TknetPeer

@Suite("Relay usage line-split (P1.4)")
struct RelayUsageSplitTests {
    @Test("usage object split across two body reads still counts")
    func splitUsage() {
        let first = Relay.consumeUsage(
            incoming: Data(#"data: {"usage":{"prompt_tokens":"#.utf8),
            remainder: "",
            previous: nil)
        #expect(first.usage == nil)
        #expect(first.remainder == "data: {\"usage\":{\"prompt_tokens\":")

        let second = Relay.consumeUsage(
            incoming: Data("5,\"completion_tokens\":2}}\n".utf8),
            remainder: first.remainder,
            previous: first.usage)
        #expect(second.usage?.prompt == 5)
        #expect(second.usage?.completion == 2)
        #expect(second.remainder == "")
    }

    @Test("complete line in one read still works")
    func singleRead() {
        let r = Relay.consumeUsage(
            incoming: Data("data: {\"id\":1,\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\n".utf8),
            remainder: "",
            previous: nil)
        #expect(r.usage?.prompt == 7)
        #expect(r.usage?.completion == 3)
        #expect(r.remainder == "")
    }

    @Test("split must not double-count when a later complete usage follows")
    func lastWins() {
        let first = Relay.consumeUsage(
            incoming: Data("data: {\"usage\":{\"prompt_tokens\":1,\"com".utf8),
            remainder: "",
            previous: nil)
        let second = Relay.consumeUsage(
            incoming: Data("pletion_tokens\":1}}\ndata: {\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n".utf8),
            remainder: first.remainder,
            previous: first.usage)
        #expect(second.usage?.prompt == 9)
        #expect(second.usage?.completion == 4)
    }

    @Test("trailing partial without newline parses on the final flush")
    func trailingFlush() {
        let first = Relay.consumeUsage(
            incoming: Data("data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}".utf8),
            remainder: "",
            previous: nil)
        #expect(first.usage == nil) // no newline yet — not a complete line
        let tail = Relay.consumeUsage(incoming: Data(), remainder: first.remainder + "\n", previous: first.usage)
        #expect(tail.usage?.prompt == 5)
        #expect(tail.usage?.completion == 2)
    }

    @Test("anthropic message_start and message_delta usage count")
    func anthropicUsage() {
        let start = Relay.consumeUsage(
            incoming: Data("data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":11,\"cache_read_input_tokens\":4,\"output_tokens\":1}}}\n".utf8),
            remainder: "",
            previous: nil)
        #expect(start.usage?.prompt == 11)
        #expect(start.usage?.cached == 4)
        let delta = Relay.consumeUsage(
            incoming: Data("data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":6}}\n".utf8),
            remainder: "",
            previous: start.usage)
        #expect(delta.usage?.prompt == 11)
        #expect(delta.usage?.completion == 6)
        #expect(delta.usage?.cached == 4)
    }
}
