import Foundation
import Testing

@testable import NovaMLXEngine

@Suite("Flint session prefix")
struct FlintPrefixTests {

    @Test("select_prefix rewinds to the checkpoint at or before the mismatch")
    func testOfficialRewind() {
        let cache = FlintPrefixEngine.Cache(
            tokens: Array(0..<13),
            checkpoints: [
                .init(position: 8),
                .init(position: 11),
            ])
        let selected = FlintPrefixEngine.select(
            prefix: cache,
            tokens: Array(0..<9) + [100, 101],
            trimmable: false)
        #expect(selected.common == 9)
        #expect(selected.cached == 8)
        #expect(selected.cache?.tokens == Array(0..<8))
        #expect(FlintPrefixEngine.prefillLog(prompt: 11, common: 9, reused: 8)
            == "Prefill: prompt=11 common_prefix=9 reused=8 new=3")
    }

    @Test("checkpoints stay at two and an early edit drops the prefix")
    func testCheckpointBound() {
        var cache = FlintPrefixEngine.Cache()
        for value in 0..<12 {
            cache.checkpoint()
            cache.tokens.append(value)
            #expect(cache.checkpoints.count <= 2)
        }
        let exact = FlintPrefixEngine.select(
            prefix: cache, tokens: Array(0..<13), trimmable: false)
        #expect(exact.common == 12)
        #expect(exact.cached == 12)
        let shortened = FlintPrefixEngine.select(
            prefix: cache, tokens: Array(0..<10), trimmable: false)
        #expect(shortened.common == 10)
        #expect(shortened.cached == 10)
        let miss = FlintPrefixEngine.select(
            prefix: shortened.cache, tokens: [99], trimmable: false)
        #expect(miss.cache == nil)
        #expect(miss.cached == 0)
    }

    @Test("a trimmable shared prefix reports cached tokens without a checkpoint")
    func testSharedPrefix() {
        var session = FlintPrefixSession()
        let first = [1, 2, 3, 4, 5]
        session.commit(tokens: first)
        let second = [1, 2, 3, 9, 9]
        let taken = session.takePrefix(ids: second, trimmable: true)
        let selected = FlintPrefixEngine.select(prefix: taken, tokens: second, trimmable: true)
        #expect(selected.common == 3)
        #expect(selected.cached > 0)
        #expect(selected.cached == 3)

        let official = FlintPrefixEngine.select(prefix: taken, tokens: second, trimmable: false)
        #expect(official.cached == 0)

        var extended = FlintPrefixSession()
        extended.commit(tokens: [1, 2, 3])
        let follow = [1, 2, 3, 4]
        let whole = extended.takePrefix(ids: follow, trimmable: false)
        let hit = FlintPrefixEngine.select(prefix: whole, tokens: follow, trimmable: false)
        #expect(hit.common == 3)
        #expect(hit.cached == 3)
        #expect(extended.chunkSize == FlintPrefixEngine.productionChunkSize)
    }

    @Test("prepare_prefix keeps full chunks and a cancel leaves the anchor")
    func testPreparePrefix() throws {
        let origin = [10, 11]
        let generated = Array(0..<13)
        var session = FlintPrefixSession(chunkSize: 4)
        session.promptCache = FlintPrefixEngine.Cache(tokens: origin)
        session.schedulePrefix(ids: origin, generated: generated)
        var calls: [[Int]] = []
        let metrics = session.preparePrefix(shouldStop: { false }) { prefix, target in
            let start = prefix.tokens.count
            let next = try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: origin + generated, cancel: false)
            calls.append(Array(next.tokens[start..<target]))
            return next
        }
        #expect(calls == [Array(0..<4), Array(4..<8), Array(8..<12)])
        #expect(session.preparation?.cache?.tokens == origin + Array(0..<12))
        #expect(session.promptCache?.tokens == origin)
        #expect(metrics?.preparedTokens == 12)

        var short = FlintPrefixSession(chunkSize: 4)
        short.promptCache = FlintPrefixEngine.Cache(tokens: origin)
        short.schedulePrefix(ids: origin, generated: [1, 2, 3])
        #expect(short.preparePrefix(shouldStop: { false }) { prefix, _ in prefix } == nil)

        var stopped = FlintPrefixSession(chunkSize: 4)
        stopped.promptCache = FlintPrefixEngine.Cache(tokens: origin)
        stopped.schedulePrefix(ids: origin, generated: Array(0..<10))
        let none = stopped.preparePrefix(shouldStop: { true }) { prefix, target in
            try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: origin + Array(0..<10), cancel: false)
        }
        #expect(none?.preparedTokens == 0)
        #expect(stopped.preparation?.cache == nil)
        #expect(stopped.promptCache?.tokens == origin)

        var checks = [false, true].makeIterator()
        let partial = stopped.preparePrefix(shouldStop: { checks.next() ?? true }) { prefix, target in
            try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: origin + Array(0..<10), cancel: false)
        }
        #expect(partial?.preparedTokens == 4)
        #expect(stopped.preparation?.cache?.tokens == origin + Array(0..<4))
        let resumed = stopped.preparePrefix(shouldStop: { false }) { prefix, target in
            try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: origin + Array(0..<10), cancel: false)
        }
        #expect(resumed?.preparedTokens == 4)
        #expect(stopped.promptCache?.tokens == origin)

        var cancelled = FlintPrefixSession(chunkSize: 4)
        cancelled.promptCache = FlintPrefixEngine.Cache(tokens: origin)
        cancelled.schedulePrefix(ids: origin, generated: Array(0..<10))
        var steps = 0
        _ = cancelled.preparePrefix(shouldStop: { false }) { prefix, target in
            steps += 1
            if steps == 2 { throw FlintPrefixCancellation() }
            return try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: origin + Array(0..<10), cancel: false)
        }
        #expect(cancelled.preparation?.cache?.tokens == origin + Array(0..<4))
        #expect(cancelled.promptCache?.tokens == origin)
    }

    @Test("only a matching prepared prefix replaces the anchor")
    func testTakePrefix() throws {
        func prepared(changed: Bool) throws -> FlintPrefixEngine.Cache? {
            let origin = [10, 11]
            let generated = Array(0..<10)
            var session = FlintPrefixSession(chunkSize: 4)
            session.promptCache = FlintPrefixEngine.Cache(tokens: origin)
            session.schedulePrefix(ids: origin, generated: generated)
            _ = session.preparePrefix(shouldStop: { false }) { prefix, target in
                try FlintPrefixSession.advanceChunk(
                    prefix: prefix, target: target, tokens: origin + generated, cancel: false)
            }
            var follow = origin + generated + [99]
            if changed { follow[3] = -1 }
            return session.takePrefix(ids: follow, trimmable: false)
        }
        #expect(try prepared(changed: false)?.tokens == [10, 11] + Array(0..<8))
        #expect(try prepared(changed: true)?.tokens == [10, 11])

        var edited = FlintPrefixSession(chunkSize: 4)
        let origin = [10, 11]
        let generated = Array(0..<14)
        edited.promptCache = FlintPrefixEngine.Cache(tokens: origin)
        edited.schedulePrefix(ids: origin, generated: generated)
        _ = edited.preparePrefix(shouldStop: { false }) { prefix, target in
            try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: origin + generated, cancel: false)
        }
        var follow = origin + generated + [99]
        follow[origin.count + 10] = -1
        let selected = edited.takePrefix(ids: follow, trimmable: false)
        #expect(selected?.tokens == origin + Array(generated.prefix(8)))
        #expect(edited.promptCache == nil)
    }

    @Test("a cancelled prefill restores the original anchor")
    func testPrefillCancelled() throws {
        var session = FlintPrefixSession(chunkSize: 4)
        let originTokens = [10, 11]
        let generated = Array(0..<14)
        session.promptCache = FlintPrefixEngine.Cache(tokens: originTokens)
        session.schedulePrefix(ids: originTokens, generated: generated)
        _ = session.preparePrefix(shouldStop: { false }) { prefix, target in
            try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: originTokens + generated, cancel: false)
        }
        let pending = session.preparation
        let prepared = pending?.cache
        let follow = originTokens + generated + [99]
        let taken = session.takePrefix(ids: follow, trimmable: false)
        #expect(taken?.tokens == prepared?.tokens)
        #expect(session.promptCache == nil)
        session.prefillCancelled(
            pending: pending, prefixWasPrepared: true, sessionCache: prepared ?? FlintPrefixEngine.Cache())
        #expect(session.promptCache?.tokens == originTokens)
        #expect(session.preparation?.cache?.tokens == prepared?.tokens)

        var edited = follow
        edited[3] = -1
        #expect(session.takePrefix(ids: edited, trimmable: false)?.tokens == originTokens)
    }
}
