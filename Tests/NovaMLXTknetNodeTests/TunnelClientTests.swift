import Foundation
import os
import Testing
@testable import NovaMLXTknetNode

@Suite("Tunnel client")
struct TunnelClientTests {
    /// Injected `delay` replacement: records every requested duration, then
    /// sleeps a small fixed slice so heartbeat loops don't spin hot. Recorded
    /// values (not wall time) are what assertions inspect.
    private final class DelayRecorder: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: [Double]())
        private let paceSeconds: Double
        init(paceSeconds: Double = 0.02) { self.paceSeconds = paceSeconds }

        func record(thenSleep seconds: Double) async throws {
            state.withLock { $0.append(seconds) }
            try await Task.sleep(nanoseconds: UInt64(paceSeconds * 1_000_000_000))
        }

        var values: [Double] { state.withLock { $0 } }

        /// Backoff delays only: heartbeat sleeps are heartbeatInterval-scaled
        /// (27–33 s here), well above any first-attempt backoff.
        var backoffValues: [Double] { values.filter { $0 < 5 } }
    }

    /// Transport factory that fails the first dial, then hands out the pair.
    /// Drives the reconnect/backoff path without real sockets.
    private final class FlakyFactory: @unchecked Sendable {
        private let pair: InMemoryTransportPair
        private let lock = NSLock()
        private var failedOnce = false
        init(pair: InMemoryTransportPair) { self.pair = pair }

        func make() throws -> TunnelTransport {
            lock.lock(); defer { lock.unlock() }
            guard failedOnce else {
                failedOnce = true
                throw TunnelError.connectionClosed
            }
            return pair.nodeSide
        }
    }

    /// Builds a client around an in-memory pair and a relay. The relay is
    /// always shut down (AsyncHTTPClient asserts in debug builds when dropped
    /// without shutdown), even when assertions fail.
    private func withClient(
        pair: InMemoryTransportPair? = nil,
        sourceEndpoint: URL = URL(string: "http://127.0.0.1:1/v1")!,
        transportFactory: (@Sendable () async throws -> TunnelTransport)? = nil,
        _ body: (_ client: TunnelClient, _ server: any TunnelTransport,
                 _ delayLog: DelayRecorder) async throws -> Void
    ) async throws {
        let thePair = pair ?? InMemoryTransportPair()
        let factory: @Sendable () async throws -> TunnelTransport =
            transportFactory ?? { thePair.nodeSide }
        let delayLog = DelayRecorder()
        var config = NodeConfig.defaultConfig()
        config.nodeId = "node-1"
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        config.sources = [SourceConfig(
            id: "s1", name: "x", type: .openaiCompatible,
            endpoint: sourceEndpoint, apiKeyRef: "s1", upstreamModel: "u")]
        let relay = Relay(config: config, secrets: FileSecretStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("tknet-tc-\(UUID().uuidString)")))
        let client = TunnelClient(
            config: config, relay: relay, transportFactory: factory,
            delay: { seconds in try await delayLog.record(thenSleep: seconds) },
            heartbeatInterval: 30, maxBackoffSeconds: 60)
        do {
            try await body(client, thePair.serverSide, delayLog)
        } catch {
            try? await relay.shutdown()
            throw error
        }
        try? await relay.shutdown()
    }

    /// Next frame that is not a heartbeat (heartbeats interleave freely with
    /// test traffic; callers care about protocol frames).
    private func nextSignificantFrame(
        from stream: AsyncStream<Frame>, timeout: TimeInterval = 10
    ) async -> Frame? {
        while true {
            guard let frame = await stream.next(timeout: timeout) else { return nil }
            if case .heartbeat = frame { continue }
            return frame
        }
    }

    /// Bounded single read from the demand stream (same race pattern as the
    /// Frame `next(timeout:)` helper, different element type).
    private func nextDemand(
        from stream: AsyncStream<[DemandEntry]>, timeout: TimeInterval
    ) async -> [DemandEntry]? {
        await withTaskGroup(of: [DemandEntry]?.self) { group in
            group.addTask {
                var iterator = stream.makeAsyncIterator()
                return await iterator.next()
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    @Test("sends hello with capabilities on start, then heartbeats")
    func helloThenHeartbeat() async throws {
        try await withClient { client, server, _ in
            let task = Task { await client.start() }
            defer { client.stop(); task.cancel() }
            let hello = await nextSignificantFrame(from: server.inbound)
            guard case .hello(let nodeId, let caps)? = hello else {
                Issue.record("expected hello, got \(String(describing: hello))")
                return
            }
            #expect(nodeId == "node-1")
            #expect(caps.count == 1)
            let hb = await server.inbound.next(timeout: 10)
            guard case .heartbeat = hb else {
                Issue.record("expected heartbeat, got \(String(describing: hb))")
                return
            }
        }
    }

    @Test("routes a request through the relay and streams frames back")
    func relaysRequest() async throws {
        try await withClient { client, server, _ in
            let task = Task { await client.start() }
            defer { client.stop(); task.cancel() }
            let hello = await nextSignificantFrame(from: server.inbound)
            guard case .hello = hello else {
                Issue.record("expected hello before request, got \(String(describing: hello))")
                return
            }
            try await server.send(.request(RequestFrame(
                reqId: "r9", model: "m", apiFormat: .openai,
                body: Data(#"{"model":"m","messages":[]}"#.utf8))))
            // The source endpoint 127.0.0.1:1 refuses connections, so the
            // relay's terminal frame is a failed end for "r9".
            var terminal: Frame?
            for _ in 0..<32 {
                guard let frame = await nextSignificantFrame(from: server.inbound, timeout: 15)
                else { break }
                if case .responseEnd("r9", _) = frame { terminal = frame; break }
            }
            guard case .responseEnd("r9", let result)? = terminal else {
                Issue.record("expected responseEnd for r9, got \(String(describing: terminal))")
                return
            }
            #expect(result.status == .failed)
            #expect(result.upstreamStatus == 0)
        }
    }

    @Test("demand.update is surfaced via demandStream")
    func demandUpdateSurfaced() async throws {
        try await withClient { client, server, _ in
            let task = Task { await client.start() }
            defer { client.stop(); task.cancel() }
            let hello = await nextSignificantFrame(from: server.inbound)
            guard case .hello = hello else {
                Issue.record("expected hello, got \(String(describing: hello))")
                return
            }
            let entries = [DemandEntry(demandId: "d2", model: "m2", modality: "language", note: nil)]
            try await server.send(.demandUpdate(entries))
            let received = await nextDemand(from: client.demandStream, timeout: 10)
            #expect(received == entries)
            #expect(client.demand == entries)
        }
    }

    @Test("reconnect uses exponential backoff with jitter")
    func backoff() async throws {
        let pair = InMemoryTransportPair()
        let flaky = FlakyFactory(pair: pair)
        try await withClient(
            pair: pair, transportFactory: { try flaky.make() }
        ) { client, server, delayLog in
            let task = Task { await client.start() }
            defer { client.stop(); task.cancel() }
            // First dial throws → one backoff delay → second dial succeeds.
            let hello = await nextSignificantFrame(from: server.inbound)
            guard case .hello = hello else {
                Issue.record("expected hello after reconnect, got \(String(describing: hello))")
                return
            }
            let backoffs = delayLog.backoffValues
            #expect(backoffs.count == 1)
            // Attempt 1: base 1 s × jitter factor 0.5–1.5.
            if let first = backoffs.first {
                #expect(first >= 0.5 && first <= 1.5)
            }
        }
    }

    @Test("concurrency cap refuses over-dispatch with a failed end frame")
    func concurrencyCap() async throws {
        let source = MockSourceServer()
        try await source.start()
        do {
            try await withClient(
                sourceEndpoint: URL(string: "http://127.0.0.1:\(source.port)/v1")!
            ) { client, server, _ in
                // Default concurrencyLimit is 1; hold r1 on the source so r2
                // arrives while the slot is still taken.
                source.delayNextResponse = 2.0
                let task = Task { await client.start() }
                defer { client.stop(); task.cancel() }
                let hello = await nextSignificantFrame(from: server.inbound)
                guard case .hello = hello else {
                    Issue.record("expected hello, got \(String(describing: hello))")
                    return
                }
                func requestFrame(_ reqId: String) -> Frame {
                    .request(RequestFrame(
                        reqId: reqId, model: "m", apiFormat: .openai,
                        body: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#.utf8)))
                }
                try await server.send(requestFrame("r1"))
                try await server.send(requestFrame("r2"))
                var r1Result: RequestResult?
                var r2Result: RequestResult?
                for _ in 0..<64 {
                    guard let frame = await nextSignificantFrame(from: server.inbound, timeout: 20)
                    else { break }
                    guard case .responseEnd(let reqId, let result) = frame else { continue }
                    if reqId == "r1" { r1Result = result }
                    if reqId == "r2" { r2Result = result }
                    if r1Result != nil && r2Result != nil { break }
                }
                guard let r2 = r2Result else {
                    Issue.record("r2 never received an end frame")
                    return
                }
                guard let r1 = r1Result else {
                    Issue.record("r1 never received an end frame")
                    return
                }
                #expect(r2.status == .failed)
                #expect(r2.errorMessage?.contains("node busy") == true)
                #expect(r1.status == .completed)
            }
        } catch {
            await source.stop()
            throw error
        }
        await source.stop()
    }
}
