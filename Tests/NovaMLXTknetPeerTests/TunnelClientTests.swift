import Foundation
import os
import Testing
@testable import NovaMLXTknetPeer

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
            return pair.peerSide
        }
    }

    /// Transport factory handing out one transport per dial, in order, so a
    /// test can close the first session server-side and observe the hello
    /// the reconnect sends on a fresh pair.
    private final class SequentialFactory: @unchecked Sendable {
        private let lock = NSLock()
        private var remaining: [any TunnelTransport]
        init(_ transports: [any TunnelTransport]) { self.remaining = transports }

        func make() throws -> TunnelTransport {
            lock.lock(); defer { lock.unlock() }
            guard !remaining.isEmpty else { throw TunnelError.connectionClosed }
            return remaining.removeFirst()
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
            transportFactory ?? { thePair.peerSide }
        let delayLog = DelayRecorder()
        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-1"
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
            guard case .hello(let peerId, let caps, _, _)? = hello else {
                Issue.record("expected hello, got \(String(describing: hello))")
                return
            }
            #expect(peerId == "peer-1")
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

    @Test("reconnect hello advertises post-applyConfig capabilities, not init's")
    func reconnectHelloUsesLatestCapabilities() async throws {
        // Drive through PeerService so the regression covers the real
        // applyConfig → updateRelay → updateConfig wiring end-to-end.
        let pair1 = InMemoryTransportPair()
        let pair2 = InMemoryTransportPair()
        let factory = SequentialFactory([pair1.peerSide, pair2.peerSide])
        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-1"
        config.sources = [SourceConfig(
            id: "s1", name: "x", type: .openaiCompatible,
            endpoint: URL(string: "http://127.0.0.1:1/v1")!, apiKeyRef: "s1",
            upstreamModel: "u")]
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        let service = PeerService(
            config: config,
            secrets: FileSecretStore(directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("tknet-tc-\(UUID().uuidString)")),
            transportFactory: { try factory.make() },
            delay: { _ in try await Task.sleep(nanoseconds: 10_000_000) })
        defer { Task { await service.stop() } }
        try await service.start()

        // Session 1: hello carries the init capability d1.
        guard case .hello(_, let firstCaps, _, _)? =
            await nextSignificantFrame(from: pair1.serverSide.inbound)
        else {
            Issue.record("expected hello on first connect")
            return
        }
        #expect(firstCaps.map(\.demandId) == ["d1"])

        // Operator edits capabilities to d2 while connected (hot
        // capabilitiesUpdate) ...
        var edited = service.currentConfig
        edited.capabilities = [Capability(
            demandId: "d2", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        await service.applyConfig(edited)
        guard case .capabilitiesUpdate(let hotCaps)? =
            await nextSignificantFrame(from: pair1.serverSide.inbound)
        else {
            Issue.record("expected capabilitiesUpdate after applyConfig")
            return
        }
        #expect(hotCaps.map(\.demandId) == ["d2"])

        // ... then the server closes and the tunnel reconnects: the second
        // hello must advertise the LATEST capabilities (d2), not the stale
        // init set (d1).
        await pair1.serverSide.close()
        guard case .hello(_, let secondCaps, _, _)? =
            await nextSignificantFrame(from: pair2.serverSide.inbound)
        else {
            Issue.record("expected hello after reconnect")
            return
        }
        #expect(secondCaps.map(\.demandId) == ["d2"])

        await service.stop()
    }


    @Test("close 4005 upgrade-required is terminal — no reconnect, terminal status")
    func upgradeRequiredStopsReconnect() async throws {
        // Decorator: peerSide of a pair, but reporting the server's close code.
        final class CodedTransport: TunnelTransport, @unchecked Sendable {
            let wrapped: TunnelTransport
            private let code = OSAllocatedUnfairLock(initialState: UInt16?.none)

            init(_ wrapped: TunnelTransport) { self.wrapped = wrapped }

            // The server "sent" close 4005 the moment its side finished —
            // TunnelClient reads closeCode after runSession returns, without
            // calling close() itself.
            lazy var inbound: AsyncStream<Frame> = {
                AsyncStream { cont in
                    let pump = Task {
                        for await f in wrapped.inbound { cont.yield(f) }
                        code.withLock { $0 = 4005 }
                        cont.finish()
                    }
                    cont.onTermination = { _ in pump.cancel() }
                }
            }()

            func send(_ frame: Frame) async throws { try await wrapped.send(frame) }
            func close() async { await wrapped.close() }
            var closeCode: UInt16? { code.withLock { $0 } }
        }

        let thePair = InMemoryTransportPair()
        let coded = CodedTransport(thePair.peerSide)
        try await withClient(
            pair: thePair,
            transportFactory: { coded }
        ) { client, server, delayLog in
            let statuses = LockedStatusLog()
            let observe = Task { for await s in client.statusStream { statuses.add(s) } }
            let runLoop = Task { await client.start() }

            // Server receives hello, then closes with 4005 (the decorator
            // reports it once close() ran).
            _ = await nextSignificantFrame(from: server.inbound)
            await server.close()

            // The client must land on .upgradeRequired and STOP without a
            // single backoff — a reconnecting loop would record one (and
            // hammer the server's bcrypt forever).
            let deadline = ContinuousClock.now + .seconds(5)
            while ContinuousClock.now < deadline {
                if statuses.contains(.upgradeRequired) { break }
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            #expect(statuses.contains(.upgradeRequired))
            #expect(delayLog.backoffValues.isEmpty)
            observe.cancel()
            runLoop.cancel()
            await client.stop()
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
                #expect(r2.errorMessage?.contains("peer busy") == true)
                #expect(r1.status == .completed)
            }
        } catch {
            await source.stop()
            throw error
        }
        await source.stop()
    }
}

/// Thread-safe TunnelStatus collector (OSAllocatedUnfairLock: async-safe).
private final class LockedStatusLog: @unchecked Sendable {
    private let items = OSAllocatedUnfairLock(initialState: [TunnelStatus]())
    func add(_ s: TunnelStatus) { items.withLock { $0.append(s) } }
    func contains(_ s: TunnelStatus) -> Bool { items.withLock { $0.contains(s) } }
}
