import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import TknetPeer

@Suite("Peer service tunnel control")
struct PeerServiceTunnelControlTests {
    private static let quickDelay: @Sendable (Double) async throws -> Void = { _ in
        try await Task.sleep(nanoseconds: 10_000_000)
    }

    private final class StatusRecorder: @unchecked Sendable {
        private let state = PeerLock(initialState: [PeerStatus]())

        func record(_ stream: AsyncStream<PeerStatus>) {
            Task {
                for await status in stream {
                    self.state.withLock { $0.append(status) }
                }
            }
        }

        var last: PeerStatus? { state.withLock { $0.last } }

        @discardableResult
        func waitLast(until predicate: (PeerStatus) -> Bool, timeout: TimeInterval = 10) async -> Bool {
            let deadline = Date().addingTimeInterval(timeout)
            while Date() < deadline {
                if let last, predicate(last) { return true }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            return last.map(predicate) ?? false
        }
    }

    private final class PairLog: @unchecked Sendable {
        private let state = PeerLock(initialState: [InMemoryTransportPair]())
        func append(_ pair: InMemoryTransportPair) { state.withLock { $0.append(pair) } }
        var all: [InMemoryTransportPair] { state.withLock { $0 } }
    }

    private func makeService(
        factory: @escaping TransportFactory,
        peerId: String? = "peer-1",
        endpoint: URL = URL(string: "http://10.0.0.9:8080/v1")!
    ) -> PeerService {
        var config = PeerConfig.defaultConfig()
        config.peerId = peerId
        config.gatewayPort = 0
        config.members = [MemberConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: endpoint, apiKeyRef: "s1", upstreamModel: "u")]
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-tunnel-\(UUID().uuidString)"))
        secrets.save("k", for: "s1")
        return PeerService(
            config: config, secrets: secrets,
            transportFactory: factory,
            delay: Self.quickDelay)
    }

    private func freshFactory(_ log: PairLog) -> TransportFactory {
        { @Sendable in
            let pair = InMemoryTransportPair()
            log.append(pair)
            return pair.peerSide
        }
    }

    private func nextSignificant(from stream: AsyncStream<Frame>, timeout: TimeInterval = 10) async -> Frame? {
        while true {
            guard let frame = await stream.next(timeout: timeout) else { return nil }
            if case .heartbeat = frame { continue }
            return frame
        }
    }

    private func waitPair(_ log: PairLog, timeout: TimeInterval = 5) async -> InMemoryTransportPair? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let pair = log.all.first { return pair }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return log.all.first
    }

    private func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 2
        return URLSession(configuration: configuration)
    }

    @Test("stopTunnel leaves gateway listening")
    func stopTunnelKeepsGateway() async throws {
        let log = PairLog()
        let recorder = StatusRecorder()
        let service = makeService(factory: freshFactory(log))
        recorder.record(service.statusStream)
        try await service.start(tunnel: false)
        let port = try #require(service.gatewayListenPort)
        #expect(await recorder.waitLast { $0.gatewayListening && $0.gatewayPort == port })

        try await service.startTunnel()
        let pair = try #require(await waitPair(log))
        guard case .hello? = await nextSignificant(from: pair.serverSide.inbound) else {
            Issue.record("expected hello after startTunnel")
            await service.stop()
            return
        }
        #expect(await recorder.waitLast { $0.connection == .connected && $0.gatewayListening })

        await service.stopTunnel()
        #expect(await recorder.waitLast { $0.connection == .idle && $0.gatewayListening && $0.gatewayPort == port })
        #expect(service.gatewayListenPort == port)

        let (data, response) = try await session().data(from: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let body = String(data: data, encoding: .utf8) ?? ""
        #expect(body.contains("\"id\":\"m\""))

        await service.stop()
    }

    @Test("startTunnel after gateway-only start dials once")
    func startTunnelAfterGateway() async throws {
        let log = PairLog()
        let service = makeService(factory: freshFactory(log))
        try await service.start(tunnel: false)
        #expect(log.all.isEmpty)
        try await service.startTunnel()
        try await service.startTunnel()
        let pair = try #require(await waitPair(log))
        #expect(log.all.count == 1)
        guard case .hello(let peerId, _, _, _)? = await nextSignificant(from: pair.serverSide.inbound) else {
            Issue.record("expected a single hello")
            await service.stop()
            return
        }
        #expect(peerId == "peer-1")
        let deadline = Date().addingTimeInterval(0.4)
        while Date() < deadline {
            guard let frame = await pair.serverSide.inbound.next(timeout: 0.1) else { break }
            if case .hello = frame {
                Issue.record("startTunnel dialed a second hello")
                break
            }
        }
        await service.stop()
    }

    @Test("full stop shuts gateway")
    func fullStopShutsGateway() async throws {
        let log = PairLog()
        let recorder = StatusRecorder()
        let service = makeService(factory: freshFactory(log))
        recorder.record(service.statusStream)
        try await service.start(tunnel: false)
        let port = try #require(service.gatewayListenPort)
        await service.stop()
        #expect(await recorder.waitLast { $0.connection == .idle && $0.gatewayListening == false })
        #expect(service.gatewayListenPort == nil)
        do {
            _ = try await session().data(from: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
            Issue.record("gateway still accepted a request after stop()")
        } catch {
            // Connection refused — the listener is gone.
        }
    }

    @Test("startTunnel before start throws")
    func startTunnelRequiresGateway() async {
        let service = makeService(factory: { InMemoryTransportPair().peerSide })
        await #expect(throws: PeerServiceError.gatewayNotRunning) {
            try await service.startTunnel()
        }
    }

    @Test("startTunnel without a peer id throws and leaves the gateway up")
    func startTunnelRequiresRegistration() async throws {
        let service = makeService(factory: { InMemoryTransportPair().peerSide }, peerId: nil)
        try await service.start(tunnel: false)
        let port = service.gatewayListenPort
        await #expect(throws: PeerServiceError.notRegistered) {
            try await service.startTunnel()
        }
        #expect(service.gatewayListenPort == port)
        await service.stop()
    }

    @Test("stopTunnel during hung dial publishes idle promptly")
    func stopTunnelDuringHungDialPublishesIdle() async throws {
        let dialStarted = PeerLock(initialState: false)
        let factory: TransportFactory = {
            dialStarted.withLock { $0 = true }
            // Cooperative hang: cancelled by stopTunnel's runTask.cancel().
            while !Task.isCancelled {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            throw CancellationError()
        }
        let recorder = StatusRecorder()
        let service = makeService(factory: factory)
        recorder.record(service.statusStream)
        try await service.start(tunnel: false)
        #expect(await recorder.waitLast { $0.gatewayListening })

        let startTask = Task {
            try await service.startTunnel()
        }
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, dialStarted.withLock({ !$0 }) {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(dialStarted.withLock { $0 })
        #expect(await recorder.waitLast { $0.connection == .connecting && $0.gatewayListening })

        let stoppedAt = Date()
        await service.stopTunnel()
        #expect(Date().timeIntervalSince(stoppedAt) < 2.0)
        #expect(await recorder.waitLast { $0.connection == .idle && $0.gatewayListening })

        _ = await startTask.result
        await service.stop()
    }

    @Test("stopTunnel then startTunnel dials again on the same port")
    func redialKeepsPort() async throws {
        let log = PairLog()
        let service = makeService(factory: freshFactory(log))
        try await service.start(tunnel: false)
        let port = service.gatewayListenPort
        try await service.startTunnel()
        let first = try #require(await waitPair(log))
        _ = await nextSignificant(from: first.serverSide.inbound)
        await service.stopTunnel()
        try await service.startTunnel()
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, log.all.count < 2 {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        guard log.all.count >= 2 else {
            Issue.record("expected a second dial, saw \(log.all.count)")
            await service.stop()
            return
        }
        guard case .hello? = await nextSignificant(from: log.all[1].serverSide.inbound) else {
            Issue.record("expected hello on the second dial")
            await service.stop()
            return
        }
        #expect(service.gatewayListenPort == port)
        await service.stop()
    }
}
