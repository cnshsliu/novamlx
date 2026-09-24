import Foundation
import os
import Testing
@testable import NovaMLXTknetPeer

/// Task 12 release gate: the whole chain over real sockets —
/// MockTunnelServer (tknet.ai) ← WebSocket ← TunnelClient ← Relay ←
/// MockSourceServer (upstream), responses streamed back the same way.
/// Nothing is in-memory here: the transport is WSTransport, the tunnel is a
/// real WS upgrade, and the source relay is real HTTP through AsyncHTTPClient.
@Suite("End-to-end")
struct EndToEndTests {
    /// Fixed 10 ms sleep regardless of the requested duration: reconnect
    /// backoffs and heartbeat spacing stay fast (requested values are the
    /// client's concern, not ours).
    private static let quickDelay: @Sendable (Double) async throws -> Void = { _ in
        try await Task.sleep(nanoseconds: 10_000_000)
    }

    /// Single-consumer tap on the tunnel's replaying frame recorder. Exactly
    /// one task iterates the stream (AsyncStream is not multi-consumer, and
    /// `receivedFrames.stream` replays recorded frames to each new subscriber,
    /// so per-read subscriptions would see the hello replayed forever).
    /// Consumers advance a cursor index and poll for the next non-heartbeat
    /// frame — generous timeouts, no tight spins.
    private final class FrameCollector: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: [Frame]())
        private var task: Task<Void, Never>?

        func collect(_ stream: AsyncStream<Frame>) {
            task = Task {
                for await frame in stream {
                    state.withLock { $0.append(frame) }
                }
            }
        }

        var all: [Frame] { state.withLock { $0 } }

        /// Next frame after `index` that is not a heartbeat (heartbeats
        /// interleave freely with request traffic at the 10 ms test pace).
        /// Returns the frame plus the advanced cursor, or nil on timeout.
        func nextSignificant(after index: Int, timeout: TimeInterval = 20) async -> (frame: Frame, next: Int)? {
            let deadline = Date().addingTimeInterval(timeout)
            var cursor = index
            while true {
                for frame in all.dropFirst(cursor) {
                    cursor += 1
                    if case .heartbeat = frame { continue }
                    return (frame, cursor)
                }
                guard Date() < deadline else { return nil }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
        }
    }

    @Test("user request flows tknet→peer→source and streams back")
    func fullChain() async throws {
        let source = MockSourceServer()
        try await source.start()
        let tunnel = MockTunnelServer()
        try await tunnel.start()
        // Safety net for early returns; the explicit teardown at the end is
        // the ordered one (service first, then both servers) and every stop()
        // is idempotent.
        defer { Task { await source.stop(); await tunnel.stop() } }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-e2e-\(UUID().uuidString)")
        let secrets = FileSecretStore(directory: dir.appendingPathComponent("secrets"))
        secrets.save("k", for: "s1")
        secrets.save("tok", for: "peer/token")

        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-1"
        config.serverURL = URL(string: "ws://127.0.0.1:\(tunnel.port)")!
        config.sources = [SourceConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: URL(string: "http://127.0.0.1:\(source.port)/v1")!,
            apiKeyRef: "s1", upstreamModel: "u")]
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]

        let service = PeerService(
            config: config, secrets: secrets,
            transportFactory: WSTransport.factory(
                server: config.serverURL, tokenRef: "peer/token", secrets: secrets),
            delay: Self.quickDelay)
        let collector = FrameCollector()
        collector.collect(tunnel.receivedFrames.stream)
        try await service.start()

        // The peer identified itself over the real tunnel.
        guard case .hello(let peerId, let caps)? = await collector.nextSignificant(after: 0)?.frame
        else {
            Issue.record("expected hello as the first significant tunnel frame")
            await service.stop(); await source.stop(); await tunnel.stop()
            return
        }
        #expect(peerId == "peer-1")
        #expect(caps.map(\.demandId) == ["d1"])
        // The peer token rode the real WS upgrade request.
        #expect(tunnel.lastAuthorization == "Bearer tok")

        // tknet.ai pushes an end-user request down the live tunnel.
        let cursor = 1
        await tunnel.push(.request(RequestFrame(
            reqId: "r1", model: "m", apiFormat: .openai,
            body: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#.utf8))))

        // Chunks stream back, then the terminal responseEnd for r1.
        var chunks = 0
        var end: RequestResult?
        var scan = cursor
        while let (frame, next) = await collector.nextSignificant(after: scan) {
            scan = next
            if frame.isChunk { chunks += 1 }
            if case .responseEnd("r1", let result) = frame { end = result; break }
        }
        #expect(chunks >= 1)
        guard let end else {
            Issue.record("expected terminal responseEnd for r1, got \(collector.all.count) frames")
            await service.stop(); await source.stop(); await tunnel.stop()
            return
        }
        #expect(end.status == .completed)
        #expect(end.completionTokens == 2)  // usage from the mock source's last SSE event

        // The relay translated the demand model to the source's upstream model
        // and attached the source key — to the source only, never onto the
        // tunnel (the tunnel saw the peer token, asserted above).
        #expect(source.lastBodyModel == "u")
        #expect(source.lastAuthorizationHeader == "Bearer k")

        // Ordered teardown: service (relay's AsyncHTTPClient) before servers.
        await service.stop()
        await source.stop()
        await tunnel.stop()
    }
}
