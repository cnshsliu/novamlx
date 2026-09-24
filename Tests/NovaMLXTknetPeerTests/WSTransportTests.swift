import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("WS transport")
struct WSTransportTests {
    @Test("heartbeat travels over a real WebSocket to the mock server")
    func heartbeatOverRealSocket() async throws {
        let tunnel = MockTunnelServer()
        try await tunnel.start()
        defer { Task { await tunnel.stop() } }

        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-ws-\(UUID().uuidString)"))
        secrets.save("tok", for: "peer/token")
        let factory = WSTransport.factory(
            server: URL(string: "ws://127.0.0.1:\(tunnel.port)")!,
            tokenRef: "peer/token", secrets: secrets)
        let transport = try await factory()

        try await transport.send(.heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))
        let seen = await tunnel.receivedFrames.stream.next(timeout: 10)
        #expect(seen == .heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))

        // The peer token rode the Authorization header of the WS upgrade request.
        #expect(tunnel.lastAuthorization == "Bearer tok")

        await transport.close()
        // Inbound finishes on close so the session loop notices, and send()
        // throws instead of silently no-op'ing (Task 6 review note).
        let afterClose = await transport.inbound.next(timeout: 5)
        #expect(afterClose == nil)
        await #expect(throws: WSTransportError.self) {
            try await transport.send(.heartbeat(Heartbeat(activeReq: 1, queueDepth: 0)))
        }

        await tunnel.stop()
    }

    @Test("server push reaches the client inbound stream")
    func serverPushOverRealSocket() async throws {
        let tunnel = MockTunnelServer()
        try await tunnel.start()
        defer { Task { await tunnel.stop() } }

        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-ws-\(UUID().uuidString)"))
        secrets.save("tok", for: "peer/token")
        let transport = try await WSTransport.factory(
            server: URL(string: "ws://127.0.0.1:\(tunnel.port)")!,
            tokenRef: "peer/token", secrets: secrets)()

        let demand = [DemandEntry(demandId: "d1", model: "m", modality: "language", note: nil)]
        await tunnel.push(.demandUpdate(demand))
        let received = await transport.inbound.next(timeout: 10)
        #expect(received == .demandUpdate(demand))

        await transport.close()
        await tunnel.stop()
    }
}
