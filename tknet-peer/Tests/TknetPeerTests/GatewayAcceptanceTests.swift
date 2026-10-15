import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import TknetPeer

/// Segment 1 gate: two fake HTTP upstreams in one pool, first 500 → second,
/// reached through the local OpenAI port rather than the forwarder alone.
@Suite("Gateway acceptance")
struct GatewayAcceptanceTests {
    @Test("first upstream 500 falls through to the second via the local gateway")
    func failoverThroughGateway() async throws {
        let a = MockSourceServer()
        let b = MockSourceServer()
        try await a.start()
        try await b.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-acc-\(UUID().uuidString)"))
        var config = PeerConfig.defaultConfig()
        config.gatewayPort = 0
        config.factories = [
            FactoryConfig(id: "fa", name: "a", baseUrl: URL(string: "http://127.0.0.1:\(a.port)")!,
                          interfaces: [InterfaceConfig(id: "fa-chat", kind: .openaiChat)]),
            FactoryConfig(id: "fb", name: "b", baseUrl: URL(string: "http://127.0.0.1:\(b.port)")!,
                          interfaces: [InterfaceConfig(id: "fb-chat", kind: .openaiChat)]),
        ]
        config.members = [
            MemberConfig(id: "a", name: "a", interfaceId: "fa-chat", upstreamModel: "ua"),
            MemberConfig(id: "b", name: "b", interfaceId: "fb-chat", upstreamModel: "ub"),
        ]
        config.pools = [PoolConfig(id: "coding", name: "coding", memberIds: ["a", "b"])]
        let forwarder = PoolForwarder(config: config, secrets: secrets)
        let gateway = LocalGateway()
        do {
            try await gateway.start(host: "127.0.0.1", port: 0, forwarder: forwarder)
            a.failNextWithStatus = 500
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.httpBody = Data(#"{"model":"coding","messages":[]}"#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            #expect(!data.isEmpty)
            #expect(a.hitCount == 1)
            #expect(b.hitCount == 1)
            #expect(b.lastBodyModel == "ub")

            let (models, modelsResponse) = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(gateway.port)/v1/models")!)
            #expect((modelsResponse as? HTTPURLResponse)?.statusCode == 200)
            #expect(String(data: models, encoding: .utf8)?.contains("coding") == true)
        } catch {
            await gateway.stop()
            try? await forwarder.shutdown()
            await a.stop(); await b.stop()
            throw error
        }
        await gateway.stop()
        try? await forwarder.shutdown()
        await a.stop(); await b.stop()
    }
}
