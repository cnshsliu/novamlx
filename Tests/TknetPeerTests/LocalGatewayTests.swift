import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import TknetPeer

@Suite("Local gateway")
struct LocalGatewayTests {
    private func endpoint(_ server: MockSourceServer) -> URL {
        URL(string: "http://127.0.0.1:\(server.port)/v1")!
    }

    private func withGateway(
        _ body: (LocalGateway, PoolForwarder, MockSourceServer, MockSourceServer) async throws -> Void
    ) async throws {
        let a = MockSourceServer()
        let b = MockSourceServer()
        try await a.start()
        try await b.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-lgw-\(UUID().uuidString)"))
        var config = PeerConfig.defaultConfig()
        config.members = [
            MemberConfig(id: "a", name: "a", type: .openaiCompatible, endpoint: endpoint(a),
                         apiKeyRef: "src/a", upstreamModel: "ua"),
            MemberConfig(id: "b", name: "b", type: .openaiCompatible, endpoint: endpoint(b),
                         apiKeyRef: "src/b", upstreamModel: "ub"),
        ]
        config.pools = [
            PoolConfig(id: "coding", name: "coding", strategy: .roundRobin, memberIds: ["a", "b"]),
            PoolConfig(id: "off", name: "off", memberIds: ["a"], isEnabled: false),
        ]
        let forwarder = PoolForwarder(config: config, secrets: secrets)
        let gateway = LocalGateway()
        do {
            try await gateway.start(host: "127.0.0.1", port: 0, forwarder: forwarder)
            try await body(gateway, forwarder, a, b)
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

    private func post(_ gateway: LocalGateway, model: String) async throws -> (Int, Data) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (status, data)
    }

    @Test("models list names enabled pools and an unknown model is 404")
    func modelsAndUnknown() async throws {
        try await withGateway { gateway, _, _, _ in
            let (data, response) = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(gateway.port)/v1/models")!)
            let status = (response as? HTTPURLResponse)?.statusCode
            #expect(status == 200)
            let text = String(data: data, encoding: .utf8) ?? ""
            #expect(text.contains("\"id\":\"coding\""))
            #expect(text.contains("tknet-peer"))
            #expect(!text.contains("\"id\":\"off\""))

            let (code, body) = try await post(gateway, model: "nope")
            #expect(code == 404)
            #expect(String(data: body, encoding: .utf8)?.contains("Unknown pool") == true)
        }
    }

    @Test("chat completions fails over from a 500 to the second member")
    func failover() async throws {
        try await withGateway { gateway, _, a, b in
            a.failNextWithStatus = 500
            let (code, data) = try await post(gateway, model: "coding")
            #expect(code == 200)
            #expect(!(String(data: data, encoding: .utf8) ?? "").isEmpty)
            #expect(a.hitCount == 1)
            #expect(b.hitCount == 1)
            #expect(b.lastBodyModel == "ub")
        }
    }
}
