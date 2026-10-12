import Foundation
import Testing
@testable import TknetPeer

@Suite("PoolForwarder")
struct PoolForwarderTests {
    private func endpoint(_ server: MockSourceServer) -> URL {
        URL(string: "http://127.0.0.1:\(server.port)/v1")!
    }

    private func member(_ id: String, _ server: MockSourceServer, upstream: String) -> MemberConfig {
        MemberConfig(
            id: id, name: id, type: .openaiCompatible, endpoint: endpoint(server),
            apiKeyRef: "src/\(id)", upstreamModel: upstream)
    }

    private func withForwarder(
        _ body: (PoolForwarder, MockSourceServer, MockSourceServer) async throws -> Void
    ) async throws {
        let a = MockSourceServer()
        let b = MockSourceServer()
        try await a.start()
        try await b.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-fwd-\(UUID().uuidString)"))
        secrets.save("k-a", for: "src/a")
        secrets.save("k-b", for: "src/b")
        var config = PeerConfig.defaultConfig()
        config.members = [member("a", a, upstream: "ua"), member("b", b, upstream: "ub")]
        config.pools = [PoolConfig(
            id: "coding", name: "coding", strategy: .roundRobin, memberIds: ["a", "b"], maxRetries: 3)]
        let forwarder = PoolForwarder(config: config, secrets: secrets)
        do {
            try await body(forwarder, a, b)
        } catch {
            try? await forwarder.shutdown()
            await a.stop(); await b.stop()
            throw error
        }
        try? await forwarder.shutdown()
        await a.stop(); await b.stop()
    }

    private func chatBody(model: String = "coding") -> Data {
        Data(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#.utf8)
    }

    private func collect(_ body: AsyncStream<Data>) async -> Data {
        var data = Data()
        for await chunk in body { data.append(chunk) }
        return data
    }

    @Test("two upstreams in one pool; first 500 falls through to second")
    func failoverOn5xx() async throws {
        try await withForwarder { forwarder, a, b in
            a.failNextWithStatus = 500
            let outcome = await forwarder.forward(poolName: "coding", body: chatBody(), timeoutSeconds: 30)
            guard case .success(let upstream) = outcome else {
                Issue.record("expected success, got \(outcome)")
                return
            }
            #expect(upstream.status == 200)
            let bytes = await collect(upstream.body)
            #expect(!bytes.isEmpty)
            #expect(a.hitCount == 1)
            #expect(b.hitCount == 1)
            #expect(a.lastBodyModel == "ua")
            #expect(b.lastBodyModel == "ub")
            #expect(a.lastNoAutoloadHeader == "1")
            #expect(b.lastAuthorizationHeader == "Bearer k-b")
        }
    }

    @Test("4xx from the first member is returned and the second is not called")
    func passThrough4xx() async throws {
        try await withForwarder { forwarder, a, b in
            a.failNextWithStatus = 400
            a.failNextBody = #"{"error":{"message":"bad request","type":"invalid_request_error"}}"#
            let outcome = await forwarder.forward(poolName: "coding", body: chatBody(), timeoutSeconds: 30)
            guard case .success(let upstream) = outcome else {
                Issue.record("expected 4xx success, got \(outcome)")
                return
            }
            #expect(upstream.status == 400)
            let text = String(data: await collect(upstream.body), encoding: .utf8) ?? ""
            #expect(text.contains("bad request"))
            #expect(a.hitCount == 1)
            #expect(b.hitCount == 0)
        }
    }

    @Test("unknown and disabled pools fail before any upstream call")
    func unknownAndDisabled() async throws {
        try await withForwarder { forwarder, a, _ in
            let unknown = await forwarder.forward(poolName: "missing", body: chatBody(model: "missing"), timeoutSeconds: 30)
            guard case .unknownPool = unknown else {
                Issue.record("expected unknownPool")
                return
            }
            var config = forwarder.currentConfig
            config.pools[0].isEnabled = false
            forwarder.updateConfig(config)
            let disabled = await forwarder.forward(poolName: "coding", body: chatBody(), timeoutSeconds: 30)
            guard case .poolDisabled = disabled else {
                Issue.record("expected poolDisabled")
                return
            }
            #expect(a.hitCount == 0)
        }
    }

    @Test("a timeout on the first member retries the second")
    func timeoutRetries() async throws {
        try await withForwarder { forwarder, a, b in
            a.delayNextResponse = 3
            let outcome = await forwarder.forward(poolName: "coding", body: chatBody(), timeoutSeconds: 1)
            guard case .success(let upstream) = outcome else {
                Issue.record("expected failover after timeout, got \(outcome)")
                return
            }
            #expect(upstream.status == 200)
            _ = await collect(upstream.body)
            #expect(b.hitCount == 1)
            #expect(b.lastBodyModel == "ub")
        }
    }
}
