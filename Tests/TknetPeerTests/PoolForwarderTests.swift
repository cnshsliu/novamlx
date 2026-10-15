import Foundation
import Testing
@testable import TknetPeer

@Suite("PoolForwarder")
struct PoolForwarderTests {
    /// One mock upstream = one factory (loopback base) with a chat interface.
    private func member(_ id: String, _ server: MockSourceServer, upstream: String) -> MemberConfig {
        MemberConfig(
            id: id, name: id, interfaceId: id == "a" ? "fa-chat" : "fb-chat", upstreamModel: upstream)
    }

    private func factories(_ a: MockSourceServer, _ b: MockSourceServer) -> [FactoryConfig] {
        [
            FactoryConfig(
                id: "fa", name: "a", baseUrl: URL(string: "http://127.0.0.1:\(a.port)")!,
                apiKeyRef: "src/a",
                interfaces: [InterfaceConfig(id: "fa-chat", kind: .openaiChat)]),
            FactoryConfig(
                id: "fb", name: "b", baseUrl: URL(string: "http://127.0.0.1:\(b.port)")!,
                apiKeyRef: "src/b",
                interfaces: [InterfaceConfig(id: "fb-chat", kind: .openaiChat)]),
        ]
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
        config.factories = factories(a, b)
        config.members = [member("a", a, upstream: "ua"), member("b", b, upstream: "ub")]
        config.pools = [PoolConfig(
            id: "coding", name: "coding", strategy: .roundRobin, kind: .openaiChat,
            memberIds: ["a", "b"], maxRetries: 3)]
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

    @Test("embeddings keep the pool and land on the embeddings path")
    func embeddingsPath() async throws {
        try await withForwarder { forwarder, a, _ in
            var config = forwarder.currentConfig
            config.pools[0].kind = .openaiEmbeddings
            config.members[0].interfaceId = "fa-emb"
            config.factories[0].interfaces.append(InterfaceConfig(id: "fa-emb", kind: .openaiEmbeddings))
            forwarder.updateConfig(config)
            let body = Data(#"{"model":"coding","input":"hello"}"#.utf8)
            let outcome = await forwarder.forward(
                poolName: "coding", body: body, timeoutSeconds: 30,
                gatewayPath: "/v1/embeddings", contentType: "application/json")
            guard case .success(let upstream) = outcome else {
                Issue.record("expected success, got \(outcome)")
                return
            }
            #expect(upstream.status == 200)
            _ = await collect(upstream.body)
            #expect(a.lastPath == "/v1/embeddings")
            #expect(a.lastBodyModel == "ua")
        }
    }

    @Test("wrong path is rejected before any member is hit")
    func pathRejectedBeforePick() async throws {
        try await withForwarder { forwarder, a, b in
            var config = forwarder.currentConfig
            config.pools[0].kind = .openaiEmbeddings
            forwarder.updateConfig(config)
            let outcome = await forwarder.forward(
                poolName: "coding", body: chatBody(), timeoutSeconds: 30,
                gatewayPath: "/v1/chat/completions")
            guard case .pathNotAllowed(let path) = outcome else {
                Issue.record("expected pathNotAllowed, got \(outcome)")
                return
            }
            #expect(path == "/v1/chat/completions")
            #expect(a.hitCount == 0)
            #expect(b.hitCount == 0)
        }
    }

    @Test("generations pool rejects edits")
    func generationsRejectsEdits() async throws {
        try await withForwarder { forwarder, a, _ in
            var config = forwarder.currentConfig
            config.pools[0].kind = .openaiImages
            config.factories[0].interfaces.append(InterfaceConfig(id: "fa-gen", kind: .openaiImages))
            config.members[0].interfaceId = "fa-gen"
            forwarder.updateConfig(config)
            let body = Data(#"{"model":"coding","prompt":"hi"}"#.utf8)
            let bad = await forwarder.forward(
                poolName: "coding", body: body, timeoutSeconds: 30,
                gatewayPath: "/v1/images/edits", contentType: "application/json")
            guard case .pathNotAllowed = bad else {
                Issue.record("expected pathNotAllowed, got \(bad)")
                return
            }
            #expect(a.hitCount == 0)
            let ok = await forwarder.forward(
                poolName: "coding", body: body, timeoutSeconds: 30,
                gatewayPath: "/v1/images/generations", contentType: "application/json")
            guard case .success = ok else {
                Issue.record("expected success, got \(ok)")
                return
            }
            #expect(a.lastPath == "/v1/images/generations")
        }
    }

    @Test("members authenticate with the factory key")
    func factoryKeyUsed() async throws {
        let server = MockSourceServer()
        try await server.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-fwd-\(UUID().uuidString)"))
        secrets.save("k-f", for: "factory/f1")
        var config = PeerConfig.defaultConfig()
        config.factories = [FactoryConfig(
            id: "f1", name: "F1",
            baseUrl: URL(string: "http://127.0.0.1:\(server.port)")!,
            apiKeyRef: "factory/f1",
            interfaces: [InterfaceConfig(id: "f1-chat", kind: .openaiChat)])]
        config.members = [
            MemberConfig(id: "m", name: "m", interfaceId: "f1-chat", upstreamModel: "u"),
        ]
        config.pools = [
            PoolConfig(id: "p", name: "p", kind: .openaiChat, memberIds: ["m"]),
        ]
        let forwarder = PoolForwarder(config: config, secrets: secrets)
        do {
            _ = await forwarder.forward(poolName: "p", body: chatBody(), timeoutSeconds: 30)
            #expect(server.lastAuthorizationHeader == "Bearer k-f")
        } catch {
            try? await forwarder.shutdown()
            await server.stop()
            throw error
        }
        try? await forwarder.shutdown()
        await server.stop()
    }

    @Test("ALL-UPPERCASE factory key resolves from the env file; missing reference fails the member")
    func envFileKeyReference() async throws {
        let server = MockSourceServer()
        try await server.start()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-fwd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let envURL = dir.appendingPathComponent("env")
        try Data("FACTORY_LIVE=sk-env-live\n".utf8).write(to: envURL)
        let secrets = FileSecretStore(directory: dir)
        // f1's key is a resolvable reference; f2's points at nothing.
        secrets.save("FACTORY_LIVE", for: "factory/f1")
        secrets.save("NOPE_KEY", for: "factory/f2")
        let base = URL(string: "http://127.0.0.1:\(server.port)")!
        var config = PeerConfig.defaultConfig()
        config.factories = [
            FactoryConfig(id: "f1", name: "live", baseUrl: base, apiKeyRef: "factory/f1",
                          interfaces: [InterfaceConfig(id: "f1-chat", kind: .openaiChat)]),
            FactoryConfig(id: "f2", name: "missing", baseUrl: base, apiKeyRef: "factory/f2",
                          interfaces: [InterfaceConfig(id: "f2-chat", kind: .openaiChat)]),
        ]
        config.members = [
            MemberConfig(id: "env", name: "env", interfaceId: "f1-chat", upstreamModel: "u"),
            MemberConfig(id: "missing", name: "missing", interfaceId: "f2-chat", upstreamModel: "u"),
        ]
        config.pools = [
            PoolConfig(id: "p-env", name: "p-env", kind: .openaiChat, memberIds: ["env"]),
            PoolConfig(id: "p-missing", name: "p-missing", kind: .openaiChat, memberIds: ["missing"]),
        ]
        let forwarder = PoolForwarder(config: config, secrets: secrets, envFile: envURL)
        do {
            _ = await forwarder.forward(poolName: "p-env", body: chatBody(), timeoutSeconds: 30)
            #expect(server.lastAuthorizationHeader == "Bearer sk-env-live")

            let outcome = await forwarder.forward(poolName: "p-missing", body: chatBody(), timeoutSeconds: 30)
            if case .allMembersFailed(let lastError, _, _) = outcome {
                #expect(lastError.contains("NOPE_KEY"))
            } else {
                Issue.record("expected allMembersFailed, got \(outcome)")
            }
        } catch {
            try? await forwarder.shutdown()
            await server.stop()
            throw error
        }
        try? await forwarder.shutdown()
        await server.stop()
    }
}
