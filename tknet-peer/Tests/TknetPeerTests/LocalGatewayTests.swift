import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import TknetPeer

@Suite("Local gateway")
struct LocalGatewayTests {

    private func withGateway(
        _ body: (LocalGateway, PoolForwarder, MockSourceServer, MockSourceServer) async throws -> Void
    ) async throws {
        let a = MockSourceServer()
        let b = MockSourceServer()
        try await a.start()
        try await b.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-lgw-\(UUID().uuidString)"))
        secrets.save("sk-ant", for: "factory/fa")
        var config = PeerConfig.defaultConfig()
        let interfaces: (String) -> [InterfaceConfig] = { prefix in
            [InterfaceConfig(id: "\(prefix)-chat", kind: .openaiChat),
             InterfaceConfig(id: "\(prefix)-speech", kind: .openaiSpeech),
             InterfaceConfig(id: "\(prefix)-emb", kind: .openaiEmbeddings),
             InterfaceConfig(id: "\(prefix)-gen", kind: .openaiImages)]
        }
        config.factories = [
            FactoryConfig(id: "fa", name: "a", baseUrl: URL(string: "http://127.0.0.1:\(a.port)")!,
                          apiKeyRef: "factory/fa", interfaces: interfaces("fa")),
            FactoryConfig(id: "fb", name: "b", baseUrl: URL(string: "http://127.0.0.1:\(b.port)")!,
                          interfaces: interfaces("fb")),
        ]
        config.members = [
            MemberConfig(id: "a", name: "a", interfaceId: "fa-chat", upstreamModel: "ua"),
            MemberConfig(id: "b", name: "b", interfaceId: "fb-chat", upstreamModel: "ub"),
        ]
        config.pools = [
            PoolConfig(id: "coding", name: "coding", strategy: .roundRobin, kind: .openaiChat,
                       memberIds: ["a", "b"]),
            PoolConfig(id: "off", name: "off", kind: .openaiChat, memberIds: ["a"], isEnabled: false),
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

    @Test("audio speech is forwarded on its own path")
    func audioSpeech() async throws {
        try await withGateway { gateway, forwarder, a, _ in
            var config = forwarder.currentConfig
            config.members[0].interfaceId = "fa-speech"
            config.pools = [
                PoolConfig(id: "tts", name: "tts", kind: .openaiSpeech, memberIds: ["a"]),
            ]
            forwarder.updateConfig(config)
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/audio/speech")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(#"{"model":"tts","input":"hello","voice":"alloy"}"#.utf8)
            let (_, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            #expect(a.lastPath == "/v1/audio/speech")
            #expect(a.lastBodyModel == "ua")
        }
    }

    @Test("embeddings pool rejects chat; variations is 404")
    func pathGating() async throws {
        try await withGateway { gateway, forwarder, a, _ in
            var config = forwarder.currentConfig
            config.members[0].interfaceId = "fa-emb"
            config.pools = [
                PoolConfig(id: "emb", name: "emb", kind: .openaiEmbeddings, memberIds: ["a"]),
                PoolConfig(id: "gen", name: "gen", kind: .openaiEmbeddings, memberIds: ["a"]),
            ]
            forwarder.updateConfig(config)

            var chat = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/chat/completions")!)
            chat.httpMethod = "POST"
            chat.setValue("application/json", forHTTPHeaderField: "Content-Type")
            chat.httpBody = Data(#"{"model":"emb","messages":[{"role":"user","content":"hi"}]}"#.utf8)
            let (chatData, chatRes) = try await URLSession.shared.data(for: chat)
            #expect((chatRes as? HTTPURLResponse)?.statusCode == 400)
            #expect(String(data: chatData, encoding: .utf8)?.contains("does not serve") == true)
            #expect(a.hitCount == 0)

            var emb = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/embeddings")!)
            emb.httpMethod = "POST"
            emb.setValue("application/json", forHTTPHeaderField: "Content-Type")
            emb.httpBody = Data(#"{"model":"emb","input":"hi"}"#.utf8)
            let (_, embRes) = try await URLSession.shared.data(for: emb)
            #expect((embRes as? HTTPURLResponse)?.statusCode == 200)
            #expect(a.lastPath == "/v1/embeddings")

            var edits = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/images/edits")!)
            edits.httpMethod = "POST"
            edits.setValue("application/json", forHTTPHeaderField: "Content-Type")
            edits.httpBody = Data(#"{"model":"gen","prompt":"x"}"#.utf8)
            let (_, editsRes) = try await URLSession.shared.data(for: edits)
            #expect((editsRes as? HTTPURLResponse)?.statusCode == 400)

            var variations = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/images/variations")!)
            variations.httpMethod = "POST"
            variations.setValue("application/json", forHTTPHeaderField: "Content-Type")
            variations.httpBody = Data(#"{"model":"gen"}"#.utf8)
            let (_, varRes) = try await URLSession.shared.data(for: variations)
            #expect((varRes as? HTTPURLResponse)?.statusCode == 404)
        }
    }

    @Test("anthropic messages forwards with x-api-key and rejects chat")
    func anthropicMessages() async throws {
        try await withGateway { gateway, forwarder, a, _ in
            var config = forwarder.currentConfig
            config.factories[0].interfaces.append(
                InterfaceConfig(id: "fa-ant", kind: .anthropicMessages))
            config.members[0].interfaceId = "fa-ant"
            config.pools = [
                PoolConfig(id: "claude", name: "claude", kind: .anthropicMessages, memberIds: ["a"]),
            ]
            forwarder.updateConfig(config)

            var chat = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/chat/completions")!)
            chat.httpMethod = "POST"
            chat.setValue("application/json", forHTTPHeaderField: "Content-Type")
            chat.httpBody = Data(#"{"model":"claude","messages":[{"role":"user","content":"hi"}]}"#.utf8)
            let (_, chatRes) = try await URLSession.shared.data(for: chat)
            #expect((chatRes as? HTTPURLResponse)?.statusCode == 400)
            #expect(a.hitCount == 0)

            var messages = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/messages")!)
            messages.httpMethod = "POST"
            messages.setValue("application/json", forHTTPHeaderField: "Content-Type")
            messages.setValue("prompt-caching-2024-07-31", forHTTPHeaderField: "anthropic-beta")
            messages.httpBody = Data(#"{"model":"claude","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#.utf8)
            let (_, msgRes) = try await URLSession.shared.data(for: messages)
            #expect((msgRes as? HTTPURLResponse)?.statusCode == 200)
            #expect(a.lastPath == "/v1/messages")
            #expect(a.lastBodyModel == "ua")
            #expect(a.lastXAPIKey == "sk-ant")
            #expect(a.lastAnthropicVersion == "2023-06-01")
            #expect(a.lastAnthropicBeta == "prompt-caching-2024-07-31")
            #expect(a.lastAuthorizationHeader == nil)
        }
    }
}
