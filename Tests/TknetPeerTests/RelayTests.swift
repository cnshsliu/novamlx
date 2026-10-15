import Foundation
import Testing
@testable import TknetPeer

@Suite("Relay")
struct RelayTests {
    /// Starts a mock source on an ephemeral port, builds a relay bound to it
    /// (endpoint built from the server's real port), and always tears both
    /// down — AsyncHTTPClient asserts in debug builds if it is dropped without
    /// shutdown, so cleanup must run even when test assertions throw.
    private func withRelay(
        requestTimeoutSeconds: Double = 300,
        interfaceKind: InterfaceKind = .openaiChat,
        _ body: (Relay, MockSourceServer) async throws -> Void
    ) async throws {
        let server = MockSourceServer()
        try await server.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-relay-\(UUID().uuidString)"))
        secrets.save("sk-source-key", for: "factory/fx")
        var config = PeerConfig.defaultConfig()
        config.requestTimeoutSeconds = requestTimeoutSeconds
        config.factories = [FactoryConfig(
            id: "fx", name: "mock", baseUrl: URL(string: "http://127.0.0.1:\(server.port)")!,
            apiKeyRef: "factory/fx",
            interfaces: [InterfaceConfig(id: "fx-chat", kind: interfaceKind)])]
        config.members = [MemberConfig(
            id: "s1", name: "mock", interfaceId: "fx-chat", upstreamModel: "upstream-model-x"
        )]
        config.pools = [PoolConfig(id: "s1", name: "demand-model", kind: interfaceKind, memberIds: ["s1"])]
        config.capabilities = [Capability(
            demandId: "d1", model: "demand-model", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0
        )]
        let relay = Relay(config: config, secrets: secrets)
        do {
            try await body(relay, server)
        } catch {
            try? await relay.shutdown()
            await server.stop()
            throw error
        }
        try? await relay.shutdown()
        await server.stop()
    }

    private func makeRequest(
        model: String = "demand-model",
        apiFormat: APIFormat = .openai
    ) -> RequestFrame {
        let body = #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}],"stream":true}"#
        return RequestFrame(reqId: "r1", model: model, apiFormat: apiFormat,
                            body: body.data(using: .utf8)!)
    }

    /// Bounded read of every frame the relay emits. The relay always finishes
    /// its stream after one terminal `responseEnd`, so `next(timeout:)`
    /// returns nil immediately once the lifecycle is over; the generous
    /// timeout only guards against a broken relay hanging the suite.
    private func collect(_ stream: AsyncStream<Frame>, limit: Int = 16) async -> [Frame] {
        var frames: [Frame] = []
        while frames.count < limit, let frame = await stream.next(timeout: 30) {
            frames.append(frame)
        }
        return frames
    }

    private func endResult(_ frame: Frame?) throws -> RequestResult {
        guard case .responseEnd(_, let result)? = frame else {
            throw ExpectationFailed("expected terminal responseEnd, got \(String(describing: frame))")
        }
        return result
    }

    private struct ExpectationFailed: Error {
        let message: String
        init(_ message: String) { self.message = message }
    }

    @Test("streams SSE chunks then one terminal end frame with telemetry")
    func streamsSSE() async throws {
        try await withRelay { relay, _ in
            let frames = await collect(relay.handle(makeRequest()))
            #expect(frames.filter(\.isChunk).count >= 2)
            #expect(frames.count == frames.filter(\.isChunk).count + 1)
            #expect(frames.last?.isCompletedEnd == true)
            // TTFT/total are wall-clock dependent, so assert fields, not equality.
            let result = try endResult(frames.last)
            #expect(result.status == .completed)
            #expect(result.upstreamStatus == 200)
            #expect(result.promptTokens == 5)
            #expect(result.completionTokens == 2)
            #expect(result.errorMessage == nil)
        }
    }

    @Test("rewrites model to the upstream model and injects the source key")
    func rewritesAndAuthenticates() async throws {
        try await withRelay { relay, server in
            let frames = await collect(relay.handle(makeRequest()))
            #expect(frames.last?.isCompletedEnd == true)
            #expect(server.lastBodyModel == "upstream-model-x")
            #expect(server.lastAuthorizationHeader == "Bearer sk-source-key")
        }
    }

    @Test("upstream failure becomes a failed end frame, not a crash")
    func upstreamFailure() async throws {
        try await withRelay { relay, server in
            server.failNextWithStatus = 503
            let frames = await collect(relay.handle(makeRequest()))
            #expect(frames.count == 1)
            let result = try endResult(frames.first)
            #expect(result.status == .failed)
            #expect(result.upstreamStatus == 503)
        }
    }

    @Test("sends X-No-Autoload and extracts NovaMLX's nested error.message")
    func noAutoloadAndNestedError() async throws {
        try await withRelay { relay, server in
            // The relay must opt out of queue-behind-load on every request.
            _ = await collect(relay.handle(makeRequest()))
            #expect(server.lastNoAutoloadHeader == "1")

            // NovaMLX answers {"error":{"message":"model not loaded"}} — the
            // relay must surface the nested string verbatim so the gateway's
            // reputation skip (exact match) sees it.
            server.failNextWithStatus = 503
            server.failNextBody = #"{"error":{"message":"model not loaded","type":"server_error"}}"#
            let frames = await collect(relay.handle(makeRequest()))
            #expect(frames.count == 1)
            let result = try endResult(frames.first)
            #expect(result.status == .failed)
            #expect(result.upstreamStatus == 503)
            #expect(result.errorMessage == "model not loaded")
        }
    }

    @Test("unknown model becomes a failed end frame")
    func unknownModel() async throws {
        try await withRelay { relay, _ in
            let frames = await collect(relay.handle(makeRequest(model: "nope")))
            #expect(frames.count == 1)
            let result = try endResult(frames.first)
            #expect(result.status == .failed)
            #expect(result.upstreamStatus == 0)
        }
    }

    @Test("a chat pool rejects an anthropic messages path before upstream")
    func chatPoolRejectsAnthropicPath() async throws {
        try await withRelay { relay, server in
            let frames = await collect(relay.handle(makeRequest(apiFormat: .anthropic)))
            #expect(frames.count == 1)
            let result = try endResult(frames.first)
            #expect(result.status == .failed)
            #expect(result.upstreamStatus == 400)
            #expect(result.errorMessage?.contains("/v1/messages") == true)
            #expect(server.lastBodyModel == nil)
            #expect(server.lastAuthorizationHeader == nil)
        }
    }

    @Test("anthropic tunnel frames go to /v1/messages with x-api-key")
    func anthropicFormatForwarded() async throws {
        try await withRelay(interfaceKind: .anthropicMessages) { relay, server in
            let frames = await collect(relay.handle(makeRequest(apiFormat: .anthropic)))
            #expect(frames.last?.isCompletedEnd == true)
            #expect(server.lastPath == "/v1/messages")
            #expect(server.lastBodyModel == "upstream-model-x")
            #expect(server.lastXAPIKey == "sk-source-key")
            #expect(server.lastAnthropicVersion == "2023-06-01")
            #expect(server.lastAuthorizationHeader == nil)
        }
    }

    @Test("non-loopback factories do not receive X-No-Autoload")
    func nonLoopbackOmitsNoAutoload() throws {
        let route = MemberRoute(
            url: URL(string: "http://10.1.2.3:8080/v1/chat/completions")!,
            kind: .openaiChat, isLocal: false)
        let body = Data(#"{"model":"demand-model","messages":[]}"#.utf8)
        let request = try PoolForwarder.makeUpstreamRequest(
            route: route, body: body, upstreamModel: "upstream-model-x", apiKey: "sk")
        #expect(request.headers.first(name: "X-No-Autoload") == nil)
        #expect(request.headers.first(name: "authorization") == "Bearer sk")
        #expect(request.url == "http://10.1.2.3:8080/v1/chat/completions")
    }

    @Test("anthropic interfaces use x-api-key, not Bearer")
    func anthropicAuthStyle() throws {
        let route = MemberRoute(
            url: URL(string: "https://api.anthropic.com/v1/messages")!,
            kind: .anthropicMessages, isLocal: false)
        let body = Data(#"{"model":"claude-x","messages":[]}"#.utf8)
        let request = try PoolForwarder.makeUpstreamRequest(
            route: route, body: body, upstreamModel: "claude-x", apiKey: "sk-ant")
        #expect(request.headers.first(name: "x-api-key") == "sk-ant")
        #expect(request.headers.first(name: "anthropic-version") == "2023-06-01")
        #expect(request.headers.first(name: "authorization") == nil)
    }

    @Test("deadline exceeded becomes a timeout end frame, not a failure")
    func deadlineExceededTimesOut() async throws {
        try await withRelay(requestTimeoutSeconds: 0.5) { relay, server in
            server.delayNextResponse = 2.0
            let frames = await collect(relay.handle(makeRequest()))
            #expect(frames.count == 1)
            let result = try endResult(frames.last)
            #expect(result.status == .timeout)
            #expect(result.upstreamStatus == 0)
        }
    }
}
