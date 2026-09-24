import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Relay")
struct RelayTests {
    /// Starts a mock source on an ephemeral port, builds a relay bound to it
    /// (endpoint built from the server's real port), and always tears both
    /// down — AsyncHTTPClient asserts in debug builds if it is dropped without
    /// shutdown, so cleanup must run even when test assertions throw.
    private func withRelay(_ body: (Relay, MockSourceServer) async throws -> Void) async throws {
        let server = MockSourceServer()
        try await server.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-relay-\(UUID().uuidString)"))
        secrets.save("sk-source-key", for: "s1")
        var config = NodeConfig.defaultConfig()
        config.sources = [SourceConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: URL(string: "http://127.0.0.1:\(server.port)/v1")!,
            apiKeyRef: "s1", upstreamModel: "upstream-model-x"
        )]
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

    private func makeRequest(model: String = "demand-model") -> RequestFrame {
        let body = #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}],"stream":true}"#
        return RequestFrame(reqId: "r1", model: model, apiFormat: .openai,
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
}
