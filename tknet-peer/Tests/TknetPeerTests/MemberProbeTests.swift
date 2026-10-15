import Foundation
import Testing
@testable import TknetPeer

/// MemberProbe: URL joining, model-list parsing, and a real HTTP round trip
/// against MockSourceServer's `/v1/models` route.
@Suite struct MemberProbeTests {
    @Test func modelsURLJoinsWithoutDoubleSlash() {
        let plain = URL(string: "http://127.0.0.1:6590/v1")!
        #expect(MemberProbe.modelsURL(endpoint: plain) == "http://127.0.0.1:6590/v1/models")
        let trailing = URL(string: "http://127.0.0.1:6590/v1/")!
        #expect(MemberProbe.modelsURL(endpoint: trailing) == "http://127.0.0.1:6590/v1/models")
    }

    @Test func parseModelsOpenAIShape() throws {
        let data = Data(#"{"object":"list","data":[{"id":"b"},{"id":"a"}]}"#.utf8)
        #expect(MemberProbe.parseModels(data) == ["a", "b"])
    }

    @Test func parseModelsDeduplicatesAndSkipsEmpty() throws {
        let data = Data(#"{"data":[{"id":"x"},{"id":"x"},{"id":""},{"id":"y"}]}"#.utf8)
        #expect(MemberProbe.parseModels(data) == ["x", "y"])
    }

    @Test func parseModelsMalformedBodiesYieldEmpty() {
        #expect(MemberProbe.parseModels(Data("not json".utf8)) == [])
        #expect(MemberProbe.parseModels(Data(#"{"data":"nope"}"#.utf8)) == [])
        #expect(MemberProbe.parseModels(Data()) == [])
    }

    @Test func probePassesAndListsModels() async throws {
        let server = MockSourceServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let endpoint = URL(string: "http://127.0.0.1:\(server.port)/v1")!

        let result = await MemberProbe.probe(endpoint: endpoint, apiKey: "sk-test")

        #expect(result.ok)
        #expect(result.httpStatus == 200)
        // Duplicate id in the fixture proves dedup on the wire path too.
        #expect(result.models == ["llama-4-8b", "qwen3.5-14b"])
        #expect(server.lastProbePath == "/v1/models")
        #expect(server.lastProbeAuthorization == "Bearer sk-test")
        await server.stop()
    }

    @Test func probeWithoutKeySendsNoAuthorization() async throws {
        let server = MockSourceServer()
        try await server.start()
        let endpoint = URL(string: "http://127.0.0.1:\(server.port)/v1")!

        _ = await MemberProbe.probe(endpoint: endpoint, apiKey: nil)

        #expect(server.lastProbeAuthorization == nil)
        await server.stop()
    }

    @Test func probeUnreachableEndpointFails() async {
        // Port 1 on loopback: reserved, nothing listens there.
        let endpoint = URL(string: "http://127.0.0.1:1/v1")!
        let result = await MemberProbe.probe(endpoint: endpoint, apiKey: nil, timeoutSeconds: 2)
        #expect(!result.ok)
        #expect(result.httpStatus == 0)
        #expect(!result.message.isEmpty)
    }
}
