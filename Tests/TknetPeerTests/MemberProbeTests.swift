import Foundation
import Testing
@testable import TknetPeer

/// MemberProbe: URL joining, model-list parsing, and a real HTTP round trip
/// against MockSourceServer's `/v1/models` route.
@Suite struct MemberProbeTests {
    @Test func modelsBaseNormalizes() {
        #expect(MemberProbe.modelsBase(URL(string: "http://127.0.0.1:6590")!).absoluteString
            == "http://127.0.0.1:6590/v1")
        #expect(MemberProbe.modelsBase(URL(string: "http://127.0.0.1:6590/")!).absoluteString
            == "http://127.0.0.1:6590/v1")
        #expect(MemberProbe.modelsBase(URL(string: "http://127.0.0.1:6590/v1")!).absoluteString
            == "http://127.0.0.1:6590/v1")
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

        let result = await MemberProbe.discoverModels(base: URL(string: "http://127.0.0.1:\(server.port)")!, apiKey: "sk-test")

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

        _ = await MemberProbe.discoverModels(base: URL(string: "http://127.0.0.1:\(server.port)")!, apiKey: nil)

        #expect(server.lastProbeAuthorization == nil)
        await server.stop()
    }

    @Test func probeUnreachableEndpointFails() async {
        // Port 1 on loopback: reserved, nothing listens there.
        let result = await MemberProbe.discoverModels(
            base: URL(string: "http://127.0.0.1:1")!, apiKey: nil)
        #expect(!result.ok)
        #expect(result.httpStatus == 0)
        #expect(!result.message.isEmpty)
    }

    @Test func draftModelDetection() {
        #expect(MemberProbe.isDraftModel("RedHatAI/Qwen3.8-27B-speculator.dspark"))
        #expect(MemberProbe.isDraftModel("deepseek-ai/dspark_qwen3_14b_block7"))
        #expect(MemberProbe.isDraftModel("deepseek-ai/DSPARK_qwen3_8b_block7_novamlx"))
        #expect(!MemberProbe.isDraftModel("mlx-community/Qwen3-8B-8bit"))
        #expect(!MemberProbe.isDraftModel("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"))
        #expect(!MemberProbe.isDraftModel("rufang/laya-hknews-tags-v4"))
    }
}
