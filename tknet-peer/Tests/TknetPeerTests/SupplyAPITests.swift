import Foundation
import Testing
@testable import TknetPeer

@Suite("Supply API")
struct SupplyAPITests {
    @Test("language and vision use chat; image is generations only")
    func kinds() {
        #expect(SupplyAPI.contract(modelId: "org/Qwen3-8B", modelType: "llm", family: "qwen")?.kind == "openai.chat")
        #expect(SupplyAPI.contract(modelId: "org/Qwen-VL", modelType: "vlm", family: "qwen")?.operations.first?.path == "/v1/chat/completions")
        let image = SupplyAPI.contract(modelId: "org/Qwen-Image", modelType: "image", family: "qwenImage")
        #expect(image?.kind == "openai.image.generations")
        #expect(image?.operations.map(\.path) == ["/v1/images/generations"])
        let edits = SupplyAPI.contract(modelId: "org/edit", modelType: "image_edit", family: "other")
        #expect(edits?.kind == "openai.image.edits")
        #expect(edits?.operations.map(\.path) == ["/v1/images/edits"])
        #expect(SupplyAPI.contract(modelId: "org/Qwen3-TTS", modelType: "audio", family: "qwen3Tts")?.kind == "openai.audio.speech")
        #expect(SupplyAPI.contract(modelId: "org/Qwen3-ASR", modelType: "audio", family: "qwen3Asr")?.operations.first?.path == "/v1/audio/transcriptions")
        #expect(SupplyAPI.contract(modelId: "org/bge", modelType: "embedding", family: "other")?.kind == "openai.embeddings")
        #expect(SupplyAPI.contract(modelId: "org/Laya", modelType: "decision", family: "laya")?.operations.first?.path == "/v1/decisions")
    }

    @Test("variations is not an interface")
    func noVariations() {
        #expect(SupplyAPI.contract(kind: "openai.image") == nil)
        for kind in ["openai.image.generations", "openai.image.edits", "openai.chat", "openai.embeddings"] {
            let api = SupplyAPI.contract(kind: kind)
            #expect(api != nil)
            #expect(api?.operations.contains { $0.path == "/v1/images/variations" } != true)
        }
    }

    @Test("nil api allows only chat")
    func nilApiChatOnly() {
        #expect(SupplyAPI.allows(nil, gatewayPath: "/v1/chat/completions"))
        #expect(!SupplyAPI.allows(nil, gatewayPath: "/v1/embeddings"))
        #expect(SupplyAPI.allows(SupplyAPI.embeddings, gatewayPath: "/v1/embeddings"))
        #expect(!SupplyAPI.allows(SupplyAPI.embeddings, gatewayPath: "/v1/chat/completions"))
        #expect(SupplyAPI.allows(SupplyAPI.imageGenerations, gatewayPath: "/v1/images/generations"))
        #expect(!SupplyAPI.allows(SupplyAPI.imageGenerations, gatewayPath: "/v1/images/edits"))
    }

    @Test("background removal is chosen from the model id")
    func matting() {
        let api = SupplyAPI.contract(modelId: "birefnet-portrait", modelType: "llm", family: "other")
        #expect(api?.kind == "novamlx.background-removal")
        #expect(api?.operations.first?.path == "/v1/images/background-removals")
        #expect(api?.operations.first?.contentType == "multipart/form-data")
        #expect(SupplyAPI.contract(modelId: "weights/bria-rmbg", modelType: "", family: "")?.kind == "novamlx.background-removal")
    }

    @Test("an unrecognized audio model does not borrow the chat path")
    func unknownAudio() {
        #expect(SupplyAPI.contract(modelId: "org/mystery-audio", modelType: "audio", family: "other") == nil)
    }

    @Test("hello carries the contract and an old capability still decodes")
    func wire() throws {
        let api = SupplyAPI.contract(modelId: "qwen", modelType: "llm", family: "qwen")
        let cap = Capability(
            demandId: "d1", model: "qwen", sourceId: "s1",
            sourceType: .localNovaMLX, priceIn: 0, priceOut: 0, api: api)
        let decoded = try FrameCodec.decode(FrameCodec.encode(.capabilitiesUpdate([cap])))
        #expect(decoded == .capabilitiesUpdate([cap]))

        let legacy = #"{"type":"capabilities.update","data":{"capabilities":[{"demandId":"d1","model":"m","sourceId":"s","sourceType":"openaiCompatible","priceIn":0,"priceOut":0}]}}"#
        let old = try FrameCodec.decode(legacy)
        guard case .capabilitiesUpdate(let caps) = old else {
            Issue.record("expected capabilities")
            return
        }
        #expect(caps.first?.api == nil)
    }

    @Test("wireCapabilities uses pool.api and defaults missing to chat")
    func wireCapabilities() {
        var config = PeerConfig.defaultConfig()
        config.members = [
            MemberConfig(
                id: "s1", name: "emb", type: .openaiCompatible,
                endpoint: URL(string: "http://127.0.0.1:9/v1")!,
                apiKeyRef: "src/s1", upstreamModel: "u"),
        ]
        config.pools = [
            PoolConfig(id: "s1", name: "emb", memberIds: ["s1"], api: SupplyAPI.embeddings),
            PoolConfig(id: "chat-pool", name: "chatty", memberIds: ["s1"]),
        ]
        config.capabilities = [
            Capability(demandId: "d1", model: "emb", sourceId: "s1",
                       sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
            Capability(demandId: "d2", model: "chatty", sourceId: "chat-pool",
                       sourceType: .openaiCompatible, priceIn: 0, priceOut: 0,
                       api: SupplyAPI.speech),
        ]
        let wire = config.wireCapabilities
        #expect(wire.first { $0.demandId == "d1" }?.api?.kind == "openai.embeddings")
        #expect(wire.first { $0.demandId == "d2" }?.api?.kind == "openai.chat")
    }
}
