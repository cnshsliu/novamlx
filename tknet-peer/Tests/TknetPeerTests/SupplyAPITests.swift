import Foundation
import Testing
@testable import TknetPeer

@Suite("Supply API")
struct SupplyAPITests {
    @Test("language and vision use chat, image does not")
    func kinds() {
        #expect(SupplyAPI.contract(modelId: "org/Qwen3-8B", modelType: "llm", family: "qwen")?.kind == "openai.chat")
        #expect(SupplyAPI.contract(modelId: "org/Qwen-VL", modelType: "vlm", family: "qwen")?.operations.first?.path == "/v1/chat/completions")
        let image = SupplyAPI.contract(modelId: "org/Qwen-Image", modelType: "image", family: "qwenImage")
        #expect(image?.operations.map(\.path) == [
            "/v1/images/generations", "/v1/images/edits", "/v1/images/variations"
        ])
        #expect(SupplyAPI.contract(modelId: "org/Qwen3-TTS", modelType: "audio", family: "qwen3Tts")?.kind == "openai.audio.speech")
        #expect(SupplyAPI.contract(modelId: "org/Qwen3-ASR", modelType: "audio", family: "qwen3Asr")?.operations.first?.path == "/v1/audio/transcriptions")
        #expect(SupplyAPI.contract(modelId: "org/bge", modelType: "embedding", family: "other")?.kind == "openai.embeddings")
        #expect(SupplyAPI.contract(modelId: "org/Laya", modelType: "decision", family: "laya")?.operations.first?.path == "/v1/decisions")
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
}
