import Foundation
import Testing
@testable import NovaMLXEngine

@Suite("Qwen3-TTS Base clone")
struct Qwen3TTSCloneServiceTests {
    @Test("Base ids are accepted, CustomVoice/VoiceDesign rejected")
    func baseIdFilter() {
        #expect(Qwen3TTSCloneService.isBaseModelId("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"))
        #expect(Qwen3TTSCloneService.isBaseModelId("Qwen3-TTS-12Hz-0.6B-Base-bf16"))
        #expect(!Qwen3TTSCloneService.isBaseModelId("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"))
        #expect(!Qwen3TTSCloneService.isBaseModelId("Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"))
        #expect(!Qwen3TTSCloneService.isBaseModelId("smcleod/dots.tts-soar-mlx"))
    }

    @Test("directory detection reads model_type and speech_tokenizer")
    func directoryDetect() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-qwen3tts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = """
        {"model_type":"qwen3_tts","architectures":["Qwen3TTSForConditionalGeneration"]}
        """
        try cfg.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(Qwen3TTSCloneService.isQwen3TTSDirectory(dir))

        let other = dir.appendingPathComponent("not-tts")
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try "{\"model_type\":\"qwen3\"}".write(
            to: other.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(!Qwen3TTSCloneService.isQwen3TTSDirectory(other))
    }

    @Test("clone engine list never substitutes CustomVoice or a missing pick")
    func engineListNoFallback() {
        let pinned = [
            Qwen3TTSCloneService.EngineOption(id: Qwen3TTSCloneService.defaultModelId, family: .qwen3Tts),
            Qwen3TTSCloneService.EngineOption(id: Qwen3TTSCloneService.dotsModelId, family: .dotsTts),
        ]
        let extra = [
            Qwen3TTSCloneService.EngineOption(id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", family: .qwen3Tts),
            Qwen3TTSCloneService.EngineOption(id: Qwen3TTSCloneService.dotsModelId, family: .dotsTts),
        ]
        let list = Qwen3TTSCloneService.mergeEngineList(pinned: pinned, extra: extra)
        #expect(list.map(\.id) == [
            Qwen3TTSCloneService.defaultModelId,
            Qwen3TTSCloneService.dotsModelId,
        ])
        #expect(Qwen3TTSCloneService.isAllowedCloneEngine(id: Qwen3TTSCloneService.dotsModelId, family: .dotsTts))
        #expect(!Qwen3TTSCloneService.isAllowedCloneEngine(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", family: .qwen3Tts))
    }

    @Test("clone script never names a stock speaker")
    func cloneScriptOmitsChelsie() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/NovaMLXUtils/Resources/scripts/qwen3_tts_clone.py")
        let src = try String(contentsOf: url, encoding: .utf8)
        #expect(!src.contains("voice=\"Chelsie\""))
        #expect(src.contains("patch_icl_decode_generated_only"))
        #expect(src.contains("unique_ref_prefix"))
        #expect(src.contains("full_codes = gen_codes"))
    }

    @Test("UI error strips transformers noise")
    func shortCloneError() {
        let stderr = """
        [transformers] You are using a model of type `qwen3_tts` to instantiate a model of type ``.
        {"error": "Clone decoder returned the reference recording"}
        """
        let msg = Qwen3TTSCloneService.shortCloneError(stderr: stderr, stdout: "")
        #expect(msg.contains("reference recording"))
        #expect(!msg.contains("transformers"))
    }

    @Test("prefetch JSON decoder")
    func prefetchJSON() throws {
        let data = "{\"ok\":true,\"model\":\"x\"}".data(using: .utf8)!
        #expect(try Qwen3TTSCloneService.decodePrefetchJSON(data))
        let bad = "{\"ok\":false}".data(using: .utf8)!
        #expect(try Qwen3TTSCloneService.decodePrefetchJSON(bad) == false)
    }
}
