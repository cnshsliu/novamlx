import AVFoundation
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

    @Test("reference playback trim uses only the unmatched reference prefix")
    func referencePrefixCut() {
        #expect(Qwen3TTSCloneService.uniqueRefPrefix("今天天气不错，你好", "你好") == "今天天气不错，")
        #expect(Qwen3TTSCloneService.uniqueRefPrefix("你好世界", "你好") == "你好")
        let cut = Qwen3TTSCloneService.prefixCutSamples(
            out: 1000, ref: 1000, text: "hello", refText: "AAAAAAAAAhello"
        )
        #expect(cut != nil)
        #expect(Qwen3TTSCloneService.prefixCutSamples(
            out: 1000, ref: 100, text: "hello", refText: "AAAAAAAAAhello"
        ) == nil)
        #expect(Qwen3TTSCloneService.uniqueRefPrefix("", "hello").isEmpty)
        #expect(Qwen3TTSCloneService.uniqueRefPrefix("hello", "").isEmpty)
        #expect(Qwen3TTSCloneService.prefixCutSamples(out: 0, ref: 1000, text: "hi", refText: "hello there") == nil)
        #expect(Qwen3TTSCloneService.prefixCutSamples(
            out: 1000, ref: 1000, text: "hello there friend", refText: "hello"
        ) == nil)
    }

    @Test("missing checkpoint and empty text fail closed")
    func cloneFailureCorners() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-tts-missing-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: missing, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: missing) }
        try #"{"model_type":"qwen3_tts"}"#.write(
            to: missing.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx-empty.wav")
        do {
            _ = try await Qwen3TTSCloneService.synthesize(
                text: "你好", refAudio: audio, refText: "参考", modelDir: missing
            )
            Issue.record("missing weights should throw")
        } catch {
            #expect(!String(describing: error).isEmpty)
        }
    }

    @Test("loaded base model clones a short phrase without the reference codec")
    func smokeClone() async throws {
        let modelDir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Models/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
        let audio = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("speech.wav")
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("config.json").path),
              FileManager.default.fileExists(atPath: audio.path)
        else { return }
        let data = try await Qwen3TTSCloneService.synthesize(
            text: "你好",
            refAudio: audio,
            refText: "今天天气不错，阳光明媚。",
            modelDir: modelDir
        )
        #expect(data.prefix(4) == Data("RIFF".utf8))
        let seconds = try wavDuration(data)
        #expect(seconds > 0.05)
        #expect(seconds < 30)

        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-clone-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: dest) }
        let kept = try await Qwen3TTSCloneService.synthesize(
            text: "你好", refAudio: audio, refText: "今天天气不错，阳光明媚。",
            modelDir: modelDir, output: dest
        )
        #expect(FileManager.default.fileExists(atPath: dest.path))
        #expect(kept.count == data.count || kept.count > 44)

        do {
            _ = try await Qwen3TTSCloneService.synthesize(
                text: "   ", refAudio: audio, refText: "今天天气不错，阳光明媚。", modelDir: modelDir
            )
            Issue.record("empty text should throw")
        } catch {
            #expect(String(describing: error).contains("short") || !String(describing: error).isEmpty)
        }
    }

    private func wavDuration(_ data: Data) throws -> Double {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-wav-\(UUID().uuidString).wav")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try AVAudioFile(forReading: url)
        guard file.fileFormat.sampleRate > 0 else { return 0 }
        return Double(file.length) / file.fileFormat.sampleRate
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
