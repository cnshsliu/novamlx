import Foundation
import Testing
@testable import NovaMLXEngine

@Suite("Forced aligner")
struct ForcedAlignerServiceTests {
    @Test("decodeWords parses aligner JSON")
    func decodeWords() throws {
        let json = """
            [{"text":"今","start":0.0,"end":0.12},{"text":"天","start":0.12,"end":0.24}]
            """.data(using: .utf8)!
        let words = try ForcedAlignerService.decodeWords(json)
        #expect(words.count == 2)
        #expect(words[0].text == "今")
        #expect(words[1].start == 0.12)
    }

    @Test("Chinese characters and English words split the way the aligner expects")
    func wordSplit() {
        #expect(ForcedAlignerService.alignedWords(in: "今天天气", language: "Chinese") == ["今", "天", "天", "气"])
        #expect(ForcedAlignerService.alignedWords(in: "Hello, world!", language: "English") == ["Hello", "world"])
        #expect(ForcedAlignerService.fixAlignTimestamps([1, 3, 2, 4]) == [1, 3, 3, 4])
        #expect(ForcedAlignerService.fixAlignTimestamps([0, 80, 160]) == [0, 80, 160])
    }

    @Test("loaded aligner returns a timestamp per word")
    func smokeAlign() async throws {
        let modelDir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Models/mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        let audio = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("speech.wav")
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("config.json").path),
              FileManager.default.fileExists(atPath: audio.path)
        else { return }
        let words = try await ForcedAlignerService.align(
            audioURL: audio, text: "hello", language: "English"
        )
        #expect(words.count == 1)
        #expect(words[0].text == "hello")
        #expect(words[0].end >= words[0].start)
    }

    @Test("splitter covers empty, mixed, and script corners")
    func wordCorners() {
        #expect(ForcedAlignerService.alignedWords(in: "", language: "Chinese").isEmpty)
        #expect(ForcedAlignerService.alignedWords(in: "!!!", language: "English").isEmpty)
        #expect(ForcedAlignerService.alignedWords(in: "   ", language: "English").isEmpty)
        #expect(ForcedAlignerService.alignedWords(in: "it's fine", language: "english") == ["it's", "fine"])
        #expect(ForcedAlignerService.alignedWords(in: "Hello今天", language: "Chinese") == ["Hello", "今", "天"])
        #expect(ForcedAlignerService.alignedWords(in: "Hello 今天", language: "English") == ["Hello", "今", "天"])
        #expect(ForcedAlignerService.alignedWords(in: "あ漢", language: "Japanese") == ["あ", "漢"])
        #expect(ForcedAlignerService.alignedWords(in: "한글A", language: "Korean") == ["한", "글", "A"])
        let prompt = ForcedAlignerService.alignPrompt(words: ["a", "b"], audioTokens: 0)
        #expect(prompt == "<|audio_start|><|audio_pad|><|audio_end|>a<timestamp><timestamp>b<timestamp><timestamp>")
        #expect(ForcedAlignerService.alignPrompt(words: ["今"], audioTokens: 3).contains(
            String(repeating: "<|audio_pad|>", count: 3)))
    }

    @Test("timestamp repair interpolates a long glitch and an edge glitch")
    func timestampCorners() {
        #expect(ForcedAlignerService.fixAlignTimestamps([]).isEmpty)
        #expect(ForcedAlignerService.fixAlignTimestamps([7]) == [7])
        #expect(ForcedAlignerService.fixAlignTimestamps([0, 100, 50, 40, 30, 200]) == [0, 100, 125, 150, 175, 200])
        #expect(ForcedAlignerService.fixAlignTimestamps([5, 0, 10]) == [5, 5, 10])
    }

    @Test("missing weights and bad JSON fail closed")
    func failureCorners() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-align-missing-\(UUID().uuidString)")
        do {
            try await ForcedAlignerService.ensureModel(at: missing)
            Issue.record("missing directory should throw")
        } catch {
            #expect(String(describing: error).contains("not found"))
        }
        #expect(throws: Error.self) {
            try ForcedAlignerService.decodeWords(Data("{\"text\":\"x\"}".utf8))
        }
        #expect(try ForcedAlignerService.decodeWords(Data("[]".utf8)).isEmpty)
    }

    @Test("loaded aligner covers several words, Chinese, and punctuation")
    func alignCorners() async throws {
        let modelDir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Models/mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        let audio = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("speech.wav")
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("config.json").path),
              FileManager.default.fileExists(atPath: audio.path)
        else { return }
        try await ForcedAlignerService.ensureModel(at: modelDir)
        let english = try await ForcedAlignerService.align(
            audioURL: audio, text: "hello there", language: "English"
        )
        #expect(english.map(\.text) == ["hello", "there"])
        #expect(english.allSatisfy { $0.end >= $0.start && $0.start >= 0 })
        let chinese = try await ForcedAlignerService.align(audioURL: audio, text: "你好")
        #expect(chinese.map(\.text) == ["你", "好"])
        #expect(chinese.allSatisfy { $0.end >= $0.start })
        let blank = try await ForcedAlignerService.align(audioURL: audio, text: "!!!", language: "English")
        #expect(blank.isEmpty)
        #expect(ForcedAlignerService.resolvedLanguage(text: "hi", requested: "") == "English")
        #expect(ForcedAlignerService.resolvedLanguage(text: "こんにちは", requested: nil) == "Chinese")
    }

    @Test("Chinese text selects Chinese language")
    func languageDetect() {
        #expect(ForcedAlignerService.resolvedLanguage(text: "今天天气很好", requested: nil) == "Chinese")
        #expect(ForcedAlignerService.resolvedLanguage(text: "Hello world", requested: nil) == "English")
        #expect(ForcedAlignerService.resolvedLanguage(text: "Hello", requested: "Korean") == "Korean")
    }

    @Test("clone language defaults and replace only stock scripts")
    func cloneLanguageScripts() {
        #expect(!VoiceCloneLanguage.chinese.defaultTranscript.isEmpty)
        #expect(!VoiceCloneLanguage.english.defaultTranscript.isEmpty)
        #expect(VoiceCloneLanguage.chinese.alignerLanguage == "Chinese")
        #expect(VoiceCloneLanguage.english.asrCode == "en")
        #expect(VoiceCloneLanguage.shouldReplaceTranscript("", switchingFrom: .english))
        #expect(VoiceCloneLanguage.shouldReplaceTranscript(
            VoiceCloneLanguage.english.defaultTranscript, switchingFrom: .english))
        #expect(!VoiceCloneLanguage.shouldReplaceTranscript("my custom take", switchingFrom: .english))
    }
}
