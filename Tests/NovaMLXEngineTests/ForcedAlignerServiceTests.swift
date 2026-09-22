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
