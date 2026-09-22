import Foundation
import Testing
@testable import NovaMLXEngine

@Suite("Voice profiles")
struct VoiceProfileManagerTests {
    @Test("uniqueName increments without colliding")
    func uniqueName() {
        #expect(VoiceProfileManager.uniqueName(base: "Voice", existing: []) == "Voice")
        #expect(VoiceProfileManager.uniqueName(base: "Voice", existing: ["Voice"]) == "Voice 2")
        #expect(VoiceProfileManager.uniqueName(base: "Voice", existing: ["Voice", "Voice 2"]) == "Voice 3")
        #expect(VoiceProfileManager.uniqueName(base: "新声音", existing: ["新声音"]) == "新声音 2")
    }
}
