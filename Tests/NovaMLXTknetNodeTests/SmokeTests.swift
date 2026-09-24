import Testing
@testable import NovaMLXTknetNode

@Suite("Tknet Node smoke")
struct SmokeTests {
    @Test("module exposes a version")
    func versionExists() {
        #expect(!TknetNode.version.isEmpty)
    }
}
