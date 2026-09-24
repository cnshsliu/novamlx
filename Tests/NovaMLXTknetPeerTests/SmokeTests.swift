import Testing
@testable import NovaMLXTknetPeer

@Suite("Tknet Peer smoke")
struct SmokeTests {
    @Test("module exposes a version")
    func versionExists() {
        #expect(!TknetPeer.version.isEmpty)
    }
}
