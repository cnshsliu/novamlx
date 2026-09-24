import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("CLI config paths")
struct CLIConfigTests {
    @Test("expands tilde to the real home directory")
    func tilde() {
        let expanded = CLIConfig.expandTilde("~/.config/tknet-peer/peer.json")
        #expect(expanded.hasPrefix(NSHomeDirectory()))
        #expect(!expanded.contains("~"))
    }

    @Test("bare tilde expands to the home directory itself")
    func bareTilde() {
        #expect(CLIConfig.expandTilde("~") == NSHomeDirectory())
    }

    @Test("non-tilde paths pass through untouched")
    func passthrough() {
        #expect(CLIConfig.expandTilde("/etc/tknet/peer.json") == "/etc/tknet/peer.json")
        #expect(CLIConfig.expandTilde("relative/peer.json") == "relative/peer.json")
    }

    @Test("another user's tilde form is left literal (not this CLI's home)")
    func otherUserLiteral() {
        #expect(CLIConfig.expandTilde("~root/config.json") == "~root/config.json")
    }

    @Test("default config path is tilde-rooted so it works for any user")
    func defaultPath() {
        #expect(CLIConfig.defaultConfigPath.hasPrefix("~/"))
        #expect(CLIConfig.defaultConfigPath.contains("tknet-peer"))
    }
}
