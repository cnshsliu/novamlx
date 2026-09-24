import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("Peer config")
struct PeerConfigTests {
    private func tmpPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-\(UUID().uuidString).json")
    }

    @Test("config round-trips through disk with 0600 permissions")
    func roundTrip() throws {
        let url = tmpPath()
        defer { try? FileManager.default.removeItem(at: url) }
        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-9"
        config.sources = [SourceConfig(
            id: "s1", name: "Local NovaMLX", type: .localNovaMLX,
            endpoint: URL(string: "http://127.0.0.1:6590/v1")!,
            apiKeyRef: "src/s1", upstreamModel: "Qwen/Qwen-Image-2.1"
        )]
        config.capabilities = [Capability(
            demandId: "d1", model: "qwen", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0
        )]
        try ConfigStore.save(config, to: url)
        let loaded = try ConfigStore.load(from: url)
        #expect(loaded == config)
        #expect(try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int == 0o600)
    }

    @Test("retired demands drop out of active capabilities but keep source configs")
    func retirement() {
        var config = PeerConfig.defaultConfig()
        config.capabilities = [
            Capability(demandId: "d1", model: "a", sourceId: "s1", sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
            Capability(demandId: "d2", model: "b", sourceId: "s1", sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
        ]
        let retired = config.markRetired(demandIds: ["d1"])   // d1 gone server-side
        #expect(retired == ["d1"])
        #expect(config.activeCapabilities.map(\.demandId) == ["d2"])
        #expect(config.capabilities.count == 2)               // nothing deleted
    }

    @Test("file secret store round-trips and never logs")
    func secretStore() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tknet-sec-\(UUID().uuidString)")
        let store = FileSecretStore(directory: dir)
        defer { try? FileManager.default.removeItem(at: dir) }
        store.save("sk-source-abc", for: "src/s1")
        #expect(store.load("src/s1") == "sk-source-abc")
        store.delete("src/s1")
        #expect(store.load("src/s1") == nil)
    }

    @Test("default config has the spec defaults")
    func defaults() {
        let c = PeerConfig.defaultConfig()
        #expect(c.concurrencyLimit == 1)
        #expect(c.requestTimeoutSeconds == 300)
        #expect(c.idleMinutesBeforeSlowPoll == 5)
        #expect(c.serverURL == URL(string: "wss://tknet.ai")!)
    }
}
