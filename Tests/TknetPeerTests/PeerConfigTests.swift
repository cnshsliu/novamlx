import Foundation
import Testing
@testable import TknetPeer

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
        config.members = [MemberConfig(
            id: "s1", name: "Local NovaMLX", type: .localNovaMLX,
            endpoint: URL(string: "http://127.0.0.1:6590/v1")!,
            apiKeyRef: "src/s1", upstreamModel: "Qwen/Qwen-Image-2.1"
        )]
        config.capabilities = [Capability(
            demandId: "d1", model: "qwen", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0
        )]
        config.migrateLegacyPoolsIfNeeded()
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
        #expect(c.serverMemory.isEmpty)
    }

    @Test("https and wss are the same saved server")
    func sameSavedServer() {
        let https = PeerConfig.normalizedREST(URL(string: "https://tknet.ai/")!)
        let wss = PeerConfig.normalizedREST(URL(string: "wss://tknet.ai")!)
        #expect(https == "https://tknet.ai")
        #expect(https == wss)
    }

    @Test("old config without server memory still loads")
    func legacyDecodes() throws {
        let json = #"{"serverURL":"wss://tknet.ai","peerId":"p","registrations":{"wss://tknet.ai":"p"}}"#
        let config = try JSONDecoder().decode(PeerConfig.self, from: Data(json.utf8))
        #expect(config.serverMemory.isEmpty)
        #expect(config.peerId == "p")
        #expect(config.registrations["wss://tknet.ai"] == "p")
    }

    @Test("adopt keeps the peer id and leaves the name empty")
    func adoptKeepsPeerId() {
        var config = PeerConfig.defaultConfig()
        config.peerId = "p"
        config.registrations["wss://tknet.ai"] = "p"
        let adopted = config.adoptRegistrationsIntoMemory()
        #expect(adopted)
        #expect(config.serverMemory.first?.server == "https://tknet.ai")
        #expect(config.serverMemory.first?.peerName == "")
        #expect(config.peerId == "p")
        let again = config.adoptRegistrationsIntoMemory()
        #expect(!again)
    }

    @Test("activate restores each peer id and keeps each name")
    func activateRestoresIdentity() {
        var config = PeerConfig.defaultConfig()
        config.registrations["wss://tknet.ai"] = "peer-a"
        config.registrations["wss://other.example:8443"] = "peer-b"
        config.peerId = "peer-a"
        let source = MemberConfig(
            id: "s1", name: "Local", type: .localNovaMLX,
            endpoint: URL(string: "http://127.0.0.1:6590/v1")!,
            apiKeyRef: "src/s1", upstreamModel: "m")
        config.serverMemory = [
            ServerMemory(server: "https://tknet.ai", peerName: "alpha", members: [source]),
            ServerMemory(server: "https://other.example:8443", peerName: "beta"),
        ]
        config.members = [source]
        let toOther = config.activateServer(URL(string: "https://other.example:8443")!)
        #expect(toOther)
        #expect(config.peerId == "peer-b")
        #expect(config.members.isEmpty)
        #expect(config.serverMemory.first { $0.server == "https://tknet.ai" }?.peerName == "alpha")
        let back = config.activateServer(URL(string: "https://tknet.ai")!)
        #expect(back)
        #expect(config.peerId == "peer-a")
        #expect(config.members.map(\.id) == ["s1"])
        #expect(config.serverMemory.first { $0.server == "https://other.example:8443" }?.peerName == "beta")
    }

    @Test("forget drops that server and keeps the other peer id")
    func forgetKeepsTheOther() {
        var config = PeerConfig.defaultConfig()
        config.registrations["wss://tknet.ai"] = "peer-a"
        config.registrations["ws://127.0.0.1:9000"] = "peer-b"
        config.peerId = "peer-a"
        config.serverMemory = [
            ServerMemory(server: "https://tknet.ai", peerName: "alpha"),
            ServerMemory(server: "http://127.0.0.1:9000", peerName: "beta"),
        ]
        config.forgetServer("https://tknet.ai")
        #expect(config.registrations["wss://tknet.ai"] == nil)
        #expect(config.peerId == "peer-b")
        #expect(config.serverMemory.map(\.peerName) == ["beta"])
    }

    @Test("remember name does not change the peer id")
    func rememberNameLeavesPeerId() {
        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-a"
        config.registrations["wss://tknet.ai"] = "peer-a"
        config.rememberPeerName("https://tknet.ai", name: "lucas-mac")
        #expect(config.peerId == "peer-a")
        #expect(config.serverMemory.first?.peerName == "lucas-mac")
        #expect(config.registrations["wss://tknet.ai"] == "peer-a")
    }

    @Test("a demand matches a downloaded model id")
    func demandMatchesLocalModel() {
        let downloaded = ["mlx-community/Qwen3-8B-4bit", "org/Other"]
        #expect(DemandModelMatch.localId(for: "missing", downloadedIds: downloaded, loadedIds: []) == nil)
        #expect(DemandModelMatch.localId(
            for: "Qwen3-8B-4bit", downloadedIds: downloaded, loadedIds: []) == "mlx-community/Qwen3-8B-4bit")
        #expect(DemandModelMatch.localId(
            for: "mlx-community/Qwen3-8B-4bit", downloadedIds: downloaded, loadedIds: [])
                == "mlx-community/Qwen3-8B-4bit")
        #expect(!DemandModelMatch.isLoaded("mlx-community/Qwen3-8B-4bit", loadedIds: []))
        #expect(DemandModelMatch.isLoaded("mlx-community/Qwen3-8B-4bit", loadedIds: ["mlx-community/Qwen3-8B-4bit"]))
    }

    @Test("localhost is available only inside a novamlx checkout")
    func localServerOnlyInCheckout() {
        #expect(TknetServers.allowsLocalServer(bundlePath: "/Users/lucas/dev/novamlx/dist/NovaMLX.app"))
        #expect(!TknetServers.allowsLocalServer(bundlePath: "/Applications/NovaMLX.app"))
        #expect(TknetServers.choices(allowsLocal: false) == [TknetServers.official])
        #expect(TknetServers.choices(allowsLocal: true) == [TknetServers.official, TknetServers.localDev])
    }
}
