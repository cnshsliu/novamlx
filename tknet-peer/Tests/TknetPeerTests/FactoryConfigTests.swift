import Foundation
import Testing
@testable import TknetPeer

@Suite("Factories")
struct FactoryConfigTests {
    private func tmpPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-factory-\(UUID().uuidString).json")
    }

    private func secrets() -> FileSecretStore {
        FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-factory-sec-\(UUID().uuidString)"))
    }

    private func member(_ id: String, endpoint: String, type: SourceType = .openaiCompatible) -> MemberConfig {
        MemberConfig(
            id: id, name: id, type: type,
            endpoint: URL(string: endpoint)!,
            apiKeyRef: "src/\(id)", upstreamModel: "model-\(id)")
    }

    private func preFactoryJSON() -> String {
        """
        {
          "serverURL": "wss://tknet.ai",
          "members": [
            {"id": "oa1", "name": "oa1", "type": "openaiCompatible",
             "endpoint": "https://api.one.com/v1", "apiKeyRef": "src/oa1", "upstreamModel": "m"},
            {"id": "oa2", "name": "oa2", "type": "openaiCompatible",
             "endpoint": "https://api.one.com/v2", "apiKeyRef": "src/oa2", "upstreamModel": "m"},
            {"id": "ob1", "name": "ob1", "type": "anthropic",
             "endpoint": "http://two.example.com:8080/x", "apiKeyRef": "src/ob1", "upstreamModel": "m"},
            {"id": "loc", "name": "loc", "type": "localNovaMLX",
             "endpoint": "http://127.0.0.1:6590/v1", "apiKeyRef": "src/loc", "upstreamModel": "m"}
          ]
        }
        """
    }

    @Test("pre-factory file derives factories by host:port; local NovaMLX stays ungrouped")
    func migrationGroupsByBase() throws {
        let config = try JSONDecoder().decode(PeerConfig.self, from: Data(preFactoryJSON().utf8))
        #expect(config.factories.count == 2)
        let one = config.factories.first { $0.baseUrl.host == "api.one.com" }
        let two = config.factories.first { $0.baseUrl.host == "two.example.com" }
        #expect(one != nil)
        #expect(two?.name == "two.example.com:8080")
        #expect(config.member(id: "oa1")?.factoryId == one?.id)
        #expect(config.member(id: "oa2")?.factoryId == one?.id)
        #expect(config.member(id: "ob1")?.factoryId == two?.id)
        #expect(config.member(id: "loc")?.factoryId == nil)
        #expect(config.members(in: one!).count == 2)
    }

    @Test("the factories key itself is the one-time latch: deleting the last factory sticks")
    func latchPreventsResurrection() throws {
        let url = tmpPath()
        defer { try? FileManager.default.removeItem(at: url) }

        var config = try JSONDecoder().decode(PeerConfig.self, from: Data(preFactoryJSON().utf8))
        #expect(config.factories.count == 2)

        // Legitimately delete every factory, keeping members orphaned.
        for factory in config.factories { config.removeFactory(id: factory.id) }
        #expect(config.factories.isEmpty)
        #expect(config.members.count == 4)
        #expect(config.members.allSatisfy { $0.factoryId == nil })

        try ConfigStore.save(config, to: url)
        let reloaded = try ConfigStore.load(from: url)
        #expect(reloaded.factories.isEmpty)   // not resurrected
        #expect(reloaded.members.allSatisfy { $0.factoryId == nil })
    }

    @Test("round-trip keeps factories and factoryIds")
    func roundTrip() throws {
        let url = tmpPath()
        defer { try? FileManager.default.removeItem(at: url) }
        var config = PeerConfig.defaultConfig()
        let factory = FactoryConfig(id: "f1", name: "One", baseUrl: URL(string: "https://api.one.com")!)
        config.factories = [factory]
        config.members = [member("oa1", endpoint: "https://api.one.com/v1")]
        config.members[0].factoryId = "f1"
        try ConfigStore.save(config, to: url)
        let loaded = try ConfigStore.load(from: url)
        #expect(loaded == config)
        #expect(loaded.factories.first?.apiKeyRef == nil)
        #expect(loaded.member(id: "oa1")?.factoryId == "f1")
    }

    @Test("saveFactory strips path/query and rejects duplicate bases")
    func saveFactoryValidation() throws {
        let store = secrets()
        var config = PeerConfig.defaultConfig()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "One",
            baseUrl: URL(string: "https://api.one.com/some/path?key=1")!, apiKey: "sk-1")
        #expect(config.factories.count == 1)
        #expect(config.factories[0].baseUrl.absoluteString == "https://api.one.com")
        #expect(store.load(config.factories[0].apiKeyRef!) == "sk-1")

        #expect(throws: PeerAdminError.self) {
            try PeerAdmin.saveFactory(
                &config, secrets: store, id: nil, name: "One again",
                baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        }
        // A different base is fine, and updating the same factory is fine.
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "Two",
            baseUrl: URL(string: "http://two.example.com:8080")!, apiKey: nil)
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: config.factories[0].id, name: "One renamed",
            baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        #expect(config.factories.count == 2)
        #expect(config.factories.first { $0.name == "One renamed" } != nil)
    }

    @Test("editing a factory URL rebases member endpoints, path intact")
    func factoryRebase() throws {
        let store = secrets()
        var config = PeerConfig.defaultConfig()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "One",
            baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        let factoryId = config.factories[0].id
        try PeerAdmin.addMember(
            &config, secrets: store, id: "oa1", name: "oa1",
            endpoint: URL(string: "https://api.one.com/v1")!,
            upstreamModel: "m", apiKey: "k", weight: nil,
            type: .openaiCompatible, factoryId: factoryId)

        try PeerAdmin.saveFactory(
            &config, secrets: store, id: factoryId, name: "One",
            baseUrl: URL(string: "https://api.moved.com:9000/base")!, apiKey: nil)
        // The pasted /base path is stripped (factories are host:port only);
        // the member keeps its own /v1 path on the new base.
        #expect(config.member(id: "oa1")?.endpoint.absoluteString == "https://api.moved.com:9000/v1")
        #expect(config.member(id: "oa1")?.factoryId == factoryId)
    }

    @Test("removeFactory orphans by default; deleteMembers strips pools too")
    func removeFactoryModes() throws {
        let store = secrets()
        var config = PeerConfig.defaultConfig()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "One",
            baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        let factoryId = config.factories[0].id
        try PeerAdmin.addMember(
            &config, secrets: store, id: "oa1", name: "oa1",
            endpoint: URL(string: "https://api.one.com/v1")!,
            upstreamModel: "m", apiKey: "k", weight: nil,
            type: .openaiCompatible, factoryId: factoryId)
        try PeerAdmin.addPool(&config, id: nil, name: "pool", memberIds: ["oa1"])

        // Orphan mode (default): member survives, pool membership survives.
        try PeerAdmin.removeFactory(&config, id: factoryId, deleteMembers: false)
        #expect(config.factories.isEmpty)
        #expect(config.member(id: "oa1") != nil)
        #expect(config.member(id: "oa1")?.factoryId == nil)
        #expect(config.pool(id: "pool")?.memberIds == ["oa1"])

        // Cascade mode: member and its pool rows go too.
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "One",
            baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        let factoryId2 = config.factories[0].id
        config.upsertMember(MemberConfig(
            id: "oa2", name: "oa2", type: .openaiCompatible,
            endpoint: URL(string: "https://api.one.com/v1")!,
            apiKeyRef: "src/oa2", upstreamModel: "m", factoryId: factoryId2))
        try PeerAdmin.updatePool(&config, id: "pool", addMembers: ["oa2"])
        try PeerAdmin.removeFactory(&config, id: factoryId2, deleteMembers: true)
        #expect(config.member(id: "oa2") == nil)
        #expect(config.pool(id: "pool")?.memberIds == ["oa1"])
    }

    @Test("addMember rejects an unknown factory")
    func addMemberUnknownFactory() throws {
        let store = secrets()
        var config = PeerConfig.defaultConfig()
        #expect(throws: PeerAdminError.self) {
            try PeerAdmin.addMember(
                &config, secrets: store, id: "x", name: "x",
                endpoint: URL(string: "https://api.one.com/v1")!,
                upstreamModel: "m", apiKey: nil, weight: nil,
                type: .openaiCompatible, factoryId: "missing")
        }
    }

    @Test("server switching carries factories through ServerMemory")
    func serverMemoryCarriesFactories() throws {
        var config = PeerConfig.defaultConfig()
        let factory = FactoryConfig(id: "f1", name: "One", baseUrl: URL(string: "https://api.one.com")!)
        config.factories = [factory]
        config.members = [member("oa1", endpoint: "https://api.one.com/v1")]
        config.members[0].factoryId = "f1"

        config.rememberPeerName("https://tknet.ai", name: "main")   // snapshot row
        let switched = config.activateServer(URL(string: "https://other.example.com")!)
        #expect(switched)
        #expect(config.factories.isEmpty)
        #expect(config.members.isEmpty)

        let back = config.activateServer(URL(string: "https://tknet.ai")!)
        #expect(back)
        #expect(config.factories.map(\.name) == ["One"])
        #expect(config.member(id: "oa1")?.factoryId == "f1")
    }
}
