import Foundation
import Testing
@testable import TknetPeer

@Suite("Factories and interfaces")
struct FactoryConfigTests {
    private func tmpPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-factory-\(UUID().uuidString).json")
    }

    private func secrets() -> FileSecretStore {
        FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-factory-sec-\(UUID().uuidString)"))
    }

    private func factory(_ id: String = "f1", base: String, kinds: [InterfaceKind] = [.openaiChat]) -> FactoryConfig {
        FactoryConfig(
            id: id, name: id, baseUrl: URL(string: base)!,
            interfaces: kinds.enumerated().map { InterfaceConfig(id: "\(id)-i\($0.offset)", kind: $0.element) })
    }

    private func member(_ id: String, interfaceId: String) -> MemberConfig {
        MemberConfig(id: id, name: id, interfaceId: interfaceId, upstreamModel: "model-\(id)")
    }

    private func legacyJSON() -> String {
        """
        {
          "serverURL": "wss://tknet.ai",
          "factories": [{"id": "f1", "name": "One", "baseUrl": "https://api.one.com"}],
          "members": [
            {"id": "oa1", "name": "oa1", "type": "openaiCompatible",
             "endpoint": "https://api.one.com/v1", "apiKeyRef": "src/oa1", "upstreamModel": "m"}
          ]
        }
        """
    }

    @Test("legacy members (own endpoint) are wiped on the interface-schema cutover")
    func legacyMembersAreWiped() throws {
        var config = try JSONDecoder().decode(PeerConfig.self, from: Data(legacyJSON().utf8))
        #expect(config.members.count == 1)
        #expect(config.members[0].interfaceId.isEmpty)

        let changed = config.migrateToInterfaceSchemaIfNeeded()
        #expect(changed)
        #expect(config.members.isEmpty)
        #expect(config.pools.isEmpty)
        // Factories survive for re-import.
        #expect(config.factories.map(\.id) == ["f1"])

        // Idempotent.
        var mutable = config
        let changedAgain = mutable.migrateToInterfaceSchemaIfNeeded()
        #expect(!changedAgain)
    }

    @Test("interface URL = factory base + canonical kind path")
    func interfaceURLDerivation() {
        let one = factory(base: "https://api.one.com", kinds: [.openaiChat, .anthropicMessages])
        #expect(one.endpointURL(of: .openaiChat)?.absoluteString == "https://api.one.com/v1/chat/completions")
        #expect(one.endpointURL(of: .anthropicMessages)?.absoluteString == "https://api.one.com/v1/messages")
        let local = factory("f2", base: "http://127.0.0.1:6590/", kinds: [.openaiSpeech])
        #expect(local.endpointURL(of: .openaiSpeech)?.absoluteString == "http://127.0.0.1:6590/v1/audio/speech")
    }

    @Test("ensureInterface is idempotent per kind and factory-scoped")
    func ensureInterfaceIdempotent() {
        var config = PeerConfig.defaultConfig()
        config.factories = [factory("f1", base: "https://api.one.com"), factory("f2", base: "https://two.com")]
        let first = config.ensureInterface(kind: .openaiChat, on: "f1")
        let again = config.ensureInterface(kind: .openaiChat, on: "f1")
        let other = config.ensureInterface(kind: .openaiChat, on: "f2")
        #expect(first == again)
        #expect(first != other)
        #expect(config.factory(id: "f1")?.interfaces.count == 1)
        // Unknown factory -> nil.
        #expect(config.ensureInterface(kind: .openaiChat, on: "nope") == nil)
    }

    @Test("member endpoint resolves through its interface")
    func memberEndpointResolution() {
        var config = PeerConfig.defaultConfig()
        config.factories = [factory(base: "https://api.one.com", kinds: [.openaiEmbeddings])]
        config.members = [member("oa1", interfaceId: "f1-i0")]
        #expect(config.memberEndpointURL(config.members[0])?.absoluteString
            == "https://api.one.com/v1/embeddings")
        #expect(config.memberFactoryBase(config.members[0])?.host == "api.one.com")
        // Dangling interface -> nil.
        config.members.append(member("dangling", interfaceId: "gone"))
        #expect(config.memberEndpointURL(config.members[1]) == nil)
    }

    @Test("round-trip keeps factories, interfaces, and interfaceIds")
    func roundTrip() throws {
        let url = tmpPath()
        defer { try? FileManager.default.removeItem(at: url) }
        var config = PeerConfig.defaultConfig()
        config.factories = [factory(base: "https://api.one.com", kinds: [.openaiChat])]
        config.members = [member("oa1", interfaceId: "f1-i0")]
        try ConfigStore.save(config, to: url)
        let loaded = try ConfigStore.load(from: url)
        #expect(loaded == config)
        #expect(loaded.member(id: "oa1")?.interfaceId == "f1-i0")
    }

    @Test("saveFactory strips path/query, keeps interfaces, rejects duplicate bases")
    func saveFactoryValidation() throws {
        let store = secrets()
        var config = PeerConfig.defaultConfig()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "One",
            baseUrl: URL(string: "https://api.one.com/some/path?key=1")!, apiKey: "sk-1")
        #expect(config.factories.count == 1)
        #expect(config.factories[0].baseUrl.absoluteString == "https://api.one.com")
        #expect(store.load(config.factories[0].apiKeyRef!) == "sk-1")

        try PeerAdmin.ensureInterface(&config, factoryId: config.factories[0].id, kind: .openaiChat)
        // Rename keeps interfaces (and the key ref) intact.
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: config.factories[0].id, name: "One renamed",
            baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        #expect(config.factories[0].interfaces.count == 1)

        #expect(throws: PeerAdminError.self) {
            try PeerAdmin.saveFactory(
                &config, secrets: store, id: nil, name: "One again",
                baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        }
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "Two",
            baseUrl: URL(string: "http://two.example.com:8080")!, apiKey: nil)
        #expect(config.factories.count == 2)
    }

    @Test("base change needs no member rewrite — URLs always derive")
    func factoryBaseChange() throws {
        let store = secrets()
        var config = PeerConfig.defaultConfig()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "One",
            baseUrl: URL(string: "https://api.one.com")!, apiKey: nil)
        let factoryId = config.factories[0].id
        let interfaceId = try PeerAdmin.ensureInterface(&config, factoryId: factoryId, kind: .openaiChat)
        try PeerAdmin.addMember(
            &config, id: "oa1", name: "oa1", interfaceId: interfaceId, upstreamModel: "m")

        try PeerAdmin.saveFactory(
            &config, secrets: store, id: factoryId, name: "One",
            baseUrl: URL(string: "https://api.moved.com:9000/base")!, apiKey: nil)
        // The pasted /base path is stripped; the member's URL follows the new
        // base with no rewrite step.
        #expect(config.memberEndpointURL(config.member(id: "oa1")!)?.absoluteString
            == "https://api.moved.com:9000/v1/chat/completions")
    }

    @Test("removeFactory removes its interfaces and members; other factories stay")
    func removeFactoryCascades() throws {
        var config = PeerConfig.defaultConfig()
        config.factories = [
            factory("f1", base: "https://api.one.com"),
            factory("f2", base: "https://two.com"),
        ]
        config.members = [
            member("oa1", interfaceId: "f1-i0"),
            member("t1", interfaceId: "f2-i0"),
        ]
        config.pools = [PoolConfig(id: "pool", name: "pool", kind: .openaiChat, memberIds: ["oa1", "t1"])]
        config.removeFactory(id: "f1")
        #expect(config.factories.map(\.id) == ["f2"])
        #expect(config.member(id: "oa1") == nil)
        #expect(config.member(id: "t1") != nil)
        #expect(config.pool(id: "pool")?.memberIds == ["t1"])
    }

    @Test("addMember rejects an unknown interface")
    func addMemberUnknownInterface() throws {
        var config = PeerConfig.defaultConfig()
        #expect(throws: PeerAdminError.self) {
            try PeerAdmin.addMember(
                &config, id: "x", name: "x", interfaceId: "missing", upstreamModel: "m")
        }
    }

    @Test("pools are kind-homogeneous: cross-kind joins are rejected")
    func poolKindHomogeneity() throws {
        var config = PeerConfig.defaultConfig()
        config.factories = [
            factory("f1", base: "https://api.one.com", kinds: [.openaiChat]),
            factory("f2", base: "https://two.com", kinds: [.openaiEmbeddings]),
        ]
        config.members = [
            member("oa1", interfaceId: "f1-i0"),
            member("oa2", interfaceId: "f1-i0"),
            member("e1", interfaceId: "f2-i0"),
        ]
        try PeerAdmin.ensureModelPool(&config, model: "shared", memberId: "oa1")
        try PeerAdmin.ensureModelPool(&config, model: "shared", memberId: "oa2")   // same kind: fine
        #expect(config.pool(named: "shared")?.kind == .openaiChat)
        #expect(config.pool(named: "shared")?.memberIds == ["oa1", "oa2"])
        #expect(throws: PeerAdminError.self) {
            try PeerAdmin.ensureModelPool(&config, model: "shared", memberId: "e1")
        }
        #expect(throws: PeerAdminError.self) {
            try PeerAdmin.addPool(&config, id: nil, name: "mixed", memberIds: ["oa1", "e1"])
        }
    }

    @Test("server switching carries factories through ServerMemory")
    func serverMemoryCarriesFactories() throws {
        var config = PeerConfig.defaultConfig()
        config.factories = [factory(base: "https://api.one.com")]
        config.members = [member("oa1", interfaceId: "f1-i0")]

        config.rememberPeerName("https://tknet.ai", name: "main")   // snapshot row
        let switched = config.activateServer(URL(string: "https://other.example.com")!)
        #expect(switched)
        #expect(config.factories.isEmpty)
        #expect(config.members.isEmpty)

        let back = config.activateServer(URL(string: "https://tknet.ai")!)
        #expect(back)
        #expect(config.factories.map(\.name) == ["f1"])
        #expect(config.member(id: "oa1")?.interfaceId == "f1-i0")
    }
}
