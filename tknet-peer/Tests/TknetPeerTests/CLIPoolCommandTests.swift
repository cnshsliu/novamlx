import Foundation
import Testing
@testable import TknetPeer

@Suite("CLI member and pool edits")
struct CLIPoolCommandTests {
    private func secrets() -> FileSecretStore {
        FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-admin-\(UUID().uuidString)"))
    }

    @Test("add member stores the key and list text omits it")
    func addMemberHidesKey() throws {
        var config = PeerConfig.defaultConfig()
        let store = secrets()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "Ten",
            baseUrl: URL(string: "http://10.0.0.2:8080")!, apiKey: "super-secret-key")
        let factoryId = config.factories[0].id
        let interfaceId = try PeerAdmin.ensureInterface(&config, factoryId: factoryId, kind: .openaiChat)
        try PeerAdmin.addMember(
            &config, id: "m1", name: "alpha", interfaceId: interfaceId,
            upstreamModel: "u1", weight: 2)
        #expect(store.load(config.factory(id: factoryId)!.apiKeyRef!) == "super-secret-key")
        let text = PeerAdmin.memberList(config)
        #expect(text.contains("m1"))
        #expect(text.contains("upstream=u1"))
        #expect(!text.contains("super-secret-key"))
    }

    @Test("remove member strips it from every pool")
    func removeMemberStripsPools() throws {
        var config = PeerConfig.defaultConfig()
        let store = secrets()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "two",
            baseUrl: URL(string: "http://10.0.0.2:1")!, apiKey: nil)
        let interface1 = try PeerAdmin.ensureInterface(
            &config, factoryId: config.factories[0].id, kind: .openaiChat)
        try PeerAdmin.addMember(&config, id: "m1", name: "a", interfaceId: interface1, upstreamModel: "u")
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "three",
            baseUrl: URL(string: "http://10.0.0.3:1")!, apiKey: nil)
        let interface2 = try PeerAdmin.ensureInterface(
            &config, factoryId: config.factories[1].id, kind: .openaiChat)
        try PeerAdmin.addMember(&config, id: "m2", name: "b", interfaceId: interface2, upstreamModel: "u")
        try PeerAdmin.addPool(&config, id: nil, name: "coding", memberIds: ["m1", "m2"])
        try PeerAdmin.removeMember(&config, id: "m1")
        #expect(config.member(id: "m1") == nil)
        #expect(config.pool(named: "coding")?.memberIds == ["m2"])
    }

    @Test("pool update adds and removes members")
    func updatePoolMembership() throws {
        var config = PeerConfig.defaultConfig()
        let store = secrets()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "one",
            baseUrl: URL(string: "http://10.0.0.2:1")!, apiKey: nil)
        let shared = try PeerAdmin.ensureInterface(
            &config, factoryId: config.factories[0].id, kind: .openaiChat)
        for id in ["m1", "m2"] {
            try PeerAdmin.addMember(
                &config, id: id, name: id, interfaceId: shared, upstreamModel: "u")
        }
        try PeerAdmin.addPool(
            &config, id: "coding", name: "coding", strategy: .roundRobin,
            maxRetries: 2, memberIds: ["m1"])
        try PeerAdmin.updatePool(&config, id: "coding", addMembers: ["m2"], removeMembers: ["m1"])
        #expect(config.pool(id: "coding")?.memberIds == ["m2"])
        #expect(config.pool(id: "coding")?.strategy == .roundRobin)
        let listed = PeerAdmin.poolList(config)
        #expect(listed.contains("strategy=round_robin"))
        #expect(!listed.contains("super-secret-key"))
    }

    @Test("unknown strategy is rejected")
    func unknownStrategy() {
        #expect(throws: PeerAdminError.self) { try PeerAdmin.strategy("fastest") }
    }

    @Test("same model joins one pool")
    func ensureModelPoolSharesName() throws {
        var config = PeerConfig.defaultConfig()
        let store = secrets()
        try PeerAdmin.saveFactory(
            &config, secrets: store, id: nil, name: "one",
            baseUrl: URL(string: "http://10.0.0.2:1")!, apiKey: nil)
        let shared = try PeerAdmin.ensureInterface(
            &config, factoryId: config.factories[0].id, kind: .openaiChat)
        for id in ["m1", "m2"] {
            try PeerAdmin.addMember(
                &config, id: id, name: "Qwen3-8B", interfaceId: shared,
                upstreamModel: "Qwen3-8B")
            try PeerAdmin.ensureModelPool(&config, model: "Qwen3-8B", memberId: id)
        }
        #expect(config.pools.count == 1)
        #expect(config.pool(named: "Qwen3-8B")?.id == "Qwen3-8B")
        #expect(config.pool(named: "Qwen3-8B")?.memberIds == ["m1", "m2"])
        PeerAdmin.detachMember(&config, memberId: "m1", fromPoolNamed: "Qwen3-8B")
        #expect(config.pool(named: "Qwen3-8B")?.memberIds == ["m2"])
    }
}
