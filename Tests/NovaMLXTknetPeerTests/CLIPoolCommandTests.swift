import Foundation
import Testing
@testable import NovaMLXTknetPeer

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
        try PeerAdmin.addMember(
            &config, secrets: store, id: "m1", name: "alpha",
            endpoint: URL(string: "http://10.0.0.2:8080/v1")!, upstreamModel: "u1",
            apiKey: "super-secret-key", weight: 2, isFree: true, type: .openaiCompatible)
        #expect(config.member(id: "m1")?.apiKeyRef == "src/m1")
        #expect(store.load("src/m1") == "super-secret-key")
        let text = PeerAdmin.memberList(config)
        #expect(text.contains("m1"))
        #expect(text.contains("upstream=u1"))
        #expect(!text.contains("super-secret-key"))
    }

    @Test("remove member strips it from every pool")
    func removeMemberStripsPools() throws {
        var config = PeerConfig.defaultConfig()
        let store = secrets()
        try PeerAdmin.addMember(
            &config, secrets: store, id: "m1", name: "a",
            endpoint: URL(string: "http://10.0.0.2:1/v1")!, upstreamModel: "u",
            apiKey: nil, weight: nil, isFree: false, type: .openaiCompatible)
        try PeerAdmin.addMember(
            &config, secrets: store, id: "m2", name: "b",
            endpoint: URL(string: "http://10.0.0.3:1/v1")!, upstreamModel: "u",
            apiKey: nil, weight: nil, isFree: false, type: .openaiCompatible)
        try PeerAdmin.addPool(&config, id: nil, name: "coding", memberIds: ["m1", "m2"])
        try PeerAdmin.removeMember(&config, id: "m1")
        #expect(config.member(id: "m1") == nil)
        #expect(config.pool(named: "coding")?.memberIds == ["m2"])
    }

    @Test("pool update adds and removes members")
    func updatePoolMembership() throws {
        var config = PeerConfig.defaultConfig()
        let store = secrets()
        for id in ["m1", "m2"] {
            try PeerAdmin.addMember(
                &config, secrets: store, id: id, name: id,
                endpoint: URL(string: "http://10.0.0.2:1/v1")!, upstreamModel: "u",
                apiKey: nil, weight: nil, isFree: false, type: .openaiCompatible)
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
}
