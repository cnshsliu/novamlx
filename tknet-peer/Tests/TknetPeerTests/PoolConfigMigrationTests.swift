import Foundation
import Testing
@testable import TknetPeer

@Suite("Pool config migration")
struct PoolConfigMigrationTests {
    @Test("legacy sources-only file decodes, then the interface cutover wipes it")
    func migratesLegacySources() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "sources": [{
            "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
            "endpoint": "http://127.0.0.1:6590/v1",
            "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
          }],
          "capabilities": [{
            "demandId": "d1", "model": "qwen-3-8b", "sourceId": "src-d1",
            "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
          }]
        }
        """.data(using: .utf8)!
        var config = try JSONDecoder().decode(PeerConfig.self, from: json)
        #expect(config.members.count == 1)
        #expect(config.members[0].id == "src-d1")
        #expect(config.members[0].isEnabled == true)
        #expect(config.members[0].interfaceId.isEmpty)
        // The cutover wipes legacy members and pools wholesale.
        var mutable = config
        let changed = mutable.migrateToInterfaceSchemaIfNeeded()
        #expect(changed)
        config = mutable
        #expect(config.members.isEmpty)
        #expect(config.pools.isEmpty)
    }

    @Test("explicit pools round-trip without inventing extras")
    func poolsRoundTrip() throws {
        var config = PeerConfig.defaultConfig()
        config.factories = [FactoryConfig(
            id: "fx", name: "x", baseUrl: URL(string: "http://10.0.0.2:8080")!,
            interfaces: [InterfaceConfig(id: "fx-chat", kind: .openaiChat)])]
        config.members = [
            MemberConfig(id: "m1", name: "a", interfaceId: "fx-chat",
                         upstreamModel: "x", weight: 2, isEnabled: true)
        ]
        config.pools = [
            PoolConfig(id: "coding", name: "coding", strategy: .roundRobin,
                       kind: .openaiChat, memberIds: ["m1"], maxRetries: 2)
        ]
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(PeerConfig.self, from: data)
        #expect(decoded.pools == config.pools)
        #expect(decoded.members == config.members)
    }

    @Test("capabilities no longer invent pools on decode — the cutover wipes legacy rows")
    func capabilitiesDoNotInventPools() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "sources": [{
            "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
            "endpoint": "http://127.0.0.1:6590/v1",
            "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
          }],
          "capabilities": [{
            "demandId": "d1", "model": "qwen-3-8b", "sourceId": "src-d1",
            "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
          }]
        }
        """.data(using: .utf8)!
        var config = try JSONDecoder().decode(PeerConfig.self, from: json)
        #expect(config.pools.isEmpty)
        let changed = config.migrateToInterfaceSchemaIfNeeded()
        #expect(changed)
        #expect(config.members.isEmpty)
        #expect(config.pools.isEmpty)
    }

    @Test("members key wins when a file still has sources")
    func membersPreferredOverSources() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "sources": [{
            "id": "old", "name": "old", "type": "openaiCompatible",
            "endpoint": "http://10.0.0.1:1/v1",
            "apiKeyRef": "src/old", "upstreamModel": "old"
          }],
          "members": [{
            "id": "new", "name": "new", "type": "openaiCompatible",
            "endpoint": "http://10.0.0.2:1/v1",
            "apiKeyRef": "src/new", "upstreamModel": "new"
          }],
          "pools": [{
            "id": "coding", "name": "coding", "strategy": "tiered",
            "memberIds": ["new"], "maxRetries": 3, "isEnabled": true, "requestCount": 0
          }]
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(PeerConfig.self, from: json)
        #expect(config.members.map(\.id) == ["new"])
        #expect(config.pools.map(\.id) == ["coding"])
        #expect(config.pools[0].strategy == .roundRobin)
        let encoded = String(data: try JSONEncoder().encode(config), encoding: .utf8) ?? ""
        #expect(!encoded.contains("\"sources\""))
    }

    @Test("saved server rows wipe legacy members the same way")
    func serverRowsWipeLegacy() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "sources": [{
            "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
            "endpoint": "http://127.0.0.1:6590/v1",
            "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
          }],
          "serverMemory": [{
            "server": "https://tknet.ai", "peerName": "alpha",
            "members": [{
              "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
              "endpoint": "http://127.0.0.1:6590/v1",
              "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
            }]
          }]
        }
        """.data(using: .utf8)!
        var config = try JSONDecoder().decode(PeerConfig.self, from: json)
        let row = try #require(config.serverMemory.first)
        #expect(row.members.map(\.id) == ["src-d1"])
        let changed = config.migrateToInterfaceSchemaIfNeeded()
        #expect(changed)
        #expect(config.serverMemory.first?.members.isEmpty == true)
    }
}
