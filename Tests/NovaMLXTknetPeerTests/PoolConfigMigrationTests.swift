import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("Pool config migration")
struct PoolConfigMigrationTests {
    @Test("legacy sources-only file becomes one pool per capability")
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
        let config = try JSONDecoder().decode(PeerConfig.self, from: json)
        #expect(config.members.count == 1)
        #expect(config.members[0].id == "src-d1")
        #expect(config.members[0].isEnabled == true)
        #expect(config.members[0].isFree == false)
        #expect(config.pools.count == 1)
        #expect(config.pools[0].id == "src-d1")
        #expect(config.pools[0].name == "qwen-3-8b")
        #expect(config.pools[0].memberIds == ["src-d1"])
        #expect(config.pools[0].strategy == .tiered)
        #expect(config.pools[0].maxRetries == 3)
        #expect(config.capabilities[0].sourceId == "src-d1")
    }

    @Test("explicit pools round-trip without inventing extras")
    func poolsRoundTrip() throws {
        var config = PeerConfig.defaultConfig()
        config.members = [
            MemberConfig(id: "m1", name: "a", type: .openaiCompatible,
                         endpoint: URL(string: "http://10.0.0.2:8080/v1")!,
                         apiKeyRef: "src/m1", upstreamModel: "x",
                         weight: 2, isEnabled: true, isFree: true)
        ]
        config.pools = [
            PoolConfig(id: "coding", name: "coding", strategy: .roundRobin,
                       memberIds: ["m1"], maxRetries: 2)
        ]
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(PeerConfig.self, from: data)
        #expect(decoded.pools == config.pools)
        #expect(decoded.members == config.members)
    }

    @Test("one published model is one pool; colliding ids get their own pool")
    func onePoolPerPublishedModel() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "sources": [{
            "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
            "endpoint": "http://127.0.0.1:6590/v1",
            "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
          }],
          "capabilities": [
            {
              "demandId": "d1", "model": "qwen-3-8b", "sourceId": "src-d1",
              "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
            },
            {
              "demandId": "d2", "model": "qwen-3-8b-fast", "sourceId": "src-d1",
              "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
            },
            {
              "demandId": "d3", "model": "qwen-3-8b", "sourceId": "src-d1",
              "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
            }
          ]
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(PeerConfig.self, from: json)
        #expect(config.pools.count == 2)
        #expect(config.pools[0].id == "src-d1")
        #expect(config.pools[0].name == "qwen-3-8b")
        #expect(config.pools[0].memberIds == ["src-d1"])
        #expect(config.pools[1].id == "src-d1-qwen-3-8b-fast")
        #expect(config.pools[1].name == "qwen-3-8b-fast")
        #expect(config.pools[1].memberIds == ["src-d1"])
        #expect(config.capabilities.map(\.sourceId) == [
            "src-d1", "src-d1-qwen-3-8b-fast", "src-d1"
        ])
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
        let encoded = String(data: try JSONEncoder().encode(config), encoding: .utf8) ?? ""
        #expect(!encoded.contains("\"sources\""))
    }

    @Test("saved server rows migrate sources the same way")
    func serverMemoryMigrates() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "serverMemory": [{
            "server": "https://tknet.ai",
            "peerName": "alpha",
            "sources": [{
              "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
              "endpoint": "http://127.0.0.1:6590/v1",
              "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
            }],
            "capabilities": [{
              "demandId": "d1", "model": "qwen-3-8b", "sourceId": "src-d1",
              "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
            }]
          }]
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(PeerConfig.self, from: json)
        let row = try #require(config.serverMemory.first)
        #expect(row.members.map(\.id) == ["src-d1"])
        #expect(row.pools.map(\.name) == ["qwen-3-8b"])
        #expect(row.pools.first?.memberIds == ["src-d1"])
    }
}
