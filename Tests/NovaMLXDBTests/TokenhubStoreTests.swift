import Testing
import Foundation
import GRDB
import NovaMLXDB
import NovaMLXCore

@Suite("TokenhubStore", .serialized)
struct TokenhubStoreTests {

    private func makeTmpDir() throws -> URL {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-tk-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        return tmp
    }

    @Test("v2 migration adds all 15 tokenhub provider columns")
    func v2MigrationAddsColumns() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // Insert a baseline row (uses only v1 columns) to verify defaults backfill
        _ = try await nova.configDB.write { db in
            try db.execute(sql: """
                INSERT INTO tokenhub_providers (name, endpoint, is_enabled, is_managed, load_balance_weight)
                VALUES ('test-provider', 'https://example.com', 1, 0, 1.0)
                """)
            return 0
        }

        let record = try nova.tokenhubStore.get(name: "test-provider")
        #expect(record != nil)
        #expect(record?.includeInLoadBalance == true)
        #expect(record?.tags == nil)
        #expect(record?.isLocal == false)
        #expect(record?.isFree == false)
        #expect(record?.isManaged == false)
        #expect(record?.supportsResponsesAPI == false)
        #expect(record?.supportsVision == false)
        #expect(record?.visionStrategy == nil)
        #expect(record?.anthropicEndpoint == nil)
        #expect(record?.visionCompanionModel == nil)
        #expect(record?.requestCount == 0)
        #expect(record?.successCount == 0)
        #expect(record?.lastTestedAt == nil)
        #expect(record?.lastStatus == nil)
        #expect(record?.contextWindowOverride == nil)
        #expect(record?.providerId == nil)
    }

    @Test("tokenhubStore upsert + get round-trips new fields")
    func upsertRoundTrip() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        var record = TokenhubProviderRecord(
            name: "round-trip",
            endpoint: "https://rt.example.com",
            apiKey: "secret",
            remoteModel: "gpt-rt",
            isEnabled: true,
            isManaged: false,
            loadBalanceWeight: 1.0,
            totalRequests: 0,
            totalTokens: 0,
            avgLatencyMs: nil,
            lastUsedAt: nil,
            extraConfig: nil
        )
        record.providerId = "round-trip"
        record.includeInLoadBalance = true
        record.tags = "[\"local\",\"fast\"]"
        record.isLocal = true
        record.isFree = true
        record.supportsResponsesAPI = true
        record.supportsVision = true
        record.visionStrategy = "companion"
        record.anthropicEndpoint = "https://anthropic.example.com"
        record.visionCompanionModel = "gpt-4o"
        record.requestCount = 42
        record.successCount = 40
        record.lastTestedAt = Date(timeIntervalSince1970: 1_700_000_000)
        record.lastStatus = "ok"
        record.contextWindowOverride = 32768

        try nova.tokenhubStore.upsert(record)

        let fetched = try nova.tokenhubStore.get(name: "round-trip")
        #expect(fetched?.endpoint == "https://rt.example.com")
        #expect(fetched?.apiKey == "secret")
        #expect(fetched?.remoteModel == "gpt-rt")
        #expect(fetched?.includeInLoadBalance == true)
        #expect(fetched?.tags == "[\"local\",\"fast\"]")
        #expect(fetched?.isLocal == true)
        #expect(fetched?.isFree == true)
        #expect(fetched?.supportsResponsesAPI == true)
        #expect(fetched?.supportsVision == true)
        #expect(fetched?.visionStrategy == "companion")
        #expect(fetched?.anthropicEndpoint == "https://anthropic.example.com")
        #expect(fetched?.visionCompanionModel == "gpt-4o")
        #expect(fetched?.requestCount == 42)
        #expect(fetched?.successCount == 40)
        #expect(fetched?.lastStatus == "ok")
        #expect(fetched?.contextWindowOverride == 32768)
    }

    @Test("tokenhubStore upsert + get round-trips domain provider", .serialized)
    func domainProviderRoundTrip() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // The TokenhubManager singleton reads from NovaMLXPaths.tokenhubProvidersFile.
        // For this test we go provider-by-provider through the store extension.
        let provider = TokenhubProvider(
            name: "Bridge Test",
            endpoint: "https://bridge.example.com",
            apiKey: "bridge-key",
            remoteModel: "bridge-model",
            isEnabled: true,
            includeInLoadBalance: true,
            tags: ["test", "bridge"],
            isLocal: false,
            isFree: true,
            isManaged: false,
            supportsResponsesAPI: true,
            supportsVision: false,
            visionStrategy: nil,
            anthropicEndpoint: nil,
            visionCompanionModel: nil,
            requestCount: 10,
            successCount: 8,
            avgLatencyMs: 250.0,
            contextWindowOverride: 16384
        )

        try nova.tokenhubStore.upsertProvider(provider)

        let fetched = try nova.tokenhubStore.getProvider(name: "Bridge Test")
        #expect(fetched?.name == "Bridge Test")
        #expect(fetched?.endpoint == "https://bridge.example.com")
        #expect(fetched?.apiKey == "bridge-key")
        #expect(fetched?.remoteModel == "bridge-model")
        #expect(fetched?.includeInLoadBalance == true)
        #expect(fetched?.tags == ["test", "bridge"])
        #expect(fetched?.isFree == true)
        #expect(fetched?.supportsResponsesAPI == true)
        #expect(fetched?.requestCount == 10)
        #expect(fetched?.successCount == 8)
        #expect(fetched?.avgLatencyMs == 250.0)
        #expect(fetched?.contextWindowOverride == 16384)
    }

    @Test("TokenhubStore.replaceAll deletes removed + upserts kept", .serialized)
    func replaceAllSemantics() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // Seed with two providers
        try nova.tokenhubStore.upsertProvider(TokenhubProvider(
            name: "Keep", endpoint: "https://keep.example.com",
            apiKey: "k", remoteModel: "k-model"
        ))
        try nova.tokenhubStore.upsertProvider(TokenhubProvider(
            name: "Drop", endpoint: "https://drop.example.com",
            apiKey: "d", remoteModel: "d-model"
        ))

        // replaceAll with only "Keep" + new "Add"
        try nova.tokenhubStore.replaceAll(with: [
            TokenhubProvider(name: "Keep", endpoint: "https://keep.example.com",
                             apiKey: "k", remoteModel: "k-model-updated"),
            TokenhubProvider(name: "Add", endpoint: "https://add.example.com",
                             apiKey: "a", remoteModel: "a-model")
        ])

        let all = try nova.tokenhubStore.listAsProviders()
        #expect(all.count == 2)
        #expect(all.contains { $0.name == "Keep" })
        #expect(all.contains { $0.name == "Add" })
        #expect(!all.contains { $0.name == "Drop" })

        let kept = try nova.tokenhubStore.getProvider(name: "Keep")
        #expect(kept?.remoteModel == "k-model-updated")
    }

    @Test("Legacy providers.json is imported into tokenhubStore on first run", .serialized)
    func legacyProvidersImport() async throws {
        let tmp = try makeTmpDir()
        let providersDir = tmp.appendingPathComponent("tokenhub", isDirectory: true)
        try FileManager.default.createDirectory(at: providersDir, withIntermediateDirectories: true)

        let providersURL = providersDir.appendingPathComponent("providers.json")
        let legacyJSON = """
        [
            {
                "id": "legacy-1",
                "name": "Legacy Provider",
                "endpoint": "https://legacy.example.com",
                "apiKey": "legacy-key",
                "remoteModel": "legacy-model",
                "isEnabled": true,
                "includeInLoadBalance": true,
                "tags": ["legacy", "test"],
                "isLocal": false,
                "isFree": true,
                "isManaged": false,
                "supportsResponsesAPI": true,
                "supportsVision": false,
                "requestCount": 100,
                "successCount": 95,
                "avgLatencyMs": 120.5,
                "contextWindowOverride": 32768
            }
        ]
        """
        try legacyJSON.write(to: providersURL, atomically: true, encoding: .utf8)

        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let fetched = try nova.tokenhubStore.getProvider(name: "Legacy Provider")
        #expect(fetched?.id == "legacy-1")
        #expect(fetched?.endpoint == "https://legacy.example.com")
        #expect(fetched?.apiKey == "legacy-key")
        #expect(fetched?.remoteModel == "legacy-model")
        #expect(fetched?.tags == ["legacy", "test"])
        #expect(fetched?.isFree == true)
        #expect(fetched?.supportsResponsesAPI == true)
        #expect(fetched?.requestCount == 100)
        #expect(fetched?.successCount == 95)
        #expect(fetched?.avgLatencyMs == 120.5)
        #expect(fetched?.contextWindowOverride == 32768)

        #expect(FileManager.default.fileExists(atPath: providersURL.appendingPathExtension("migrated").path))
    }
}
