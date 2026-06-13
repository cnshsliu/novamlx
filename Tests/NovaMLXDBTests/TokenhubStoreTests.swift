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
}
