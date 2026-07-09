import Testing
import Foundation
import GRDB
import NovaMLXDB
import NovaMLXCore

@Suite("APIKeyStore + Domain")
struct APIKeyStoreTests {

    /// Build a store backed by a fresh temp-file pool with the `api_keys`
    /// schema mirroring `ConfigDBSchema.createAPIKeysTable`. GRDB's
    /// `DatabasePool` requires WAL, which doesn't work with `:memory:`, so we
    /// drop a file in the system temp dir and let it be cleaned up by the OS.
    private func makeStore() throws -> APIKeyStore {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-apikeystore-test-\(UUID().uuidString).sqlite")
        let db = try DatabasePool(path: tmp.path)
        try db.write { d in
            try d.create(table: "api_keys") { t in
                t.column("id", .text).primaryKey()
                t.column("name", .text).notNull()
                t.column("key_hash", .text).notNull()
                t.column("raw_key", .text).notNull()
                t.column("key_prefix", .text).notNull()
                t.column("key_suffix", .text).notNull().defaults(to: "")
                t.column("created_at", .datetime).notNull()
                t.column("expires_at", .datetime)
                t.column("is_enabled", .boolean).notNull().defaults(to: true)
                t.column("rate_limit_per_second", .double)
                t.column("rate_limit_burst", .integer)
                t.column("allowed_models", .text)
                t.column("allowed_endpoints", .text)
                t.column("max_tokens_per_period", .integer)
                t.column("max_requests_per_period", .integer)
                t.column("usage_reset_period", .text).notNull().defaults(to: "daily")
                t.column("total_tokens_used", .integer).notNull().defaults(to: 0)
                t.column("total_requests", .integer).notNull().defaults(to: 0)
                t.column("last_used_at", .datetime)
                t.column("period_tokens", .integer).notNull().defaults(to: 0)
                t.column("period_requests", .integer).notNull().defaults(to: 0)
                t.column("period_reset_date", .text)
                t.column("per_model_tokens", .text).defaults(to: "{}")
            }
            try d.create(index: "idx_api_keys_hash", on: "api_keys", columns: ["key_hash"])
            try d.create(table: "api_key_usage_events") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("key_id", .text)
                t.column("recorded_at", .datetime).notNull()
                t.column("model", .text)
                t.column("endpoint", .text).notNull()
                t.column("prompt_tokens", .integer).notNull().defaults(to: 0)
                t.column("completion_tokens", .integer).notNull().defaults(to: 0)
                t.column("total_tokens", .integer).notNull()
            }
        }
        return APIKeyStore(db: db)
    }

    @Test("create + listAsAPIKey returns domain type")
    func createAndListDomain() throws {
        let store = try makeStore()
        let (record, raw) = try store.create(name: "test-key")

        let domainKeys = try store.listAsAPIKey()
        let found = domainKeys.first { $0.id == record.id }

        #expect(found != nil)
        #expect(found?.name == "test-key")
        #expect(found?.keyHash == APIKeyStore.hashRawKey(raw))
        #expect(found?.isLegacyImport == false)
    }

    @Test("findAPIKeyByRawToken hashes input")
    func findByRawToken() throws {
        let store = try makeStore()
        let (_, raw) = try store.create(name: "lookup-test")

        let found = try store.findAPIKeyByRawToken(raw)
        #expect(found != nil)
        #expect(found?.name == "lookup-test")

        let miss = try store.findAPIKeyByRawToken("sk-novamlx-deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        #expect(miss == nil)
    }

    @Test("getAsAPIKey returns nil for unknown id")
    func getAsAPIKeyMissing() throws {
        let store = try makeStore()
        #expect(try store.getAsAPIKey(id: "key-nonexistent") == nil)
    }

    @Test("getAsAPIKey round-trips for a known id")
    func getAsAPIKeyHit() throws {
        let store = try makeStore()
        let (record, _) = try store.create(name: "fetch-by-id")
        let fetched = try store.getAsAPIKey(id: record.id)
        #expect(fetched?.id == record.id)
        #expect(fetched?.name == "fetch-by-id")
    }

    @Test("legacy placeholder record is flagged as legacy")
    func legacyPlaceholderDetection() throws {
        // Direct helper check with a record carrying the placeholder rawKey.
        let placeholder = "sk-novamlx-" + String(repeating: "0", count: 64)
        let fakeRecord = APIKeyRecord(
            id: "key-legacy-test",
            name: "legacy",
            keyHash: "deadbeef",
            rawKey: placeholder,
            keyPrefix: "sk-novamlx-deadbeef",
            keySuffix: "1234",
            createdAt: Date(),
            expiresAt: nil,
            isEnabled: true,
            rateLimitPerSecond: nil,
            rateLimitBurst: nil,
            allowedModels: nil,
            allowedEndpoints: nil,
            maxTokensPerPeriod: nil,
            maxRequestsPerPeriod: nil,
            usageResetPeriod: "daily",
            totalTokensUsed: 0,
            totalRequests: 0,
            lastUsedAt: nil,
            periodTokens: 0,
            periodRequests: 0,
            periodResetDate: nil,
            perModelTokens: nil
        )
        #expect(APIKeyStore._isLegacyRecord(fakeRecord) == true)

        // Fresh keys created through the store are NOT flagged as legacy.
        let store = try makeStore()
        let (record, _) = try store.create(name: "fresh")
        let fresh = try store.getAsAPIKey(id: record.id)
        #expect(fresh?.isLegacyImport == false)
    }

    @Test("JSON-encoded allowedModels/endpoints/perModelTokens round-trip")
    func jsonFieldsRoundTrip() throws {
        let store = try makeStore()
        let (record, _) = try store.create(
            name: "json-key",
            rateLimitPerSecond: 2.5,
            rateLimitBurst: 10,
            allowedModels: ["gpt-4", "claude"],
            allowedEndpoints: ["/v1/chat/completions"],
            maxTokensPerPeriod: 1_000_000,
            maxRequestsPerPeriod: 500,
            usageResetPeriod: "weekly"
        )

        let domain = try store.getAsAPIKey(id: record.id)
        #expect(domain?.allowedModels == ["gpt-4", "claude"])
        #expect(domain?.allowedEndpoints == ["/v1/chat/completions"])
        #expect(domain?.rateLimitPerSecond == 2.5)
        #expect(domain?.rateLimitBurst == 10)
        #expect(domain?.maxTokensPerPeriod == 1_000_000)
        #expect(domain?.maxRequestsPerPeriod == 500)
        #expect(domain?.usageResetPeriod == .weekly)
        #expect(domain?.usage.perModelTokens.isEmpty == true)
    }

    @Test("recordUsage is reflected in domain KeyUsage")
    func usagePropagation() throws {
        let store = try makeStore()
        let (record, _) = try store.create(name: "usage-key")

        try store.recordUsage(keyId: record.id, tokens: 1234, model: "gpt-4")
        try store.recordUsage(keyId: record.id, tokens: 766, model: "gpt-4")

        let domain = try store.getAsAPIKey(id: record.id)!
        #expect(domain.usage.totalTokensUsed == 2000)
        #expect(domain.usage.totalRequests == 2)
        #expect(domain.usage.periodTokens == 2000)
        #expect(domain.usage.periodRequests == 2)
        #expect(domain.usage.perModelTokens["gpt-4"] == 2000)
        #expect(domain.usage.lastUsedAt != nil)
    }

    @Test("unknown usageResetPeriod falls back to daily")
    func unknownResetPeriodFallback() throws {
        let store = try makeStore()
        // Insert a record with a bogus reset period directly via the store's db.
        let (record, _) = try store.create(name: "bogus-period")
        try store.update(id: record.id) { rec in
            rec.usageResetPeriod = "totally-bogus"
        }

        let domain = try store.getAsAPIKey(id: record.id)
        #expect(domain?.usageResetPeriod == .daily)
    }

    @Test("usage ledger supports time-range and per-model breakdown")
    func usageLedgerReport() throws {
        let store = try makeStore()
        let (recordA, _) = try store.create(name: "dept-a")
        let (recordB, _) = try store.create(name: "dept-b")

        let now = Date()
        let from = now.addingTimeInterval(-3600)
        let to = now.addingTimeInterval(3600)

        try store.recordUsage(
            keyId: recordA.id, promptTokens: 100, completionTokens: 50,
            model: "mlx-community/Qwen3-4B-4bit", endpoint: "/v1/chat/completions"
        )
        try store.recordUsage(
            keyId: recordA.id, promptTokens: 20, completionTokens: 10,
            model: "mlx-community/gemma-4-26b-a4b-it-4bit", endpoint: "/v1/chat/completions"
        )
        try store.recordUsage(
            keyId: recordB.id, promptTokens: 30, completionTokens: 15,
            model: "mlx-community/Qwen3-4B-4bit", endpoint: "/v1/messages"
        )
        try store.recordUsage(
            keyId: nil, promptTokens: 5, completionTokens: 5,
            model: "mlx-community/Qwen3-4B-4bit", endpoint: "/v1/chat/completions"
        )

        let report = try store.usageReport(from: from, to: to)
        #expect(report.total.requests == 4)
        #expect(report.total.totalTokens == 235)
        #expect(report.attributed.totalTokens == 225)
        #expect(report.unattributed.totalTokens == 10)
        #expect(report.byKey.count == 3)

        let keyA = report.byKey.first { $0.keyId == recordA.id }
        #expect(keyA?.usage.totalTokens == 180)
        #expect(keyA?.byModel.count == 2)

        let perKeyA = try store.usageReport(keyId: recordA.id, from: from, to: to)
        #expect(perKeyA.byKey.count == 1)
        #expect(perKeyA.byKey.first?.usage.totalTokens == 180)
    }

    @Test("malformed JSON string fields decode to nil/empty without throwing")
    func malformedJSONFieldsDegradeGracefully() throws {
        let store = try makeStore()
        let (record, _) = try store.create(name: "malformed-test")
        try store.update(id: record.id) { rec in
            rec.allowedModels = "{not valid json"
            rec.allowedEndpoints = ""
            rec.perModelTokens = "<<<broken>>>"
        }
        let domain = try store.getAsAPIKey(id: record.id)
        #expect(domain != nil)
        #expect(domain?.allowedModels == nil)
        #expect(domain?.allowedEndpoints == nil)
        #expect(domain?.usage.perModelTokens == [:])
    }
}
