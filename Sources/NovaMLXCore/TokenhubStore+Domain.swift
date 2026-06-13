import Foundation
import NovaMLXDB

// TokenhubStore lives in NovaMLXDB; TokenhubProvider lives here in NovaMLXCore.
// Declaring this extension in NovaMLXCore keeps the dependency direction one-way
// (NovaMLXCore -> NovaMLXDB) while still giving the DB store the ability to
// hand back the domain type the rest of the app already uses.

extension TokenhubStore {
    // MARK: - Domain-facing accessors

    /// All providers, converted to the domain `TokenhubProvider` type.
    public func listAsProviders() throws -> [TokenhubProvider] {
        try list().map { Self.toDomain($0) }
    }

    /// Fetch a single provider by name, returning the domain type.
    public func getProvider(name: String) throws -> TokenhubProvider? {
        guard let record = try get(name: name) else { return nil }
        return Self.toDomain(record)
    }

    /// Upsert a domain provider into the store.
    public func upsertProvider(_ provider: TokenhubProvider) throws {
        try upsert(Self.toRecord(provider))
    }

    /// Delete by provider name.
    public func deleteProvider(name: String) throws {
        try delete(name: name)
    }

    /// Replace the entire provider set atomically: delete rows not in `providers`,
    /// upsert the rest. Used by TokenhubManager.syncToStore during the Bridge
    /// phase so the SQLite shadow stays in lock-step with the authoritative
    /// JSON file. The whole replacement runs in a single GRDB write
    /// transaction so a partial failure cannot leave the store half-migrated.
    public func replaceAll(with providers: [TokenhubProvider]) throws {
        let desiredNames = Set(providers.map { $0.name })
        let existing = try list().map { $0.name }
        let toDelete = existing.filter { !desiredNames.contains($0) }
        try write { db in
            for name in toDelete {
                try TokenhubProviderRecord.deleteOne(db, key: name)
            }
            for provider in providers {
                try Self.toRecord(provider).save(db)
            }
        }
    }

    // MARK: - Mapping

    /// Convert a DB-layer `TokenhubProviderRecord` into the domain
    /// `TokenhubProvider`. JSON-encoded `tags` are decoded inline; missing or
    /// malformed tags decode to `[]`. Empty `apiKey` / `remoteModel` strings
    /// are reconstructed from NULL columns so callers see the same shape they
    /// would have written.
    public static func toDomain(_ record: TokenhubProviderRecord) -> TokenhubProvider {
        var provider = TokenhubProvider(
            name: record.name,
            endpoint: record.endpoint,
            apiKey: record.apiKey ?? "",
            remoteModel: record.remoteModel ?? "",
            isEnabled: record.isEnabled,
            includeInLoadBalance: record.includeInLoadBalance,
            tags: decodeTags(record.tags),
            isLocal: record.isLocal,
            isFree: record.isFree,
            isManaged: record.isManaged,
            supportsResponsesAPI: record.supportsResponsesAPI,
            supportsVision: record.supportsVision,
            visionStrategy: record.visionStrategy,
            anthropicEndpoint: record.anthropicEndpoint,
            visionCompanionModel: record.visionCompanionModel,
            requestCount: record.requestCount,
            successCount: record.successCount,
            avgLatencyMs: record.avgLatencyMs ?? 0,
            contextWindowOverride: record.contextWindowOverride
        )
        // Override the derived id with the stored one if present so managed
        // providers (e.g. "cloud-gpt-4o", "nova-qwen3") keep their canonical id
        // across the JSON -> DB -> JSON round-trip.
        if let pid = record.providerId { provider.id = pid }
        provider.lastTestedAt = record.lastTestedAt
        provider.lastStatus = record.lastStatus
        return provider
    }

    /// Convert a domain `TokenhubProvider` into a DB-layer record. Empty
    /// `apiKey` / `remoteModel` strings are stored as NULL so the column
    /// matches its optional schema; empty `tags` arrays likewise become NULL.
    public static func toRecord(_ provider: TokenhubProvider) -> TokenhubProviderRecord {
        var record = TokenhubProviderRecord(
            name: provider.name,
            endpoint: provider.endpoint,
            apiKey: provider.apiKey.isEmpty ? nil : provider.apiKey,
            remoteModel: provider.remoteModel.isEmpty ? nil : provider.remoteModel,
            isEnabled: provider.isEnabled,
            isManaged: provider.isManaged,
            loadBalanceWeight: provider.includeInLoadBalance ? 1.0 : 0.0,
            totalRequests: Int64(provider.requestCount),
            totalTokens: 0,
            avgLatencyMs: provider.avgLatencyMs == 0 ? nil : provider.avgLatencyMs,
            lastUsedAt: provider.lastTestedAt,
            extraConfig: nil
        )
        record.providerId = provider.id
        record.includeInLoadBalance = provider.includeInLoadBalance
        record.tags = encodeTags(provider.tags)
        record.isLocal = provider.isLocal
        record.isFree = provider.isFree
        record.supportsResponsesAPI = provider.supportsResponsesAPI
        record.supportsVision = provider.supportsVision
        record.visionStrategy = provider.visionStrategy
        record.anthropicEndpoint = provider.anthropicEndpoint
        record.visionCompanionModel = provider.visionCompanionModel
        record.requestCount = provider.requestCount
        record.successCount = provider.successCount
        record.lastTestedAt = provider.lastTestedAt
        record.lastStatus = provider.lastStatus
        record.contextWindowOverride = provider.contextWindowOverride
        return record
    }

    // MARK: - Private JSON helpers

    /// Encode `[String]` as a JSON string for the `tags` column. Empty arrays
    /// return nil so the column stays NULL (matches `decodeTags` semantics).
    private static func encodeTags(_ tags: [String]) -> String? {
        guard !tags.isEmpty else { return nil }
        guard let data = try? JSONEncoder().encode(tags) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Decode the `tags` column back to `[String]`. Returns `[]` for nil,
    /// empty, or malformed JSON.
    private static func decodeTags(_ value: String?) -> [String] {
        guard let value, !value.isEmpty,
              let data = value.data(using: .utf8),
              let tags = try? JSONDecoder().decode([String].self, from: data) else { return [] }
        return tags
    }
}
