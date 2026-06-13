import Foundation
import NovaMLXDB

// APIKeyStore lives in NovaMLXDB; APIKey lives here in NovaMLXCore.
// Declaring this extension in NovaMLXCore keeps the dependency direction one-way
// (NovaMLXCore -> NovaMLXDB) while still giving the DB store the ability to
// hand back the domain type the rest of the app already uses.

extension APIKeyStore {
    // MARK: - Domain-facing accessors

    /// All keys, converted to the domain `APIKey` type.
    public func listAsAPIKey() throws -> [APIKey] {
        try list().map { Self.toDomain($0) }
    }

    /// Look up a key by its raw plaintext token, returning the domain type.
    /// Hashes the input before querying the store, so the raw token never
    /// leaves this call in plaintext form on the wire to SQLite.
    public func findAPIKeyByRawToken(_ raw: String) throws -> APIKey? {
        guard let record = try findByRawKey(raw) else { return nil }
        return Self.toDomain(record)
    }

    /// Single-key fetch by id, returning the domain type, or nil if missing.
    public func getAsAPIKey(id: String) throws -> APIKey? {
        guard let record = try get(id: id) else { return nil }
        return Self.toDomain(record)
    }

    // MARK: - Limit Checks

    /// Check if a key has exceeded its period (daily/weekly/monthly) token or
    /// request limits. Returns `true` if the key is within limits OR if the
    /// key uses the `.never` reset period (no limits). Returns `false` if the
    /// key is unknown (conservative deny).
    public func isWithinLimits(keyId: String) -> Bool {
        guard let key = (try? getAsAPIKey(id: keyId)) ?? nil else { return false }
        return Self.computeIsWithinLimits(key)
    }

    /// Get the period usage as a fraction (0.0 - 1.0) for progress display.
    /// Returns 0 if the key is unknown or has no token cap configured.
    public func periodUsageFraction(keyId: String) -> Double {
        guard let key = (try? getAsAPIKey(id: keyId)) ?? nil else { return 0 }
        return Self.computePeriodUsageFraction(key)
    }

    /// Pure rate-limit computation. "never" means no limits. Period counters
    /// are treated as zero if the stored period-reset date no longer matches
    /// the current period (i.e. the period rolled over since last write).
    private static func computeIsWithinLimits(_ key: APIKey) -> Bool {
        if key.usageResetPeriod == .never { return true }

        let periodKey = periodDate(for: key.usageResetPeriod)
        var periodTokens = key.usage.periodTokens
        var periodRequests = key.usage.periodRequests
        if key.usage.periodResetDate != periodKey {
            periodTokens = 0
            periodRequests = 0
        }

        if let maxTokens = key.maxTokensPerPeriod, periodTokens >= maxTokens { return false }
        if let maxRequests = key.maxRequestsPerPeriod, periodRequests >= maxRequests { return false }
        return true
    }

    private static func computePeriodUsageFraction(_ key: APIKey) -> Double {
        guard let max = key.maxTokensPerPeriod, max > 0 else { return 0 }

        let periodKey = periodDate(for: key.usageResetPeriod)
        var tokens = key.usage.periodTokens
        if key.usage.periodResetDate != periodKey { tokens = 0 }

        return min(1.0, Double(tokens) / Double(max))
    }

    /// Delegates to `APIKeyStore.periodDate(for:)` in NovaMLXDB so the domain
    /// layer and DB agree exactly on when a period resets.
    private static func periodDate(for period: UsageResetPeriod) -> String {
        APIKeyStore.periodDate(for: period.rawValue)
    }

    // MARK: - Conversion

    /// Convert a DB-layer `APIKeyRecord` into the domain `APIKey`.
    /// Decodes the JSON-string fields (`allowedModels`, `allowedEndpoints`,
    /// `perModelTokens`) inline rather than reaching for the private helpers
    /// in `APIKeyStore.swift`.
    public static func toDomain(_ record: APIKeyRecord) -> APIKey {
        let allowedModels: [String]? = decodeJSONString(record.allowedModels)
        let allowedEndpoints: [String]? = decodeJSONString(record.allowedEndpoints)
        let perModelTokens: [String: Int64] = decodeJSONString(record.perModelTokens) ?? [:]

        let usage = APIKey.KeyUsage(
            totalTokensUsed: record.totalTokensUsed,
            totalRequests: record.totalRequests,
            lastUsedAt: record.lastUsedAt,
            periodTokens: record.periodTokens,
            periodRequests: record.periodRequests,
            periodResetDate: record.periodResetDate,
            perModelTokens: perModelTokens
        )

        let resetPeriod = UsageResetPeriod(rawValue: record.usageResetPeriod) ?? .daily

        var key = APIKey(
            id: record.id,
            name: record.name,
            keyHash: record.keyHash,
            keyPrefix: record.keyPrefix,
            keySuffix: record.keySuffix,
            createdAt: record.createdAt,
            expiresAt: record.expiresAt,
            isEnabled: record.isEnabled,
            rateLimitPerSecond: record.rateLimitPerSecond,
            rateLimitBurst: record.rateLimitBurst,
            allowedModels: allowedModels,
            allowedEndpoints: allowedEndpoints,
            maxTokensPerPeriod: record.maxTokensPerPeriod,
            maxRequestsPerPeriod: record.maxRequestsPerPeriod,
            usageResetPeriod: resetPeriod,
            usage: usage
        )
        key.isLegacyImportValue = Self._isLegacyRecord(record)
        return key
    }

    /// True if the record was produced by `NovaDB.importLegacyJSON` — i.e. its
    /// `rawKey` was filled with the all-zero placeholder because the plaintext
    /// key was never recoverable from the legacy hash-only JSON format.
    public static func _isLegacyRecord(_ record: APIKeyRecord) -> Bool {
        record.rawKey == legacyPlaceholderRawKey
    }

    /// Placeholder written into `rawKey` for legacy imports. Mirrors the value
    /// produced by `NovaDB.importLegacyJSON` (see NovaDB.swift).
    static let legacyPlaceholderRawKey: String = "sk-novamlx-" + String(repeating: "0", count: 64)

    // MARK: - Private JSON helpers

    /// Decodes a JSON-encoded string field into the requested Decodable type.
    /// Returns nil for nil/empty input or decode failures.
    private static func decodeJSONString<T: Decodable>(_ value: String?) -> T? {
        guard let value, !value.isEmpty else { return nil }
        guard let data = value.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }
}
