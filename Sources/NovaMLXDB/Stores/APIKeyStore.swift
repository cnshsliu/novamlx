import Foundation
import GRDB
import CryptoKit
import Logging

public final class APIKeyStore: Sendable {
    private let db: DatabasePool
    private let log = Logger(label: "APIKeyStore")

    public init(db: DatabasePool) {
        self.db = db
    }

    // MARK: - CRUD

    public func list() throws -> [APIKeyRecord] {
        try db.read { db in
            try APIKeyRecord
                .order(Column("created_at").desc)
                .fetchAll(db)
        }
    }

    public func get(id: String) throws -> APIKeyRecord? {
        try db.read { db in
            try APIKeyRecord.fetchOne(db, key: id)
        }
    }

    public func create(name: String, rateLimitPerSecond: Double? = nil, rateLimitBurst: Int? = nil, allowedModels: [String]? = nil, allowedEndpoints: [String]? = nil, maxTokensPerPeriod: Int64? = nil, maxRequestsPerPeriod: Int64? = nil, usageResetPeriod: String = "daily") throws -> (record: APIKeyRecord, rawKey: String) {
        let raw = Self.generateRawKey()
        let hash = Self.hashRawKey(raw)
        let prefix = String(raw.prefix(19))
        let suffix = String(raw.suffix(4))

        let record = APIKeyRecord(
            id: "key-\(UUID().uuidString)",
            name: name,
            keyHash: hash,
            rawKey: raw,
            keyPrefix: prefix,
            keySuffix: suffix,
            createdAt: Date(),
            expiresAt: nil,
            isEnabled: true,
            rateLimitPerSecond: rateLimitPerSecond,
            rateLimitBurst: rateLimitBurst,
            allowedModels: encodeJSON(allowedModels),
            allowedEndpoints: encodeJSON(allowedEndpoints),
            maxTokensPerPeriod: maxTokensPerPeriod,
            maxRequestsPerPeriod: maxRequestsPerPeriod,
            usageResetPeriod: usageResetPeriod,
            totalTokensUsed: 0,
            totalRequests: 0,
            lastUsedAt: nil,
            periodTokens: 0,
            periodRequests: 0,
            periodResetDate: nil,
            perModelTokens: "{}"
        )

        try db.write { db in
            try record.insert(db)
        }

        log.info("[APIKeyStore] Created key '\(name)' (\(prefix)...\(suffix))")
        return (record, raw)
    }

    public func update(id: String, _ updates: @Sendable (inout APIKeyRecord) -> Void) throws {
        try db.write { db in
            guard var record = try APIKeyRecord.fetchOne(db, key: id) else {
                throw NSError(domain: "APIKeyStore", code: 404, userInfo: [NSLocalizedDescriptionKey: "Key not found: \(id)"])
            }
            updates(&record)
            try record.update(db)
        }
    }

    public func delete(id: String) throws {
        _ = try db.write { db in
            try APIKeyRecord.deleteOne(db, key: id)
        }
        log.info("[APIKeyStore] Deleted key \(id)")
    }

    // MARK: - Auth Lookup

    public func findByHash(_ hash: String) throws -> APIKeyRecord? {
        try db.read { db in
            try APIKeyRecord
                .filter(Column("key_hash") == hash)
                .fetchOne(db)
        }
    }

    public func findByRawKey(_ raw: String) throws -> APIKeyRecord? {
        let hash = Self.hashRawKey(raw)
        return try findByHash(hash)
    }

    /// Get the plaintext key for display purposes
    public func getRawKey(id: String) throws -> String? {
        try db.read { db in
            try APIKeyRecord
                .select(Column("raw_key"))
                .filter(Column("id") == id)
                .fetchOne(db) as String?
        }
    }

    // MARK: - Usage Tracking

    public func recordUsage(keyId: String, tokens: Int64, model: String?) throws {
        try db.write { db in
            guard var record = try APIKeyRecord.fetchOne(db, key: keyId) else { return }

            let periodKey = Self.periodDate(for: record.usageResetPeriod)
            if record.periodResetDate != periodKey {
                record.periodTokens = 0
                record.periodRequests = 0
                record.periodResetDate = periodKey
            }

            record.totalTokensUsed += tokens
            record.totalRequests += 1
            record.periodTokens += tokens
            record.periodRequests += 1
            record.lastUsedAt = Date()

            if let model {
                var perModel: [String: Int64] = decodeJSON(record.perModelTokens ?? "{}") ?? [:]
                perModel[model, default: 0] += tokens
                record.perModelTokens = encodeJSON(perModel) ?? "{}"
            }

            try record.update(db)
        }
    }

    /// Returns the period-date string (e.g. "2026-06-13") used to key usage-reset periods.
    /// Canonical period-key implementation used by both the DB layer and the NovaMLXCore domain extension.
    public static func periodDate(for usageResetPeriod: String) -> String {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        switch usageResetPeriod {
        case "daily", "never":
            return formatter.string(from: now)
        case "weekly":
            let week = calendar.dateInterval(of: .weekOfYear, for: now)?.start ?? now
            return formatter.string(from: week)
        case "monthly":
            let month = calendar.dateInterval(of: .month, for: now)?.start ?? now
            return formatter.string(from: month)
        default:
            return formatter.string(from: now)
        }
    }

    // MARK: - Rotate

    public func rotate(id: String) throws -> (record: APIKeyRecord, rawKey: String) {
        let raw = Self.generateRawKey()
        let hash = Self.hashRawKey(raw)
        let prefix = String(raw.prefix(19))
        let suffix = String(raw.suffix(4))

        try db.write { db in
            guard var record = try APIKeyRecord.fetchOne(db, key: id) else {
                throw NSError(domain: "APIKeyStore", code: 404, userInfo: [NSLocalizedDescriptionKey: "Key not found: \(id)"])
            }
            record.keyHash = hash
            record.rawKey = raw
            record.keyPrefix = prefix
            record.keySuffix = suffix
            try record.update(db)
        }

        let record = try get(id: id)!
        log.info("[APIKeyStore] Rotated key '\(record.name)' (\(prefix)...\(suffix))")
        return (record, raw)
    }

    // MARK: - Helpers

    public static func hashRawKey(_ rawKey: String) -> String {
        let data = Data(rawKey.utf8)
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }

    public static func generateRawKey() -> String {
        let bytes = (0..<32).map { _ in UInt8.random(in: 0...255) }
        let hex = bytes.compactMap { String(format: "%02x", $0) }.joined()
        return "sk-novamlx-\(hex)"
    }
}

private func encodeJSON<T: Encodable>(_ value: T?) -> String? {
    guard let value else { return nil }
    guard let data = try? JSONEncoder().encode(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

private func decodeJSON<T: Decodable>(_ value: String) -> T? {
    guard let data = value.data(using: .utf8) else { return nil }
    return try? JSONDecoder().decode(T.self, from: data)
}
