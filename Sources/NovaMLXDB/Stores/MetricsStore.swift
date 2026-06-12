import Foundation
import GRDB

public final class MetricsDBStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func get() throws -> MetricsRecord {
        if let record = try db.read({ db in try MetricsRecord.fetchOne(db, key: 1) }) {
            return record
        }
        let record = MetricsRecord(
            totalRequests: 0, totalTokens: 0, totalInferenceTimeMs: 0,
            cacheHits: 0, cacheMisses: 0, evictions: 0
        )
        try db.write { db in
            try record.insert(db)
        }
        return record
    }

    public func increment(requests: Int64 = 0, tokens: Int64 = 0, inferenceTimeMs: Int64 = 0, cacheHits: Int64 = 0, cacheMisses: Int64 = 0, evictions: Int64 = 0, model: String? = nil) throws {
        try db.write { db in
            var record = try MetricsRecord.fetchOne(db, key: 1) ?? MetricsRecord(
                totalRequests: 0, totalTokens: 0, totalInferenceTimeMs: 0,
                cacheHits: 0, cacheMisses: 0, evictions: 0
            )
            record.totalRequests += requests
            record.totalTokens += tokens
            record.totalInferenceTimeMs += inferenceTimeMs
            record.cacheHits += cacheHits
            record.cacheMisses += cacheMisses
            record.evictions += evictions
            record.updatedAt = Date()

            if let model {
                var stats: [String: [String: Int64]] = (try? JSONDecoder().decode([String: [String: Int64]].self, from: Data((record.perModelStats ?? "{}").utf8))) ?? [:]
                var entry = stats[model] ?? [:]
                entry["requests", default: 0] += requests
                entry["tokens", default: 0] += tokens
                entry["inferenceTimeMs", default: 0] += inferenceTimeMs
                stats[model] = entry
                record.perModelStats = (try? JSONEncoder().encode(stats)).flatMap { String(data: $0, encoding: .utf8) }
            }

            try record.save(db)
        }
    }

    public func reset() throws {
        try db.write { db in
            var record = try MetricsRecord.fetchOne(db, key: 1) ?? MetricsRecord(
                totalRequests: 0, totalTokens: 0, totalInferenceTimeMs: 0,
                cacheHits: 0, cacheMisses: 0, evictions: 0
            )
            record.totalRequests = 0
            record.totalTokens = 0
            record.totalInferenceTimeMs = 0
            record.cacheHits = 0
            record.cacheMisses = 0
            record.evictions = 0
            record.perModelStats = "{}"
            record.perModelCache = "{}"
            record.updatedAt = Date()
            try record.save(db)
        }
    }
}
