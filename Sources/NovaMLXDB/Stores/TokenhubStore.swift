import Foundation
import GRDB

public final class TokenhubStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list() throws -> [TokenhubProviderRecord] {
        try db.read { db in
            try TokenhubProviderRecord.fetchAll(db)
        }
    }

    public func get(name: String) throws -> TokenhubProviderRecord? {
        try db.read { db in
            try TokenhubProviderRecord.fetchOne(db, key: name)
        }
    }

    public func upsert(_ record: TokenhubProviderRecord) throws {
        try db.write { db in
            try record.save(db)
        }
    }

    public func delete(name: String) throws {
        try db.write { db in
            try TokenhubProviderRecord.deleteOne(db, key: name)
        }
    }

    public func recordUsage(name: String, tokens: Int64, latencyMs: Double?) throws {
        try db.write { db in
            guard var record = try TokenhubProviderRecord.fetchOne(db, key: name) else { return }
            record.totalRequests += 1
            record.totalTokens += tokens
            record.lastUsedAt = Date()
            if let latency = latencyMs {
                let count = max(1, record.totalRequests)
                record.avgLatencyMs = ((record.avgLatencyMs ?? 0) * Double(count - 1) + latency) / Double(count)
            }
            try record.update(db)
        }
    }

    /// Internal-write passthrough for cross-module extensions.
    /// The `TokenhubStore+Domain` extension lives in NovaMLXCore (a different
    /// module), so it cannot see this class's private `db` pool. Exposing a
    /// typed `write` helper lets that extension participate in GRDB
    /// transactions (e.g. atomic set replacement) without leaking the pool.
    public func write<T>(_ block: @Sendable (Database) throws -> T) throws -> T {
        try db.write(block)
    }
}
