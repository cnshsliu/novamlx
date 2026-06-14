import Foundation
import GRDB

// NovaMLXDB cannot import NovaMLXCore (NovaMLXCore depends on NovaMLXDB).
// These stores expose record-level CRUD mirroring the TokenhubStore pattern.
// Domain-typed accessors live in Sources/NovaMLXCore/LoadBalancerStore+Domain.swift.

// MARK: - LoadBalancerStore

public final class LoadBalancerStore: Sendable {
    private let db: DatabasePool
    public init(db: DatabasePool) { self.db = db }

    public func list() throws -> [LoadBalancerRow] {
        try db.read { db in
            try LoadBalancerRow
                .order(Column("created_at").desc)
                .fetchAll(db)
        }
    }

    public func get(_ id: UUID) throws -> LoadBalancerRow? {
        try db.read { db in
            try LoadBalancerRow.fetchOne(db, key: id.uuidString)
        }
    }

    public func getBySlug(_ slug: String) throws -> LoadBalancerRow? {
        try db.read { db in
            try LoadBalancerRow
                .filter(Column("slug") == slug)
                .fetchOne(db)
        }
    }

    public func upsert(_ row: LoadBalancerRow) throws {
        try db.write { db in
            try row.save(db)
        }
    }

    public func delete(_ id: UUID) throws {
        try db.write { db in
            _ = try LoadBalancerRow.deleteOne(db, key: id.uuidString)
        }
    }

    /// Atomically increment the per-LB request counter.
    public func incrementRequestCount(_ id: UUID) throws {
        try db.write { db in
            if var row = try LoadBalancerRow.fetchOne(db, key: id.uuidString) {
                row.requestCount += 1
                try row.save(db)
            }
        }
    }

    /// Internal-write passthrough for cross-module extensions. The
    /// `LoadBalancerStore+Domain` extension lives in NovaMLXCore (a different
    /// module) and cannot see this class's private `db` pool. Mirrors the
    /// TokenhubStore.write helper.
    public func write<T>(_ block: @Sendable (Database) throws -> T) throws -> T {
        try db.write(block)
    }
}

// MARK: - LBMemberStore

public final class LBMemberStore: Sendable {
    private let db: DatabasePool
    public init(db: DatabasePool) { self.db = db }

    public func listByLB(_ lbId: UUID) throws -> [LBMemberRow] {
        try db.read { db in
            try LBMemberRow
                .filter(Column("lb_id") == lbId.uuidString)
                .fetchAll(db)
        }
    }

    public func get(_ id: UUID) throws -> LBMemberRow? {
        try db.read { db in
            try LBMemberRow.fetchOne(db, key: id.uuidString)
        }
    }

    public func upsert(_ row: LBMemberRow) throws {
        try db.write { db in
            try row.save(db)
        }
    }

    public func delete(_ id: UUID) throws {
        try db.write { db in
            _ = try LBMemberRow.deleteOne(db, key: id.uuidString)
        }
    }
}

// MARK: - LBMemberStatsStore

public final class LBMemberStatsStore: Sendable {
    private let db: DatabasePool
    public init(db: DatabasePool) { self.db = db }

    public func get(_ memberId: UUID) throws -> LBMemberStatsRow? {
        try db.read { db in
            try LBMemberStatsRow.fetchOne(db, key: memberId.uuidString)
        }
    }

    /// Fetch stats for all members of an LB. Takes a memberStore so we don't
    /// need to re-derive the membership inside the read transaction.
    public func listByLB(_ lbId: UUID, memberStore: LBMemberStore) throws -> [LBMemberStatsRow] {
        let members = try memberStore.listByLB(lbId)
        return try db.read { db in
            try members.compactMap { m in
                try LBMemberStatsRow.fetchOne(db, key: m.id)
            }
        }
    }

    /// Atomically record a request outcome. Lazily creates the row on first call.
    public func recordRequest(
        memberId: UUID,
        succeeded: Bool,
        latencyMs: Int64,
        httpStatus: Int,
        errorMessage: String?
    ) throws {
        try db.write { db in
            let id = memberId.uuidString
            if var row = try LBMemberStatsRow.fetchOne(db, key: id) {
                row.requestCount += 1
                if succeeded {
                    row.successCount += 1
                    row.totalLatencyMs += latencyMs
                } else {
                    row.failureCount += 1
                    if httpStatus >= 500 { row.count5xx += 1 }
                    row.lastError = errorMessage?.prefix(500).description
                }
                row.lastUsedAt = Date()
                row.updatedAt = Date()
                try row.save(db)
            } else {
                let now = Date()
                let row = LBMemberStatsRow(
                    memberId: id,
                    requestCount: 1,
                    successCount: succeeded ? 1 : 0,
                    failureCount: succeeded ? 0 : 1,
                    count5xx: (!succeeded && httpStatus >= 500) ? 1 : 0,
                    totalLatencyMs: succeeded ? latencyMs : 0,
                    lastUsedAt: now,
                    lastError: succeeded ? nil : errorMessage?.prefix(500).description,
                    updatedAt: now
                )
                try row.save(db)
            }
        }
    }
}

// MARK: - GRDB Row types
// Records mirror the snake_case column layout defined in ConfigDBSchema.
// The LB table migration is registered in Task 3; these records just map
// the columns back to Swift types.

public struct LoadBalancerRow: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "load_balancers"

    public let id: String
    public var name: String
    public var slug: String
    public var strategy: String
    public var maxRetries: Int
    public var isEnabled: Bool
    public var requestCount: Int
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String,
        name: String,
        slug: String,
        strategy: String,
        maxRetries: Int,
        isEnabled: Bool,
        requestCount: Int,
        createdAt: Date,
        updatedAt: Date
    ) {
        self.id = id
        self.name = name
        self.slug = slug
        self.strategy = strategy
        self.maxRetries = maxRetries
        self.isEnabled = isEnabled
        self.requestCount = requestCount
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case id, name, slug, strategy
        case maxRetries = "max_retries"
        case isEnabled = "is_enabled"
        case requestCount = "request_count"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

extension LoadBalancerRow: FetchableRecord, MutablePersistableRecord {}

public struct LBMemberRow: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "lb_members"

    public let id: String
    public var lbId: String
    public var kind: String
    public var ref: String
    public var weight: Int?
    public var isEnabled: Bool

    public init(
        id: String,
        lbId: String,
        kind: String,
        ref: String,
        weight: Int? = nil,
        isEnabled: Bool
    ) {
        self.id = id
        self.lbId = lbId
        self.kind = kind
        self.ref = ref
        self.weight = weight
        self.isEnabled = isEnabled
    }

    enum CodingKeys: String, CodingKey {
        case id
        case lbId = "lb_id"
        case kind, ref, weight
        case isEnabled = "is_enabled"
    }
}

extension LBMemberRow: FetchableRecord, MutablePersistableRecord {}

public struct LBMemberStatsRow: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "lb_member_stats"

    public let memberId: String
    public var requestCount: Int
    public var successCount: Int
    public var failureCount: Int
    public var count5xx: Int
    public var totalLatencyMs: Int64
    public var lastUsedAt: Date?
    public var lastError: String?
    public var updatedAt: Date

    public init(
        memberId: String,
        requestCount: Int,
        successCount: Int,
        failureCount: Int,
        count5xx: Int,
        totalLatencyMs: Int64,
        lastUsedAt: Date? = nil,
        lastError: String? = nil,
        updatedAt: Date
    ) {
        self.memberId = memberId
        self.requestCount = requestCount
        self.successCount = successCount
        self.failureCount = failureCount
        self.count5xx = count5xx
        self.totalLatencyMs = totalLatencyMs
        self.lastUsedAt = lastUsedAt
        self.lastError = lastError
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case memberId = "member_id"
        case requestCount = "request_count"
        case successCount = "success_count"
        case failureCount = "failure_count"
        case count5xx = "count_5xx"
        case totalLatencyMs = "total_latency_ms"
        case lastUsedAt = "last_used_at"
        case lastError = "last_error"
        case updatedAt = "updated_at"
    }
}

extension LBMemberStatsRow: FetchableRecord, MutablePersistableRecord {}
