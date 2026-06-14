# TokenHub + Multi-Load-Balancer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NovaMLX's implicit single-LB TokenHub model with an explicit multi-Load-Balancer system where each LB is a first-class entity that can include local models and/or remote providers as members, addressable via `lb:<slug>` model prefix.

**Architecture:** Three new SQLite tables (`load_balancers`, `lb_members`, `lb_member_stats`) backed by three new stores on `NovaDB.shared`. A pure `LBRouter` function applies the per-LB strategy to pick a member. A new `LBProxy` layer wraps request dispatch with retry-on-failure semantics. The existing `tknet:` / bare-model dispatch is preserved unchanged. TokenHub providers lose their `includeInLoadBalance` and `isManaged` fields; locals are no longer auto-inserted as virtual providers.

**Tech Stack:** Swift 5.10+, SwiftUI, GRDB 7.11, Hummingbird 2.x, Swift Testing (`@Test`), NovaMLX's existing modular SwiftPM layout (`NovaMLXCore`, `NovaMLXDB`, `NovaMLXAPI`, `NovaMLXMenuBar`).

**Spec:** `docs/superpowers/specs/2026-06-14-tokenhub-loadbalance-design.md` — all design decisions documented there. Read it first.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `Sources/NovaMLXCore/LoadBalancerTypes.swift` | Domain types: `LoadBalancer`, `LBMember`, `LBMemberStats`, `LBStrategy`, `MemberKind` |
| `Sources/NovaMLXDB/Stores/LoadBalancerStore.swift` | `LoadBalancerStore`, `LBMemberStore`, `LBMemberStatsStore` — three GRDB stores |
| `Sources/NovaMLXDB/Migrations/LBMigration.swift` | `v<next>_load_balancers` migration + provider column-drop |
| `Sources/NovaMLXCore/LBRouter.swift` | Pure selection function per strategy; no I/O |
| `Sources/NovaMLXAPI/LBProxy.swift` | Request dispatch + retry loop + stats recording |
| `Sources/NovaMLXAPI/APIServer+LoadBalancerAdmin.swift` | `/admin/load-balancers` REST endpoints |
| `Sources/NovaMLXMenuBar/LoadBalancersPageView.swift` | SwiftUI list + edit views |
| `Sources/NovaMLXMenuBar/LBMemberPickerSheet.swift` | Add-member multi-select sheet |
| `Tests/NovaMLXCoreTests/LBRouterTests.swift` | Router unit tests (one per strategy + edge cases) |
| `Tests/NovaMLXDBTests/LoadBalancerStoreTests.swift` | Store CRUD + cascade + migration tests |

### Modified files

| Path | Change |
|---|---|
| `Sources/NovaMLXDB/NovaDB.swift` | Add `loadBalancerStore`, `lbMemberStore`, `lbMemberStatsStore`; register `v<next>_load_balancers` migration |
| `Sources/NovaMLXMenuBar/NovaAppView.swift` | Rename `AppPage.models` → "Local Inference"; add `AppPage.loadBalancers = "Load Balancers"` as sibling of `tokenhub` |
| `Sources/NovaMLXCore/TokenhubTypes.swift` | Drop `includeInLoadBalance` and `isManaged` from `TokenhubProvider` |
| `Sources/NovaMLXCore/TokenhubStore+Domain.swift` | Delete `provisionLocalProviders()` and the priority-tiered `resolve()` method |
| `Sources/NovaMLXDB/Stores/TokenhubStore.swift` | Drop `includeInLoadBalance` and `isManaged` from `TokenhubProviderRecord` |
| `Sources/NovaMLXMenuBar/TokenhubPageView.swift` | Remove `formIncludeInLB` binding (~line 995), `lbProviders` filter (~line 406), LB checkbox from provider card |
| `Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift` | Add `lb:` prefix branch → call `LBProxy.handle(...)`; existing `tknet:` branch unchanged |

---

## Task 1: Core Domain Types

**Files:**
- Create: `Sources/NovaMLXCore/LoadBalancerTypes.swift`

- [ ] **Step 1: Create the types file**

```swift
// Sources/NovaMLXCore/LoadBalancerTypes.swift
import Foundation

/// Strategy an LB uses to pick a member when multiple are healthy.
public enum LBStrategy: String, Codable, Sendable, CaseIterable {
    /// Priority tiers: local+free > local > free > paid. Round-robin within tier. (default)
    case tiered
    /// Equal rotation across all healthy members.
    case roundRobin
    /// Probability proportional to member.weight. Treats nil weight as 1.
    case weighted
    /// Lowest avg_latency_ms over last 20 successes. Cold-start (no successes) treated as latency=0.
    case lowestLatency
    /// Uniform random across healthy members.
    case random
}

/// Whether an LB member is a local MLX model or a remote TokenHub provider.
public enum MemberKind: String, Codable, Sendable {
    case local    // ref = model_id
    case remote   // ref = provider_id (TokenhubProvider.id)
}

/// A named load balancer with its own selection strategy.
public struct LoadBalancer: Codable, Sendable, Identifiable {
    public let id: UUID
    public var name: String
    public var slug: String
    public var strategy: LBStrategy
    public var maxRetries: Int
    public var isEnabled: Bool
    public var requestCount: Int
    public let createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        name: String,
        slug: String,
        strategy: LBStrategy = .tiered,
        maxRetries: Int = 3,
        isEnabled: Bool = true,
        requestCount: Int = 0,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id; self.name = name; self.slug = slug
        self.strategy = strategy; self.maxRetries = maxRetries
        self.isEnabled = isEnabled; self.requestCount = requestCount
        self.createdAt = createdAt; self.updatedAt = updatedAt
    }
}

/// A member of an LB. Either a local model or a remote provider, referenced by ID.
public struct LBMember: Codable, Sendable, Identifiable {
    public let id: UUID
    public var lbId: UUID
    public var kind: MemberKind
    public var ref: String
    public var weight: Int?
    public var isEnabled: Bool

    public init(
        id: UUID = UUID(),
        lbId: UUID,
        kind: MemberKind,
        ref: String,
        weight: Int? = nil,
        isEnabled: Bool = true
    ) {
        self.id = id; self.lbId = lbId; self.kind = kind
        self.ref = ref; self.weight = weight; self.isEnabled = isEnabled
    }
}

/// Per-member routing statistics. 1:1 with LBMember.
public struct LBMemberStats: Codable, Sendable {
    public let memberId: UUID
    public var requestCount: Int
    public var successCount: Int
    public var failureCount: Int
    public var count5xx: Int
    public var totalLatencyMs: Int64
    public var lastUsedAt: Date?
    public var lastError: String?
    public var updatedAt: Date

    public var avgLatencyMs: Int64 {
        successCount > 0 ? totalLatencyMs / Int64(successCount) : 0
    }
    public var successRate: Double {
        requestCount > 0 ? Double(successCount) / Double(requestCount) : 0
    }

    public init(
        memberId: UUID,
        requestCount: Int = 0,
        successCount: Int = 0,
        failureCount: Int = 0,
        count5xx: Int = 0,
        totalLatencyMs: Int64 = 0,
        lastUsedAt: Date? = nil,
        lastError: String? = nil,
        updatedAt: Date = Date()
    ) {
        self.memberId = memberId; self.requestCount = requestCount
        self.successCount = successCount; self.failureCount = failureCount
        self.count5xx = count5xx; self.totalLatencyMs = totalLatencyMs
        self.lastUsedAt = lastUsedAt; self.lastError = lastError
        self.updatedAt = updatedAt
    }
}
```

- [ ] **Step 2: Build to verify compiles**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!` with no errors in the new file.

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXCore/LoadBalancerTypes.swift
git commit -m "feat(lb): add LoadBalancer domain types"
```

---

## Task 2: Storage Layer — Three GRDB Stores

**Files:**
- Create: `Sources/NovaMLXDB/Stores/LoadBalancerStore.swift`
- Modify: `Sources/NovaMLXDB/NovaDB.swift` — add three store properties + initialization

- [ ] **Step 1: Write the store file**

Mirror the existing `TokenhubStore` pattern: `public final class XStore: Sendable { private let db: DatabasePool; init(db:) }`. Each method uses `db.write { db in ... }` for writes and `db.read { db in ... }` for reads.

```swift
// Sources/NovaMLXDB/Stores/LoadBalancerStore.swift
import Foundation
import GRDB
import NovaMLXCore

// MARK: - LoadBalancerStore

public final class LoadBalancerStore: Sendable {
    private let db: DatabasePool
    public init(db: DatabasePool) { self.db = db }

    public func list() throws -> [LoadBalancer] {
        try db.read { db in
            try LoadBalancerRow
                .order(LoadBalancerRow.Columns.createdAt.desc)
                .fetchAll(db)
                .map(LoadBalancerRow.toDomain)
        }
    }

    public func get(_ id: UUID) throws -> LoadBalancer? {
        try db.read { db in
            try LoadBalancerRow.fetchOne(
                db, key: id.uuidString
            ).map(LoadBalancerRow.toDomain)
        }
    }

    public func getBySlug(_ slug: String) throws -> LoadBalancer? {
        try db.read { db in
            let row = try LoadBalancerRow
                .filter(LoadBalancerRow.Columns.slug == slug)
                .fetchOne(db)
            return row.map(LoadBalancerRow.toDomain)
        }
    }

    public func upsert(_ lb: LoadBalancer) throws {
        try db.write { db in
            try LoadBalancerRow.fromDomain(lb).save(db)
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
            if let row = try LoadBalancerRow.fetchOne(db, key: id.uuidString) {
                row.requestCount += 1
                try row.save(db)
            }
        }
    }
}

// MARK: - LBMemberStore

public final class LBMemberStore: Sendable {
    private let db: DatabasePool
    public init(db: DatabasePool) { self.db = db }

    public func listByLB(_ lbId: UUID) throws -> [LBMember] {
        try db.read { db in
            try LBMemberRow
                .filter(LBMemberRow.Columns.lbId == lbId.uuidString)
                .fetchAll(db)
                .map(LBMemberRow.toDomain)
        }
    }

    public func get(_ id: UUID) throws -> LBMember? {
        try db.read { db in
            try LBMemberRow.fetchOne(db, key: id.uuidString).map(LBMemberRow.toDomain)
        }
    }

    public func upsert(_ member: LBMember) throws {
        try db.write { db in
            try LBMemberRow.fromDomain(member).save(db)
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

    public func get(_ memberId: UUID) throws -> LBMemberStats? {
        try db.read { db in
            try LBMemberStatsRow.fetchOne(db, key: memberId.uuidString)
                .map(LBMemberStatsRow.toDomain)
        }
    }

    public func listByLB(_ lbId: UUID, memberStore: LBMemberStore) throws -> [LBMemberStats] {
        let members = try memberStore.listByLB(lbId)
        return try db.read { db in
            try members.compactMap { m in
                try LBMemberStatsRow.fetchOne(db, key: m.id.uuidString)
                    .map(LBMemberStatsRow.toDomain)
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
```

Now add the GRDB `Row` types (Codable structs that map 1:1 to table columns). They live in the same file:

```swift
// Sources/NovaMLXDB/Stores/LoadBalancerStore.swift (continued)

// MARK: - GRDB Row types

struct LoadBalancerRow: Codable, FetchableRecord, MutablePersistableRecord {
    static let databaseTableName = "load_balancers"

    let id: String
    var name: String
    var slug: String
    var strategy: String
    var maxRetries: Int
    var isEnabled: Int
    var requestCount: Int
    var createdAt: String
    var updatedAt: String

    enum Columns: String, ColumnExpression {
        case id, name, slug, strategy
        case maxRetries = "max_retries"
        case isEnabled = "is_enabled"
        case requestCount = "request_count"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    static func toDomain(_ row: LoadBalancerRow) -> LoadBalancer {
        LoadBalancer(
            id: UUID(uuidString: row.id) ?? UUID(),
            name: row.name,
            slug: row.slug,
            strategy: LBStrategy(rawValue: row.strategy) ?? .tiered,
            maxRetries: row.maxRetries,
            isEnabled: row.isEnabled != 0,
            requestCount: row.requestCount,
            createdAt: ISO8601DateFormatter().date(from: row.createdAt) ?? Date(),
            updatedAt: ISO8601DateFormatter().date(from: row.updatedAt) ?? Date()
        )
    }

    static func fromDomain(_ lb: LoadBalancer) -> LoadBalancerRow {
        let iso = ISO8601DateFormatter()
        return LoadBalancerRow(
            id: lb.id.uuidString,
            name: lb.name,
            slug: lb.slug,
            strategy: lb.strategy.rawValue,
            maxRetries: lb.maxRetries,
            isEnabled: lb.isEnabled ? 1 : 0,
            requestCount: lb.requestCount,
            createdAt: iso.string(from: lb.createdAt),
            updatedAt: iso.string(from: lb.updatedAt)
        )
    }
}

struct LBMemberRow: Codable, FetchableRecord, MutablePersistableRecord {
    static let databaseTableName = "lb_members"

    let id: String
    var lbId: String
    var kind: String
    var ref: String
    var weight: Int?
    var isEnabled: Int

    enum Columns: String, ColumnExpression {
        case id
        case lbId = "lb_id"
        case kind, ref, weight
        case isEnabled = "is_enabled"
    }

    static func toDomain(_ row: LBMemberRow) -> LBMember {
        LBMember(
            id: UUID(uuidString: row.id) ?? UUID(),
            lbId: UUID(uuidString: row.lbId) ?? UUID(),
            kind: MemberKind(rawValue: row.kind) ?? .remote,
            ref: row.ref,
            weight: row.weight,
            isEnabled: row.isEnabled != 0
        )
    }

    static func fromDomain(_ m: LBMember) -> LBMemberRow {
        LBMemberRow(
            id: m.id.uuidString,
            lbId: m.lbId.uuidString,
            kind: m.kind.rawValue,
            ref: m.ref,
            weight: m.weight,
            isEnabled: m.isEnabled ? 1 : 0
        )
    }
}

struct LBMemberStatsRow: Codable, FetchableRecord, MutablePersistableRecord {
    static let databaseTableName = "lb_member_stats"

    let memberId: String
    var requestCount: Int
    var successCount: Int
    var failureCount: Int
    var count5xx: Int
    var totalLatencyMs: Int64
    var lastUsedAt: String?
    var lastError: String?
    var updatedAt: String

    enum Columns: String, ColumnExpression {
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

    static func toDomain(_ row: LBMemberStatsRow) -> LBMemberStats {
        let iso = ISO8601DateFormatter()
        return LBMemberStats(
            memberId: UUID(uuidString: row.memberId) ?? UUID(),
            requestCount: row.requestCount,
            successCount: row.successCount,
            failureCount: row.failureCount,
            count5xx: row.count5xx,
            totalLatencyMs: row.totalLatencyMs,
            lastUsedAt: row.lastUsedAt.flatMap { iso.date(from: $0) },
            lastError: row.lastError,
            updatedAt: iso.date(from: row.updatedAt) ?? Date()
        )
    }
}
```

- [ ] **Step 2: Wire stores into NovaDB**

Open `Sources/NovaMLXDB/NovaDB.swift`. Add three stored properties alongside the existing `tokenhubStore`, and initialize them in `initStores()` (the existing method that creates `tokenhubStore`).

Add properties next to `tokenhubStore`:

```swift
public private(set) var loadBalancerStore: LoadBalancerStore!
public private(set) var lbMemberStore: LBMemberStore!
public private(set) var lbMemberStatsStore: LBMemberStatsStore!
```

In `initStores()`:

```swift
self.loadBalancerStore = LoadBalancerStore(db: configDB)
self.lbMemberStore = LBMemberStore(db: configDB)
self.lbMemberStatsStore = LBMemberStatsStore(db: configDB)
```

- [ ] **Step 3: Build to verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!`. If you see errors about NovaDB.shared being nil in tests, that's pre-existing — ignore.

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXDB/Stores/LoadBalancerStore.swift Sources/NovaMLXDB/NovaDB.swift
git commit -m "feat(db): add LoadBalancer/Member/Stats stores"
```

---

## Task 3: Database Migration

**Files:**
- Create: `Sources/NovaMLXDB/Migrations/LBMigration.swift`
- Modify: `Sources/NovaMLXDB/NovaDB.swift` — register migration
- Test: `Tests/NovaMLXDBTests/LoadBalancerStoreTests.swift` (created here; extended in Task 4)

- [ ] **Step 1: Write the migration file**

```swift
// Sources/NovaMLXDB/Migrations/LBMigration.swift
import GRDB

public enum LBMigration {
    /// Create the three LB tables and clean up legacy provider columns/rows.
    public static func vLoadBalancers(_ db: Database) throws {
        // 1. Create new tables (idempotent — CREATE IF NOT EXISTS)
        try db.create(table: "load_balancers", ifNotExists: true) { t in
            t.column("id", .text).primaryKey()
            t.column("name", .text).notNull()
            t.column("slug", .text).notNull().unique()
            t.column("strategy", .text).notNull().defaults(to: "tiered")
            t.column("max_retries", .integer).notNull().defaults(to: 3)
            t.column("is_enabled", .integer).notNull().defaults(to: 1)
            t.column("request_count", .integer).notNull().defaults(to: 0)
            t.column("created_at", .text).notNull()
            t.column("updated_at", .text).notNull()
        }

        try db.create(table: "lb_members", ifNotExists: true) { t in
            t.column("id", .text).primaryKey()
            t.column("lb_id", .text).notNull()
                .references("load_balancers", onDelete: .cascade)
            t.column("kind", .text).notNull()
            t.column("ref", .text).notNull()
            t.column("weight", .integer)
            t.column("is_enabled", .integer).notNull().defaults(to: 1)
        }
        try db.create(index: "idx_lb_members_lb_id",
                      on: "lb_members", columns: ["lb_id"], ifNotExists: true)

        try db.create(table: "lb_member_stats", ifNotExists: true) { t in
            t.column("member_id", .text).primaryKey()
                .references("lb_members", onDelete: .cascade)
            t.column("request_count", .integer).notNull().defaults(to: 0)
            t.column("success_count", .integer).notNull().defaults(to: 0)
            t.column("failure_count", .integer).notNull().defaults(to: 0)
            t.column("count_5xx", .integer).notNull().defaults(to: 0)
            t.column("total_latency_ms", .integer).notNull().defaults(to: 0)
            t.column("last_used_at", .text)
            t.column("last_error", .text)
            t.column("updated_at", .text).notNull()
        }

        // 2. Delete legacy local-virtual-provider rows
        try db.execute(sql: "DELETE FROM tokenhub_providers WHERE is_managed = 1")

        // 3. Drop legacy columns via table-rewrite pattern.
        //    SQLite < 3.35 (and GRDB versions before 6.x) don't support DROP COLUMN.
        //    Even where supported, the existing migration pattern here uses rewrite.
        let cols = try String.fetchCursor(
            db, sql: "PRAGMA table_info(tokenhub_providers)"
        )
        var allCols: [String] = []
        while let row = try cols.next() { allCols.append(row) }
        // allCols is "name|type|notnull|default|pk" — extract just the column names
        // PRAGMA table_info returns one row per column with fields:
        // cid, name, type, notnull, dflt_value, pk
        // GRDB's String.fetchCursor on PRAGMA returns the "name" field only here.
        // To be safe, fetch records:
        let colRecords = try Row.fetchAll(db, sql: "PRAGMA table_info(tokenhub_providers)")
        let names = colRecords.map { $0["name"] as String }

        let dropped = Set(["includeInLoadBalance", "isManaged"])
        let keep = names.filter { !dropped.contains($0) }
        let keepList = keep.joined(separator: ", ")

        if !dropped.isDisjoint(with: names) {
            try db.create(table: "tokenhub_providers_new", ifNotExists: true) { t in
                // Mirror existing schema for kept columns. We don't know the exact types
                // at compile time, so we copy via CREATE TABLE ... AS SELECT for shape
                // and add constraints back via ALTER.
                _ = t
            }
            // Use raw SQL for the rewrite (GRDB's table-builder doesn't easily mirror
            // an existing table). CREATE TABLE new AS SELECT preserves data; we then
            // recreate indexes.
            try db.execute(sql: """
                DROP TABLE IF EXISTS tokenhub_providers_new;
                CREATE TABLE tokenhub_providers_new AS SELECT \(keepList) FROM tokenhub_providers;
                DROP TABLE tokenhub_providers;
                ALTER TABLE tokenhub_providers_new RENAME TO tokenhub_providers;
                """)
            // Note: constraints (PK, UNIQUE, FK) are lost by CREATE AS SELECT.
            // This is acceptable — provider rows are simple value rows with no FKs.
            // The id column is still indexed by the slug-lookup query in TokenhubStore.
        }
    }
}
```

> **Note on the table-rewrite:** SQLite's `CREATE TABLE x AS SELECT` creates a copy without constraints/indexes. The `tokenhub_providers` table has no foreign keys (verified by inspecting the existing schema), and the slug lookup uses a full-scan which is fine for ~10 rows. If performance matters later, re-add `CREATE INDEX tokenhub_providers_name_idx ON tokenhub_providers(name)` in the migration.

- [ ] **Step 2: Register the migration in NovaDB.swift**

Find the existing `configMigrator.registerMigration(...)` calls (around `NovaDB.swift:107`). Add after the latest one:

```swift
configMigrator.registerMigration("v_load_balancers") { db in
    try LBMigration.vLoadBalancers(db)
}
```

- [ ] **Step 3: Write the migration test**

```swift
// Tests/NovaMLXDBTests/LoadBalancerStoreTests.swift
import Testing
import Foundation
@testable import NovaMLXDB
import NovaMLXCore

@Suite("LoadBalancer Store + Migration")
struct LoadBalancerStoreTests {
    private func makeTmpDB() throws -> URL {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: tmp, withIntermediateDirectories: true
        )
        try NovaDB.shared.setup(baseDir: tmp)
        return tmp
    }

    @Test("Migration creates all 3 LB tables")
    func migrationCreatesTables() throws {
        _ = try makeTmpDB()
        // Force-read via store — should not throw "no such table"
        let lbs = try NovaDB.shared.loadBalancerStore.list()
        #expect(lbs.isEmpty)
        let members = try NovaDB.shared.lbMemberStore.listByLB(UUID())
        #expect(members.isEmpty)
    }

    @Test("Migration drops legacy provider columns")
    @Test(.disabled("Enable after Task 12 changes TokenhubProviderRecord"))
    func migrationDropsLegacyColumns() throws {
        // Stub: full verification happens after the TokenhubProviderRecord
        // struct is updated to drop includeInLoadBalance/isManaged.
    }
}
```

- [ ] **Step 4: Run tests, verify migration test passes**

Run: `swift test --filter LoadBalancerStoreTests 2>&1 | tail -15`
Expected: `migrationCreatesTables` passes; `migrationDropsLegacyColumns` is skipped (disabled).

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDB/Migrations/LBMigration.swift \
        Sources/NovaMLXDB/NovaDB.swift \
        Tests/NovaMLXDBTests/LoadBalancerStoreTests.swift
git commit -m "feat(db): add v_load_balancers migration"
```

---

## Task 4: Store CRUD Tests + Slug Validation

**Files:**
- Extend: `Tests/NovaMLXDBTests/LoadBalancerStoreTests.swift`

- [ ] **Step 1: Add CRUD and cascade tests**

Append to the `LoadBalancerStoreTests` suite:

```swift
extension LoadBalancerStoreTests {
    @Test("Upsert + get LoadBalancer by id and slug")
    func upsertAndGet() throws {
        _ = try makeTmpDB()
        let lb = LoadBalancer(name: "Coding Pool", slug: "coding-pool", strategy: .roundRobin)
        try NovaDB.shared.loadBalancerStore.upsert(lb)

        let byId = try NovaDB.shared.loadBalancerStore.get(lb.id)
        #expect(byId?.slug == "coding-pool")
        #expect(byId?.strategy == .roundRobin)

        let bySlug = try NovaDB.shared.loadBalancerStore.getBySlug("coding-pool")
        #expect(bySlug?.id == lb.id)
    }

    @Test("Delete LoadBalancer cascades to members + stats")
    func deleteCascades() throws {
        _ = try makeTmpDB()
        let lb = LoadBalancer(name: "X", slug: "x")
        try NovaDB.shared.loadBalancerStore.upsert(lb)

        let m = LBMember(lbId: lb.id, kind: .remote, ref: "provider-1")
        try NovaDB.shared.lbMemberStore.upsert(m)
        try NovaDB.shared.lbMemberStatsStore.recordRequest(
            memberId: m.id, succeeded: true, latencyMs: 50, httpStatus: 200, errorMessage: nil
        )

        // Sanity
        let membersBefore = try NovaDB.shared.lbMemberStore.listByLB(lb.id)
        #expect(membersBefore.count == 1)
        let statsBefore = try NovaDB.shared.lbMemberStatsStore.get(m.id)
        #expect(statsBefore?.requestCount == 1)

        try NovaDB.shared.loadBalancerStore.delete(lb.id)

        // Cascade
        let membersAfter = try NovaDB.shared.lbMemberStore.listByLB(lb.id)
        #expect(membersAfter.isEmpty)
        let statsAfter = try NovaDB.shared.lbMemberStatsStore.get(m.id)
        #expect(statsAfter == nil)
    }

    @Test("recordRequest lazily creates stats row and aggregates correctly")
    func recordRequestAggregation() throws {
        _ = try makeTmpDB()
        let memberId = UUID()
        try NovaDB.shared.lbMemberStatsStore.recordRequest(
            memberId: memberId, succeeded: true, latencyMs: 100,
            httpStatus: 200, errorMessage: nil
        )
        try NovaDB.shared.lbMemberStatsStore.recordRequest(
            memberId: memberId, succeeded: true, latencyMs: 200,
            httpStatus: 200, errorMessage: nil
        )
        try NovaDB.shared.lbMemberStatsStore.recordRequest(
            memberId: memberId, succeeded: false, latencyMs: 0,
            httpStatus: 503, errorMessage: "timeout"
        )

        let stats = try NovaDB.shared.lbMemberStatsStore.get(memberId)
        #expect(stats?.requestCount == 3)
        #expect(stats?.successCount == 2)
        #expect(stats?.failureCount == 1)
        #expect(stats?.count5xx == 1)
        #expect(stats?.totalLatencyMs == 300)
        #expect(stats?.avgLatencyMs == 150)  // 300 / 2 successes
        #expect(stats?.lastError == "timeout")
    }
}
```

- [ ] **Step 2: Run tests**

Run: `swift test --filter LoadBalancerStoreTests 2>&1 | tail -20`
Expected: All 4 tests pass (1 from Task 3 + 3 here).

- [ ] **Step 3: Commit**

```bash
git add Tests/NovaMLXDBTests/LoadBalancerStoreTests.swift
git commit -m "test(db): LB store CRUD + cascade + stats aggregation"
```

---

## Task 5: LBRouter — Pure Strategy Functions

**Files:**
- Create: `Sources/NovaMLXCore/LBRouter.swift`
- Create: `Tests/NovaMLXCoreTests/LBRouterTests.swift`

The router is a **pure function** over a snapshot of inputs. It does no I/O. This makes it trivially testable.

- [ ] **Step 1: Define the router interface**

```swift
// Sources/NovaMLXCore/LBRouter.swift
import Foundation

/// Read-only inputs the router needs to make a decision.
public struct LBRouterInput: Sendable {
    public let lb: LoadBalancer
    public let members: [LBMember]
    public let stats: [UUID: LBMemberStats]      // keyed by member.id
    public let isLocalLoaded: (String) -> Bool   // model_id -> loaded?
    public let isProviderFree: (String) -> Bool  // provider_id -> isFree?

    public init(
        lb: LoadBalancer,
        members: [LBMember],
        stats: [UUID: LBMemberStats] = [:],
        isLocalLoaded: @escaping (String) -> Bool,
        isProviderFree: @escaping (String) -> Bool
    ) {
        self.lb = lb; self.members = members; self.stats = stats
        self.isLocalLoaded = isLocalLoaded
        self.isProviderFree = isProviderFree
    }
}

/// The ordered candidate list the LB will try. First element is preferred.
public typealias LBCandidateList = [LBMember]

public enum LBRouter {
    /// Filter + apply strategy. Returned list is ordered (preferred first).
    /// Empty list means "no healthy members".
    public static func plan(_ input: LBRouterInput) -> LBCandidateList {
        let healthy = input.members.filter { member in
            guard member.isEnabled else { return false }
            switch member.kind {
            case .local:
                return input.isLocalLoaded(member.ref)
            case .remote:
                return true  // remotes are assumed healthy; failures handled at proxy layer
            }
        }
        guard !healthy.isEmpty else { return [] }

        switch input.lb.strategy {
        case .tiered:      return applyTiered(healthy, input)
        case .roundRobin:  return applyRoundRobin(healthy, input)
        case .weighted:    return applyWeighted(healthy, input)
        case .lowestLatency: return applyLowestLatency(healthy, input)
        case .random:      return applyRandom(healthy, input)
        }
    }

    // MARK: - Strategies

    /// Tiers: local+free=3, local=2, free=1, paid=0. Higher tier first.
    /// Within a tier, stable order by member.id (deterministic for tests).
    /// For round-robin within tier, the proxy layer rotates the returned
    /// list by `lb.requestCount` to spread load over time.
    private static func applyTiered(
        _ members: [LBMember], _ input: LBRouterInput
    ) -> LBCandidateList {
        func tier(_ m: LBMember) -> Int {
            switch m.kind {
            case .local:
                return 2  // locals are free by definition
            case .remote:
                return input.isProviderFree(m.ref) ? 1 : 0
            }
        }
        let rotated = members.rotated(by: input.lb.requestCount)
        return rotated.sorted { a, b in
            let ta = tier(a), tb = tier(b)
            return ta != tb ? ta > tb : a.id.uuidString < b.id.uuidString
        }
    }

    /// Equal rotation. Proxy rotates the full list by requestCount so each
    /// request hits a different starting member.
    private static func applyRoundRobin(
        _ members: [LBMember], _ input: LBRouterInput
    ) -> LBCandidateList {
        members.rotated(by: input.lb.requestCount)
    }

    /// Probability ∝ weight. Build candidate list by repeating each member
    /// `weight` times (nil = 1), then interleave + rotate by requestCount.
    /// Weight 0 is rejected at write time; defensive: treat 0 as 1.
    private static func applyWeighted(
        _ members: [LBMember], _ input: LBRouterInput
    ) -> LBCandidateList {
        var expanded: [LBMember] = []
        for m in members {
            let w = max(1, m.weight ?? 1)
            expanded.append(contentsOf: Array(repeating: m, count: w))
        }
        return expanded.rotated(by: input.lb.requestCount)
    }

    /// Lowest avg_latency_ms first. Cold-start (successCount == 0) treated as 0.
    /// Ties broken by success_rate desc, then lastUsedAt asc.
    private static func applyLowestLatency(
        _ members: [LBMember], _ input: LBRouterInput
    ) -> LBCandidateList {
        members.sorted { a, b in
            let sa = input.stats[a.id], sb = input.stats[b.id]
            let la = sa?.avgLatencyMs ?? 0
            let lb = sb?.avgLatencyMs ?? 0
            if la != lb { return la < lb }
            let ra = sa?.successRate ?? 0
            let rb = sb?.successRate ?? 0
            if ra != rb { return ra > rb }
            let ta = sa?.lastUsedAt ?? Date.distantPast
            let tb = sb?.lastUsedAt ?? Date.distantPast
            return ta < tb
        }
    }

    /// Uniform random shuffle. Use each member's UUID hash as deterministic
    /// seed (so tests are reproducible).
    private static func applyRandom(
        _ members: [LBMember], _ input: LBRouterInput
    ) -> LBCandidateList {
        members.shuffled(by: { a, b in
            a.id.uuidString.hashValue < b.id.uuidString.hashValue
        })
    }
}

// MARK: - Array rotation helper

extension Array {
    /// Rotate left by `n`. `n` may be larger than count.
    fileprivate func rotated(by n: Int) -> [Element] {
        guard !isEmpty else { return [] }
        let shift = ((n % count) + count) % count
        return Array(self[shift..<count] + self[0..<shift])
    }
}
```

- [ ] **Step 2: Write router tests**

```swift
// Tests/NovaMLXCoreTests/LBRouterTests.swift
import Testing
import Foundation
@testable import NovaMLXCore

@Suite("LBRouter")
struct LBRouterTests {
    // Fixtures: a local model (loaded) + two remotes (one free, one paid)
    private func makeFixture() -> (LoadBalancer, [LBMember]) {
        let lb = LoadBalancer(name: "X", slug: "x", strategy: .tiered, requestCount: 0)
        let local = LBMember(lbId: lb.id, kind: .local, ref: "model-a")
        let freeRemote = LBMember(lbId: lb.id, kind: .remote, ref: "prov-free")
        let paidRemote = LBMember(lbId: lb.id, kind: .remote, ref: "prov-paid")
        return (lb, [local, freeRemote, paidRemote])
    }

    private func makeInput(
        lb: LoadBalancer, members: [LBMember],
        loaded: [String: Bool] = [:],
        free: [String: Bool] = [:]
    ) -> LBRouterInput {
        LBRouterInput(
            lb: lb, members: members,
            isLocalLoaded: { loaded[$0] ?? false },
            isProviderFree: { free[$0] ?? false }
        )
    }

    @Test("Empty member list returns empty plan")
    func emptyMembers() {
        let lb = LoadBalancer(name: "X", slug: "x")
        let input = makeInput(lb: lb, members: [])
        #expect(LBRouter.plan(input).isEmpty)
    }

    @Test("Unloaded local is skipped")
    func unloadedLocalSkipped() {
        let (lb, members) = makeFixture()
        // local unloaded, both remotes available
        let input = makeInput(
            lb: lb, members: members,
            loaded: ["model-a": false],
            free: ["prov-free": true, "prov-paid": false]
        )
        let plan = LBRouter.plan(input)
        // Should be 2 remotes; tiered sorts free(1) above paid(0)
        #expect(plan.count == 2)
        #expect(plan[0].ref == "prov-free")
        #expect(plan[1].ref == "prov-paid")
    }

    @Test("Tiered: local preferred when loaded, then free, then paid")
    func tieredOrdering() {
        let (lb, members) = makeFixture()
        let input = makeInput(
            lb: lb, members: members,
            loaded: ["model-a": true],
            free: ["prov-free": true, "prov-paid": false]
        )
        let plan = LBRouter.plan(input)
        #expect(plan.count == 3)
        #expect(plan[0].kind == .local)         // tier 2
        #expect(plan[1].ref == "prov-free")     // tier 1
        #expect(plan[2].ref == "prov-paid")     // tier 0
    }

    @Test("Disabled member is skipped even if loaded")
    func disabledMemberSkipped() {
        let (lb, members) = makeFixture()
        var local = members[0]
        local.isEnabled = false
        let input = makeInput(
            lb: lb, members: [local, members[1], members[2]],
            loaded: ["model-a": true],
            free: ["prov-free": true, "prov-paid": false]
        )
        let plan = LBRouter.plan(input)
        #expect(plan.count == 2)
        #expect(plan.first { $0.kind == .local } == nil)
    }

    @Test("Round-robin rotates by requestCount")
    func roundRobinRotates() {
        let (lb, members) = makeFixture()
        var lb0 = lb; lb0.strategy = .roundRobin; lb0.requestCount = 0
        var lb1 = lb; lb1.strategy = .roundRobin; lb1.requestCount = 1
        var lb2 = lb; lb2.strategy = .roundRobin; lb2.requestCount = 2

        let loaded: [String: Bool] = ["model-a": true]
        let free: [String: Bool] = ["prov-free": true, "prov-paid": false]

        let p0 = LBRouter.plan(makeInput(lb: lb0, members: members, loaded: loaded, free: free))
        let p1 = LBRouter.plan(makeInput(lb: lb1, members: members, loaded: loaded, free: free))
        let p2 = LBRouter.plan(makeInput(lb: lb2, members: members, loaded: loaded, free: free))

        // Same members, different starting points
        #expect(p0.first?.id == members[0].id)
        #expect(p1.first?.id == members[1].id)
        #expect(p2.first?.id == members[2].id)
    }

    @Test("Weighted: nil weight = 1, weight 0 rejected defensively")
    func weightedNilAndZero() {
        let (lb, members) = makeFixture()
        var lb = lb; lb.strategy = .weighted
        var m0 = members[0]; m0.weight = nil     // → 1
        var m1 = members[1]; m1.weight = 3
        var m2 = members[2]; m2.weight = 0       // → 1 defensively
        let input = makeInput(
            lb: lb, members: [m0, m1, m2],
            loaded: ["model-a": true],
            free: ["prov-free": true, "prov-paid": false]
        )
        let plan = LBRouter.plan(input)
        // Expanded: m0×1, m1×3, m2×1 = 5 entries. m1 should appear 3 times.
        #expect(plan.count == 5)
        #expect(plan.filter { $0.id == m1.id }.count == 3)
    }

    @Test("Lowest latency: cold-start (no stats) treated as latency=0 → preferred")
    func lowestLatencyColdStart() {
        let (lb, members) = makeFixture()
        var lb = lb; lb.strategy = .lowestLatency
        let stats: [UUID: LBMemberStats] = [
            members[1].id: LBMemberStats(memberId: members[1].id, successCount: 5, totalLatencyMs: 500),  // avg 100
            members[2].id: LBMemberStats(memberId: members[2].id, successCount: 10, totalLatencyMs: 1000) // avg 100
        ]
        // members[0] has no stats → cold → avg 0 → preferred
        let input = LBRouterInput(
            lb: lb, members: members, stats: stats,
            isLocalLoaded: { _ in true },
            isProviderFree: { _ in true }
        )
        let plan = LBRouter.plan(input)
        #expect(plan.first?.id == members[0].id)
    }

    @Test("Lowest latency: picks lowest avg after warmup")
    func lowestLatencyWarmed() {
        let (lb, members) = makeFixture()
        var lb = lb; lb.strategy = .lowestLatency
        let stats: [UUID: LBMemberStats] = [
            members[0].id: LBMemberStats(memberId: members[0].id, successCount: 10, totalLatencyMs: 2000), // avg 200
            members[1].id: LBMemberStats(memberId: members[1].id, successCount: 5,  totalLatencyMs: 500),  // avg 100
            members[2].id: LBMemberStats(memberId: members[2].id, successCount: 8,  totalLatencyMs: 1600)  // avg 200
        ]
        let input = LBRouterInput(
            lb: lb, members: members, stats: stats,
            isLocalLoaded: { _ in true },
            isProviderFree: { _ in true }
        )
        #expect(LBRouter.plan(input).first?.id == members[1].id)
    }
}
```

- [ ] **Step 3: Run tests**

Run: `swift test --filter LBRouterTests 2>&1 | tail -20`
Expected: all 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXCore/LBRouter.swift Tests/NovaMLXCoreTests/LBRouterTests.swift
git commit -m "feat(core): LBRouter with 5 strategies + tests"
```

---

## Task 6: TokenHub Provider Cleanup

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift` — drop fields
- Modify: `Sources/NovaMLXDB/Stores/TokenhubStore.swift` — drop fields from Record
- Modify: `Sources/NovaMLXCore/TokenhubStore+Domain.swift` — drop from mapper
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift` — remove LB checkbox + filter
- Modify: `Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift` — remove `provisionLocalProviders` call (if any)

**Order matters:** do this BEFORE the LBProxy task so the legacy LB code paths are gone before the new one is wired up.

- [ ] **Step 1: Inspect current TokenhubProvider struct**

Run: `grep -n "includeInLoadBalance\|isManaged\|isLocal" Sources/NovaMLXCore/TokenhubTypes.swift Sources/NovaMLXDB/Stores/TokenhubStore.swift Sources/NovaMLXCore/TokenhubStore+Domain.swift`

Note every line that references these fields. You'll need to remove all of them.

- [ ] **Step 2: Drop fields from TokenhubProvider (domain type)**

In `Sources/NovaMLXCore/TokenhubTypes.swift`, delete:
- `var includeInLoadBalance: Bool`
- `var isManaged: Bool`

And from the `init(...)`, remove those parameters. Update the doc comment if it mentions them.

Also delete `var isLocal: Bool` if it's computed from `isManaged`. Locals are no longer in TokenHub.

- [ ] **Step 3: Drop fields from TokenhubProviderRecord**

In `Sources/NovaMLXDB/Stores/TokenhubStore.swift`, delete the corresponding columns from `TokenhubProviderRecord` (the GRDB PersistableRecord).

- [ ] **Step 4: Drop fields from the domain mapper**

In `Sources/NovaMLXCore/TokenhubStore+Domain.swift`:
- Remove `includeInLoadBalance` / `isManaged` / `isLocal` from `toDomain(...)`.
- Remove the same from `fromDomain(...)` if it exists.
- **Delete the `provisionLocalProviders()` method entirely** (around the file).
- **Delete the `resolve()` method entirely** (around line 146-180). LBRouter + LBProxy replace it.

- [ ] **Step 5: Find and fix every caller of removed APIs**

Run: `grep -rn "includeInLoadBalance\|isManaged\|provisionLocalProviders\|\.resolve(" Sources/ Tests/`

For each hit:
- `includeInLoadBalance` / `isManaged` field reference → remove the line.
- `provisionLocalProviders()` call → delete the call site entirely.
- `resolve(...)` call on TokenhubManager → delete the call; it's replaced by LBRouter + LBProxy in Task 7.

If any caller is in the API proxy (`APIServer+TokenhubProxy.swift`), leave a `// TODO(Task 7): replace with LBProxy` marker at that line so the next task picks it up.

- [ ] **Step 6: Remove LB checkbox from TokenhubPageView**

In `Sources/NovaMLXMenuBar/TokenhubPageView.swift`:
- Delete the `formIncludeInLB` State binding (~line 995).
- Delete the `lbProviders` filter (~line 406) — provider list shows ALL providers directly.
- Delete the "Load Balance" checkbox from the provider form.

- [ ] **Step 7: Build to verify**

Run: `./build.sh 2>&1 | tail -10`
Expected: `Build complete!`. Any errors are missed callers — fix them.

If `APIServer+TokenhubProxy.swift` now has a missing routing path for the case that previously called `resolve()`, comment out that branch with `// TODO(Task 7): wire LBProxy`. The compile error becomes a runtime no-op temporarily.

- [ ] **Step 8: Commit**

```bash
git add -u Sources/NovaMLXCore/TokenhubTypes.swift \
            Sources/NovaMLXDB/Stores/TokenhubStore.swift \
            Sources/NovaMLXCore/TokenhubStore+Domain.swift \
            Sources/NovaMLXMenuBar/TokenhubPageView.swift \
            Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift
git commit -m "refactor(tokenhub): drop includeInLB/isManaged; remove resolve() + provisionLocalProviders"
```

---

## Task 7: LBProxy — Request Dispatch + Retry

**Files:**
- Create: `Sources/NovaMLXAPI/LBProxy.swift`
- Modify: `Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift` — dispatch `lb:` prefix

- [ ] **Step 1: Define LBProxy**

```swift
// Sources/NovaMLXAPI/LBProxy.swift
import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXDB
import Logging

/// Outcome of routing one request through an LB.
public enum LBProxyOutcome: Sendable {
    case success(HTTPResponse)
    case allMembersFailed(lastError: String)
    case noHealthyMembers
    case unknownLB(slug: String)
}

/// Single shot: pick a member, send request, retry on failure.
/// Stateless — caller (APIServer) holds the persistence coupling.
public actor LBProxy {
    private let lbStore: LoadBalancerStore
    private let memberStore: LBMemberStore
    private let statsStore: LBMemberStatsStore
    private let isLocalLoaded: @Sendable (String) -> Bool
    private let isProviderFree: @Sendable (String) -> Bool

    public init(
        lbStore: LoadBalancerStore,
        memberStore: LBMemberStore,
        statsStore: LBMemberStatsStore,
        isLocalLoaded: @escaping @Sendable (String) -> Bool,
        isProviderFree: @escaping @Sendable (String) -> Bool
    ) {
        self.lbStore = lbStore
        self.memberStore = memberStore
        self.statsStore = statsStore
        self.isLocalLoaded = isLocalLoaded
        self.isProviderFree = isProviderFree
    }

    public func handle(
        slug: String,
        payload: RequestPayload,
        sendToMember: @Sendable (LBMember, RequestPayload) async throws -> HTTPResponse
    ) async -> LBProxyOutcome {
        // Store methods are sync (GRDB db.read/db.write block). From inside
        // an actor we can call them directly without `await`.
        guard let lb = try? lbStore.getBySlug(slug) else {
            return .unknownLB(slug: slug)
        }
        guard lb.isEnabled else {
            return .allMembersFailed(lastError: "load balancer '\(slug)' is disabled")
        }

        let members = (try? memberStore.listByLB(lb.id)) ?? []
        let statsMap = loadStats(for: members)

        let input = LBRouterInput(
            lb: lb, members: members, stats: statsMap,
            isLocalLoaded: isLocalLoaded, isProviderFree: isProviderFree
        )
        let candidates = LBRouter.plan(input)
        guard !candidates.isEmpty else {
            return .noHealthyMembers
        }

        // Bump per-LB counter once per request (not per retry).
        try? lbStore.incrementRequestCount(lb.id)

        var lastError = "no attempts made"
        let maxAttempts = min(lb.maxRetries, candidates.count)
        for member in candidates.prefix(maxAttempts) {
            let started = Date()
            do {
                let resp = try await sendToMember(member, payload)
                let latencyMs = Int64(Date().timeIntervalSince(started) * 1000)
                recordOutcome(
                    memberId: member.id,
                    succeeded: resp.status.code < 500,
                    latencyMs: latencyMs,
                    httpStatus: Int(resp.status.code),
                    errorMessage: resp.status.code >= 500 ? "HTTP \(resp.status.code)" : nil
                )
                if resp.status.code < 500 {
                    return .success(resp)
                }
                lastError = "HTTP \(resp.status.code)"
            } catch {
                let latencyMs = Int64(Date().timeIntervalSince(started) * 1000)
                recordOutcome(
                    memberId: member.id,
                    succeeded: false,
                    latencyMs: latencyMs,
                    httpStatus: 0,
                    errorMessage: String(describing: error).prefix(500).description
                )
                lastError = String(describing: error).prefix(200).description
            }
        }
        return .allMembersFailed(lastError: lastError)
    }

    private func loadStats(for members: [LBMember]) -> [UUID: LBMemberStats] {
        var map: [UUID: LBMemberStats] = [:]
        for m in members {
            if let s = try? statsStore.get(m.id) {
                map[m.id] = s
            }
        }
        return map
    }

    private func recordOutcome(
        memberId: UUID, succeeded: Bool, latencyMs: Int64,
        httpStatus: Int, errorMessage: String?
    ) {
        try? statsStore.recordRequest(
            memberId: memberId, succeeded: succeeded,
            latencyMs: latencyMs, httpStatus: httpStatus,
            errorMessage: errorMessage
        )
    }
}

/// Minimal payload abstraction for testing. Real impl uses Hummingbird's
/// Request/Response types.
public struct RequestPayload: Sendable {
    public let body: Data
    public let headers: [String: String]
    public init(body: Data, headers: [String: String] = [:]) {
        self.body = body; self.headers = headers
    }
}

public struct HTTPResponse: Sendable {
    public let status: HTTPStatus
    public let body: Data
    public init(status: HTTPStatus, body: Data = Data()) {
        self.status = status; self.body = body
    }
}

public struct HTTPStatus: Sendable, Equatable {
    public let code: Int
    public init(_ code: Int) { self.code = code }
    public static let ok = HTTPStatus(200)
    public static let badRequest = HTTPStatus(400)
    public static let notFound = HTTPStatus(404)
    public static let serverError = HTTPStatus(500)
    public static let badGateway = HTTPStatus(502)
    public static let serviceUnavailable = HTTPStatus(503)
}
```

> **Note:** `HTTPStatus` / `HTTPResponse` shown above are minimal stand-ins. If the codebase already imports Hummingbird's `HBResponse` / `HBHTTPResponseContext`, use those instead and delete the local `HTTPStatus` struct. Check by running `grep -n "HBResponse\|HBHTTPResponse" Sources/NovaMLXAPI/` before writing this file.

- [ ] **Step 2: Build**

Run: `./build.sh 2>&1 | tail -5`
Expected: compiles. Resolve any naming conflicts with existing Hummingbird types.

- [ ] **Step 3: Commit (LBProxy in isolation)**

```bash
git add Sources/NovaMLXAPI/LBProxy.swift
git commit -m "feat(api): LBProxy with retry loop + stats recording"
```

- [ ] **Step 4: Wire LBProxy into APIServer**

Open `Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift`. Find where the model prefix dispatch lives (around `ChatPageView.swift:121` was the SwiftUI side; the API server equivalent uses `model.hasPrefix("tknet:")`).

Add a new branch for `lb:` prefix BEFORE the existing branches:

```swift
// pseudocode — adapt to actual APIServer routing style
if model.hasPrefix("lb:") {
    let slug = String(model.dropFirst("lb:".count))
    let outcome = await lbProxy.handle(
        slug: slug,
        payload: RequestPayload(body: req.body, headers: req.headers.dictionary)
    ) { member, payload in
        switch member.kind {
        case .local:
            // dispatch to local inference engine
            return try await self.localInferenceProxy(model: member.ref, payload: payload)
        case .remote:
            // dispatch to remote provider
            return try await self.remoteProviderProxy(providerId: member.ref, payload: payload)
        }
    }
    switch outcome {
    case .success(let resp): return resp
    case .unknownLB: return .init(status: .notFound, body: ...)
    case .noHealthyMembers: return .init(status: .serviceUnavailable, body: ...)
    case .allMembersFailed(let err): return .init(status: .badGateway, body: ...)
    }
}
```

You also need to instantiate `LBProxy` once at server startup. Find where `TokenhubManager` is constructed and add a sibling:

```swift
let lbProxy = LBProxy(
    lbStore: NovaDB.shared.loadBalancerStore,
    memberStore: NovaDB.shared.lbMemberStore,
    statsStore: NovaDB.shared.lbMemberStatsStore,
    isLocalLoaded: { modelId in MLXEngine.shared.isModelLoaded(modelId) },
    isProviderFree: { providerId in
        // sync read from tokenhubStore
        (try? NovaDB.shared.tokenhubStore.get(providerId).map { $0.isFree }) ?? false
    }
)
```

- [ ] **Step 5: Build + smoke test**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!`.

Then manual smoke test: start app, create an LB via REST API (next task), POST `/v1/chat/completions` with `model="lb:nonexistent"` → expect 404.

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift
git commit -m "feat(api): dispatch lb: prefix through LBProxy"
```

---

## Task 8: Admin REST API

**Files:**
- Create: `Sources/NovaMLXAPI/APIServer+LoadBalancerAdmin.swift`

- [ ] **Step 1: Write the endpoints**

Mirror the existing admin route registration style in NovaMLXAPI. The file should export a single function that registers routes on the HBApplication.

```swift
// Sources/NovaMLXAPI/APIServer+LoadBalancerAdmin.swift
import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXDB
import Logging

extension APIServer {
    public func registerLoadBalancerAdminRoutes(_ router: HBRouter) {
        // All routes require admin API key middleware (applied at router group level).
        let group = router.group()
            .add(middleware: AdminAPIKeyMiddleware())  // existing

        // GET /admin/load-balancers
        group.get("admin/load-balancers") { _, _ in
            let lbs = try NovaDB.shared.loadBalancerStore.list()
            return lbs.map(LBDTO.init)
        }

        // POST /admin/load-balancers
        group.post("admin/load-balancers") { request, _ in
            let input = try request.decode(as: CreateLBInput.self)
            // Validate slug
            guard Self.isValidSlug(input.slug) else {
                throw HBHTTPError(.badRequest, message: "slug must match ^[a-z0-9-]+$")
            }
            if (try NovaDB.shared.loadBalancerStore.getBySlug(input.slug)) != nil {
                throw HBHTTPError(.badRequest, message: "slug already exists")
            }
            let lb = LoadBalancer(
                name: input.name, slug: input.slug,
                strategy: input.strategy ?? .tiered,
                maxRetries: input.maxRetries ?? 3
            )
            try NovaDB.shared.loadBalancerStore.upsert(lb)
            return LBDTO(lb)
        }

        // GET /admin/load-balancers/:id
        group.get("admin/load-balancers/:id") { request, _ in
            guard let id = request.parameters.uuid("id"),
                  let lb = try NovaDB.shared.loadBalancerStore.get(id) else {
                throw HBHTTPError(.notFound, message: "LB not found")
            }
            let members = try NovaDB.shared.lbMemberStore.listByLB(id)
            let memberStats = members.compactMap {
                try? NovaDB.shared.lbMemberStatsStore.get($0.id)
            }
            return LBDetailDTO(lb: lb, members: members.map(MemberDTO.init),
                               stats: memberStats.map(StatsDTO.init))
        }

        // PATCH /admin/load-balancers/:id
        group.patch("admin/load-balancers/:id") { request, _ in
            guard let id = request.parameters.uuid("id"),
                  var lb = try NovaDB.shared.loadBalancerStore.get(id) else {
                throw HBHTTPError(.notFound, message: "LB not found")
            }
            let patch = try request.decode(as: PatchLBInput.self)
            if let v = patch.name { lb.name = v }
            if let v = patch.slug {
                guard Self.isValidSlug(v) else {
                    throw HBHTTPError(.badRequest, message: "slug must match ^[a-z0-9-]+$")
                }
                if (try NovaDB.shared.loadBalancerStore.getBySlug(v))?.id != lb.id {
                    throw HBHTTPError(.badRequest, message: "slug already exists")
                }
                lb.slug = v
            }
            if let v = patch.strategy { lb.strategy = v }
            if let v = patch.maxRetries { lb.maxRetries = v }
            if let v = patch.isEnabled { lb.isEnabled = v }
            lb.updatedAt = Date()
            try NovaDB.shared.loadBalancerStore.upsert(lb)
            return LBDTO(lb)
        }

        // DELETE /admin/load-balancers/:id  (cascade via FK)
        group.delete("admin/load-balancers/:id") { request, _ in
            guard let id = request.parameters.uuid("id") else {
                throw HBHTTPError(.badRequest, message: "invalid id")
            }
            try NovaDB.shared.loadBalancerStore.delete(id)
            return ["ok": true]
        }

        // POST /admin/load-balancers/:id/members
        group.post("admin/load-balancers/:id/members") { request, _ in
            guard let lbId = request.parameters.uuid("id"),
                  try NovaDB.shared.loadBalancerStore.get(lbId) != nil else {
                throw HBHTTPError(.notFound, message: "LB not found")
            }
            let input = try request.decode(as: AddMemberInput.self)

            // Validate weight
            if let w = input.weight, w <= 0 {
                throw HBHTTPError(.badRequest, message: "weight must be > 0 (use is_enabled=false to exclude)")
            }

            // Validate ref exists (lazy on remote, sync on local)
            switch input.kind {
            case .local:
                // Caller must ensure model is loaded before adding; we don't enforce here.
                break
            case .remote:
                let exists = (try? NovaDB.shared.tokenhubStore.get(input.ref)) != nil
                if !exists {
                    throw HBHTTPError(.badRequest, message: "remote provider not found: \(input.ref)")
                }
            }

            let member = LBMember(
                lbId: lbId, kind: input.kind, ref: input.ref,
                weight: input.weight, isEnabled: true
            )
            try NovaDB.shared.lbMemberStore.upsert(member)
            return MemberDTO(member)
        }

        // PATCH /admin/load-balancers/:id/members/:memberId
        group.patch("admin/load-balancers/:id/members/:memberId") { request, _ in
            guard let memberId = request.parameters.uuid("memberId"),
                  var member = try NovaDB.shared.lbMemberStore.get(memberId) else {
                throw HBHTTPError(.notFound, message: "member not found")
            }
            let patch = try request.decode(as: PatchMemberInput.self)
            if let v = patch.weight {
                guard v > 0 else {
                    throw HBHTTPError(.badRequest, message: "weight must be > 0")
                }
                member.weight = v
            }
            if let v = patch.isEnabled { member.isEnabled = v }
            try NovaDB.shared.lbMemberStore.upsert(member)
            return MemberDTO(member)
        }

        // DELETE /admin/load-balancers/:id/members/:memberId
        group.delete("admin/load-balancers/:id/members/:memberId") { request, _ in
            guard let memberId = request.parameters.uuid("memberId") else {
                throw HBHTTPError(.badRequest, message: "invalid memberId")
            }
            try NovaDB.shared.lbMemberStore.delete(memberId)  // cascades stats
            return ["ok": true]
        }

        // POST /admin/load-balancers/:id/test
        // Invokes `lb:<slug>` with a sample payload and returns a trace:
        // which member was picked, latency, outcome. Does NOT record stats
        // (test is observational, not real traffic).
        group.post("admin/load-balancers/:id/test") { request, _ in
            guard let id = request.parameters.uuid("id"),
                  let lb = try NovaDB.shared.loadBalancerStore.get(id) else {
                throw HBHTTPError(.notFound, message: "LB not found")
            }
            let input = try request.decode(as: TestPayloadInput.self)
            let members = try NovaDB.shared.lbMemberStore.listByLB(id)
            let statsMap = Dictionary(
                uniqueKeysWithValues: members.compactMap {
                    (try? NovaDB.shared.lbMemberStatsStore.get($0.id)).map { ($0.memberId, $0) }
                }
            )

            let routerInput = LBRouterInput(
                lb: lb, members: members, stats: statsMap,
                isLocalLoaded: { modelId in MLXEngine.shared.isModelLoaded(modelId) },
                isProviderFree: { provId in
                    (try? NovaDB.shared.tokenhubStore.get(provId).map { $0.isFree }) ?? false
                }
            )
            let candidates = LBRouter.plan(routerInput)
            let trace = LBTestTrace(
                slug: lb.slug,
                candidates: candidates.map { c in
                    LBTestTrace.Candidate(
                        id: c.id, kind: c.kind, ref: c.ref,
                        loaded: c.kind == .local
                            ? MLXEngine.shared.isModelLoaded(c.ref) : nil
                    )
                },
                firstChoice: candidates.first.map { c in
                    LBTestTrace.Candidate(
                        id: c.id, kind: c.kind, ref: c.ref,
                        loaded: c.kind == .local
                            ? MLXEngine.shared.isModelLoaded(c.ref) : nil
                    )
                },
                noHealthyMembers: candidates.isEmpty
            )
            return trace
        }
    }

    /// Slug validation: ^[a-z0-9-]+$
    public static func isValidSlug(_ s: String) -> Bool {
        guard !s.isEmpty, s.count <= 64 else { return false }
        return s.allSatisfy { c in
            (c >= "a" && c <= "z") || (c >= "0" && c <= "9") || c == "-"
        }
    }
}

// MARK: - DTOs

struct CreateLBInput: Codable {
    let name: String
    let slug: String
    let strategy: LBStrategy?
    let maxRetries: Int?
}

struct PatchLBInput: Codable {
    var name: String?
    var slug: String?
    var strategy: LBStrategy?
    var maxRetries: Int?
    var isEnabled: Bool?
}

struct AddMemberInput: Codable {
    let kind: MemberKind
    let ref: String
    let weight: Int?
}

struct PatchMemberInput: Codable {
    var weight: Int?
    var isEnabled: Bool?
}

struct LBDTO: Codable {
    let id: UUID
    let name: String
    let slug: String
    let strategy: LBStrategy
    let maxRetries: Int
    let isEnabled: Bool
    let requestCount: Int
    init(_ lb: LoadBalancer) {
        self.id = lb.id; self.name = lb.name; self.slug = lb.slug
        self.strategy = lb.strategy; self.maxRetries = lb.maxRetries
        self.isEnabled = lb.isEnabled; self.requestCount = lb.requestCount
    }
}

struct MemberDTO: Codable {
    let id: UUID
    let lbId: UUID
    let kind: MemberKind
    let ref: String
    let weight: Int?
    let isEnabled: Bool
    init(_ m: LBMember) {
        self.id = m.id; self.lbId = m.lbId; self.kind = m.kind
        self.ref = m.ref; self.weight = m.weight; self.isEnabled = m.isEnabled
    }
}

struct StatsDTO: Codable {
    let memberId: UUID
    let requestCount: Int
    let successCount: Int
    let failureCount: Int
    let count5xx: Int
    let avgLatencyMs: Int64
    let successRate: Double
    let lastUsedAt: Date?
    let lastError: String?
    init(_ s: LBMemberStats) {
        self.memberId = s.memberId; self.requestCount = s.requestCount
        self.successCount = s.successCount; self.failureCount = s.failureCount
        self.count5xx = s.count5xx; self.avgLatencyMs = s.avgLatencyMs
        self.successRate = s.successRate; self.lastUsedAt = s.lastUsedAt
        self.lastError = s.lastError
    }
}

struct LBDetailDTO: Codable {
    let lb: LBDTO
    let members: [MemberDTO]
    let stats: [StatsDTO]
}

struct TestPayloadInput: Codable {
    // Empty for now — placeholder for future sample-message body.
    // Today /test just runs the router and returns the candidate trace.
    let note: String?
}

struct LBTestTrace: Codable {
    struct Candidate: Codable {
        let id: UUID
        let kind: MemberKind
        let ref: String
        let loaded: Bool?   // only set for locals
    }
    let slug: String
    let candidates: [Candidate]
    let firstChoice: Candidate?
    let noHealthyMembers: Bool
}
```

- [ ] **Step 2: Register routes in the main APIServer.setup**

Find where other `+Admin` extension files are called (e.g., `registerAPIKeyAdminRoutes`). Add:

```swift
self.registerLoadBalancerAdminRoutes(router)
```

- [ ] **Step 3: Build**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!`. Fix any naming mismatches with existing HBRouter API.

- [ ] **Step 4: Manual smoke test**

Restart app: `killall NovaMLX; sleep 2; open dist/NovaMLX.app`

Then via curl:

```bash
# Create LB
curl -X POST http://localhost:6590/admin/load-balancers \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","slug":"test","strategy":"tiered"}'

# List
curl http://localhost:6590/admin/load-balancers -H "Authorization: Bearer $ADMIN_KEY"

# 404 for unknown slug via API
curl -X POST http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"lb:nonexistent","messages":[{"role":"user","content":"hi"}]}'
# Expect: 404 with "Unknown load balancer: nonexistent"
```

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXAPI/APIServer+LoadBalancerAdmin.swift \
        Sources/NovaMLXAPI/APIServer.swift   # if setup() was modified
git commit -m "feat(api): /admin/load-balancers REST CRUD"
```

---

## Task 9: Sidebar Rename + New Menu Item

**Files:**
- Modify: `Sources/NovaMLXMenuBar/NovaAppView.swift`

- [ ] **Step 1: Update AppPage enum**

Find `public enum AppPage` (around line 8). Update:

```swift
public enum AppPage: String, CaseIterable, Identifiable, Sendable {
    case status = "Status"
    case localInference = "Local Inference"   // was: models = "Models"
    case tokenhub = "Tokenhub"
    case loadBalancers = "Load Balancers"     // NEW
    case chat = "Playground"
    case cluster = "Cluster"
    case apiKeys = "API Keys"
    case settings = "Settings"

    public var id: String { rawValue }
}
```

- [ ] **Step 2: Rename usages**

Run: `grep -rn "\.models\|AppPage.models" Sources/NovaMLXMenuBar/`

For each hit, replace `.models` → `.localInference`. The enum case name changed. Common spots:
- `selectedPage = .models` → `.localInference`
- `case .models:` → `case .localInference:`
- `AppPage.models` → `AppPage.localInference`

- [ ] **Step 3: Add the Load Balancers view switch case**

In the body of `NovaAppView`, the `switch selectedPage` block needs a new case:

```swift
case .loadBalancers:
    LoadBalancersPageView()
```

- [ ] **Step 4: Build**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!` (after creating `LoadBalancersPageView.swift` in Task 10). For now, comment out the `case .loadBalancers:` line to keep compile working, then add it back in Task 10.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXMenuBar/NovaAppView.swift
git commit -m "feat(ui): rename Models→Local Inference; add Load Balancers menu item"
```

---

## Task 10: Load Balancers List + Edit Pages

**Files:**
- Create: `Sources/NovaMLXMenuBar/LoadBalancersPageView.swift`

- [ ] **Step 1: Write the list view**

```swift
// Sources/NovaMLXMenuBar/LoadBalancersPageView.swift
import SwiftUI
import NovaMLXCore
import NovaMLXDB

struct LoadBalancersPageView: View {
    @State private var lbs: [LoadBalancer] = []
    @State private var editing: LoadBalancer?
    @State private var creating = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                if lbs.isEmpty {
                    emptyState
                } else {
                    LazyVStack(spacing: 8) {
                        ForEach(lbs) { lb in
                            LBRow(lb: lb) { editing = lb }
                        }
                    }
                }
            }
            .padding(24)
        }
        .navigationTitle("Load Balancers")
        .sheet(item: $editing) { lb in
            LBEditView(lbId: lb.id)
        }
        .sheet(isPresented: $creating) {
            LBEditView(lbId: nil)
        }
        .task { await reload() }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Load Balancers").font(.title2.bold())
                Text("Route requests across pools via `lb:<slug>`")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Button("+ New LB") { creating = true }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "scalemass")
                .font(.system(size: 32))
                .foregroundColor(.secondary)
            Text("No load balancers yet").font(.headline)
            Text("Create one to route requests across local and remote models.")
                .font(.caption).foregroundColor(.secondary)
        }
        .padding(.top, 60)
    }

    private func reload() async {
        do {
            lbs = try NovaDB.shared.loadBalancerStore.list()
        } catch {
            NovaMLXLog.error("[LB] list failed: \(error)")
        }
    }
}

struct LBRow: View {
    let lb: LoadBalancer
    let onEdit: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(lb.name).font(.headline)
                    Text("lb:\(lb.slug)")
                        .font(.caption.monospaced())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .foregroundColor(.accentColor)
                        .clipShape(Capsule())
                }
                Text("\(lb.strategy.rawValue) · \(lb.requestCount) requests")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Circle()
                .fill(lb.isEnabled ? Color.green : Color.gray.opacity(0.4))
                .frame(width: 10, height: 10)
            Button("Edit", action: onEdit)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .opacity(lb.isEnabled ? 1.0 : 0.5)
    }
}
```

- [ ] **Step 2: Write the edit view**

Append to the same file:

```swift
struct LBEditView: View {
    let lbId: UUID?

    @Environment(\.dismiss) private var dismiss
    @State private var lb: LoadBalancer?
    @State private var members: [LBMember] = []
    @State private var stats: [UUID: LBMemberStats] = [:]
    @State private var showAddMember = false
    @State private var errorMsg: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            if let lb {
                formFields(lb: lb)
                membersSection(lb: lb)
                if let errorMsg { Text(errorMsg).foregroundColor(.red).font(.caption) }
            } else {
                ProgressView()
            }
            Spacer()
        }
        .padding(24)
        .frame(minWidth: 600, minHeight: 500)
        .sheet(isPresented: $showAddMember) {
            LBMemberPickerSheet(lbId: lbId!) { added in
                Task { await reload(); showAddMember = false }
            }
        }
        .task {
            if let lbId {
                await reload()
            } else {
                // Create new LB with default values
                let new = LoadBalancer(name: "New LB", slug: "new-lb")
                try? NovaDB.shared.loadBalancerStore.upsert(new)
                self.lb = new
            }
        }
    }

    private var header: some View {
        HStack {
            Text(lbId == nil ? "New Load Balancer" : "Edit Load Balancer")
                .font(.title2.bold())
            Spacer()
            Button("Done") { dismiss() }
        }
    }

    @ViewBuilder
    private func formFields(lb: LoadBalancer) -> some View {
        let strategyBinding = Binding<LBStrategy>(
            get: { self.lb?.strategy ?? .tiered },
            set: { newStrategy in
                self.lb?.strategy = newStrategy
                save()
            }
        )
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 10) {
            GridRow {
                Text("Name").font(.caption)
                TextField("Name", text: Binding(
                    get: { self.lb?.name ?? "" },
                    set: { self.lb?.name = $0; save() }
                ))
            }
            GridRow {
                Text("Slug").font(.caption)
                TextField("Slug", text: Binding(
                    get: { self.lb?.slug ?? "" },
                    set: { newSlug in
                        guard APIServer.isValidSlug(newSlug) else {
                            self.errorMsg = "slug must match ^[a-z0-9-]+$"
                            return
                        }
                        self.errorMsg = nil
                        self.lb?.slug = newSlug
                        save()
                    }
                )).autocapitalization(.none)
            }
            GridRow {
                Text("Strategy").font(.caption)
                Picker("Strategy", selection: strategyBinding) {
                    ForEach(LBStrategy.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
            }
            GridRow {
                Text("Max retries").font(.caption)
                Stepper(value: Binding(
                    get: { self.lb?.maxRetries ?? 3 },
                    set: { self.lb?.maxRetries = $0; save() }
                ), in: 1...10) {
                    Text("\(self.lb?.maxRetries ?? 3)")
                }
            }
            GridRow {
                Text("Enabled").font(.caption)
                Toggle("", isOn: Binding(
                    get: { self.lb?.isEnabled ?? true },
                    set: { self.lb?.isEnabled = $0; save() }
                ))
            }
        }
    }

    @ViewBuilder
    private func membersSection(lb: LoadBalancer) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Members (\(members.count))").font(.headline)
                Spacer()
                Button("+ Add member") { showAddMember = true }
            }
            ForEach(members) { m in
                LBMemberRow(
                    member: m,
                    stats: stats[m.id],
                    lb: lb,
                    onChange: { Task { await reload() } }
                )
            }
        }
    }

    private func save() {
        guard var updated = lb else { return }
        updated.updatedAt = Date()
        do {
            try NovaDB.shared.loadBalancerStore.upsert(updated)
            self.lb = updated
        } catch {
            self.errorMsg = String(describing: error)
        }
    }

    private func reload() async {
        guard let lbId else { return }
        do {
            lb = try NovaDB.shared.loadBalancerStore.get(lbId)
            members = try NovaDB.shared.lbMemberStore.listByLB(lbId)
            var statsMap: [UUID: LBMemberStats] = [:]
            for m in members {
                if let s = try NovaDB.shared.lbMemberStatsStore.get(m.id) {
                    statsMap[m.id] = s
                }
            }
            self.stats = statsMap
        } catch {
            NovaMLXLog.error("[LBEdit] reload failed: \(error)")
        }
    }
}

struct LBMemberRow: View {
    let member: LBMember
    let stats: LBMemberStats?
    let lb: LoadBalancer
    let onChange: () -> Void

    var body: some View {
        HStack {
            // Kind badge
            Text(member.kind == .local ? "LOCAL" : "REMOTE")
                .font(.caption2.bold())
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(member.kind == .local
                    ? Color.green.opacity(0.15) : Color.yellow.opacity(0.15))
                .foregroundColor(member.kind == .local ? .green : .orange)
                .clipShape(Capsule())

            // Reference
            Text(member.ref).font(.caption.monospaced())
                .foregroundColor(.primary)

            // Status
            if member.kind == .local {
                let loaded = MLXEngine.shared.isModelLoaded(member.ref)
                Text(loaded ? "✓ loaded" : "⚠ not loaded")
                    .font(.caption2)
                    .foregroundColor(loaded ? .green : .orange)
            } else if let stats {
                Text("\(stats.avgLatencyMs)ms avg")
                    .font(.caption2).foregroundColor(.secondary)
            }

            Spacer()

            // Weight (only if weighted strategy)
            if lb.strategy == .weighted {
                Text("w:").font(.caption2).foregroundColor(.secondary)
                TextField("", value: Binding(
                    get: { member.weight ?? 1 },
                    set: { newWeight in
                        var updated = member
                        updated.weight = max(1, newWeight)
                        try? NovaDB.shared.lbMemberStore.upsert(updated)
                        onChange()
                    }
                ), format: .number)
                .frame(width: 40)
                .textFieldStyle(.roundedBorder)
            }

            // Enable toggle
            Toggle("", isOn: Binding(
                get: { member.isEnabled },
                set: { v in
                    var updated = member
                    updated.isEnabled = v
                    try? NovaDB.shared.lbMemberStore.upsert(updated)
                    onChange()
                }
            )).labelsHidden()

            // Remove
            Button(role: .destructive) {
                try? NovaDB.shared.lbMemberStore.delete(member.id)
                onChange()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red.opacity(0.6))
            }.buttonStyle(.plain)
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
```

- [ ] **Step 3: Re-enable the `case .loadBalancers:` in NovaAppView.swift** (commented out in Task 9 Step 3)

```swift
case .loadBalancers:
    LoadBalancersPageView()
```

- [ ] **Step 4: Build**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!`. Resolve any Swift type mismatches (the code above uses standard SwiftUI 5.x APIs).

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXMenuBar/LoadBalancersPageView.swift \
        Sources/NovaMLXMenuBar/NovaAppView.swift
git commit -m "feat(ui): LB list + edit pages with member management"
```

---

## Task 11: Add-Member Picker Sheet

**Files:**
- Create: `Sources/NovaMLXMenuBar/LBMemberPickerSheet.swift`

- [ ] **Step 1: Write the picker**

```swift
// Sources/NovaMLXMenuBar/LBMemberPickerSheet.swift
import SwiftUI
import NovaMLXCore
import NovaMLXDB

/// Multi-select sheet for adding members to an LB.
/// Shows two tabs: Local (downloaded models) and Remote (enabled providers).
struct LBMemberPickerSheet: View {
    let lbId: UUID
    let onAdded: ([LBMember]) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab: Tab = .local
    @State private var localModels: [String] = []          // downloaded model IDs
    @State private var remoteProviders: [(id: String, name: String)] = []
    @State private var existingMemberRefs: Set<String> = []
    @State private var selected: Set<String> = []

    enum Tab: String, CaseIterable, Identifiable {
        case local = "Local", remote = "Remote"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Add members").font(.title3.bold())

            Picker("", selection: $selectedTab) {
                ForEach(Tab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            switch selectedTab {
            case .local: localList
            case .remote: remoteList
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add \(selected.count)") {
                    addSelected()
                }
                .buttonStyle(.borderedProminent)
                .disabled(selected.isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 500, minHeight: 400)
        .task { await reload() }
    }

    private var localList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(localModels, id: \.self) { modelId in
                    HStack {
                        Image(systemName: selected.contains(modelId) ? "checkmark.square" : "square")
                            .foregroundColor(.accentColor)
                        Text(modelId).font(.caption.monospaced())
                        Spacer()
                        if MLXEngine.shared.isModelLoaded(modelId) {
                            Text("loaded").font(.caption2).foregroundColor(.green)
                        } else {
                            Text("not loaded").font(.caption2).foregroundColor(.orange)
                        }
                    }
                    .padding(.vertical, 4)
                    .padding(.horizontal, 8)
                    .background(existingMemberRefs.contains(modelId)
                        ? Color.gray.opacity(0.2) : Color.clear)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        guard !existingMemberRefs.contains(modelId) else { return }
                        if selected.contains(modelId) {
                            selected.remove(modelId)
                        } else {
                            selected.insert(modelId)
                        }
                    }
                }
            }
        }
    }

    private var remoteList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(remoteProviders, id: \.id) { p in
                    HStack {
                        Image(systemName: selected.contains(p.id) ? "checkmark.square" : "square")
                            .foregroundColor(.accentColor)
                        Text(p.name).font(.caption)
                        Spacer()
                        Text(p.id).font(.caption2.monospaced()).foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                    .padding(.horizontal, 8)
                    .background(existingMemberRefs.contains(p.id)
                        ? Color.gray.opacity(0.2) : Color.clear)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        guard !existingMemberRefs.contains(p.id) else { return }
                        if selected.contains(p.id) {
                            selected.remove(p.id)
                        } else {
                            selected.insert(p.id)
                        }
                    }
                }
            }
        }
    }

    private func addSelected() {
        let kind: MemberKind = (selectedTab == .local) ? .local : .remote
        var added: [LBMember] = []
        for ref in selected {
            let m = LBMember(lbId: lbId, kind: kind, ref: ref)
            do {
                try NovaDB.shared.lbMemberStore.upsert(m)
                added.append(m)
            } catch {
                NovaMLXLog.error("[LBMemberPicker] add failed: \(error)")
            }
        }
        onAdded(added)
        dismiss()
    }

    private func reload() async {
        // Local: downloaded model IDs from ModelManager
        let downloaded = ModelManager.shared.downloadedModels().map(\.id)
        localModels = downloaded

        // Remote: enabled providers
        let allProviders = (try? NovaDB.shared.tokenhubStore.list().compactMap { $0 }) ?? []
        remoteProviders = allProviders
            .filter { $0.isEnabled }
            .map { (id: $0.id, name: $0.name) }

        // Existing members (to disable in the picker)
        let existing = (try? NovaDB.shared.lbMemberStore.listByLB(lbId)) ?? []
        existingMemberRefs = Set(existing.map(\.ref))
    }
}
```

- [ ] **Step 2: Build**

Run: `./build.sh 2>&1 | tail -5`
Expected: `Build complete!`. Resolve type mismatches (e.g., `ModelManager.shared` might be an instance property; the `downloadedModels()` return type might need `.map(\.id)` adjusted based on whether ModelRecord uses `id` or `repo`).

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/LBMemberPickerSheet.swift
git commit -m "feat(ui): LB member picker sheet (local + remote tabs)"
```

---

## Task 12: Rename "Models" Page Title

**Files:**
- Modify: `Sources/NovaMLXMenuBar/ModelsPageView.swift` — change page title string only (req #1 polish)

- [ ] **Step 1: Find the page title**

Run: `grep -n "navigationTitle\|\"Models\"" Sources/NovaMLXMenuBar/ModelsPageView.swift | head -5`

- [ ] **Step 2: Update the displayed title**

Change any `Text("Models")` or `.navigationTitle("Models")` to `Text("Local Inference")` / `.navigationTitle("Local Inference")`.

Leave the file name `ModelsPageView.swift` unchanged — renaming files is out of scope and risks merge conflicts.

- [ ] **Step 3: Build + visual check**

Run: `./build.sh 2>&1 | tail -5 && killall NovaMLX; sleep 2; open dist/NovaMLX.app && echo Restarted`

Open the Local Inference page; verify the title at the top reads "Local Inference".

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/ModelsPageView.swift
git commit -m "feat(ui): Local Inference page title"
```

---

## Task 13: E2E Smoke Test

**Files:**
- Manual test — no code changes

- [ ] **Step 1: Restart app**

```bash
killall NovaMLX; sleep 2; open dist/NovaMLX.app
```

- [ ] **Step 2: Open Load Balancers page**

- Click "Load Balancers" in the sidebar (should sit between TokenHub and Playground).
- Verify empty state shows "No load balancers yet".
- Click "+ New LB".

- [ ] **Step 3: Create an LB**

- Name: `Coding Pool`
- Slug: `coding-pool`
- Strategy: `round_robin`
- Save.

- [ ] **Step 4: Add members**

- Click "+ Add member".
- In the Local tab, check a downloaded model that is currently loaded.
- Switch to Remote tab, check one enabled remote provider.
- Click "Add 2".

- [ ] **Step 5: Test the `lb:coding-pool` route**

```bash
curl -X POST http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lb:coding-pool",
    "messages": [{"role":"user","content":"hello"}],
    "stream": false
  }'
```

Verify:
- 200 response with content.
- The LB's `request_count` increments by 1 in the UI.
- The chosen member's stats row is created.

- [ ] **Step 6: Test error paths**

```bash
# Unknown slug → 404
curl -X POST http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"lb:nope","messages":[{"role":"user","content":"hi"}]}'

# Disabled LB → 503
# (disable the LB in the UI, then re-run the curl from step 5)
```

- [ ] **Step 7: Test streaming failover**

- Create a second LB with one fake-broken remote (point at a non-existent endpoint) and one good remote.
- Stream a request via `lb:failover-test` with `stream: true`.
- Verify: client receives a successful streamed response (from the good remote). Broken remote's `failure_count` increments.

- [ ] **Step 8: Document any issues**

If any test fails, file as a follow-up TODO. Don't gate the merge on minor issues.

- [ ] **Step 9: Commit test results (if any config was tweaked)**

```bash
git status  # if anything changed
git add -A
git commit -m "test: E2E smoke for multi-LB system"
```

---

## Task 14: Final Cleanup + Release Notes

**Files:**
- Modify: `README.md` (or `CHANGELOG.md` if present) — note the breaking change

- [ ] **Step 1: Check for orphaned code**

Run:
```bash
grep -rn "includeInLoadBalance\|isManaged\|provisionLocalProviders\|lbProviders\|formIncludeInLB" Sources/ Tests/
```

Expected: no hits. Any hits are missed cleanup — remove them.

- [ ] **Step 2: Update release notes**

In `README.md` (or wherever release notes live), add under the next-version section:

```markdown
## Breaking Changes

- **Multi-Load-Balancer:** The implicit single LB has been replaced with an explicit
  multi-LB system. Existing LB configs (providers with `includeInLoadBalance=true`)
  are **not** migrated — you must create Load Balancers explicitly under the new
  "Load Balancers" menu.
- **TokenHub scope:** Local inference models no longer appear in TokenHub provider
  list. TokenHub is now remote providers only.
- **Sidebar:** "Models" renamed to "Local Inference".

## New Features

- Multiple Load Balancers, each with its own members + strategy.
- Routing via `lb:<slug>` model prefix (alongside existing `tknet:` and bare).
- 5 strategies: tiered / round_robin / weighted / lowest_latency / random.
- Per-member stats: request count, success rate, avg latency, last error.
- Auto-retry on member failure (up to `maxRetries`, default 3).
- Admin REST API at `/admin/load-balancers`.
```

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: release notes for multi-LB breaking change"
```

---

## Done

All tasks complete. Final state:

- 3 new SQLite tables, 3 new stores.
- 5 selection strategies with full test coverage.
- `lb:<slug>` model-prefix routing with retry-on-failure.
- Full admin REST API for LB + member CRUD.
- SwiftUI list + edit pages, member picker sheet.
- TokenHub cleanly scoped to remote providers.
- Sidebar shows "Local Inference" + "Load Balancers".
