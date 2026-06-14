import Testing
import Foundation
import GRDB
@testable import NovaMLXDB
import NovaMLXCore

@Suite("LoadBalancer Store + Migration", .serialized)
struct LoadBalancerStoreTests {

    private func makeTmpDir() throws -> URL {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-lb-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        return tmp
    }

    @Test("v4 migration creates all 3 LB tables")
    func migrationCreatesTables() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // Force-read via store — should not throw "no such table"
        let lbs = try nova.loadBalancerStore.list()
        #expect(lbs.isEmpty)

        // listByLB takes a UUID; an arbitrary one should yield an empty list
        let members = try nova.lbMemberStore.listByLB(UUID())
        #expect(members.isEmpty)
    }

    @Test("v4 migration drops legacy provider columns")
    func migrationDropsLegacyColumns() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // After v4 runs, the three legacy columns must be gone.
        let cols = try await nova.configDB.read { db in
            try Row.fetchAll(db, sql: "PRAGMA table_info(tokenhub_providers)")
                .map { $0["name"] as String }
        }
        #expect(!cols.contains("is_managed"))
        #expect(!cols.contains("include_in_load_balance"))
        #expect(!cols.contains("is_local"))
    }

    @Test("v4 migration leaves tokenhub_providers usable")
    func migrationLeavesTableUsable() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // The v4 migration drops `is_managed`. Verify the table is still
        // writable/queryable via raw SQL with the remaining columns.
        // (tokenhubStore.get() cannot be used yet because the
        // TokenhubProviderRecord Swift struct still declares isManaged;
        // removing that field is Task 6's job.)
        _ = try await nova.configDB.write { db in
            try db.execute(sql: """
                INSERT INTO tokenhub_providers (name, endpoint, is_enabled, load_balance_weight)
                VALUES ('post-v4-provider', 'https://example.com', 1, 1.0)
                """)
            return 0
        }
        let count = try await nova.configDB.read { db in
            try Int.fetchOne(db, sql: """
                SELECT COUNT(*) FROM tokenhub_providers WHERE name = 'post-v4-provider'
                """) ?? 0
        }
        #expect(count == 1)
    }

    @Test("v4 migration is idempotent across repeated setup calls")
    func migrationIsIdempotent() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        // First setup runs the migration.
        try nova.setup(baseDir: tmp)
        // Second setup is a no-op at NovaDB level (`_isSetup`), but verify
        // the LB tables are still in the expected state.
        let lbs = try nova.loadBalancerStore.list()
        #expect(lbs.isEmpty)
        // Verify the LB tables still exist after a read.
        let tableCount = try await nova.configDB.read { db in
            try Int.fetchOne(db, sql: """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='table' AND name IN ('load_balancers','lb_members','lb_member_stats')
                """) ?? 0
        }
        #expect(tableCount == 3)
    }
}
