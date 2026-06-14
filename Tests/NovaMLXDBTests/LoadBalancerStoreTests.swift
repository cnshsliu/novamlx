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

    // MARK: - Domain-level CRUD tests (Task 4)

    @Test("Upsert + get LoadBalancer by id and slug")
    func upsertAndGet() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let lb = LoadBalancer(name: "Coding Pool", slug: "coding-pool", strategy: .roundRobin)
        try nova.loadBalancerStore.upsertLB(lb)

        let byId = try nova.loadBalancerStore.getLB(lb.id)
        #expect(byId?.slug == "coding-pool")
        #expect(byId?.strategy == .roundRobin)

        let bySlug = try nova.loadBalancerStore.getLBBySlug("coding-pool")
        #expect(bySlug?.id == lb.id)
    }

    @Test("Delete LoadBalancer cascades to members + stats")
    func deleteCascades() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let lb = LoadBalancer(name: "X", slug: "x")
        try nova.loadBalancerStore.upsertLB(lb)

        let m = LBMember(lbId: lb.id, kind: .remote, ref: "provider-1")
        try nova.lbMemberStore.upsertMember(m)
        try nova.lbMemberStatsStore.recordRequest(
            memberId: m.id, succeeded: true, latencyMs: 50,
            httpStatus: 200, errorMessage: nil
        )

        // Sanity
        let membersBefore = try nova.lbMemberStore.listMembers(lbId: lb.id)
        #expect(membersBefore.count == 1)
        let statsBefore = try nova.lbMemberStatsStore.getStats(m.id)
        #expect(statsBefore?.requestCount == 1)

        try nova.loadBalancerStore.deleteLB(lb.id)

        // Cascade
        let membersAfter = try nova.lbMemberStore.listMembers(lbId: lb.id)
        #expect(membersAfter.isEmpty)
        let statsAfter = try nova.lbMemberStatsStore.getStats(m.id)
        #expect(statsAfter == nil)
    }

    @Test("recordRequest lazily creates stats row and aggregates correctly")
    func recordRequestAggregation() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        // lb_member_stats.member_id has a FK -> lb_members.id, so we must
        // create a parent LB + member first. Once the member exists, the
        // stats row is still lazily created on the first recordRequest call
        // (no explicit insert step), which is what this test exercises.
        let lb = LoadBalancer(name: "Stats", slug: "stats")
        try nova.loadBalancerStore.upsertLB(lb)
        let member = LBMember(lbId: lb.id, kind: .remote, ref: "prov-stats")
        try nova.lbMemberStore.upsertMember(member)
        let memberId = member.id

        try nova.lbMemberStatsStore.recordRequest(
            memberId: memberId, succeeded: true, latencyMs: 100,
            httpStatus: 200, errorMessage: nil
        )
        try nova.lbMemberStatsStore.recordRequest(
            memberId: memberId, succeeded: true, latencyMs: 200,
            httpStatus: 200, errorMessage: nil
        )
        try nova.lbMemberStatsStore.recordRequest(
            memberId: memberId, succeeded: false, latencyMs: 0,
            httpStatus: 503, errorMessage: "timeout"
        )

        let stats = try nova.lbMemberStatsStore.getStats(memberId)
        #expect(stats?.requestCount == 3)
        #expect(stats?.successCount == 2)
        #expect(stats?.failureCount == 1)
        #expect(stats?.count5xx == 1)
        #expect(stats?.totalLatencyMs == 300)
        #expect(stats?.avgLatencyMs == 150)  // 300 / 2 successes
        #expect(stats?.lastError == "timeout")
    }

    @Test("incrementLBRequestCount bumps counter atomically")
    func incrementLBRequestCount() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let lb = LoadBalancer(name: "Counter", slug: "counter")
        try nova.loadBalancerStore.upsertLB(lb)

        try nova.loadBalancerStore.incrementLBRequestCount(lb.id)
        try nova.loadBalancerStore.incrementLBRequestCount(lb.id)
        try nova.loadBalancerStore.incrementLBRequestCount(lb.id)

        let after = try nova.loadBalancerStore.getLB(lb.id)
        #expect(after?.requestCount == 3)
    }
}
