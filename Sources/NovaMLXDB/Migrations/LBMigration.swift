import GRDB

/// v4 migration: creates the three load-balancer tables and cleans up the
/// legacy `tokenhub_providers` columns/rows that the LB subsystem replaces.
///
/// The LB tables live in `configDB` (not `dataDB`) because:
/// - `load_balancers` references `tokenhub_providers` (members can be remote
///   providers), which already lives in configDB.
/// - LBs are user-defined configuration, like providers.
///
/// Idempotent: safe to run multiple times. SQLite's migrator only re-runs a
/// migration if its name is missing from the migration log, but this method
/// is also defensive against partial-state re-runs (e.g. manual DB inspection).
public enum LBMigration {
    /// Create `load_balancers`, `lb_members`, `lb_member_stats`; delete legacy
    /// local-virtual-provider rows; drop the three legacy columns
    /// (`is_managed`, `include_in_load_balance`, `is_local`) from
    /// `tokenhub_providers`.
    public static func v4LoadBalancers(in db: Database) throws {
        // 1. Create the three new tables.
        try db.create(table: "load_balancers", ifNotExists: true) { t in
            t.column("id", .text).primaryKey()
            t.column("name", .text).notNull()
            t.column("slug", .text).notNull().unique()
            t.column("strategy", .text).notNull().defaults(to: "tiered")
            t.column("max_retries", .integer).notNull().defaults(to: 3)
            t.column("is_enabled", .boolean).notNull().defaults(to: true)
            t.column("request_count", .integer).notNull().defaults(to: 0)
            t.column("created_at", .datetime).notNull()
            t.column("updated_at", .datetime).notNull()
        }

        try db.create(table: "lb_members", ifNotExists: true) { t in
            t.column("id", .text).primaryKey()
            t.column("lb_id", .text).notNull()
                .references("load_balancers", onDelete: .cascade)
            t.column("kind", .text).notNull()
            t.column("ref", .text).notNull()
            t.column("weight", .integer)
            t.column("is_enabled", .boolean).notNull().defaults(to: true)
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
            t.column("last_used_at", .datetime)
            t.column("last_error", .text)
            t.column("updated_at", .datetime).notNull()
        }

        // 2. Delete legacy local-virtual-provider rows. These were the old
        //    in-process "managed" providers; the LB subsystem replaces them.
        //    Guarded by the column's existence so a re-run after the drop is
        //    a no-op rather than a SQL error.
        let existingCols = try Set(
            Row.fetchAll(db, sql: "PRAGMA table_info(tokenhub_providers)")
               .map { $0["name"] as String }
        )
        if existingCols.contains("is_managed") {
            try db.execute(sql: "DELETE FROM tokenhub_providers WHERE is_managed = 1")
        }

        // 3. Drop the three legacy columns. GRDB 7+ supports
        //    `alter(table:) { t in t.drop(column:) }` directly, which under
        //    the hood rewrites the table for SQLite versions that lack native
        //    DROP COLUMN (SQLite < 3.35). Each drop is gated on the column
        //    still existing so a re-run is idempotent.
        let toDrop = ["is_managed", "include_in_load_balance", "is_local"]
        let needDrop = toDrop.filter { existingCols.contains($0) }
        if !needDrop.isEmpty {
            try db.alter(table: "tokenhub_providers") { t in
                for col in needDrop {
                    t.drop(column: col)
                }
            }
        }
    }
}
