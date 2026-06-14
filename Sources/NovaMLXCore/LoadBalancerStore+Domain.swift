import Foundation
import NovaMLXDB

// LoadBalancerStore / LBMemberStore / LBMemberStatsStore live in NovaMLXDB;
// LoadBalancer / LBMember / LBMemberStats live here in NovaMLXCore. Declaring
// these extensions in NovaMLXCore keeps the dependency direction one-way
// (NovaMLXCore -> NovaMLXDB) while still giving the DB stores the ability to
// hand back the domain types the rest of the app uses. Mirrors the
// TokenhubStore+Domain pattern.

extension LoadBalancerStore {
    // MARK: - Domain-facing accessors

    /// All load balancers, converted to the domain `LoadBalancer` type.
    public func listLBs() throws -> [LoadBalancer] {
        try list().map(Self.toDomain)
    }

    /// Fetch a single LB by id, returning the domain type.
    public func getLB(_ id: UUID) throws -> LoadBalancer? {
        guard let row = try get(id) else { return nil }
        return Self.toDomain(row)
    }

    /// Fetch a single LB by slug, returning the domain type.
    public func getLBBySlug(_ slug: String) throws -> LoadBalancer? {
        guard let row = try getBySlug(slug) else { return nil }
        return Self.toDomain(row)
    }

    /// Upsert a domain LB into the store.
    public func upsertLB(_ lb: LoadBalancer) throws {
        try upsert(Self.toRecord(lb))
    }

    /// Delete an LB by id.
    public func deleteLB(_ id: UUID) throws {
        try delete(id)
    }

    /// Atomically bump the per-LB request counter.
    public func incrementLBRequestCount(_ id: UUID) throws {
        try incrementRequestCount(id)
    }

    // MARK: - Mapping

    private static func toDomain(_ row: LoadBalancerRow) -> LoadBalancer {
        LoadBalancer(
            id: UUID(uuidString: row.id) ?? UUID(),
            name: row.name,
            slug: row.slug,
            strategy: LBStrategy(rawValue: row.strategy) ?? .tiered,
            maxRetries: row.maxRetries,
            isEnabled: row.isEnabled,
            requestCount: row.requestCount,
            createdAt: row.createdAt,
            updatedAt: row.updatedAt
        )
    }

    private static func toRecord(_ lb: LoadBalancer) -> LoadBalancerRow {
        LoadBalancerRow(
            id: lb.id.uuidString,
            name: lb.name,
            slug: lb.slug,
            strategy: lb.strategy.rawValue,
            maxRetries: lb.maxRetries,
            isEnabled: lb.isEnabled,
            requestCount: lb.requestCount,
            createdAt: lb.createdAt,
            updatedAt: lb.updatedAt
        )
    }
}

extension LBMemberStore {
    // MARK: - Domain-facing accessors

    /// All members of an LB, converted to the domain `LBMember` type.
    public func listMembers(lbId: UUID) throws -> [LBMember] {
        try listByLB(lbId).map(Self.toDomain)
    }

    /// Fetch a single member by id, returning the domain type.
    public func getMember(_ id: UUID) throws -> LBMember? {
        guard let row = try get(id) else { return nil }
        return Self.toDomain(row)
    }

    /// Upsert a domain member into the store.
    public func upsertMember(_ member: LBMember) throws {
        try upsert(Self.toRecord(member))
    }

    /// Delete a member by id.
    public func deleteMember(_ id: UUID) throws {
        try delete(id)
    }

    // MARK: - Mapping

    private static func toDomain(_ row: LBMemberRow) -> LBMember {
        LBMember(
            id: UUID(uuidString: row.id) ?? UUID(),
            lbId: UUID(uuidString: row.lbId) ?? UUID(),
            kind: MemberKind(rawValue: row.kind) ?? .remote,
            ref: row.ref,
            weight: row.weight,
            isEnabled: row.isEnabled
        )
    }

    private static func toRecord(_ member: LBMember) -> LBMemberRow {
        LBMemberRow(
            id: member.id.uuidString,
            lbId: member.lbId.uuidString,
            kind: member.kind.rawValue,
            ref: member.ref,
            weight: member.weight,
            isEnabled: member.isEnabled
        )
    }
}

extension LBMemberStatsStore {
    // MARK: - Domain-facing accessors

    /// Stats for a single member, or nil if no row exists yet.
    public func getStats(_ memberId: UUID) throws -> LBMemberStats? {
        guard let row = try get(memberId) else { return nil }
        return Self.toDomain(row)
    }

    /// Stats for all members of an LB. Takes a memberStore so membership can be
    /// resolved outside the read transaction (matches the underlying
    /// `listByLB(_:memberStore:)` signature).
    public func listStatsByLB(lbId: UUID, memberStore: LBMemberStore) throws -> [LBMemberStats] {
        try listByLB(lbId, memberStore: memberStore).map(Self.toDomain)
    }

    // MARK: - Mapping

    private static func toDomain(_ row: LBMemberStatsRow) -> LBMemberStats {
        LBMemberStats(
            memberId: UUID(uuidString: row.memberId) ?? UUID(),
            requestCount: row.requestCount,
            successCount: row.successCount,
            failureCount: row.failureCount,
            count5xx: row.count5xx,
            totalLatencyMs: row.totalLatencyMs,
            lastUsedAt: row.lastUsedAt,
            lastError: row.lastError,
            updatedAt: row.updatedAt
        )
    }
}
