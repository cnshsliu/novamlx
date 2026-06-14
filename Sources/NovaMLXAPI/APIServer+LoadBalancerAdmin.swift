import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXDB
import NovaMLXInference

// Handler logic for /admin/load-balancers/* routes. The actual route
// registrations live inline in APIServer.swift's `adminRouter` block
// (mirrors how /admin/tokenhub/* routes are wired). Keeping the handlers
// in an extension keeps the main file small and makes the LB surface
// easy to grep.
extension NovaMLXAPIServer {

    // MARK: - Slug validation

    /// `^[a-z0-9-]+$`, length 1-64. Used for both create and slug-update paths.
    public static func isValidLBSlug(_ s: String) -> Bool {
        guard !s.isEmpty, s.count <= 64 else { return false }
        return s.allSatisfy { c in
            (c >= "a" && c <= "z") || (c >= "0" && c <= "9") || c == "-"
        }
    }

    // MARK: - LB CRUD

    // Handlers are internal (not public) because the DTOs they take and
    // return are themselves internal. Only the route closures in
    // APIServer.swift (same module) ever call these, so internal is the
    // tightest access level that works.
    static func lbAdminList() throws -> [LBDTO] {
        try NovaDB.shared.loadBalancerStore.listLBs().map(LBDTO.init)
    }

    static func lbAdminCreate(input: CreateLBInput) throws -> LBDTO {
        guard isValidLBSlug(input.slug) else {
            throw NovaMLXError.apiError("slug must match ^[a-z0-9-]+$ and be 1-64 chars")
        }
        if try NovaDB.shared.loadBalancerStore.getLBBySlug(input.slug) != nil {
            throw NovaMLXError.apiError("slug already exists: \(input.slug)")
        }
        let lb = LoadBalancer(
            name: input.name,
            slug: input.slug,
            strategy: input.strategy ?? .tiered,
            maxRetries: input.maxRetries ?? 3
        )
        try NovaDB.shared.loadBalancerStore.upsertLB(lb)
        return LBDTO(lb)
    }

    static func lbAdminDetail(id: UUID) throws -> LBDetailDTO {
        guard let lb = try NovaDB.shared.loadBalancerStore.getLB(id) else {
            throw NovaMLXError.apiError("LB not found: \(id)")
        }
        let members = try NovaDB.shared.lbMemberStore.listMembers(lbId: id)
        var stats: [LBMemberStats] = []
        for m in members {
            if let s = try NovaDB.shared.lbMemberStatsStore.getStats(m.id) {
                stats.append(s)
            }
        }
        return LBDetailDTO(
            lb: LBDTO(lb),
            members: members.map(MemberDTO.init),
            stats: stats.map(StatsDTO.init)
        )
    }

    static func lbAdminUpdate(id: UUID, patch: PatchLBInput) throws -> LBDTO {
        guard var lb = try NovaDB.shared.loadBalancerStore.getLB(id) else {
            throw NovaMLXError.apiError("LB not found: \(id)")
        }
        if let v = patch.name { lb.name = v }
        if let v = patch.slug {
            guard isValidLBSlug(v) else {
                throw NovaMLXError.apiError("slug must match ^[a-z0-9-]+$ and be 1-64 chars")
            }
            if let existing = try NovaDB.shared.loadBalancerStore.getLBBySlug(v),
               existing.id != lb.id {
                throw NovaMLXError.apiError("slug already exists: \(v)")
            }
            lb.slug = v
        }
        if let v = patch.strategy { lb.strategy = v }
        if let v = patch.maxRetries { lb.maxRetries = v }
        if let v = patch.isEnabled { lb.isEnabled = v }
        lb.updatedAt = Date()
        try NovaDB.shared.loadBalancerStore.upsertLB(lb)
        return LBDTO(lb)
    }

    static func lbAdminDelete(id: UUID) throws {
        // The schema's ON DELETE CASCADE on lb_members.lb_id +
        // lb_member_stats.member_id handles cleanup of children. We just
        // delete the LB row and trust the FK cascade.
        try NovaDB.shared.loadBalancerStore.deleteLB(id)
    }

    // MARK: - Member CRUD

    static func lbAdminAddMember(lbId: UUID, input: AddMemberInput) throws -> MemberDTO {
        guard try NovaDB.shared.loadBalancerStore.getLB(lbId) != nil else {
            throw NovaMLXError.apiError("LB not found: \(lbId)")
        }
        if let w = input.weight, w <= 0 {
            throw NovaMLXError.apiError("weight must be > 0 (use is_enabled=false to exclude a member)")
        }
        // Validate ref exists for remote members. Locals are not strictly
        // checked here — the caller is expected to load the model before
        // adding it, and the router will mark an unloaded local as unhealthy.
        switch input.kind {
        case .local:
            break
        case .remote:
            let exists = (try? NovaDB.shared.tokenhubStore.getProvider(name: input.ref)) != nil
            if !exists {
                throw NovaMLXError.apiError("remote provider not found: \(input.ref)")
            }
        }
        let member = LBMember(
            lbId: lbId,
            kind: input.kind,
            ref: input.ref,
            weight: input.weight,
            isEnabled: true
        )
        try NovaDB.shared.lbMemberStore.upsertMember(member)
        return MemberDTO(member)
    }

    static func lbAdminUpdateMember(memberId: UUID, patch: PatchMemberInput) throws -> MemberDTO {
        guard var member = try NovaDB.shared.lbMemberStore.getMember(memberId) else {
            throw NovaMLXError.apiError("member not found: \(memberId)")
        }
        if let v = patch.weight {
            guard v > 0 else {
                throw NovaMLXError.apiError("weight must be > 0")
            }
            member.weight = v
        }
        if let v = patch.isEnabled { member.isEnabled = v }
        try NovaDB.shared.lbMemberStore.upsertMember(member)
        return MemberDTO(member)
    }

    static func lbAdminDeleteMember(memberId: UUID) throws {
        // Relies on FK cascade for lb_member_stats.member_id.
        try NovaDB.shared.lbMemberStore.deleteMember(memberId)
    }

    // MARK: - Test (dry-run routing plan)

    /// Returns the ordered candidate list the LBRouter would pick right now,
    /// without actually sending a request. Useful for the UI to show "what
    /// would happen" and for ops to verify member wiring.
    static func lbAdminTest(id: UUID, inference: InferenceService) throws -> LBTestTrace {
        guard let lb = try NovaDB.shared.loadBalancerStore.getLB(id) else {
            throw NovaMLXError.apiError("LB not found: \(id)")
        }
        let members = try NovaDB.shared.lbMemberStore.listMembers(lbId: id)
        var statsMap: [UUID: LBMemberStats] = [:]
        for m in members {
            if let s = try NovaDB.shared.lbMemberStatsStore.getStats(m.id) {
                statsMap[m.id] = s
            }
        }
        let input = LBRouterInput(
            lb: lb,
            members: members,
            stats: statsMap,
            isLocalLoaded: { modelId in inference.isModelLoaded(modelId) },
            isProviderFree: { provName in
                (try? NovaDB.shared.tokenhubStore.getProvider(name: provName)?.isFree) ?? false
            }
        )
        let candidates = LBRouter.plan(input)
        let toTrace: (LBMember) -> LBTestTrace.Candidate = { c in
            LBTestTrace.Candidate(
                id: c.id,
                kind: c.kind,
                ref: c.ref,
                loaded: c.kind == .local ? inference.isModelLoaded(c.ref) : nil
            )
        }
        return LBTestTrace(
            slug: lb.slug,
            candidates: candidates.map(toTrace),
            firstChoice: candidates.first.map(toTrace),
            noHealthyMembers: candidates.isEmpty
        )
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
        self.id = lb.id
        self.name = lb.name
        self.slug = lb.slug
        self.strategy = lb.strategy
        self.maxRetries = lb.maxRetries
        self.isEnabled = lb.isEnabled
        self.requestCount = lb.requestCount
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
        self.id = m.id
        self.lbId = m.lbId
        self.kind = m.kind
        self.ref = m.ref
        self.weight = m.weight
        self.isEnabled = m.isEnabled
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
        self.memberId = s.memberId
        self.requestCount = s.requestCount
        self.successCount = s.successCount
        self.failureCount = s.failureCount
        self.count5xx = s.count5xx
        self.avgLatencyMs = s.avgLatencyMs
        self.successRate = s.successRate
        self.lastUsedAt = s.lastUsedAt
        self.lastError = s.lastError
    }
}

struct LBDetailDTO: Codable {
    let lb: LBDTO
    let members: [MemberDTO]
    let stats: [StatsDTO]
}

struct LBTestTrace: Codable {
    struct Candidate: Codable {
        let id: UUID
        let kind: MemberKind
        let ref: String
        /// For local members: whether the model is currently loaded.
        /// For remote members: nil (loaded-ness is not a local concept).
        let loaded: Bool?
    }
    let slug: String
    let candidates: [Candidate]
    let firstChoice: Candidate?
    let noHealthyMembers: Bool
}
