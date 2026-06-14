import Foundation

/// Read-only inputs the router needs to make a decision.
public struct LBRouterInput: Sendable {
    public let lb: LoadBalancer
    public let members: [LBMember]
    public let stats: [UUID: LBMemberStats]      // keyed by member.id
    public let isLocalLoaded: @Sendable (String) -> Bool   // model_id -> loaded?
    public let isProviderFree: @Sendable (String) -> Bool  // provider_id -> isFree?

    public init(
        lb: LoadBalancer,
        members: [LBMember],
        stats: [UUID: LBMemberStats] = [:],
        isLocalLoaded: @escaping @Sendable (String) -> Bool,
        isProviderFree: @escaping @Sendable (String) -> Bool
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
        case .tiered:        return applyTiered(healthy, input)
        case .roundRobin:    return applyRoundRobin(healthy, input)
        case .weighted:      return applyWeighted(healthy, input)
        case .lowestLatency: return applyLowestLatency(healthy, input)
        case .random:        return applyRandom(healthy, input)
        }
    }

    // MARK: - Strategies

    /// Tiers: local=2, free-remote=1, paid-remote=0. Higher tier first.
    /// Within a tier, stable order by member.id (deterministic for tests).
    /// For round-robin within tier, members are pre-rotated by `lb.requestCount`.
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

    /// Equal rotation. Rotate the full list by requestCount so each
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
    /// seed (so tests are reproducible). `shuffled()` (stdlib) takes no
    /// comparator; use `sorted(by:)` on hash instead.
    private static func applyRandom(
        _ members: [LBMember], _ input: LBRouterInput
    ) -> LBCandidateList {
        members.sorted { a, b in
            a.id.uuidString.hashValue < b.id.uuidString.hashValue
        }
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
