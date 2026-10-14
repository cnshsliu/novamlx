import Foundation

/// Read-only inputs for one routing decision.
/// Port of `LBRouterInput` (`Sources/NovaMLXCore/LBRouter.swift`): stats are
/// keyed by `member.id` instead of `UUID`.
public struct PoolRouterInput: Sendable {
    public var pool: PoolConfig
    public var members: [MemberConfig]
    public var stats: [String: MemberStats]

    public init(pool: PoolConfig, members: [MemberConfig], stats: [String: MemberStats] = [:]) {
        self.pool = pool
        self.members = members
        self.stats = stats
    }
}

/// Ordered candidates, preferred first. Empty means no enabled members.
public enum PoolRouter {
    public static func plan(_ input: PoolRouterInput) -> [MemberConfig] {
        let healthy = input.members.filter(\.isEnabled)
        guard !healthy.isEmpty else { return [] }

        switch input.pool.strategy {
        case .roundRobin: return applyRoundRobin(healthy, input)
        case .weighted: return applyWeighted(healthy, input)
        case .lowestLatency: return applyLowestLatency(healthy, input)
        case .random: return applyRandom(healthy, input)
        }
    }

    // MARK: - Strategies

    /// Equal rotation. Each request starts at the next member.
    private static func applyRoundRobin(_ members: [MemberConfig], _ input: PoolRouterInput) -> [MemberConfig] {
        members.rotated(by: input.pool.requestCount)
    }

    /// Probability proportional to weight. nil and 0 both expand to 1
    /// (0 is rejected at write time; this is the defensive match for LBRouter).
    private static func applyWeighted(_ members: [MemberConfig], _ input: PoolRouterInput) -> [MemberConfig] {
        var expanded: [MemberConfig] = []
        for member in members {
            let weight = max(1, member.weight ?? 1)
            expanded.append(contentsOf: Array(repeating: member, count: weight))
        }
        return expanded.rotated(by: input.pool.requestCount)
    }

    /// Lowest average latency first. Cold start (`successCount == 0`, or no
    /// row) sorts as latency 0. Ties break by success rate, then last use.
    private static func applyLowestLatency(_ members: [MemberConfig], _ input: PoolRouterInput) -> [MemberConfig] {
        members.sorted { a, b in
            let sa = input.stats[a.id]
            let sb = input.stats[b.id]
            let latencyA = sa?.avgLatencyMs ?? 0
            let latencyB = sb?.avgLatencyMs ?? 0
            if latencyA != latencyB { return latencyA < latencyB }
            let rateA = sa?.successRate ?? 0
            let rateB = sb?.successRate ?? 0
            if rateA != rateB { return rateA > rateB }
            let usedA = sa?.lastUsedAt ?? Date.distantPast
            let usedB = sb?.lastUsedAt ?? Date.distantPast
            return usedA < usedB
        }
    }

    /// Deterministic stand-in for `LBRouter.applyRandom`'s UUID hash sort.
    private static func applyRandom(_ members: [MemberConfig], _ input: PoolRouterInput) -> [MemberConfig] {
        members.sorted { $0.id.hashValue < $1.id.hashValue }
    }
}

extension Array {
    /// Rotate left by `n`. `n` may be larger than `count` or negative.
    fileprivate func rotated(by n: Int) -> [Element] {
        guard !isEmpty else { return [] }
        let shift = ((n % count) + count) % count
        return Array(self[shift..<count] + self[0..<shift])
    }
}
