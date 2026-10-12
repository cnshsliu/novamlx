import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("PoolRouter")
struct PoolRouterTests {
    private func member(_ id: String, endpoint: String, type: SourceType = .openaiCompatible,
                        free: Bool = false, weight: Int? = nil, enabled: Bool = true) -> MemberConfig {
        MemberConfig(
            id: id, name: id, type: type, endpoint: URL(string: endpoint)!,
            apiKeyRef: "src/\(id)", upstreamModel: id, weight: weight,
            isEnabled: enabled, isFree: free)
    }

    private func pool(_ strategy: PoolStrategy, requestCount: Int = 0) -> PoolConfig {
        PoolConfig(id: "p", name: "p", strategy: strategy, memberIds: [], requestCount: requestCount)
    }

    private func fixture() -> [MemberConfig] {
        [
            member("local", endpoint: "http://127.0.0.1:6590/v1", type: .localNovaMLX),
            member("free", endpoint: "http://10.0.0.2:8080/v1", free: true),
            member("paid", endpoint: "http://10.0.0.3:8080/v1", free: false),
        ]
    }

    @Test("empty member list returns an empty plan")
    func emptyMembers() {
        let input = PoolRouterInput(pool: pool(.tiered), members: [])
        #expect(PoolRouter.plan(input).isEmpty)
    }

    @Test("disabled members are skipped")
    func disabledSkipped() {
        var members = fixture()
        members[0].isEnabled = false
        let plan = PoolRouter.plan(PoolRouterInput(pool: pool(.tiered), members: members))
        #expect(plan.map(\.id) == ["free", "paid"])
    }

    @Test("tiered prefers loopback, then free, then paid")
    func tieredOrdering() {
        let plan = PoolRouter.plan(PoolRouterInput(pool: pool(.tiered), members: fixture()))
        #expect(plan.map(\.id) == ["local", "free", "paid"])
    }

    @Test("tiered rotates inside a tier by requestCount")
    func tieredRotatesInsideTier() {
        let members = [
            member("a", endpoint: "http://127.0.0.1:1/v1"),
            member("b", endpoint: "http://localhost:2/v1"),
        ]
        let first = PoolRouter.plan(PoolRouterInput(pool: pool(.tiered, requestCount: 0), members: members))
        let second = PoolRouter.plan(PoolRouterInput(pool: pool(.tiered, requestCount: 1), members: members))
        #expect(first.map(\.id) == ["a", "b"])
        #expect(second.map(\.id) == ["b", "a"])
    }

    @Test("round robin rotates by requestCount")
    func roundRobinRotates() {
        let members = fixture()
        let p0 = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin, requestCount: 0), members: members))
        let p1 = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin, requestCount: 1), members: members))
        let p2 = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin, requestCount: 2), members: members))
        #expect(p0.map(\.id) == ["local", "free", "paid"])
        #expect(p1.map(\.id) == ["free", "paid", "local"])
        #expect(p2.map(\.id) == ["paid", "local", "free"])
    }

    @Test("weighted expands nil to 1 and 0 to 1")
    func weightedNilAndZero() {
        let members = [
            member("a", endpoint: "http://10.0.0.2:1/v1", weight: nil),
            member("b", endpoint: "http://10.0.0.3:1/v1", weight: 3),
            member("c", endpoint: "http://10.0.0.4:1/v1", weight: 0),
        ]
        let plan = PoolRouter.plan(PoolRouterInput(pool: pool(.weighted), members: members))
        #expect(plan.count == 5)
        #expect(plan.filter { $0.id == "b" }.count == 3)
        #expect(plan.filter { $0.id == "a" }.count == 1)
        #expect(plan.filter { $0.id == "c" }.count == 1)
    }

    @Test("lowest latency prefers a cold start")
    func lowestLatencyColdStart() {
        let members = fixture()
        let stats: [String: MemberStats] = [
            "free": MemberStats(memberId: "free", successCount: 5, totalLatencyMs: 500),
            "paid": MemberStats(memberId: "paid", successCount: 10, totalLatencyMs: 1000),
        ]
        let plan = PoolRouter.plan(PoolRouterInput(
            pool: pool(.lowestLatency), members: members, stats: stats))
        #expect(plan.first?.id == "local")
    }

    @Test("random order is deterministic for a member id")
    func randomIsDeterministic() {
        let members = fixture()
        let input = PoolRouterInput(pool: pool(.random), members: members)
        let first = PoolRouter.plan(input).map(\.id)
        let second = PoolRouter.plan(input).map(\.id)
        let expected = members.sorted { $0.id.hashValue < $1.id.hashValue }.map(\.id)
        #expect(first == second)
        #expect(first == expected)
    }

    @Test("strategy raw values stay snake case")
    func snakeCaseRawValues() {
        #expect(PoolStrategy.tiered.rawValue == "tiered")
        #expect(PoolStrategy.roundRobin.rawValue == "round_robin")
        #expect(PoolStrategy.weighted.rawValue == "weighted")
        #expect(PoolStrategy.lowestLatency.rawValue == "lowest_latency")
        #expect(PoolStrategy.random.rawValue == "random")
    }
}
