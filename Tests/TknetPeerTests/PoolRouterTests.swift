import Foundation
import Testing
@testable import TknetPeer

@Suite("PoolRouter")
struct PoolRouterTests {
    private func member(_ id: String, weight: Int? = nil, enabled: Bool = true) -> MemberConfig {
        MemberConfig(
            id: id, name: id, interfaceId: "i-\(id)", upstreamModel: id, weight: weight,
            isEnabled: enabled)
    }

    private func pool(_ strategy: PoolStrategy, requestCount: Int = 0) -> PoolConfig {
        PoolConfig(id: "p", name: "p", strategy: strategy, memberIds: [], requestCount: requestCount)
    }

    private func fixture() -> [MemberConfig] {
        [
            member("local"),
            member("lan"),
            member("remote"),
        ]
    }

    @Test("empty member list returns an empty plan")
    func emptyMembers() {
        let input = PoolRouterInput(pool: pool(.roundRobin), members: [])
        #expect(PoolRouter.plan(input).isEmpty)
    }

    @Test("disabled members are skipped")
    func disabledSkipped() {
        var members = fixture()
        members[0].isEnabled = false
        let plan = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin), members: members))
        #expect(plan.map(\.id) == ["lan", "remote"])
    }

    @Test("round robin rotates by requestCount")
    func roundRobinRotates() {
        let members = fixture()
        let p0 = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin, requestCount: 0), members: members))
        let p1 = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin, requestCount: 1), members: members))
        let p2 = PoolRouter.plan(PoolRouterInput(pool: pool(.roundRobin, requestCount: 2), members: members))
        #expect(p0.map(\.id) == ["local", "lan", "remote"])
        #expect(p1.map(\.id) == ["lan", "remote", "local"])
        #expect(p2.map(\.id) == ["remote", "local", "lan"])
    }

    @Test("weighted expands nil to 1 and 0 to 1")
    func weightedNilAndZero() {
        let members = [
            member("a", weight: nil),
            member("b", weight: 3),
            member("c", weight: 0),
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
            "lan": MemberStats(memberId: "lan", successCount: 5, totalLatencyMs: 500),
            "remote": MemberStats(memberId: "remote", successCount: 10, totalLatencyMs: 1000),
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
        #expect(PoolStrategy.roundRobin.rawValue == "round_robin")
        #expect(PoolStrategy.weighted.rawValue == "weighted")
        #expect(PoolStrategy.lowestLatency.rawValue == "lowest_latency")
        #expect(PoolStrategy.random.rawValue == "random")
    }
}
