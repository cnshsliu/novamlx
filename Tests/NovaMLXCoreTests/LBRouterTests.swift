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
        stats: [UUID: LBMemberStats] = [:],
        loaded: [String: Bool] = [:],
        free: [String: Bool] = [:]
    ) -> LBRouterInput {
        LBRouterInput(
            lb: lb, members: members, stats: stats,
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
        let (base, members) = makeFixture()
        var lb = base; lb.strategy = .weighted
        var m0 = members[0]; m0.weight = nil     // -> 1
        var m1 = members[1]; m1.weight = 3
        var m2 = members[2]; m2.weight = 0       // -> 1 defensively
        let input = makeInput(
            lb: lb, members: [m0, m1, m2],
            loaded: ["model-a": true],
            free: ["prov-free": true, "prov-paid": false]
        )
        let plan = LBRouter.plan(input)
        // Expanded: m0x1, m1x3, m2x1 = 5 entries. m1 should appear 3 times.
        #expect(plan.count == 5)
        #expect(plan.filter { $0.id == m1.id }.count == 3)
    }

    @Test("Lowest latency: cold-start (no stats) treated as latency=0 -> preferred")
    func lowestLatencyColdStart() {
        let (base, members) = makeFixture()
        var lb = base; lb.strategy = .lowestLatency
        let stats: [UUID: LBMemberStats] = [
            members[1].id: LBMemberStats(memberId: members[1].id, successCount: 5, totalLatencyMs: 500),  // avg 100
            members[2].id: LBMemberStats(memberId: members[2].id, successCount: 10, totalLatencyMs: 1000) // avg 100
        ]
        // members[0] has no stats -> cold -> avg 0 -> preferred
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
        let (base, members) = makeFixture()
        var lb = base; lb.strategy = .lowestLatency
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

    @Suite("LBStrategy wire format")
    struct LBStrategyWireFormatTests {
        @Test("snake_case raw values match admin API contract")
        func snakeCaseRawValues() {
            // Spec contract: tiered | round_robin | weighted | lowest_latency | random
            #expect(LBStrategy.tiered.rawValue == "tiered")
            #expect(LBStrategy.roundRobin.rawValue == "round_robin")
            #expect(LBStrategy.weighted.rawValue == "weighted")
            #expect(LBStrategy.lowestLatency.rawValue == "lowest_latency")
            #expect(LBStrategy.random.rawValue == "random")
        }

        @Test("JSON decode accepts snake_case strings from admin API")
        func decodeSnakeCase() throws {
            let json = """
            {"id":"00000000-0000-0000-0000-000000000001","name":"X","slug":"x","strategy":"round_robin","maxRetries":3,"isEnabled":true,"requestCount":0,"createdAt":0,"updatedAt":0}
            """.data(using: .utf8)!
            let lb = try JSONDecoder().decode(LoadBalancer.self, from: json)
            #expect(lb.strategy == .roundRobin)
        }

        @Test("JSON encode emits snake_case strings for admin API consumers")
        func encodeSnakeCase() throws {
            let lb = LoadBalancer(name: "X", slug: "x", strategy: .lowestLatency)
            let data = try JSONEncoder().encode(lb)
            let s = String(data: data, encoding: .utf8) ?? ""
            #expect(s.contains("\"strategy\":\"lowest_latency\""))
            #expect(!s.contains("lowestLatency"))
        }
    }
}
