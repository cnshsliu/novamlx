import Foundation
import Testing
@testable import TknetPeer

@Suite("Member stats")
struct MemberStatsTests {
    @Test("success updates the average latency")
    func successUpdatesAverage() async {
        let store = MemberStatsStore()
        await store.record(memberId: "m", succeeded: true, latencyMs: 100, httpStatus: 200, errorMessage: nil)
        await store.record(memberId: "m", succeeded: true, latencyMs: 300, httpStatus: 200, errorMessage: nil)
        let row = await store.snapshot()["m"]
        #expect(row?.requestCount == 2)
        #expect(row?.successCount == 2)
        #expect(row?.failureCount == 0)
        #expect(row?.avgLatencyMs == 200)
        #expect(row?.successRate == 1)
    }

    @Test("5xx bumps count5xx and 4xx counts as a non-failover success")
    func statusClasses() async {
        let store = MemberStatsStore()
        await store.record(memberId: "m", succeeded: false, latencyMs: 10, httpStatus: 503, errorMessage: "model not loaded")
        await store.record(memberId: "m", succeeded: true, latencyMs: 20, httpStatus: 400, errorMessage: nil)
        let row = await store.snapshot()["m"]
        #expect(row?.requestCount == 2)
        #expect(row?.count5xx == 1)
        #expect(row?.failureCount == 1)
        #expect(row?.successCount == 1)
        #expect(row?.lastError == "model not loaded")
        #expect(row?.avgLatencyMs == 20)
    }
}
