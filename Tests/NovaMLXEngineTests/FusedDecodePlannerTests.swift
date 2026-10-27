import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXEngine

/// Scheduling contract for a decode step. The GPU scheduler uses
/// `DecodeFusePlanner.lanes` and then writes each sampled id back to the
/// sequence at that index. This driver does the same with a scripted model so
/// isolation, cancel, and finish reasons can be checked without Metal.
private struct ScriptedRequest: Equatable {
    var id: UUID
    var cacheOffset: Int?
    var emitted: [Int] = []
    var maxTokens: Int
    var stopToken: Int
    var finish: FinishReason?
    var cancelled = false

    var isLive: Bool { finish == nil && !cancelled }
}

private struct ScriptedForward: Equatable {
    var width: Int
    var offset: Int?
    var ids: [UUID]
}

private enum ScriptedBatch {
    static func step(
        requests: inout [ScriptedRequest],
        forwards: inout [ScriptedForward],
        nextToken: (UUID, Int) -> Int
    ) {
        let live = requests.enumerated().filter { $0.element.isLive }
        guard !live.isEmpty else { return }
        let lanes = DecodeFusePlanner.lanes(cacheOffsets: live.map { $0.element.cacheOffset })
        for lane in lanes {
            let members = lane.indices.map { live[$0].offset }
            let offset: Int? = {
                if case .cohort(let value) = lane.kind { return value }
                return requests[members[0]].cacheOffset
            }()
            forwards.append(ScriptedForward(
                width: members.count,
                offset: offset,
                ids: members.map { requests[$0].id }
            ))
            for index in members {
                let token = nextToken(requests[index].id, requests[index].emitted.count)
                if token == requests[index].stopToken {
                    requests[index].finish = .stop
                    continue
                }
                requests[index].emitted.append(token)
                if requests[index].emitted.count >= requests[index].maxTokens {
                    requests[index].finish = .length
                }
            }
        }
    }

    static func run(
        requests: [ScriptedRequest],
        steps: Int,
        nextToken: (UUID, Int) -> Int
    ) -> (requests: [ScriptedRequest], forwards: [ScriptedForward]) {
        var requests = requests
        var forwards: [ScriptedForward] = []
        for _ in 0..<steps {
            step(requests: &requests, forwards: &forwards, nextToken: nextToken)
        }
        return (requests, forwards)
    }
}

@Suite("Decode fuse planner")
struct FusedDecodePlannerTests {
    @Test("A single sequence stays serial")
    func singleSequenceStaysSerial() {
        let lanes = DecodeFusePlanner.lanes(cacheOffsets: [128])
        #expect(lanes == [DecodeFusePlanner.Lane(kind: .serial, indices: [0])])
    }

    @Test("Equal offsets share one cohort and keep index order")
    func equalOffsetsCohort() {
        let lanes = DecodeFusePlanner.lanes(cacheOffsets: [8, nil, 8, 3])
        #expect(lanes == [
            DecodeFusePlanner.Lane(kind: .cohort(offset: 8), indices: [0, 2]),
            DecodeFusePlanner.Lane(kind: .serial, indices: [1]),
            DecodeFusePlanner.Lane(kind: .serial, indices: [3]),
        ])
    }

    @Test("Unequal offsets and non-simple caches do not share a forward")
    func unequalAndIncompatibleStaySerial() {
        let lanes = DecodeFusePlanner.lanes(cacheOffsets: [10, 12, nil, nil])
        #expect(lanes.allSatisfy { $0.kind == .serial })
        #expect(lanes.map(\.indices) == [[0], [1], [2], [3]])
    }

    @Test("Empty and one-wide inputs have no cohort")
    func emptyHasNoLane() {
        #expect(DecodeFusePlanner.lanes(cacheOffsets: []).isEmpty)
        #expect(DecodeFusePlanner.lanes(cacheOffsets: [nil]).first?.kind == .serial)
    }
}

@Suite("Multi-request decode scheduling")
struct FusedDecodeSchedulingTests {
    private func script(_ table: [UUID: [Int]]) -> (UUID, Int) -> Int {
        { id, count in
            let row = table[id] ?? []
            return count < row.count ? row[count] : -1
        }
    }

    @Test("Batched token streams match solo runs and do not cross")
    func batchMatchesSoloAndIsolatesStreams() {
        let a = UUID()
        let b = UUID()
        let table = [a: [11, 12, 13, 14], b: [21, 22, 23, 24]]
        let next = script(table)
        let make: (UUID) -> ScriptedRequest = { id in
            ScriptedRequest(id: id, cacheOffset: 64, maxTokens: 4, stopToken: 99)
        }

        let soloA = ScriptedBatch.run(requests: [make(a)], steps: 4, nextToken: next)
        let soloB = ScriptedBatch.run(requests: [make(b)], steps: 4, nextToken: next)
        let both = ScriptedBatch.run(requests: [make(a), make(b)], steps: 4, nextToken: next)

        #expect(soloA.forwards.allSatisfy { $0.width == 1 })
        #expect(both.forwards.allSatisfy { $0.width == 2 && $0.ids == [a, b] })
        #expect(both.requests[0].emitted == soloA.requests[0].emitted)
        #expect(both.requests[1].emitted == soloB.requests[0].emitted)
        #expect(both.requests[0].emitted == [11, 12, 13, 14])
        #expect(both.requests[1].emitted == [21, 22, 23, 24])
        #expect(both.requests[0].finish == .length)
        #expect(both.requests[1].finish == .length)
    }

    @Test("Stop and length stay on the sequence that hit them")
    func finishReasonsDoNotCross() {
        let a = UUID()
        let b = UUID()
        let table = [a: [1, 7], b: [5, 6, 8]]
        let next = script(table)
        let ran = ScriptedBatch.run(
            requests: [
                ScriptedRequest(id: a, cacheOffset: 32, maxTokens: 8, stopToken: 7),
                ScriptedRequest(id: b, cacheOffset: 32, maxTokens: 2, stopToken: 99),
            ],
            steps: 4,
            nextToken: next
        )
        #expect(ran.requests[0].emitted == [1])
        #expect(ran.requests[0].finish == .stop)
        #expect(ran.requests[1].emitted == [5, 6])
        #expect(ran.requests[1].finish == .length)
        #expect(ran.forwards.map(\.width) == [2, 2])
    }

    @Test("Cancel freezes that stream and the survivor continues alone")
    func cancelIsolatesTheOtherStream() {
        let a = UUID()
        let b = UUID()
        let table = [a: [1, 2, 3, 4], b: [9, 8, 7, 6]]
        let next = script(table)
        var requests = [
            ScriptedRequest(id: a, cacheOffset: 16, maxTokens: 4, stopToken: 99),
            ScriptedRequest(id: b, cacheOffset: 16, maxTokens: 4, stopToken: 99),
        ]
        var forwards: [ScriptedForward] = []
        ScriptedBatch.step(requests: &requests, forwards: &forwards, nextToken: next)
        requests[0].cancelled = true
        ScriptedBatch.step(requests: &requests, forwards: &forwards, nextToken: next)
        ScriptedBatch.step(requests: &requests, forwards: &forwards, nextToken: next)

        #expect(requests[0].emitted == [1])
        #expect(requests[0].finish == nil)
        #expect(requests[1].emitted == [9, 8, 7])
        #expect(forwards.map(\.width) == [2, 1, 1])
        #expect(forwards[1].ids == [b])
        #expect(forwards[2].ids == [b])
    }

    @Test("A non-simple cache stays on its own forward beside a cohort")
    func incompatibleCacheDoesNotJoinCohort() {
        let a = UUID()
        let b = UUID()
        let c = UUID()
        let table = [a: [1, 2], b: [3, 4], c: [5, 6]]
        let ran = ScriptedBatch.run(
            requests: [
                ScriptedRequest(id: a, cacheOffset: 5, maxTokens: 2, stopToken: 99),
                ScriptedRequest(id: b, cacheOffset: nil, maxTokens: 2, stopToken: 99),
                ScriptedRequest(id: c, cacheOffset: 5, maxTokens: 2, stopToken: 99),
            ],
            steps: 1,
            nextToken: script(table)
        )
        #expect(ran.forwards.map(\.ids) == [[a, c], [b]])
        #expect(ran.requests[0].emitted == [1])
        #expect(ran.requests[1].emitted == [3])
        #expect(ran.requests[2].emitted == [5])
    }

    @Test("Different KV offsets never share a forward")
    func differentOffsetsStayOneWide() {
        let a = UUID()
        let b = UUID()
        let table = [a: [4, 5], b: [6, 7]]
        let ran = ScriptedBatch.run(
            requests: [
                ScriptedRequest(id: a, cacheOffset: 10, maxTokens: 2, stopToken: 99),
                ScriptedRequest(id: b, cacheOffset: 40, maxTokens: 2, stopToken: 99),
            ],
            steps: 2,
            nextToken: script(table)
        )
        #expect(ran.forwards.allSatisfy { $0.width == 1 })
        #expect(ran.requests[0].emitted == [4, 5])
        #expect(ran.requests[1].emitted == [6, 7])
    }

    @Test("ResumeOnce delivers a queued cancel to one waiter")
    func resumeOnceFiresOnce() {
        let gate = ResumeOnce()
        #expect(gate.tryFire())
        #expect(!gate.tryFire())
    }

    @Test("Fused metrics default the shared-forward counters to zero")
    func metricsDefaultSharedForwardToZero() {
        let metrics = FusedSchedulerMetrics()
        #expect(metrics.peakFusedForwardWidth == 0)
        #expect(metrics.totalFusedForwards == 0)
        let batch = BatchSchedulerMetrics(
            pendingCount: 0, activeSequences: 0, totalFusedSteps: 0,
            totalTokensViaFused: 0, peakBatchWidth: 0, totalBatches: 0)
        #expect(batch.peakFusedForwardWidth == 0)
        #expect(batch.totalFusedForwards == 0)
    }
}
