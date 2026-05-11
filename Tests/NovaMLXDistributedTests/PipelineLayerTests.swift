import Foundation
import Testing
import MLX
@testable import NovaMLXDistributed

@Suite("PipelineLayer")
struct PipelineLayerTests {

    @Test("PendingSend stores output, destination, and group")
    func pendingSendProperties() {
        let group = DistributedGroup.uninitialized
        let output = MLXArray([1.0, 2.0])
        let send = PendingSend(output: output, destination: 1, group: group)
        #expect(send.destination == 1)
    }

    @Test("PrefillSendQueue starts empty")
    func sendQueueStartsEmpty() {
        let queue = PrefillSendQueue()
        #expect(queue.isEmpty)
        #expect(queue.count == 0)
    }

    @Test("PrefillSendQueue enqueue increments count")
    func sendQueueEnqueue() {
        let queue = PrefillSendQueue()
        let group = DistributedGroup.uninitialized
        queue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        #expect(queue.count == 1)
        #expect(!queue.isEmpty)
    }

    @Test("PrefillSendQueue drain returns all pending sends and clears")
    func sendQueueDrain() {
        let queue = PrefillSendQueue()
        let group = DistributedGroup.uninitialized
        queue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        queue.enqueue(PendingSend(output: MLXArray([2.0]), destination: 1, group: group))
        let drained = queue.drain()
        #expect(drained.count == 2)
        #expect(queue.isEmpty)
        #expect(queue.count == 0)
    }

    @Test("PrefillSendQueue clear discards without returning")
    func sendQueueClear() {
        let queue = PrefillSendQueue()
        let group = DistributedGroup.uninitialized
        queue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        queue.clear()
        #expect(queue.isEmpty)
    }

    @Test("PrefillSendQueue drain on empty returns empty array")
    func sendQueueDrainEmpty() {
        let queue = PrefillSendQueue()
        let drained = queue.drain()
        #expect(drained.isEmpty)
    }

    @Test("Global flushPrefillSends does not crash on empty queue")
    func globalFlushEmpty() {
        clearPrefillSends()
        flushPrefillSends()
    }

    @Test("Global clearPrefillSends clears the shared queue")
    func globalClear() {
        clearPrefillSends()
        let group = DistributedGroup.uninitialized
        prefillSendQueue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        clearPrefillSends()
        #expect(prefillSendQueue.isEmpty)
    }

    @Test("PipelineFirstLayer rank 0 passes through without recv")
    func pipelineFirstLayerRank0Passthrough() async throws {
        let group = DistributedGroup.uninitialized
        let layer = PipelineFirstLayer(rank: 0, group: group)
        let input = MLXArray(0 ..< 3)
        let output = try await layer.forward(input: input)
        #expect(output.shape == input.shape)
    }

    @Test("PipelineFirstLayer non-zero rank receives (uninitialized group returns zeros)")
    func pipelineFirstLayerNonZeroRecv() async throws {
        let group = DistributedGroup.uninitialized
        let layer = PipelineFirstLayer(rank: 1, group: group)
        let input = MLXArray(0 ..< 3)
        let output = try await layer.forward(input: input)
        #expect(output.shape == input.shape)
    }

    @Test("PipelineLastLayer last rank passes through without send")
    func pipelineLastLayerLastRankPassthrough() async throws {
        let group = DistributedGroup.uninitialized
        let layer = PipelineLastLayer(rank: 1, worldSize: 2, group: group)
        let input = MLXArray(0 ..< 3)
        let output = try await layer.forward(input: input)
        #expect(output.shape == input.shape)
    }

    @Test("PipelineLastLayer non-last rank queues send when queueSends is true")
    func pipelineLastLayerQueuesSend() async throws {
        clearPrefillSends()
        let group = DistributedGroup.uninitialized
        let layer = PipelineLastLayer(rank: 0, worldSize: 2, group: group)
        pipelineQueueSends = true
        defer {
            pipelineQueueSends = false
            clearPrefillSends()
        }

        let beforeCount = prefillSendQueue.count
        let input = MLXArray(0 ..< 3)
        _ = try await layer.forward(input: input)
        #expect(prefillSendQueue.count == beforeCount + 1)
    }

    @Test("PipelineLastLayer non-last rank sends immediately when queueSends is false")
    func pipelineLastLayerImmediateSend() async throws {
        clearPrefillSends()
        let group = DistributedGroup.uninitialized
        let layer = PipelineLastLayer(rank: 0, worldSize: 2, group: group)
        pipelineQueueSends = false

        let input = MLXArray(0 ..< 3)
        _ = try await layer.forward(input: input)
        #expect(prefillSendQueue.isEmpty)
    }
}
