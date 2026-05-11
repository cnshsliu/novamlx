import Foundation
@preconcurrency import MLX

// MARK: - PendingSend

/// A queued send operation waiting to be flushed.
public struct PendingSend: @unchecked Sendable {
    public let output: MLXArray
    public let destination: Int
    public let group: DistributedGroup

    public init(output: MLXArray, destination: Int, group: DistributedGroup) {
        self.output = output
        self.destination = destination
        self.group = group
    }
}

// MARK: - PrefillSendQueue

/// Thread-safe send queue for batching prefill communication.
public final class PrefillSendQueue: @unchecked Sendable {
    private var pending: [PendingSend] = []
    private let lock = NSLock()

    public init() {}

    /// Whether the queue has no pending sends.
    public var isEmpty: Bool {
        lock.withLock { pending.isEmpty }
    }

    /// Number of pending sends.
    public var count: Int {
        lock.withLock { pending.count }
    }

    /// Add a send to the queue.
    public func enqueue(_ send: PendingSend) {
        lock.withLock { pending.append(send) }
    }

    /// Remove and return all pending sends.
    public func drain() -> [PendingSend] {
        lock.withLock {
            let result = pending
            pending.removeAll()
            return result
        }
    }

    /// Discard all pending sends without transmitting.
    public func clear() {
        lock.withLock { pending.removeAll() }
    }
}

// MARK: - Shared send queue

/// Global prefill send queue shared across all PipelineLastLayer instances.
public let prefillSendQueue = PrefillSendQueue()

/// Whether pipeline layers should queue sends (true during prefill) or send immediately.
/// - Note: Access is protected by `pipelineQueueSendsLock` for thread safety.
private let pipelineQueueSendsLock = NSLock()
public var pipelineQueueSends: Bool {
    get { pipelineQueueSendsLock.withLock { _pipelineQueueSends } }
    set { pipelineQueueSendsLock.withLock { _pipelineQueueSends = newValue } }
}
nonisolated(unsafe) private var _pipelineQueueSends = false

/// Flush all pending prefill sends via `asyncEval`.
public func flushPrefillSends() {
    let sends = prefillSendQueue.drain()
    for send in sends {
        let sent = MLXDistributedWrapper.send(
            send.output,
            to: send.destination,
            group: send.group
        )
        MLX.asyncEval(sent)
    }
}

/// Discard all pending sends without transmitting (error/cancellation path).
public func clearPrefillSends() {
    prefillSendQueue.clear()
}

// MARK: - PipelineFirstLayer

/// Wraps the first layer of a shard to receive activations from the previous rank.
///
/// - Rank 0: passes input through (generates embeddings from token IDs).
/// - Other ranks: `recv` activation from `rank - 1`.
public final class PipelineFirstLayer: @unchecked Sendable {
    public let rank: Int
    public let group: DistributedGroup

    public init(rank: Int, group: DistributedGroup) {
        self.rank = rank
        self.group = group
    }

    /// Receive activation from previous rank (if not rank 0) and return it.
    public func forward(input: MLXArray) async throws -> MLXArray {
        if rank == 0 {
            return input
        }
        MLX.eval(input)
        let received = MLXDistributedWrapper.recvLike(
            input,
            from: rank - 1,
            group: group
        )
        MLX.eval(received)
        return received
    }
}

// MARK: - PipelineLastLayer

/// Wraps the last layer of a shard to send activations to the next rank.
///
/// - Last rank: passes through (returns activation for sampling).
/// - Other ranks during prefill (`queueSends = true`): queue send.
/// - Other ranks during decode (`queueSends = false`): immediate send.
public final class PipelineLastLayer: @unchecked Sendable {
    public let rank: Int
    public let worldSize: Int
    public let group: DistributedGroup

    public init(rank: Int, worldSize: Int, group: DistributedGroup) {
        self.rank = rank
        self.worldSize = worldSize
        self.group = group
    }

    /// Whether this rank is the last in the pipeline.
    public var isLastRank: Bool {
        rank == worldSize - 1
    }

    /// Forward output through the send mechanism.
    public func forward(input: MLXArray) async throws -> MLXArray {
        if isLastRank {
            return input
        }

        MLX.eval(input)

        if pipelineQueueSends {
            prefillSendQueue.enqueue(PendingSend(
                output: input,
                destination: rank + 1,
                group: group
            ))
        } else {
            _ = MLXDistributedWrapper.send(
                input,
                to: rank + 1,
                group: group
            )
        }

        return input
    }
}
