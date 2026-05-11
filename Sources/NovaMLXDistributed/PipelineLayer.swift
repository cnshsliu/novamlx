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
