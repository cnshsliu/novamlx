import Foundation
import NovaMLXCore
import NovaMLXUtils
import AsyncAlgorithms

public enum RequestPriority: Int, Sendable, Comparable {
    case low = 0
    case normal = 1
    case high = 2

    public static func < (lhs: RequestPriority, rhs: RequestPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

struct QueuedRequest: @unchecked Sendable {
    let request: InferenceRequest
    let priority: RequestPriority
    let enqueuedAt: Date
    let continuation: CheckedContinuation<InferenceResult, Error>
}

struct QueuedStreamRequest: @unchecked Sendable {
    let request: InferenceRequest
    let priority: RequestPriority
    let enqueuedAt: Date
    let continuation: AsyncThrowingStream<Token, Error>.Continuation
}

public struct BatcherMetrics: Codable, Sendable {
    public let activeRequests: Int
    public let queueDepth: Int
    public let totalQueued: UInt64
    public let totalCompleted: UInt64
    public let totalPreempted: UInt64
    public let peakActiveCount: Int
    public let averageQueueWaitTime: TimeInterval
    public let maxBatchSize: Int
    public let uptime: TimeInterval

    public init(
        activeRequests: Int = 0,
        queueDepth: Int = 0,
        totalQueued: UInt64 = 0,
        totalCompleted: UInt64 = 0,
        totalPreempted: UInt64 = 0,
        peakActiveCount: Int = 0,
        averageQueueWaitTime: TimeInterval = 0,
        maxBatchSize: Int = 8,
        uptime: TimeInterval = 0
    ) {
        self.activeRequests = activeRequests
        self.queueDepth = queueDepth
        self.totalQueued = totalQueued
        self.totalCompleted = totalCompleted
        self.totalPreempted = totalPreempted
        self.peakActiveCount = peakActiveCount
        self.averageQueueWaitTime = averageQueueWaitTime
        self.maxBatchSize = maxBatchSize
        self.uptime = uptime
    }
}

public final class ContinuousBatcher: @unchecked Sendable {
    private let engine: MLXEngine
    public let maxBatchSize: Int
    private let lock = NovaMLXLock()

    private var generateQueue: [QueuedRequest]
    private var streamQueue: [QueuedStreamRequest]
    private var _activeCount: Int
    private var _totalQueued: UInt64
    private var _totalCompleted: UInt64
    private var _totalQueueWaitTime: TimeInterval
    private var _totalPreempted: UInt64
    private var _peakActiveCount: Int
    private var _startTime: Date
    private var activeTasks: [UUID: Task<Void, Never>]

    public init(engine: MLXEngine, maxBatchSize: Int = 8) {
        self.engine = engine
        self.maxBatchSize = maxBatchSize
        self.generateQueue = []
        self.streamQueue = []
        self._activeCount = 0
        self._totalQueued = 0
        self._totalCompleted = 0
        self._totalQueueWaitTime = 0
        self._totalPreempted = 0
        self._peakActiveCount = 0
        self._startTime = Date()
        self.activeTasks = [:]
    }

    public var activeRequests: Int { lock.withLock { _activeCount } }
    public var queueDepth: Int { lock.withLock { generateQueue.count + streamQueue.count } }
    public var totalQueued: UInt64 { lock.withLock { _totalQueued } }
    public var totalCompleted: UInt64 { lock.withLock { _totalCompleted } }
    public var totalPreempted: UInt64 { lock.withLock { _totalPreempted } }
    public var peakActiveCount: Int { lock.withLock { _peakActiveCount } }
    public var averageQueueWaitTime: TimeInterval {
        lock.withLock {
            _totalCompleted > 0 ? _totalQueueWaitTime / Double(_totalCompleted) : 0
        }
    }
    public var uptime: TimeInterval {
        Date().timeIntervalSince(lock.withLock { _startTime })
    }

    public var metrics: BatcherMetrics {
        lock.withLock {
            BatcherMetrics(
                activeRequests: _activeCount,
                queueDepth: generateQueue.count + streamQueue.count,
                totalQueued: _totalQueued,
                totalCompleted: _totalCompleted,
                totalPreempted: _totalPreempted,
                peakActiveCount: _peakActiveCount,
                averageQueueWaitTime: _totalCompleted > 0 ? _totalQueueWaitTime / Double(_totalCompleted) : 0,
                maxBatchSize: maxBatchSize,
                uptime: Date().timeIntervalSince(_startTime)
            )
        }
    }

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        let priority: RequestPriority = .normal

        guard activeRequests < maxBatchSize else {
            return try await enqueueAndWait(request: request, priority: priority)
        }

        lock.withLock {
            _activeCount += 1
            _totalQueued += 1
            if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
        }
        defer { lock.withLock { _activeCount -= 1; _totalCompleted += 1 } }

        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing generate (active: \(activeRequests))")
        return try await engine.generate(request)
    }

    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        return AsyncThrowingStream { continuation in
            let priority: RequestPriority = .normal

            let canStart = lock.withLock {
                if _activeCount < maxBatchSize {
                    _activeCount += 1
                    _totalQueued += 1
                    if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
                    return true
                }
                return false
            }

            if canStart {
                let task = Task {
                    NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing stream (active: \(self.activeRequests))")
                    do {
                        let stream = self.engine.stream(request)
                        for try await token in stream {
                            if Task.isCancelled { break }
                            continuation.yield(token)
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                    self.lock.withLock {
                        self._activeCount -= 1
                        self._totalCompleted += 1
                    }
                    self.processQueuedStream()
                }
                self.lock.withLock { self.activeTasks[request.id] = task }

                continuation.onTermination = { _ in
                    task.cancel()
                    _ = self.lock.withLock { self.activeTasks.removeValue(forKey: request.id) }
                }
            } else {
                lock.withLock {
                    streamQueue.append(QueuedStreamRequest(
                        request: request, priority: priority, enqueuedAt: Date(), continuation: continuation
                    ))
                    streamQueue.sort { $0.priority > $1.priority }
                    _totalQueued += 1
                }
                NovaMLXLog.info("Stream request queued: \(request.id), queue depth: \(queueDepth)")
            }
        }
    }

    public func abort(requestId: UUID) {
        lock.withLock {
            generateQueue.removeAll { $0.request.id == requestId }
            streamQueue.removeAll { $0.request.id == requestId }
            if activeTasks[requestId] != nil {
                activeTasks[requestId]?.cancel()
                activeTasks.removeValue(forKey: requestId)
                _totalPreempted += 1
            }
        }
        engine.abort(requestId: requestId)
    }

    private func enqueueAndWait(request: InferenceRequest, priority: RequestPriority) async throws -> InferenceResult {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock {
                generateQueue.append(QueuedRequest(
                    request: request, priority: priority, enqueuedAt: Date(), continuation: continuation
                ))
                generateQueue.sort { $0.priority > $1.priority }
                _totalQueued += 1
            }
            NovaMLXLog.info("Generate request queued: \(request.id), queue depth: \(queueDepth)")
            processQueuedGenerate()
        }
    }

    private func processQueuedGenerate() {
        let item = lock.withLock { () -> QueuedRequest? in
            guard _activeCount < maxBatchSize, !generateQueue.isEmpty else { return nil }
            let item = generateQueue.removeFirst()
            _activeCount += 1
            return item
        }

        guard let item = item else { return }

        let waitTime = Date().timeIntervalSince(item.enqueuedAt)
        lock.withLock { _totalQueueWaitTime += waitTime }

        let request = item.request
        let continuation = item.continuation

        Task {
            do {
                let result = try await engine.generate(request)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
            lock.withLock {
                _activeCount -= 1
                _totalCompleted += 1
            }
            processQueuedGenerate()
            processQueuedStream()
        }
    }

    private func processQueuedStream() {
        let item = lock.withLock { () -> QueuedStreamRequest? in
            guard _activeCount < maxBatchSize, !streamQueue.isEmpty else { return nil }
            let item = streamQueue.removeFirst()
            _activeCount += 1
            return item
        }

        guard let item = item else { return }

        let waitTime = Date().timeIntervalSince(item.enqueuedAt)
        lock.withLock { _totalQueueWaitTime += waitTime }

        let request = item.request
        let continuation = item.continuation

        let task = Task {
            NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing dequeued stream")
            do {
                let stream = engine.stream(request)
                for try await token in stream {
                    if Task.isCancelled { break }
                    continuation.yield(token)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
            lock.withLock {
                _activeCount -= 1
                _totalCompleted += 1
            }
            processQueuedStream()
            processQueuedGenerate()
        }
        lock.withLock { activeTasks[request.id] = task }
    }
}
