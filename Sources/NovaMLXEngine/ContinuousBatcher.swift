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
    public let specAcceptedTokens: UInt64
    public let specTotalDrafted: UInt64

    public init(
        activeRequests: Int = 0,
        queueDepth: Int = 0,
        totalQueued: UInt64 = 0,
        totalCompleted: UInt64 = 0,
        totalPreempted: UInt64 = 0,
        peakActiveCount: Int = 0,
        averageQueueWaitTime: TimeInterval = 0,
        maxBatchSize: Int = 8,
        uptime: TimeInterval = 0,
        specAcceptedTokens: UInt64 = 0,
        specTotalDrafted: UInt64 = 0
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
        self.specAcceptedTokens = specAcceptedTokens
        self.specTotalDrafted = specTotalDrafted
    }
}

public final class ContinuousBatcher: @unchecked Sendable {
    private let engine: MLXEngine
    private let budgetTracker: MemoryBudgetTracker
    public let maxBatchSize: Int
    private let maxConcurrentPerModel: Int
    private let lock = NovaMLXLock()

    private var generateQueue: [QueuedRequest]
    private var streamQueue: [QueuedStreamRequest]
    private var _activeCount: Int
    private var _activeModelCounts: [String: Int]
    private var _totalQueued: UInt64
    private var _totalCompleted: UInt64
    private var _totalQueueWaitTime: TimeInterval
    private var _totalPreempted: UInt64
    private var _peakActiveCount: Int
    private var _startTime: Date
    private var activeTasks: [UUID: Task<Void, Never>]

    private var _specAcceptedTokens: UInt64 = 0
    private var _specTotalDrafted: UInt64 = 0

    public init(engine: MLXEngine, maxBatchSize: Int = 8, maxConcurrentPerModel: Int = ResourceLimits.safetyConcurrentCap) {
        self.engine = engine
        self.budgetTracker = engine.budgetTracker
        self.maxBatchSize = maxBatchSize
        self.maxConcurrentPerModel = maxConcurrentPerModel
        self.generateQueue = []
        self.streamQueue = []
        self._activeCount = 0
        self._activeModelCounts = [:]
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
    public var specAcceptanceRate: Double {
        lock.withLock {
            _specTotalDrafted > 0 ? Double(_specAcceptedTokens) / Double(_specTotalDrafted) : 0
        }
    }
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
                uptime: Date().timeIntervalSince(_startTime),
                specAcceptedTokens: _specAcceptedTokens,
                specTotalDrafted: _specTotalDrafted
            )
        }
    }

    // MARK: - Memory-Aware Admission

    private func decodeCap(for request: InferenceRequest) -> Int {
        let linear = engine.getContainer(for: request.model)?.config.hasLinearAttention == true
        return ResourceLimits.decodeConcurrencyCap(
            hasDraftModel: request.draftModel != nil,
            hasLinearAttention: linear
        )
    }

    /// Atomically take a per-model decode slot. Caller must `releaseSlot` or
    /// roll the count back if it later decides not to run.
    private func tryReserveSlot(_ request: InferenceRequest) -> Bool {
        let modelId = request.model
        let cap = decodeCap(for: request)
        return lock.withLock {
            let current = _activeModelCounts[modelId] ?? 0
            guard current < cap else {
                NovaMLXLog.info(
                    "Scheduler: queuing \(request.id.uuidString.prefix(8)) — serial decode cap (\(current)/\(cap)) model=\(modelId) draft=\(request.draftModel != nil)"
                )
                return false
            }
            _activeCount += 1
            if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
            _activeModelCounts[modelId] = current + 1
            _totalQueued += 1
            return true
        }
    }

    private func releaseSlot(modelId: String, completed: Bool) {
        lock.withLock {
            _activeCount = max(0, _activeCount - 1)
            if completed { _totalCompleted += 1 }
            _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
        }
    }

    // MARK: - Submit

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        let priority: RequestPriority = .normal
        let modelId = request.model

        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)
        let memoryOk = await budgetTracker.canAdmit(
            modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )
        guard memoryOk, tryReserveSlot(request) else {
            return try await enqueueAndWait(request: request, priority: priority)
        }

        await budgetTracker.reserve(
            modelId: modelId,
            sequenceId: request.id,
            weightsBytes: 0,
            estimatedTokens: estimatedTokens,
            bytesPerToken: bytesPerToken
        )

        defer {
            releaseSlot(modelId: modelId, completed: true)
            Task { await budgetTracker.release(sequenceId: request.id) }
            processQueuedStream()
            processQueuedGenerate()
        }

        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing generate (active: \(activeRequests), model=\(modelId))")
        return try await engine.generate(request)
    }

    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        return AsyncThrowingStream { continuation in
            let priority: RequestPriority = .normal
            let modelId = request.model

            Task {
                let bytesPerToken = self.engine.effectiveBytesPerToken(modelId: modelId)
                let estimatedTokens = self.engine.estimateRequestTokens(modelId: modelId, request: request)
                let memoryOk = await self.budgetTracker.canAdmit(
                    modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
                )
                let canStart = memoryOk && self.tryReserveSlot(request)

                if canStart {
                    await self.budgetTracker.reserve(
                        modelId: modelId,
                        sequenceId: request.id,
                        weightsBytes: 0,
                        estimatedTokens: estimatedTokens,
                        bytesPerToken: bytesPerToken
                    )

                    NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing stream (active: \(self.activeRequests), model=\(modelId))")

                    let task = Task {
                        do {
                            let stream = self.engine.stream(request)
                            for try await token in stream {
                                if Task.isCancelled { break }
                                continuation.yield(token)
                            }
                            continuation.finish()
                            NovaMLXLog.info("[Batcher:\(request.id.uuidString.prefix(8))] stream finished cleanly — model=\(modelId)")
                        } catch {
                            continuation.finish(throwing: error)
                            NovaMLXLog.error("[Batcher:\(request.id.uuidString.prefix(8))] stream errored — model=\(modelId): \(error) — \(type(of: error))")
                        }

                        self.releaseSlot(modelId: modelId, completed: true)
                        await self.budgetTracker.release(sequenceId: request.id)

                        self.processQueuedStream()
                        self.processQueuedGenerate()
                    }
                    self.lock.withLock { self.activeTasks[request.id] = task }

                    continuation.onTermination = { _ in
                        task.cancel()
                        _ = self.lock.withLock { self.activeTasks.removeValue(forKey: request.id) }
                    }
                } else {
                    // Queue the request
                    self.lock.withLock {
                        self.streamQueue.append(QueuedStreamRequest(
                            request: request, priority: priority, enqueuedAt: Date(), continuation: continuation
                        ))
                        self.streamQueue.sort { $0.priority > $1.priority }
                        self._totalQueued += 1
                    }
                    NovaMLXLog.info("Stream request queued: \(request.id.uuidString.prefix(8)), queue depth: \(self.queueDepth)")
                }
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
            NovaMLXLog.info("Generate request queued: \(request.id.uuidString.prefix(8)), queue depth: \(queueDepth)")
            processQueuedGenerate()
        }
    }

    private func processQueuedGenerate() {
        let item = lock.withLock { () -> QueuedRequest? in
            guard !generateQueue.isEmpty else { return nil }
            for i in generateQueue.indices {
                let candidate = generateQueue[i]
                let cap = decodeCap(for: candidate.request)
                let currentForModel = _activeModelCounts[candidate.request.model] ?? 0
                guard currentForModel < cap else { continue }
                generateQueue.remove(at: i)
                _activeCount += 1
                if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
                _activeModelCounts[candidate.request.model] = currentForModel + 1
                return candidate
            }
            return nil
        }

        guard let item = item else { return }

        let waitTime = Date().timeIntervalSince(item.enqueuedAt)
        lock.withLock { _totalQueueWaitTime += waitTime }

        let request = item.request
        let continuation = item.continuation
        let modelId = request.model

        Task {
            let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
            let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

            let canAdmit = await budgetTracker.canAdmit(
                modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
            )

            if !canAdmit {
                lock.withLock {
                    generateQueue.insert(item, at: 0)
                    _totalQueued -= 1
                }
                releaseSlot(modelId: modelId, completed: false)
                NovaMLXLog.info("Scheduler: generate dequeue rejected — insufficient memory, re-queuing")
                return
            }

            await budgetTracker.reserve(
                modelId: modelId, sequenceId: request.id,
                weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
            )

            do {
                let result = try await engine.generate(request)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }

            releaseSlot(modelId: modelId, completed: true)
            await budgetTracker.release(sequenceId: request.id)

            processQueuedGenerate()
            processQueuedStream()
        }
    }

    private func processQueuedStream() {
        let item = lock.withLock { () -> QueuedStreamRequest? in
            guard !streamQueue.isEmpty else { return nil }
            for i in streamQueue.indices {
                let candidate = streamQueue[i]
                let cap = decodeCap(for: candidate.request)
                let currentForModel = _activeModelCounts[candidate.request.model] ?? 0
                guard currentForModel < cap else { continue }
                streamQueue.remove(at: i)
                _activeCount += 1
                if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
                _activeModelCounts[candidate.request.model] = currentForModel + 1
                return candidate
            }
            return nil
        }

        guard let item = item else { return }

        let waitTime = Date().timeIntervalSince(item.enqueuedAt)
        lock.withLock { _totalQueueWaitTime += waitTime }

        let request = item.request
        let continuation = item.continuation
        let modelId = request.model

        let task = Task {
            let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
            let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

            let canAdmit = await budgetTracker.canAdmit(
                modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
            )

            if !canAdmit {
                lock.withLock {
                    streamQueue.insert(item, at: 0)
                    _totalQueued -= 1
                }
                releaseSlot(modelId: modelId, completed: false)
                NovaMLXLog.info("Scheduler: stream dequeue rejected — insufficient memory, re-queuing")
                return
            }

            await budgetTracker.reserve(
                modelId: modelId, sequenceId: request.id,
                weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
            )

            NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing dequeued stream")
            do {
                let stream = engine.stream(request)
                for try await token in stream {
                    if Task.isCancelled {
                        NovaMLXLog.info("[Batcher:\(request.id.uuidString.prefix(8))] stream cancelled — model=\(modelId)")
                        break
                    }
                    continuation.yield(token)
                }
                continuation.finish()
                NovaMLXLog.info("[Batcher:\(request.id.uuidString.prefix(8))] dequeued stream finished cleanly — model=\(modelId)")
            } catch {
                continuation.finish(throwing: error)
                NovaMLXLog.error("[Batcher:\(request.id.uuidString.prefix(8))] dequeued stream errored — model=\(modelId): \(error) — \(type(of: error))")
            }

            releaseSlot(modelId: modelId, completed: true)
            await budgetTracker.release(sequenceId: request.id)

            processQueuedStream()
            processQueuedGenerate()
        }
        lock.withLock { activeTasks[request.id] = task }
    }
}
