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

    public init(engine: MLXEngine, maxBatchSize: Int = 8, maxConcurrentPerModel: Int = 4) {
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

    /// Check whether a request can be admitted based on memory budget and per-model concurrency.
    private func canAdmitRequest(_ request: InferenceRequest) async -> Bool {
        let modelId = request.model

        // Per-model concurrency check
        let activeForModel = lock.withLock {
            _activeModelCounts[modelId] ?? 0
        }
        if activeForModel >= maxConcurrentPerModel {
            NovaMLXLog.info("Scheduler: queuing \(request.id.uuidString.prefix(8)) — model \(modelId) at concurrency limit (\(activeForModel)/\(maxConcurrentPerModel))")
            return false
        }

        // Memory budget check
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)
        let canAdmit = await budgetTracker.canAdmit(
            modelId: modelId,
            estimatedTokens: estimatedTokens,
            bytesPerToken: bytesPerToken
        )

        if !canAdmit {
            let neededMB = UInt64(estimatedTokens) * UInt64(bytesPerToken) / 1024 / 1024
            let availableMB = await budgetTracker.availableKVBudget / 1024 / 1024
            NovaMLXLog.info("Scheduler: queuing \(request.id.uuidString.prefix(8)) — insufficient memory (need \(neededMB)MB, available \(availableMB)MB)")
        }

        return canAdmit
    }

    // MARK: - Submit

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        let priority: RequestPriority = .normal
        let modelId = request.model

        // Memory-aware admission check
        let canStart = await canAdmitRequest(request)

        guard canStart else {
            return try await enqueueAndWait(request: request, priority: priority)
        }

        // Admit: increment counts and reserve memory
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

        lock.withLock {
            _activeCount += 1
            _totalQueued += 1
            if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
            _activeModelCounts[modelId] = (_activeModelCounts[modelId] ?? 0) + 1
        }

        await budgetTracker.reserve(
            modelId: modelId,
            sequenceId: request.id,
            weightsBytes: 0,
            estimatedTokens: estimatedTokens,
            bytesPerToken: bytesPerToken
        )

        defer {
            lock.withLock {
                _activeCount -= 1
                _totalCompleted += 1
                _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
            }
            Task { await budgetTracker.release(sequenceId: request.id) }
        }

        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing generate (active: \(activeRequests), model=\(modelId))")
        return try await engine.generate(request)
    }

    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        return AsyncThrowingStream { continuation in
            let priority: RequestPriority = .normal
            let modelId = request.model

            let admissionTask = Task {
                await self.canAdmitRequest(request)
            }

            Task {
                let canStart = await admissionTask.value

                if canStart {
                    let bytesPerToken = self.engine.effectiveBytesPerToken(modelId: modelId)
                    let estimatedTokens = self.engine.estimateRequestTokens(modelId: modelId, request: request)

                    self.lock.withLock {
                        self._activeCount += 1
                        self._totalQueued += 1
                        if self._activeCount > self._peakActiveCount { self._peakActiveCount = self._activeCount }
                        self._activeModelCounts[modelId] = (self._activeModelCounts[modelId] ?? 0) + 1
                    }

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
                        } catch {
                            continuation.finish(throwing: error)
                        }

                        self.lock.withLock {
                            self._activeCount -= 1
                            self._totalCompleted += 1
                            self._activeModelCounts[modelId] = max(0, (self._activeModelCounts[modelId] ?? 1) - 1)
                        }
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
            guard _activeCount < maxBatchSize, !generateQueue.isEmpty else { return nil }
            let item = generateQueue.removeFirst()

            let currentForModel = _activeModelCounts[item.request.model] ?? 0
            guard currentForModel < maxConcurrentPerModel else { return nil }

            _activeCount += 1
            _activeModelCounts[item.request.model] = currentForModel + 1
            return item
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
                    _activeCount -= 1
                    _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
                    generateQueue.insert(item, at: 0)
                    _totalQueued -= 1
                }
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

            lock.withLock {
                _activeCount -= 1
                _totalCompleted += 1
                _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
            }
            await budgetTracker.release(sequenceId: request.id)

            processQueuedGenerate()
            processQueuedStream()
        }
    }

    private func processQueuedStream() {
        let item = lock.withLock { () -> QueuedStreamRequest? in
            guard _activeCount < maxBatchSize, !streamQueue.isEmpty else { return nil }
            let item = streamQueue.removeFirst()

            let currentForModel = _activeModelCounts[item.request.model] ?? 0
            guard currentForModel < maxConcurrentPerModel else { return nil }

            _activeCount += 1
            _activeModelCounts[item.request.model] = currentForModel + 1
            return item
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
                    _activeCount -= 1
                    _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
                    streamQueue.insert(item, at: 0)
                    _totalQueued -= 1
                }
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
                _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
            }
            await budgetTracker.release(sequenceId: request.id)

            processQueuedStream()
            processQueuedGenerate()
        }
        lock.withLock { activeTasks[request.id] = task }
    }
}
