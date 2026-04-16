import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

// MARK: - Non-Sendable Wrappers

/// Wrapper to pass non-Sendable `[KVCache]` across isolation boundaries.
/// Safe because `mlxContainer.perform` guarantees serial access.
final class FusedCachesBox: @unchecked Sendable {
    let caches: [KVCache]
    init(caches: [KVCache]) { self.caches = caches }
}

// MARK: - Active Stream Sequence

struct ActiveStreamSequence: @unchecked Sendable {
    let id: UUID
    let request: InferenceRequest
    let caches: [KVCache]
    var lastTokenId: Int
    var generatedText: String
    var completionTokens: Int
    let maxTokens: Int
    let sampler: CompiledSampler
    let continuation: AsyncThrowingStream<Token, Error>.Continuation
    let modelId: String
    let promptTokenCount: Int
    let startTime: Date
    var isFinished: Bool
}

// MARK: - Queued Stream Request

struct FusedQueuedStreamRequest: @unchecked Sendable {
    let request: InferenceRequest
    let enqueuedAt: Date
    let continuation: AsyncThrowingStream<Token, Error>.Continuation
}

struct FusedQueuedGenerateRequest: @unchecked Sendable {
    let request: InferenceRequest
    let enqueuedAt: Date
    let continuation: CheckedContinuation<InferenceResult, Error>
}

// MARK: - FusedBatchScheduler

/// Production scheduler that fuses multiple active sequences into shared GPU forward passes.
/// Each decode step stacks all sequences' last tokens, runs ONE model forward pass, then
/// samples and yields tokens per-sequence independently.
///
/// Memory-aware admission via `MemoryBudgetTracker` (from Phase 1).
/// Streaming support via per-sequence `AsyncThrowingStream.Continuation`.
public final class FusedBatchScheduler: @unchecked Sendable {

    private let engine: MLXEngine
    private let budgetTracker: MemoryBudgetTracker
    public let maxConcurrentPerModel: Int
    private let lock = NovaMLXLock()

    // Active sequences grouped by model
    private var activeByModel: [String: [ActiveStreamSequence]] = [:]
    private var activeModelCounts: [String: Int] = [:]

    // Queues
    private var streamQueue: [FusedQueuedStreamRequest] = []
    private var generateQueue: [FusedQueuedGenerateRequest] = []

    // Run loop state
    private var isRunLoopActive = false
    private var totalDecodeSteps: UInt64 = 0
    private var totalTokensViaFused: UInt64 = 0
    private var peakBatchWidth: Int = 0

    // MARK: - Init

    public init(engine: MLXEngine, maxConcurrentPerModel: Int = 4) {
        self.engine = engine
        self.budgetTracker = engine.budgetTracker
        self.maxConcurrentPerModel = maxConcurrentPerModel
    }

    // MARK: - Public API

    public var activeRequestCount: Int {
        lock.withLock { activeByModel.values.reduce(0) { $0 + $1.count } }
    }

    public var queueDepth: Int {
        lock.withLock { streamQueue.count + generateQueue.count }
    }

    public var metrics: FusedSchedulerMetrics {
        lock.withLock {
            FusedSchedulerMetrics(
                activeSequences: activeByModel.values.reduce(0) { $0 + $1.count },
                queueDepth: streamQueue.count + generateQueue.count,
                totalDecodeSteps: totalDecodeSteps,
                totalTokensViaFused: totalTokensViaFused,
                peakBatchWidth: peakBatchWidth
            )
        }
    }

    // MARK: - Submit (non-streaming)

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        let modelId = request.model

        // Admission check
        let canStart = await canAdmit(request)

        guard canStart else {
            return try await enqueueGenerateAndWait(request: request)
        }

        // Admit and prefill
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

        lock.withLock {
            activeModelCounts[modelId] = (activeModelCounts[modelId] ?? 0) + 1
        }
        await budgetTracker.reserve(
            modelId: modelId, sequenceId: request.id,
            weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        // Wrap as stream, collect all tokens
        return try await withCheckedThrowingContinuation { continuation in
            let stream = self.internalSubmitStream(request)
            Task {
                var text = ""
                var promptTokens = 0
                var completionTokens = 0
                var finishReason: FinishReason = .stop
                do {
                    for try await token in stream {
                        if let fr = token.finishReason { finishReason = fr }
                        if !token.text.isEmpty { text += token.text; completionTokens += 1 }
                    }
                    // Get prompt token count from first sequence
                    promptTokens = lock.withLock {
                        activeByModel[modelId]?.first?.promptTokenCount ?? 0
                    }
                    continuation.resume(returning: InferenceResult(
                        id: request.id, model: modelId, text: text,
                        tokensPerSecond: 0, promptTokens: promptTokens,
                        completionTokens: completionTokens, finishReason: finishReason
                    ))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Submit Stream

    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let modelId = request.model

        return AsyncThrowingStream { outerContinuation in
            let admissionTask = Task {
                await self.canAdmit(request)
            }

            Task {
                let canStart = await admissionTask.value

                guard canStart else {
                    // Queue it
                    self.lock.withLock {
                        self.streamQueue.append(FusedQueuedStreamRequest(
                            request: request, enqueuedAt: Date(), continuation: outerContinuation
                        ))
                    }
                    NovaMLXLog.info("FusedScheduler: queued stream \(request.id.uuidString.prefix(8)), depth=\(self.queueDepth)")
                    self.scheduleRunLoop()
                    return
                }

                let bytesPerToken = self.engine.effectiveBytesPerToken(modelId: modelId)
                let estimatedTokens = self.engine.estimateRequestTokens(modelId: modelId, request: request)

                self.lock.withLock {
                    self.activeModelCounts[modelId] = (self.activeModelCounts[modelId] ?? 0) + 1
                }
                await self.budgetTracker.reserve(
                    modelId: modelId, sequenceId: request.id,
                    weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
                )

                // Prefill and add to active sequences
                do {
                    let seq = try await self.prefillSequence(request: request, continuation: outerContinuation)
                    self.lock.withLock {
                        if self.activeByModel[modelId] == nil { self.activeByModel[modelId] = [] }
                        self.activeByModel[modelId]?.append(seq)
                    }
                    NovaMLXLog.request(request.id.uuidString.prefix(8).description, "FusedScheduler: admitted + prefilled (active=\(self.activeRequestCount), model=\(modelId))")
                    self.scheduleRunLoop()
                } catch {
                    outerContinuation.finish(throwing: error)
                    self.lock.withLock {
                        self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                    }
                    await self.budgetTracker.release(sequenceId: request.id)
                }
            }
        }
    }

    // MARK: - Internal stream (for non-streaming submit)

    private func internalSubmitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let modelId = request.model

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let seq = try await self.prefillSequence(request: request, continuation: continuation)
                    self.lock.withLock {
                        if self.activeByModel[modelId] == nil { self.activeByModel[modelId] = [] }
                        self.activeByModel[modelId]?.append(seq)
                    }
                    self.scheduleRunLoop()
                } catch {
                    continuation.finish(throwing: error)
                    self.lock.withLock {
                        self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                    }
                    await self.budgetTracker.release(sequenceId: request.id)
                }
            }
        }
    }

    // MARK: - Admission

    private func canAdmit(_ request: InferenceRequest) async -> Bool {
        let modelId = request.model
        let activeForModel = lock.withLock { activeModelCounts[modelId] ?? 0 }
        if activeForModel >= maxConcurrentPerModel {
            NovaMLXLog.info("FusedScheduler: queuing \(request.id.uuidString.prefix(8)) — model at concurrency limit (\(activeForModel)/\(maxConcurrentPerModel))")
            return false
        }

        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)
        let canAdmit = await budgetTracker.canAdmit(
            modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        if !canAdmit {
            NovaMLXLog.info("FusedScheduler: queuing \(request.id.uuidString.prefix(8)) — insufficient memory")
        }
        return canAdmit
    }

    // MARK: - Prefill

    private func prefillSequence(
        request: InferenceRequest,
        continuation: AsyncThrowingStream<Token, Error>.Continuation
    ) async throws -> ActiveStreamSequence {
        guard let container = engine.getContainer(for: request.model),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(request.model)
        }

        let hasImages = request.messages.contains { $0.images?.isEmpty == false }
        let userInput = try await engine.buildUserInput(request: request, hasImages: hasImages)
        let input = try await mlxContainer.prepare(input: userInput)
        let promptTokenCount = input.text.tokens.size
        let maxTokens = request.maxTokens ?? container.config.maxTokens

        try engine.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        let parameters = engine.buildGenerateParameters(
            request: request, config: container.config,
            settings: engine.settingsProvider?(request.model)
        )

        // Prefill: run generate with maxTokens=1 to process prompt and get first token
        let cachesBox = MutableSendableBox<[KVCache]?>(nil)
        var prefillParams = GenerateParameters(maxTokens: 1, temperature: parameters.temperature)
        prefillParams.topP = parameters.topP
        prefillParams.topK = parameters.topK
        prefillParams.prefillStepSize = parameters.prefillStepSize
        prefillParams.kvBits = parameters.kvBits
        prefillParams.kvGroupSize = parameters.kvGroupSize

        let inputBox = SendableBox(input)
        let paramsBox = SendableBox(prefillParams)
        let stream = try await mlxContainer.perform { context in
            let cache = context.model.newCache(parameters: parameters)
            cachesBox.value = cache
            return try MLXLMCommon.generate(
                input: inputBox.value, cache: cache,
                parameters: paramsBox.value, context: context
            )
        }

        var firstTokenText = ""
        for await generation in stream {
            switch generation {
            case .chunk(let text): firstTokenText = text
            case .info: break
            case .toolCall: break
            }
        }

        guard let caches = cachesBox.value else {
            throw NovaMLXError.inferenceFailed("Prefill produced no cache")
        }

        let firstTokenId = container.tokenizer?.encode(firstTokenText).first ?? 0

        // Yield first token to client
        if !firstTokenText.isEmpty {
            continuation.yield(Token(id: 0, text: firstTokenText))
        }

        let sampler = CompiledSampler(
            temperature: Float(request.temperature ?? container.config.temperature),
            topP: Float(request.topP ?? container.config.topP),
            topK: request.topK ?? 0,
            minP: request.minP ?? 0.0
        )

        return ActiveStreamSequence(
            id: request.id,
            request: request,
            caches: caches,
            lastTokenId: firstTokenId,
            generatedText: firstTokenText,
            completionTokens: firstTokenText.isEmpty ? 0 : 1,
            maxTokens: maxTokens,
            sampler: sampler,
            continuation: continuation,
            modelId: request.model,
            promptTokenCount: promptTokenCount,
            startTime: Date(),
            isFinished: false
        )
    }

    // MARK: - Run Loop

    private func scheduleRunLoop() {
        let shouldStart = lock.withLock {
            guard !isRunLoopActive else { return false }
            let hasActive = !activeByModel.isEmpty
            let hasQueued = !streamQueue.isEmpty || !generateQueue.isEmpty
            guard hasActive || hasQueued else { return false }
            isRunLoopActive = true
            return true
        }
        guard shouldStart else { return }
        Task { await runLoop() }
    }

    private func runLoop() async {
        while true {
            // 1. Admit queued requests
            await admitQueued()

            // 2. Run fused decode step
            let hadActive = await fusedDecodeStep()

            // 3. Check if we should continue
            let shouldContinue = lock.withLock {
                let hasActive = activeByModel.values.contains { !$0.isEmpty }
                let hasQueued = !streamQueue.isEmpty || !generateQueue.isEmpty
                if !hasActive && !hasQueued {
                    isRunLoopActive = false
                    return false
                }
                return true
            }

            if !shouldContinue { return }
            if !hadActive {
                // No active sequences but queued requests waiting for memory
                // Brief pause to avoid busy-waiting
                try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
            }
        }
    }

    private func admitQueued() async {
        // Admit generate queue
        var admittedGenerate: [FusedQueuedGenerateRequest] = []
        var remainingGenerate: [FusedQueuedGenerateRequest] = []

        let gQueue = lock.withLock { generateQueue }
        for item in gQueue {
            let canStart = await canAdmit(item.request)
            if canStart {
                admittedGenerate.append(item)
            } else {
                remainingGenerate.append(item)
            }
        }
        lock.withLock { generateQueue = remainingGenerate }

        for item in admittedGenerate {
            let modelId = item.request.model
            let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
            let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: item.request)

            lock.withLock { activeModelCounts[modelId] = (activeModelCounts[modelId] ?? 0) + 1 }
            await budgetTracker.reserve(
                modelId: modelId, sequenceId: item.request.id,
                weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
            )

            // Wrap generate as stream, collect result
            let stream = internalSubmitStream(item.request)
            Task {
                var text = ""
                var completionTokens = 0
                var finishReason: FinishReason = .stop
                do {
                    for try await token in stream {
                        if let fr = token.finishReason { finishReason = fr }
                        if !token.text.isEmpty { text += token.text; completionTokens += 1 }
                    }
                    let promptTokens = lock.withLock {
                        activeByModel[modelId]?.first?.promptTokenCount ?? 0
                    }
                    item.continuation.resume(returning: InferenceResult(
                        id: item.request.id, model: modelId, text: text,
                        tokensPerSecond: 0, promptTokens: promptTokens,
                        completionTokens: completionTokens, finishReason: finishReason
                    ))
                } catch {
                    item.continuation.resume(throwing: error)
                }
            }
        }

        // Admit stream queue
        var admittedStream: [FusedQueuedStreamRequest] = []
        var remainingStream: [FusedQueuedStreamRequest] = []

        let sQueue = lock.withLock { streamQueue }
        for item in sQueue {
            let canStart = await canAdmit(item.request)
            if canStart {
                admittedStream.append(item)
            } else {
                remainingStream.append(item)
            }
        }
        lock.withLock { streamQueue = remainingStream }

        for item in admittedStream {
            let modelId = item.request.model
            let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
            let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: item.request)

            lock.withLock { activeModelCounts[modelId] = (activeModelCounts[modelId] ?? 0) + 1 }
            await budgetTracker.reserve(
                modelId: modelId, sequenceId: item.request.id,
                weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
            )

            do {
                let seq = try await prefillSequence(request: item.request, continuation: item.continuation)
                lock.withLock {
                    if activeByModel[modelId] == nil { activeByModel[modelId] = [] }
                    activeByModel[modelId]?.append(seq)
                }
            } catch {
                item.continuation.finish(throwing: error)
                lock.withLock {
                    activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
                }
                await budgetTracker.release(sequenceId: item.request.id)
            }
        }
    }

    // MARK: - Fused Decode Step

    @discardableResult
    private func fusedDecodeStep() async -> Bool {
        let groups = lock.withLock { activeByModel }
        var anyActive = false

        for (modelId, sequences) in groups {
            let active = sequences.filter { !$0.isFinished }
            guard !active.isEmpty else {
                lock.withLock { activeByModel[modelId] = [] }
                continue
            }

            guard let container = engine.getContainer(for: modelId),
                  let mlxContainer = container.mlxContainer else {
                for seq in active {
                    seq.continuation.finish(throwing: NovaMLXError.modelNotFound(modelId))
                    await budgetTracker.release(sequenceId: seq.id)
                }
                lock.withLock { activeByModel[modelId] = []; activeModelCounts[modelId] = 0 }
                continue
            }

            let batchSize = active.count
            lock.withLock {
                totalDecodeSteps += 1
                if batchSize > peakBatchWidth { peakBatchWidth = batchSize }
            }

            // Stack last tokens into [batchSize, 1]
            let tokens = stacked(
                active.map { MLXArray(Int32($0.lastTokenId)) },
                axis: 0
            ).reshaped(batchSize, 1)

            // Build fused KV caches per layer
            guard let firstCaches = active.first?.caches, !firstCaches.isEmpty else {
                lock.withLock { activeByModel[modelId] = active }
                continue
            }
            let numLayers = firstCaches.count
            let fusedCaches: [KVCache] = (0..<numLayers).map { layerIdx in
                let fbc = FusedBatchKVCache(maxBatchSize: maxConcurrentPerModel)
                for seq in active {
                    if let simpleCache = seq.caches[layerIdx] as? KVCacheSimple {
                        _ = fbc.addSequence(id: seq.id, prefillCache: simpleCache)
                    }
                }
                return fbc
            }

            // Single forward pass (wrap non-Sendable KVCache array)
            let fusedCachesBox = FusedCachesBox(caches: fusedCaches)
            let logitsBox = await mlxContainer.perform { context in
                let out = context.model(tokens, cache: fusedCachesBox.caches)
                eval(out)
                return SendableBox(out)
            }
            let logits = logitsBox.value

            let eosId = container.tokenizer?.eosTokenId

            // Per-sequence sample and yield
            for (i, var seq) in active.enumerated() {
                let seqLogits = logits[i][0..., -1, 0...]
                let sampled = seq.sampler.sample(logits: seqLogits)
                let tokenId = sampled.item(Int.self)

                seq.lastTokenId = tokenId
                seq.completionTokens += 1

                let decoded = container.tokenizer?.decode([tokenId]) ?? ""
                seq.generatedText += decoded
                seq.continuation.yield(Token(id: 0, text: decoded))

                lock.withLock { totalTokensViaFused += 1 }

                let isEOS = eosId == tokenId
                let atMax = seq.completionTokens >= seq.maxTokens

                if isEOS || atMax {
                    let finishReason: FinishReason = atMax ? .length : .stop
                    seq.continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    seq.continuation.finish()
                    seq.isFinished = true
                }
            }

            // Update state
            let finished = active.filter { $0.isFinished }
            let surviving = active.filter { !$0.isFinished }

            lock.withLock { activeByModel[modelId] = surviving }

            for seq in finished {
                await budgetTracker.release(sequenceId: seq.id)
                lock.withLock {
                    activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
                }
                let elapsed = Date().timeIntervalSince(seq.startTime)
                engine.metricsStore.recordRequest(
                    model: seq.modelId,
                    tokens: UInt64(seq.completionTokens),
                    inferenceTime: elapsed
                )
            }

            anyActive = true
        }

        // Clean up empty model entries
        lock.withLock {
            for modelId in activeByModel.keys {
                if activeByModel[modelId]?.isEmpty ?? true {
                    activeByModel.removeValue(forKey: modelId)
                }
            }
        }

        return anyActive
    }

    // MARK: - Queue Helpers

    private func enqueueGenerateAndWait(request: InferenceRequest) async throws -> InferenceResult {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock {
                generateQueue.append(FusedQueuedGenerateRequest(
                    request: request, enqueuedAt: Date(), continuation: continuation
                ))
            }
            NovaMLXLog.info("FusedScheduler: queued generate \(request.id.uuidString.prefix(8)), depth=\(queueDepth)")
            scheduleRunLoop()
        }
    }

    // MARK: - Abort

    public func abort(requestId: UUID) {
        lock.withLock {
            streamQueue.removeAll { $0.request.id == requestId }
            generateQueue.removeAll { $0.request.id == requestId }
            for (_, var sequences) in activeByModel {
                if let idx = sequences.firstIndex(where: { $0.id == requestId }) {
                    sequences[idx].isFinished = true
                    sequences[idx].continuation.finish()
                    sequences.remove(at: idx)
                }
            }
        }
        engine.abort(requestId: requestId)
    }
}

// MARK: - Metrics

public struct FusedSchedulerMetrics: Sendable {
    public let activeSequences: Int
    public let queueDepth: Int
    public let totalDecodeSteps: UInt64
    public let totalTokensViaFused: UInt64
    public let peakBatchWidth: Int
}
