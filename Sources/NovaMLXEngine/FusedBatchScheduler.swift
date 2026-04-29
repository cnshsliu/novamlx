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

/// Result of a single-token decode (fallback path, no speculation).
struct DecodeResult: Sendable {
    let tokenId: Int
    let sequenceId: UUID
}

/// Result of a speculative decode round: accepted tokens + bonus token.
struct SpecDecodeResult: Sendable {
    /// Accepted + bonus token IDs, in order. Always >= 1 (the bonus token).
    let acceptedTokens: [Int]
    /// How many of the drafted tokens were accepted (excluding bonus).
    let draftAccepted: Int
    /// How many draft tokens were proposed.
    let draftProposed: Int
    let sequenceId: UUID
}

/// Sampler configuration passed into perform closure.
struct SamplerConfig: Sendable {
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
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
    let promptTokenIds: [Int]  // Full prompt tokens for prefix cache storage
    let startTime: Date
    let admittedAt: Date       // When this sequence was admitted (for preemption ordering)
    var isFinished: Bool
    /// Recent token IDs for N-gram speculation context.
    var recentTokenIds: [Int]
    /// Frequency penalty value — prevents repetition collapse in small quantized models.
    /// Applied inline via scatter_add before sampling. 0 = disabled.
    let frequencyPenalty: Float

    // Accumulated batch decode — avoids per-token decode that breaks SentencePiece spaces
    var accumulatedTokenIds: [Int] = []
    var lastDecodedText: String = ""

    mutating func decodeNextToken(_ tokenId: Int, tokenizer: Tokenizer) -> String? {
        accumulatedTokenIds.append(tokenId)
        let fullText = tokenizer.decode(accumulatedTokenIds)
        let deltaLength = fullText.count - lastDecodedText.count
        lastDecodedText = fullText
        guard deltaLength > 0 else { return nil }
        return String(fullText.suffix(deltaLength))
    }
}

// MARK: - Queued Stream Request

struct FusedQueuedStreamRequest: @unchecked Sendable {
    let request: InferenceRequest
    let enqueuedAt: Date
    let priority: RequestPriority
    let continuation: AsyncThrowingStream<Token, Error>.Continuation
}

struct FusedQueuedGenerateRequest: @unchecked Sendable {
    let request: InferenceRequest
    let enqueuedAt: Date
    let priority: RequestPriority
    let continuation: CheckedContinuation<InferenceResult, Error>
}

// MARK: - FusedBatchScheduler

/// Production scheduler that fuses multiple active sequences into shared GPU forward passes.
/// Each decode step stacks all sequences' last tokens, runs ONE model forward pass, then
/// samples and yields tokens per-sequence independently.
///
/// Phase 1: Memory-aware admission via `MemoryBudgetTracker`.
/// Phase 2: Fused batch decode with streaming.
/// Phase 3: Prefix cache read-path activation.
/// Phase 4: Preemption, auto-concurrency, priority queues, memory feedback.
public final class FusedBatchScheduler: @unchecked Sendable {

    private let engine: MLXEngine
    private let budgetTracker: MemoryBudgetTracker
    private let maxConcurrentPerModel: Int  // Hard upper bound
    private let lock = NovaMLXLock()

    // Active sequences grouped by model
    private var activeByModel: [String: [ActiveStreamSequence]] = [:]
    private var activeModelCounts: [String: Int] = [:]

    // Queues (sorted by priority on dequeue)
    private var streamQueue: [FusedQueuedStreamRequest] = []
    private var generateQueue: [FusedQueuedGenerateRequest] = []

    // Run loop state
    private var isRunLoopActive = false
    private var totalDecodeSteps: UInt64 = 0
    private var totalTokensViaFused: UInt64 = 0
    private var peakBatchWidth: Int = 0
    private var totalPreemptions: UInt64 = 0
    private var lastDecodeStepTime: Date = Date()
    private var lastDecodeStepTokens: UInt64 = 0

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
            let specStats = engine.specDecoder.speculatorStats
            return FusedSchedulerMetrics(
                activeSequences: activeByModel.values.reduce(0) { $0 + $1.count },
                queueDepth: streamQueue.count + generateQueue.count,
                totalDecodeSteps: totalDecodeSteps,
                totalTokensViaFused: totalTokensViaFused,
                peakBatchWidth: peakBatchWidth,
                totalPreemptions: totalPreemptions,
                specAcceptanceRate: engine.specDecoder.acceptanceRate,
                specCacheSize: specStats.cacheSize
            )
        }
    }

    // MARK: - Auto Concurrency

    /// Calculate optimal concurrent sequences for a model based on available memory.
    /// More KV-hungry models get fewer slots; small models get more.
    /// Bounded by `maxConcurrentPerModel` hard limit.
    private func optimalConcurrency(for modelId: String) async -> Int {
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let budget = await budgetTracker.availableKVBudget
        let usable = budget * 90 / 100  // 10% headroom

        // Estimate tokens per sequence: use a reasonable default context
        let estimatedTokensPerSeq = 4096
        let bytesPerSeq = UInt64(estimatedTokensPerSeq) * UInt64(bytesPerToken)

        guard bytesPerSeq > 0 else { return maxConcurrentPerModel }

        let fitCount = Int(usable / bytesPerSeq)
        return max(1, min(fitCount, maxConcurrentPerModel))
    }

    // MARK: - Submit (non-streaming)

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        let modelId = request.model
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

        // Quick admission check — if memory is tight, fall back to engine (no fused optimization)
        guard await canAdmit(request) else {
            NovaMLXLog.info("FusedScheduler: submit() can't admit — falling back to engine")
            return try await engine.generate(request)
        }

        // Reserve budget
        lock.withLock { activeModelCounts[modelId] = (activeModelCounts[modelId] ?? 0) + 1 }
        await budgetTracker.reserve(
            modelId: modelId, sequenceId: request.id,
            weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        let startTime = Date()
        let promptTokensBox = MutableSendableBox<Int>(0)

        // Create stream, prefill, add to active, collect all tokens
        let stream = AsyncThrowingStream<Token, Error> { continuation in
            Task {
                do {
                    let seq = try await self.prefillSequence(request: request, continuation: continuation)
                    promptTokensBox.value = seq.promptTokenCount
                    if !seq.isFinished {
                        self.lock.withLock {
                            if self.activeByModel[modelId] == nil { self.activeByModel[modelId] = [] }
                            self.activeByModel[modelId]?.append(seq)
                        }
                        self.scheduleRunLoop()
                    } else {
                        // EOS on first token — release budget immediately
                        self.lock.withLock {
                            self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                        }
                        await self.budgetTracker.release(sequenceId: request.id)
                    }
                } catch {
                    continuation.finish(throwing: error)
                    self.lock.withLock {
                        self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                    }
                    await self.budgetTracker.release(sequenceId: request.id)
                }
            }
        }

        var text = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop

        for try await token in stream {
            if let fr = token.finishReason { finishReason = fr }
            if !token.text.isEmpty { text += token.text; completionTokens += 1 }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        return InferenceResult(
            id: request.id, model: modelId, text: text,
            tokensPerSecond: tps, promptTokens: promptTokensBox.value,
            completionTokens: completionTokens, finishReason: finishReason
        )
    }

    // MARK: - Submit Stream

    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let modelId = request.model
        let reqTag = request.id.uuidString.prefix(8)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Check admission
                    let canStart = await self.canAdmit(request)
                    if !canStart {
                        // Queue for later admission
                        NovaMLXLog.info("FusedScheduler: queuing stream \(reqTag) — memory/concurrency limit")
                        self.lock.withLock {
                            self.streamQueue.append(FusedQueuedStreamRequest(
                                request: request, enqueuedAt: Date(),
                                priority: .normal, continuation: continuation
                            ))
                        }
                        self.scheduleRunLoop()
                        return
                    }

                    // Reserve budget
                    let bytesPerToken = self.engine.effectiveBytesPerToken(modelId: modelId)
                    let estimatedTokens = self.engine.estimateRequestTokens(modelId: modelId, request: request)
                    self.lock.withLock { self.activeModelCounts[modelId] = (self.activeModelCounts[modelId] ?? 0) + 1 }
                    await self.budgetTracker.reserve(
                        modelId: modelId, sequenceId: request.id,
                        weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
                    )

                    // Prefill and add to active sequences
                    let seq = try await self.prefillSequence(request: request, continuation: continuation)
                    if !seq.isFinished {
                        self.lock.withLock {
                            if self.activeByModel[modelId] == nil { self.activeByModel[modelId] = [] }
                            self.activeByModel[modelId]?.append(seq)
                        }
                        self.scheduleRunLoop()
                        NovaMLXLog.info("[Route:\(reqTag)] → FusedBatchScheduler (model=\(modelId), active=\(self.activeRequestCount))")
                    } else {
                        // EOS on first token — release budget
                        self.lock.withLock {
                            self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                        }
                        await self.budgetTracker.release(sequenceId: request.id)
                    }
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

        // Use auto-tuned concurrency limit instead of static max
        let concurrentLimit = await optimalConcurrency(for: modelId)
        if activeForModel >= concurrentLimit {
            NovaMLXLog.info("FusedScheduler: queuing \(request.id.uuidString.prefix(8)) — model at concurrency limit (\(activeForModel)/\(concurrentLimit))")
            return false
        }

        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)
        let canAdmitMemory = await budgetTracker.canAdmit(
            modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        if !canAdmitMemory {
            NovaMLXLog.info("FusedScheduler: queuing \(request.id.uuidString.prefix(8)) — insufficient memory")
        }
        return canAdmitMemory
    }

    // MARK: - Preemption

    /// Preempt the newest active sequence for the given model to free memory.
    /// The preempted sequence's continuation gets finished (client receives error),
    /// memory is released. The client must re-submit the request.
    ///
    /// Returns `true` if a sequence was preempted.
    @discardableResult
    private func preemptNewest(for modelId: String, toMakeRoomFor requestId: UUID) async -> Bool {
        let victim: ActiveStreamSequence? = lock.withLock {
            guard var sequences = activeByModel[modelId], !sequences.isEmpty else { return nil }

            // Find the newest sequence (last admitted) that is NOT the requesting one
            let candidates = sequences.filter { !$0.isFinished && $0.id != requestId }
            guard let newest = candidates.max(by: { $0.admittedAt < $1.admittedAt }) else { return nil }

            // Mark as finished
            if let idx = sequences.firstIndex(where: { $0.id == newest.id }) {
                sequences[idx].isFinished = true
                activeByModel[modelId] = sequences
            }
            return newest
        }

        guard let victim = victim else { return false }

        // Release the victim's memory budget
        await budgetTracker.release(sequenceId: victim.id)
        lock.withLock {
            activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
            totalPreemptions += 1
        }

        // Finish the victim's stream with a retryable error
        victim.continuation.finish(throwing: NovaMLXError.inferenceFailed(
            "Sequence preempted due to memory pressure — please retry"
        ))

        NovaMLXLog.warning("FusedScheduler: PREEMPTED sequence \(victim.id.uuidString.prefix(8)) (model=\(modelId), \(victim.completionTokens) tokens generated) to make room for \(requestId.uuidString.prefix(8))")

        return true
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

        try await engine.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        let allTokens = input.text.tokens.asArray(Int32.self).map { Int($0) }

        let parameters = engine.buildGenerateParameters(
            request: request, config: container.config,
            settings: engine.settingsProvider?(request.model)
        )

        // --- Memory feedback: update budget with actual token count ---
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: request.model)
        let actualKVBytes = UInt64(allTokens.count + maxTokens) * UInt64(bytesPerToken)
        await budgetTracker.updateActual(
            modelId: request.model,
            sequenceId: request.id,
            actualBytes: actualKVBytes
        )

        // --- Prefix cache check ---
        let prefixCacheManager = engine.getOrCreatePrefixCacheManager(modelId: request.model)
        let prefixResult = prefixCacheManager?.fetchPrefix(tokenIds: allTokens)

        if let prefixResult = prefixResult,
           let restoredCaches = prefixResult.cache,
           prefixResult.cachedTokenCount > 0,
           !prefixResult.remainingTokenIds.isEmpty {

            // Validate restored caches are compatible with the current model
            // Check layer count matches and caches are non-empty
            let freshCachesBox = await mlxContainer.perform { context in
                FusedCachesBox(caches: context.model.newCache(parameters: parameters))
            }
            let freshCaches = freshCachesBox.caches
            let cacheCompatible = restoredCaches.count == freshCaches.count &&
                restoredCaches.allSatisfy { !$0.state.isEmpty }

            if cacheCompatible {
                // CACHE HIT: Restore KV caches, only prefill remaining tokens
                NovaMLXLog.info("FusedScheduler: prefix cache HIT — \(prefixResult.cachedTokenCount) tokens cached, \(prefixResult.remainingTokenIds.count) remaining to prefill")

                return try await prefillRemaining(
                    request: request,
                    continuation: continuation,
                    container: container,
                    mlxContainer: mlxContainer,
                    parameters: parameters,
                    restoredCaches: restoredCaches,
                    remainingTokenIds: prefixResult.remainingTokenIds,
                    cachedTokenCount: prefixResult.cachedTokenCount,
                    allTokens: allTokens,
                    promptTokenCount: promptTokenCount,
                    maxTokens: maxTokens
                )
            } else {
                NovaMLXLog.warning("FusedScheduler: prefix cache HIT but shapes incompatible — falling back to full prefill (restored=\(restoredCaches.count) layers, expected=\(freshCaches.count))")
                // Clear stale cache to prevent future mismatches
                prefixCacheManager?.clear()
            }
        }

        // CACHE MISS (or no prefix cache): Manual full prefill
        if let prefixResult = prefixResult, prefixResult.cachedTokenCount == 0 {
            NovaMLXLog.debug("FusedScheduler: prefix cache miss — full prefill (\(allTokens.count) tokens)")
        }

        // Manual prefill: create caches and run forward pass INSIDE mlxContainer.perform.
        // Running model() outside perform causes NaN/Inf logits because MLX lazy arrays
        // and compiled functions require the SerialAccessContainer context for correct
        // evaluation. ContinuousBatcher works because it calls model() inside perform.
        let reqTag = request.id.uuidString.prefix(8)
        NovaMLXLog.info("[Prefill:\(reqTag)] Manual prefill — \(allTokens.count) tokens")

        // Keep caches as KVCacheSimple for fused decode compatibility (no kvBits).
        var prefillParamsMut = parameters
        prefillParamsMut.kvBits = nil
        let prefillParams = prefillParamsMut

        let chunkSize = prefillParams.prefillStepSize
        let tokenIds = input.text.tokens.asArray(Int32.self)
        let prefillTimeoutSeconds: Double = 120  // 2 minute timeout for prefill

        let sampler = CompiledSampler(
            temperature: Float(request.temperature ?? container.config.temperature),
            topP: Float(request.topP ?? container.config.topP),
            topK: request.topK ?? container.config.topK,
            minP: request.minP ?? container.config.minP
        )

        // Frequency penalty — per-model default from generation_config.json + family defaults
        let freqPenaltyValue = request.frequencyPenalty ?? container.config.frequencyPenalty

        // Run entire prefill + first token sampling inside perform to ensure
        // correct MLX lazy evaluation context.
        let cachesBox = MutableSendableBox<[KVCache]?>(nil)
        let prefillResultBox = await mlxContainer.perform { context in
            let model = context.model
            let caches = model.newCache(parameters: prefillParams)

            // Chunked forward pass with timeout checks between chunks
            var lastLogits: MLXArray = MLXArray()
            var offset = 0
            let prefillStart = Date()

            while offset < tokenIds.count {
                let elapsed = Date().timeIntervalSince(prefillStart)
                if elapsed > prefillTimeoutSeconds {
                    NovaMLXLog.error("[Prefill:\(reqTag)] Timeout after \(String(format: "%.1f", elapsed))s at chunk offset \(offset)/\(tokenIds.count)")
                    return SendableBox((-1, "Timeout after \(Int(elapsed))s (\(offset)/\(tokenIds.count) tokens processed)"))
                }
                let end = min(offset + chunkSize, tokenIds.count)
                let chunk = MLXArray(Array(tokenIds[offset..<end])).reshaped(1, end - offset)
                lastLogits = model(chunk, cache: caches)
                eval(lastLogits)
                eval(caches)
                offset = end
            }
            let prefillElapsed = Date().timeIntervalSince(prefillStart)
            NovaMLXLog.info("[Prefill:\(reqTag)] Forward pass completed in \(String(format: "%.1f", prefillElapsed))s for \(tokenIds.count) tokens")

            // Sample first token — check for NaN/inf logits before sampling.
            // IMPORTANT: Cast to float32 before computing stats. Float16 logits
            // have a max value of ~65504, and summing 32K+ vocab logits trivially
            // overflows to inf even with perfectly valid values. Using max/min in
            // float32 avoids false positives.
            let logitsToSample = lastLogits[0..., -1, 0...]
            let logitsF32 = logitsToSample.asType(.float32)
            let maxLogit = logitsF32.max().item(Float.self)
            let minLogit = logitsF32.min().item(Float.self)
            if maxLogit.isNaN || maxLogit.isInfinite || minLogit.isNaN || minLogit.isInfinite {
                NovaMLXLog.error("[Prefill:\(reqTag)] Logits contain NaN/Inf (max=\(maxLogit), min=\(minLogit)) — sequence too long or model unstable")
                return SendableBox((-1, "NaN/Inf"))
            }
            let stats = "max=\(String(format: "%.2f", maxLogit)), min=\(String(format: "%.2f", minLogit)), shape=\(logitsToSample.shape), dtype=\(logitsToSample.dtype)"
            let prefillSampled = sampler.sample(logits: logitsToSample)
            eval(prefillSampled)

            cachesBox.value = caches
            return SendableBox((prefillSampled.item(Int.self), stats))
        }
        let (firstTokenId, logitStats) = prefillResultBox.value
        let caches = cachesBox.value ?? []
        let firstTokenText: String
        var initialAccumulatedIds: [Int] = []
        var initialDecodedText: String = ""
        if firstTokenId >= 0, let tok = container.tokenizer {
            initialAccumulatedIds = [firstTokenId]
            firstTokenText = tok.decode(initialAccumulatedIds)
            initialDecodedText = firstTokenText
        } else {
            firstTokenText = "<invalid>"
        }
        NovaMLXLog.info("[Prefill:\(reqTag)] First token: id=\(firstTokenId), text='\(firstTokenText)', logits=[\(logitStats)]")

        guard firstTokenId >= 0 else {
            throw NovaMLXError.inferenceFailed(
                "Prefill produced invalid logits (NaN/inf) — prompt may be too long for this model"
            )
        }

        guard !caches.isEmpty else {
            throw NovaMLXError.inferenceFailed("Prefill produced no cache")
        }

        // Check if first token is EOS
        let eosId = container.tokenizer?.eosTokenId
        let isEOS = eosId == firstTokenId

        // Yield first token to client (unless it's EOS and we finish immediately)
        let scrubbedFirstToken = MLXEngine.scrubControlTokens(firstTokenText)
        if !isEOS && !scrubbedFirstToken.isEmpty {
            continuation.yield(Token(id: 0, text: scrubbedFirstToken))
        }

        let completionCount: Int
        let isFinished: Bool
        if isEOS {
            continuation.yield(Token(id: 0, text: "", finishReason: .stop))
            continuation.finish()
            completionCount = 0
            isFinished = true
        } else {
            completionCount = firstTokenText.isEmpty ? 0 : 1
            isFinished = false
        }

        return ActiveStreamSequence(
            id: request.id,
            request: request,
            caches: caches,
            lastTokenId: firstTokenId,
            generatedText: firstTokenText,
            completionTokens: completionCount,
            maxTokens: maxTokens,
            sampler: sampler,
            continuation: continuation,
            modelId: request.model,
            promptTokenCount: promptTokenCount,
            promptTokenIds: allTokens,
            startTime: Date(),
            admittedAt: Date(),
            isFinished: isFinished,
            recentTokenIds: Array(allTokens.suffix(20)),
            frequencyPenalty: freqPenaltyValue,
            accumulatedTokenIds: initialAccumulatedIds,
            lastDecodedText: initialDecodedText
        )
    }

    /// Prefill only the remaining tokens after a prefix cache hit.
    /// Runs model forward pass on remaining tokens with restored KV caches.
    private func prefillRemaining(
        request: InferenceRequest,
        continuation: AsyncThrowingStream<Token, Error>.Continuation,
        container: ModelContainer,
        mlxContainer: MLXLMCommon.ModelContainer,
        parameters: GenerateParameters,
        restoredCaches: [KVCache],
        remainingTokenIds: [Int],
        cachedTokenCount: Int,
        allTokens: [Int],
        promptTokenCount: Int,
        maxTokens: Int
    ) async throws -> ActiveStreamSequence {
        // Run forward pass inside perform — model() must be called within
        // SerialAccessContainer to ensure correct MLX lazy evaluation context.
        let sampler = CompiledSampler(
            temperature: Float(request.temperature ?? container.config.temperature),
            topP: Float(request.topP ?? container.config.topP),
            topK: request.topK ?? container.config.topK,
            minP: request.minP ?? container.config.minP
        )
        let freqPenaltyValue = request.frequencyPenalty ?? container.config.frequencyPenalty
        let reqTag = request.id.uuidString.prefix(8)
        let remainingArray = MLXArray(remainingTokenIds.map { Int32($0) })
        let firstTokenIdBox = await mlxContainer.perform(nonSendable: (remainingArray, restoredCaches)) { context, values in
            let (remaining, caches) = values
            let logits = context.model(remaining.reshaped(1, remainingTokenIds.count), cache: caches)
            eval(logits)
            let logitsToSample = logits[0..., -1, 0...]
            // NaN/inf check — cast to float32 to avoid false positives from
            // float16 sum overflow (32K+ vocab logits easily exceed float16 max).
            let logitsF32 = logitsToSample.asType(.float32)
            let maxLogit = logitsF32.max().item(Float.self)
            let minLogit = logitsF32.min().item(Float.self)
            if maxLogit.isNaN || maxLogit.isInfinite || minLogit.isNaN || minLogit.isInfinite {
                NovaMLXLog.error("[Prefill:\(reqTag)] Remaining prefill logits contain NaN/Inf (max=\(maxLogit), min=\(minLogit))")
                return SendableBox(-1)
            }
            let sampled = sampler.sample(logits: logitsToSample)
            eval(sampled)
            return SendableBox(sampled.item(Int.self))
        }
        let firstTokenId = firstTokenIdBox.value

        guard firstTokenId >= 0 else {
            throw NovaMLXError.inferenceFailed(
                "Prefill (remaining) produced invalid logits (NaN/inf) — prompt may be too long for this model"
            )
        }

        let firstTokenText: String
        var prefixAccumulatedIds: [Int] = []
        var prefixDecodedText: String = ""
        if let tok = container.tokenizer {
            prefixAccumulatedIds = [firstTokenId]
            firstTokenText = tok.decode(prefixAccumulatedIds)
            prefixDecodedText = firstTokenText
        } else {
            firstTokenText = ""
        }
        NovaMLXLog.info("[Prefill:\(reqTag)] First token (remaining): id=\(firstTokenId), text='\(firstTokenText)'")

        let scrubbedPrefixToken = MLXEngine.scrubControlTokens(firstTokenText)
        if !scrubbedPrefixToken.isEmpty {
            continuation.yield(Token(id: 0, text: scrubbedPrefixToken))
        }

        return ActiveStreamSequence(
            id: request.id,
            request: request,
            caches: restoredCaches,
            lastTokenId: firstTokenId,
            generatedText: firstTokenText,
            completionTokens: firstTokenText.isEmpty ? 0 : 1,
            maxTokens: maxTokens,
            sampler: sampler,
            continuation: continuation,
            modelId: request.model,
            promptTokenCount: promptTokenCount,
            promptTokenIds: allTokens,
            startTime: Date(),
            admittedAt: Date(),
            isFinished: false,
            recentTokenIds: Array(allTokens.suffix(20)),
            frequencyPenalty: freqPenaltyValue,
            accumulatedTokenIds: prefixAccumulatedIds,
            lastDecodedText: prefixDecodedText
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
        var iteration = 0
        while true {
            iteration += 1
            // 1. Admit queued requests (sorted by priority)
            await admitQueued()

            // Snapshot for logging
            let snapshot = lock.withLock { activeByModel.mapValues { $0.count } }
            if iteration <= 5 || iteration % 20 == 0 {
                NovaMLXLog.info("[RunLoop] iteration=\(iteration), activeByModel=\(snapshot)")
            }

            // 2. Run fused decode step
            let hadActive = await fusedDecodeStep()

            // 3. Check if we should continue
            let shouldContinue = lock.withLock {
                let hasActive = activeByModel.values.contains { !$0.isEmpty }
                let hasQueued = !streamQueue.isEmpty || !generateQueue.isEmpty
                if !hasActive && !hasQueued {
                    NovaMLXLog.info("[RunLoop] Exiting — no active/queued")
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

    /// Sort queued items by priority (high first), then by enqueue time (oldest first).
    private func prioritizedQueue<T>(
        _ items: [T],
        priority: (T) -> RequestPriority,
        enqueuedAt: (T) -> Date
    ) -> [T] {
        items.sorted { a, b in
            let pa = priority(a), pb = priority(b)
            if pa != pb { return pa > pb }
            return enqueuedAt(a) < enqueuedAt(b)
        }
    }

    private func admitQueued() async {
        // Admit generate queue (sorted by priority, then FIFO)
        var admittedGenerate: [FusedQueuedGenerateRequest] = []
        var remainingGenerate: [FusedQueuedGenerateRequest] = []

        let gQueue = lock.withLock { generateQueue }
        let sortedGenerate = prioritizedQueue(
            gQueue,
            priority: { $0.priority },
            enqueuedAt: { $0.enqueuedAt }
        )

        for item in sortedGenerate {
            let canStart = await canAdmit(item.request)
            if canStart {
                admittedGenerate.append(item)
            } else {
                // Try preemption for high-priority items
                if item.priority == .high {
                    let preempted = await preemptNewest(for: item.request.model, toMakeRoomFor: item.request.id)
                    if preempted {
                        // Re-check admission after preemption freed some memory
                        let retryCanAdmit = await canAdmit(item.request)
                        if retryCanAdmit {
                            admittedGenerate.append(item)
                            continue
                        }
                    }
                }
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

        // Admit stream queue (sorted by priority, then FIFO)
        var admittedStream: [FusedQueuedStreamRequest] = []
        var remainingStream: [FusedQueuedStreamRequest] = []

        let sQueue = lock.withLock { streamQueue }
        let sortedStream = prioritizedQueue(
            sQueue,
            priority: { $0.priority },
            enqueuedAt: { $0.enqueuedAt }
        )

        for item in sortedStream {
            let canStart = await canAdmit(item.request)
            if canStart {
                admittedStream.append(item)
            } else {
                // Try preemption for high-priority items
                if item.priority == .high {
                    let preempted = await preemptNewest(for: item.request.model, toMakeRoomFor: item.request.id)
                    if preempted {
                        let retryCanAdmit = await canAdmit(item.request)
                        if retryCanAdmit {
                            admittedStream.append(item)
                            continue
                        }
                    }
                }
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
                await self.budgetTracker.release(sequenceId: item.request.id)
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

            guard let firstCaches = active.first?.caches, !firstCaches.isEmpty else {
                lock.withLock { activeByModel[modelId] = active }
                continue
            }

            // Decode step: run forward pass + sample inside mlxContainer.perform.
            // model() must be called inside perform to ensure correct MLX lazy
            // evaluation context — calling outside causes NaN/Inf logits.
            let eosId = container.tokenizer?.eosTokenId

            // Run decode step per-sequence inside perform.
            // N-gram speculation: propose draft tokens, verify in one forward pass.

            let specDecoder = engine.specDecoder
            var specResults: [SpecDecodeResult] = []

            for seq in active {
                let draftTokens = specDecoder.speculate(context: seq.recentTokenIds)

                if draftTokens.isEmpty {
                    // No draft — standard single-token decode (zero overhead).
                    let token = MLXArray(Int32(seq.lastTokenId)).reshaped(1, 1)
                    let seqCaches = seq.caches
                    let seqSampler = seq.sampler
                    let recentIds = seq.recentTokenIds
                    let freqPen = seq.frequencyPenalty
                    let tidBox = await mlxContainer.perform(nonSendable: (token, seqCaches)) { context, values in
                        let (tokenInput, caches) = values
                        let logits = context.model(tokenInput, cache: caches)
                        eval(logits)
                        var lastLogits = logits[0..., -1, 0...]

                        // Apply frequency penalty to prevent repetition collapse
                        if freqPen > 0, !recentIds.isEmpty {
                            let vocabSize = lastLogits.dim(-1)
                            let ids = MLXArray(recentIds.map { Int32($0) })
                            let ones = MLXArray.ones([recentIds.count], type: Float32.self)
                            let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
                                .at[ids].add(ones)
                            lastLogits = lastLogits - (histogram * freqPen).reshaped(1, -1)
                        }

                        let sampled = seqSampler.sample(logits: lastLogits)
                        eval(sampled)
                        return SendableBox(sampled.item(Int.self))
                    }
                    let tid = tidBox.value
                    specResults.append(SpecDecodeResult(
                        acceptedTokens: [tid], draftAccepted: 0, draftProposed: 0, sequenceId: seq.id
                    ))
                } else {
                    // Speculative decode: [lastToken] + draftTokens in one forward pass.
                    let allTokens = [seq.lastTokenId] + draftTokens
                    let tokenInput = MLXArray(allTokens.map { Int32($0) }).reshaped(1, allTokens.count)
                    let seqCaches = seq.caches
                    let numDraft = draftTokens.count
                    let recentIds = seq.recentTokenIds
                    let freqPen = seq.frequencyPenalty

                    let verifyBox = await mlxContainer.perform(nonSendable: (tokenInput, seqCaches, numDraft)) { context, values in
                        let (input, caches, nDraft) = values
                        let logits = context.model(input, cache: caches)
                        eval(logits)

                        // Verify each draft position against the main model's greedy argmax.
                        var accepted = 0
                        var allAccepted: [Int] = []
                        for pos in 0..<(1 + nDraft) {
                            var posLogits = logits[0..., pos, 0...]
                            // Apply frequency penalty at position 0 (real decode position)
                            if pos == 0, freqPen > 0, !recentIds.isEmpty {
                                let vocabSize = posLogits.dim(-1)
                                let ids = MLXArray(recentIds.map { Int32($0) })
                                let ones = MLXArray.ones([recentIds.count], type: Float32.self)
                                let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
                                    .at[ids].add(ones)
                                posLogits = posLogits - (histogram * freqPen).reshaped(1, -1)
                            }
                            let mainToken = argMax(posLogits, axis: -1).item(Int.self)
                            if pos == 0 {
                                // Position 0: this is the "free" bonus token from the real last token.
                                allAccepted.append(mainToken)
                            } else {
                                // Compare main's greedy output vs the draft at this position.
                                if mainToken == draftTokens[pos - 1] {
                                    accepted += 1
                                    allAccepted.append(mainToken)
                                } else {
                                    // Rejection — keep the main model's token as bonus, stop here.
                                    allAccepted.append(mainToken)
                                    break
                                }
                            }
                        }

                        // If all drafts accepted, the bonus token is already the last in allAccepted.
                        // Trim KV cache to remove rejected positions.
                        let excessDraft = nDraft - accepted
                        if excessDraft > 0 {
                            for cache in caches {
                                let trimmed = cache.trim(excessDraft)
                                NovaMLXLog.debug("[Spec] Trimmed \(trimmed) excess KV entries from \(type(of: cache))")
                            }
                        }

                        return SendableBox((allAccepted, accepted))
                    }
                    let (acceptedTokens, draftAccepted) = verifyBox.value
                    specResults.append(SpecDecodeResult(
                        acceptedTokens: acceptedTokens,
                        draftAccepted: draftAccepted,
                        draftProposed: numDraft,
                        sequenceId: seq.id
                    ))
                }
            }

            // Apply spec decode results: yield accepted tokens per sequence.
            var updatedActive = active
            for (i, var seq) in updatedActive.enumerated() {
                guard i < specResults.count else { break }
                let result = specResults[i]

                // Record speculation stats and update N-gram cache.
                if result.draftProposed > 0 {
                    specDecoder.recordAccepted(tokens: result.acceptedTokens, accepted: result.draftAccepted)
                }

                // Yield all accepted tokens.
                var sequenceFinished = false
                for tokenId in result.acceptedTokens {
                    seq.lastTokenId = tokenId
                    seq.completionTokens += 1

                    // Update N-gram context buffer.
                    seq.recentTokenIds.append(tokenId)
                    if seq.recentTokenIds.count > 128 {
                        seq.recentTokenIds = Array(seq.recentTokenIds.suffix(64))
                    }

                    // Record into N-gram cache for future speculation.
                    specDecoder.recordToken(tokenId)

                    let decoded: String
                    if let tok = container.tokenizer, let delta = seq.decodeNextToken(tokenId, tokenizer: tok) {
                        decoded = MLXEngine.scrubControlTokens(delta)
                    } else {
                        decoded = ""
                    }
                    seq.generatedText = seq.lastDecodedText
                    if !decoded.isEmpty {
                        seq.continuation.yield(Token(id: 0, text: decoded))
                    }

                    lock.withLock { totalTokensViaFused += 1 }

                    let isEOS = eosId == tokenId
                    let atMax = seq.completionTokens >= seq.maxTokens

                    // Check custom stop sequences
                    var hitStop = false
                    if let stopSeqs = seq.request.stop {
                        for stopStr in stopSeqs where !stopStr.isEmpty {
                            if seq.generatedText.hasSuffix(stopStr) {
                                hitStop = true
                                break
                            }
                        }
                    }

                    if isEOS || atMax || hitStop {
                        let finishReason: FinishReason = atMax ? .length : .stop
                        seq.continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                        seq.continuation.finish()
                        seq.isFinished = true
                        sequenceFinished = true
                        break
                    }
                }

                if !sequenceFinished {
                    updatedActive[i] = seq
                } else {
                    updatedActive[i] = seq
                }
            }

            // Update state — use updatedActive (with new lastTokenIds, completionTokens, etc.)
            let finished = updatedActive.filter { $0.isFinished }
            let surviving = updatedActive.filter { !$0.isFinished }

            // CRITICAL: Write back ALL sequences (both surviving and finished).
            // Surviving sequences have updated lastTokenId, completionTokens, generatedText.
            // Without this write-back, the next runLoop iteration reads stale data and
            // sequences never reach their maxTokens limit.
            lock.withLock {
                let finishedIds = Set(finished.map { $0.id })
                if var current = activeByModel[modelId] {
                    // Remove finished
                    current.removeAll { finishedIds.contains($0.id) }
                    // Update surviving sequences with new state (lastTokenId, completionTokens, etc.)
                    let survivingMap = Dictionary(uniqueKeysWithValues: surviving.map { ($0.id, $0) })
                    for i in current.indices {
                        if let updated = survivingMap[current[i].id] {
                            current[i] = updated
                        }
                    }
                    activeByModel[modelId] = current
                }
            }

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

                // Store prefix cache blocks for this sequence's prompt
                if !seq.promptTokenIds.isEmpty {
                    if let prefixCache = engine.getOrCreatePrefixCacheManager(modelId: seq.modelId) {
                        prefixCache.storeCache(tokenIds: seq.promptTokenIds, cache: seq.caches)
                    }
                }
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

        // Report live TPS to MetricsStore so the status chart updates during generation
        let tokensNow = lock.withLock { totalTokensViaFused }
        let tokensThisStep = tokensNow - lastDecodeStepTokens
        let now = Date()
        let stepElapsed = now.timeIntervalSince(lastDecodeStepTime)
        if tokensThisStep > 0 && stepElapsed > 0.01 {
            engine.metricsStore.updateLiveTps(Double(tokensThisStep) / stepElapsed)
        }
        lastDecodeStepTime = now
        lastDecodeStepTokens = tokensNow

        return anyActive
    }

    // MARK: - Queue Helpers

    private func enqueueGenerateAndWait(
        request: InferenceRequest,
        priority: RequestPriority
    ) async throws -> InferenceResult {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock {
                generateQueue.append(FusedQueuedGenerateRequest(
                    request: request, enqueuedAt: Date(), priority: priority,
                    continuation: continuation
                ))
            }
            NovaMLXLog.info("FusedScheduler: queued generate \(request.id.uuidString.prefix(8)), depth=\(queueDepth), priority=\(priority)")
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
    public let totalPreemptions: UInt64
    public let specAcceptanceRate: Double
    public let specCacheSize: Int

    public init(
        activeSequences: Int = 0,
        queueDepth: Int = 0,
        totalDecodeSteps: UInt64 = 0,
        totalTokensViaFused: UInt64 = 0,
        peakBatchWidth: Int = 0,
        totalPreemptions: UInt64 = 0,
        specAcceptanceRate: Double = 0,
        specCacheSize: Int = 0
    ) {
        self.activeSequences = activeSequences
        self.queueDepth = queueDepth
        self.totalDecodeSteps = totalDecodeSteps
        self.totalTokensViaFused = totalTokensViaFused
        self.peakBatchWidth = peakBatchWidth
        self.totalPreemptions = totalPreemptions
        self.specAcceptanceRate = specAcceptanceRate
        self.specCacheSize = specCacheSize
    }
}
