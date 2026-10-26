import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
@preconcurrency import MLXNN
import NovaMLXCore
import NovaMLXUtils

// MARK: - Non-Sendable Wrappers

/// Wrapper to pass non-Sendable `[KVCache]` across isolation boundaries.
/// Safe because `mlxContainer.perform` guarantees serial access.
final class FusedCachesBox: @unchecked Sendable {
    let caches: [KVCache]
    init(caches: [KVCache]) { self.caches = caches }
}

/// One fused decode launch: a Flint resident packet, or one host sample.
private enum FusedDecodeHit: Sendable {
    case packet([Int])
    case single(Int, RawLogprobData?)
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
    /// Logprob data for the first accepted token (the real decode token).
    let logprobs: (logprob: Float, topLogprobs: [TopLogprob])?
    /// Flint greedy resident packet. Argmax ran on device for every id.
    var fromResidentPacket: Bool = false
}

/// Sampler configuration passed into perform closure.
struct SamplerConfig: Sendable {
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
}

/// Raw logprob data extracted inside `perform` — token texts resolved outside.
struct RawLogprobData: Sendable {
    let logprob: Float
    let topLogprobs: [(tokenId: Int, logprob: Float)]
}

// MARK: - FinishGuard

/// Reference-type finish guard for `ActiveStreamSequence`. Because
/// `ActiveStreamSequence` is a value type, writes to its `isFinished` field
/// through one copy are not observable through other copies — but every
/// copy holds the same `AsyncThrowingStream.Continuation` reference, so
/// without a shared guard, two concurrent code paths can both call
/// `continuation.finish()` and double-finish (use-after-free / SIGSEGV).
///
/// `tryMarkFinished()` is the only safe way to call `continuation.finish()`:
/// it returns `true` exactly once across all copies.
public final class FinishGuard: @unchecked Sendable {
    private var _lock = os_unfair_lock()
    private var _done = false

    public init() {}

    @discardableResult
    public func tryMarkFinished() -> Bool {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        if _done { return false }
        _done = true
        return true
    }

    public var isDone: Bool {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        return _done
    }
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
    /// Shared across all copies (reference type). Authoritative for stream-side
    /// finish — `isFinished` is advisory for fast reads, `finishGuard` is the
    /// truth for continuation lifecycle and budget release ownership.
    let finishGuard: FinishGuard
    /// True only when *this* decode step was the one that successfully
    /// called `safeFinish()` — prevents double budget release when
    /// preempt/abort already released the sequence.
    var _finishedByDecodeStep: Bool = false
    /// Recent token IDs for N-gram speculation context.
    var recentTokenIds: [Int]
    /// Frequency penalty value — prevents repetition collapse in small quantized models.
    /// Applied inline via scatter_add before sampling. 0 = disabled.
    let frequencyPenalty: Float

    // Accumulated batch decode — avoids per-token decode that breaks SentencePiece spaces
    var accumulatedTokenIds: [Int] = []
    var lastDecodedText: String = ""

    /// Reference box: the sequence struct is copied across the decode loop
    /// and prefill, so tool-parsing state must be shared, not value-copied.
    /// Nil when the model declares no tool-call format — identical to the
    /// pre-tool-parsing behavior.
    final class ToolStreamState: @unchecked Sendable {
        let processor: ToolCallProcessor
        /// Flint fused path. Types arguments from the tool schema and
        /// derives call ids from the seed (`rec` `chat.py`). Nil keeps the
        /// stock processor for every other family.
        var flintExtractor: FlintToolCallExtractor?
        var emittedToolCall = false
        init(processor: ToolCallProcessor) {
            self.processor = processor
        }

        enum Piece {
            case text(String)
            case call(ToolCallResult)
        }

        func feedPieces(_ text: String) -> [Piece] {
            if let flintExtractor {
                return flintExtractor.feed(text).map(Self.piece)
            }
            var pieces: [Piece] = []
            for out in processor.processChunkOutputs(text) {
                switch out {
                case .response(let text):
                    pieces.append(.text(text))
                case .toolCall(let tc):
                    pieces.append(.call(ActiveStreamSequence.toolCallResult(tc)))
                case .rejectedToolCall:
                    break
                }
            }
            return pieces
        }

        func finishPieces() -> [Piece] {
            if let flintExtractor {
                return flintExtractor.finish().map(Self.piece)
            }
            var pieces: [Piece] = []
            for out in processor.processEOSOutputs() {
                switch out {
                case .response(let text):
                    pieces.append(.text(text))
                case .toolCall(let tc):
                    pieces.append(.call(ActiveStreamSequence.toolCallResult(tc)))
                case .rejectedToolCall:
                    break
                }
            }
            return pieces
        }

        private static func piece(_ event: FlintToolCallExtractor.Event) -> Piece {
            switch event {
            case .response(let text):
                return .text(text)
            case .tool(let name, let arguments, let id):
                return .call(ToolCallResult(id: id, functionName: name, arguments: arguments))
            }
        }
    }

    var toolStream: ToolStreamState? = nil

    /// Wall clock for decode steps only. Nil until the first decode launch.
    var decodeWindowStart: Date? = nil
    /// Tokens emitted by decode steps, excluding the prefill-sampled token and EOS.
    var decodeTokens: Int = 0
    var packetTokens: Int = 0
    var singleTokens: Int = 0
    var packetLaunches: Int = 0

    /// Route one decoded text chunk through the tool parser and yield response
    /// text / tool calls. Falls through untouched when no state is attached.
    /// `logprob` attaches to the first response output.
    func yieldProcessed(_ decoded: String, logprob: (Float, [TopLogprob])? = nil) {
        guard let state = toolStream else {
            if !decoded.isEmpty {
                safeYield(Token(id: 0, text: decoded, logprob: logprob?.0, topLogprobs: logprob?.1))
            }
            return
        }
        var first = true
        for piece in state.feedPieces(decoded) {
            switch piece {
            case .text(let text):
                if !text.isEmpty {
                    safeYield(Token(id: 0, text: text,
                        logprob: first ? logprob?.0 : nil,
                        topLogprobs: first ? logprob?.1 : nil))
                }
            case .call(let call):
                state.emittedToolCall = true
                safeYield(Token(id: 0, text: "", toolCall: call))
            }
            first = false
        }
    }

    /// Flush buffered text / pending calls at end of stream.
    func drainToolProcessor() {
        guard let state = toolStream else { return }
        for piece in state.finishPieces() {
            switch piece {
            case .text(let text):
                if !text.isEmpty { safeYield(Token(id: 0, text: text)) }
            case .call(let call):
                state.emittedToolCall = true
                safeYield(Token(id: 0, text: "", toolCall: call))
            }
        }
    }

    /// Mirror of MLXEngine's iterator mapping: mlx-lm ToolCall → NovaMLX
    /// ToolCallResult with stable id and JSON-string arguments.
    static func toolCallResult(_ tc: MLXLMCommon.ToolCall) -> ToolCallResult {
        let argsData = try? JSONSerialization.data(
            withJSONObject: tc.function.arguments.mapValues { $0.anyValue })
        return ToolCallResult(
            id: "call_\(UUID().uuidString.prefix(8).lowercased())",
            functionName: tc.function.name,
            arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        )
    }

    /// Harmony channel state: tracks whether the previous token was a scrubbed control token.
    /// When the model outputs <|channel|>final<|message|>, these arrive as 3 separate tokens.
    /// <|channel|> is scrubbed to "", "final" passes through, <|message|> is scrubbed to "".
    /// This flag lets us suppress "final"/"analysis" when they follow a control token.
    var previousTokenWasControl: Bool = false

    /// Harmony channel type seen between a `<|channel|>` and `<|message|>`. When a
    /// channel-type residue word ("analysis" / "final" / "commentary") arrives
    /// immediately after the scrubbed `<|channel|>`, we remember it so the
    /// subsequent scrubbed `<|message|>` can emit the appropriate `<think>` /
    /// `</think>` boundary instead of an empty string. This is the streaming
    /// counterpart of `MLXEngine.scrubControlTokens`'s text-level Harmony
    /// normalization — necessary because in streaming the 3-token boundary
    /// (`<|channel|>`, channel-type, `<|message|>`) is split across deltas.
    var pendingHarmonyChannel: String? = nil

    /// Tracks whether we have emitted an opening `<think>` tag without yet
    /// emitting `</think>`. Guards against malformed output if the model
    /// emits `<|channel|>final<|message|>` without a preceding analysis
    /// channel: in that case we strip the boundary instead of emitting a
    /// stray closing `</think>`.
    var harmonyInThinking: Bool = false

    /// Flint only. Stops a turn that already showed an answer and then opens
    /// another `<think>` when structural grammar is off. Other families leave
    /// this at the default.
    var flintThink = FlintChatDecode.ThinkReentryGuard()
    /// Flint chat/agent grammar. Nil when grammar is off; the think guard is
    /// the fallback in that case. The mask is applied before each sample.
    var flintGrammar: FlintStructuralGrammar? = nil
    /// Generated token ids penalized on the Flint filtered packet.
    /// Official presence is a 0/1 mask over sampled tokens, not the prompt.
    /// Empty when `flintPresencePenalty` is 0.
    var flintPresenceSeen: Set<Int> = []
    var flintPresencePenalty: Float = 0

    /// Yield a token to the stream IFF the sequence has not been finished
    /// by any concurrent path. Silent no-op if already finished.
    func safeYield(_ token: Token) {
        guard !finishGuard.isDone else { return }
        continuation.yield(token)
    }

    /// Atomically transition the sequence to "finished" exactly once
    /// across all copies, then call `continuation.finish(...)`.
    /// Returns `true` if this call performed the finish (caller should
    /// release budget, record metrics); `false` if another path already
    /// finished it (caller should skip cleanup).
    @discardableResult
    func safeFinish(throwing error: Error? = nil) -> Bool {
        guard finishGuard.tryMarkFinished() else { return false }
        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
        return true
    }

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
    private let maxConcurrentPerModel: Int  // Safety bound; GPU/RAM sliders are the real limit
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
    /// Token-level Flint prefix (`prefix.py`). Not the block-hash SSD cache.
    private var flintPrefixes: [String: FlintPrefixSession] = [:]
    private var flintKVTokens: [String: [Int]] = [:]
    private var flintKV: [String: [KVCache]] = [:]
    private var flintMetrics: [UUID: (cached: Int, common: Int)] = [:]

    /// `prepare_prefix` after a finished Flint generation. Short outputs never
    /// schedule. A cancel inside a chunk leaves the last committed cache.
    /// Decode KV is not stored here; the next prefill reuses the prompt snapshot.
    private func noteFlintPreparedPrefix(modelId: String, prompt: [Int], generated: [Int]) {
        guard var session = flintPrefixes[modelId] else { return }
        if generated.count >= session.chunkSize {
            session.schedulePrefix(ids: prompt, generated: generated)
        }
        guard let pending = session.preparation else {
            flintPrefixes[modelId] = session
            return
        }
        let full = pending.tokens
        _ = session.preparePrefix(shouldStop: { false }) { prefix, target in
            try FlintPrefixSession.advanceChunk(
                prefix: prefix, target: target, tokens: full, cancel: false)
        }
        flintPrefixes[modelId] = session
    }

    // MARK: - Init

    public init(engine: MLXEngine, maxConcurrentPerModel: Int = ResourceLimits.safetyConcurrentCap) {
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

    /// Concurrent sequences that fit in the remaining GPU KV budget.
    /// Safety-capped only — GPU/RAM sliders are the real limit.
    private func optimalConcurrency(for modelId: String) async -> Int {
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let usable = await budgetTracker.availableKVBudget

        if engine.getContainer(for: modelId)?.config.hasLinearAttention == true {
            return 1
        }

        let estimatedTokensPerSeq = 4096
        let bytesPerSeq = UInt64(estimatedTokensPerSeq) * UInt64(bytesPerToken)

        guard bytesPerSeq > 0 else { return maxConcurrentPerModel }

        let fitCount = Int(usable / bytesPerSeq)
        return max(1, min(fitCount, maxConcurrentPerModel))
    }

    // MARK: - Submit (non-streaming)

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        let modelId = request.model

        // Atomic admission check + slot reserve — if GPU/RAM is tight, queue
        guard await canAdmitAndReserve(request) else {
            NovaMLXLog.info("FusedScheduler: submit(\(request.id.uuidString.prefix(8))) can't admit — queuing")
            return try await enqueueGenerateAndWait(request: request, priority: .normal)
        }

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
                            assert((self.activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow in EOS-first-token path")
                        }
                        await self.budgetTracker.release(sequenceId: request.id)
                    }
                } catch {
                    // SAFE: pre-prefill error path, continuation is exclusively owned
                    continuation.finish(throwing: error)
                    NovaMLXLog.error("[Fused:\(request.id.uuidString.prefix(8))] prefill error — model=\(modelId): \(error) — \(type(of: error))")
                    self.lock.withLock {
                        self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                        assert((self.activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow in submit() prefill error path")
                    }
                    await self.budgetTracker.release(sequenceId: request.id)
                }
            }
        }

        var text = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var collectedToolCalls: [ToolCallResult] = []

        for try await token in stream {
            if let fr = token.finishReason { finishReason = fr }
            if let tc = token.toolCall { collectedToolCalls.append(tc) }
            if !token.text.isEmpty { text += token.text; completionTokens += 1 }
        }

        // Scrub any leaked control tokens from the final text
        let controlPatterns = MLXEngine.controlTokensForModel(modelId: modelId)
        if !controlPatterns.isEmpty {
            text = MLXEngine.scrubControlTokens(
                MLXEngine.trimControlTokens(text, patterns: controlPatterns)
            )
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        let prefixMetrics = flintMetrics.removeValue(forKey: request.id)
        return InferenceResult(
            id: request.id, model: modelId, text: text,
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
            tokensPerSecond: tps, promptTokens: promptTokensBox.value,
            completionTokens: completionTokens, finishReason: finishReason,
            cachedTokens: prefixMetrics?.cached ?? 0,
            commonPrefixTokens: prefixMetrics?.common ?? 0
        )
    }

    // MARK: - Submit Stream

    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let modelId = request.model
        let reqTag = request.id.uuidString.prefix(8)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Atomic admission check + slot reserve
                    let canStart = await self.canAdmitAndReserve(request)
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
                            assert((self.activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow in submitStream EOS path")
                        }
                        await self.budgetTracker.release(sequenceId: request.id)
                    }
                } catch {
                    // SAFE: pre-prefill error path, continuation is exclusively owned
                    continuation.finish(throwing: error)
                    NovaMLXLog.error("[Fused:\(request.id.uuidString.prefix(8))] submitStream prefill error — model=\(modelId): \(error) — \(type(of: error))")
                    self.lock.withLock {
                        self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                        assert((self.activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow in submitStream error path")
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
                    if !seq.isFinished {
                        self.lock.withLock {
                            if self.activeByModel[modelId] == nil { self.activeByModel[modelId] = [] }
                            self.activeByModel[modelId]?.append(seq)
                        }
                        self.scheduleRunLoop()
                    } else {
                        self.lock.withLock {
                            self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                        }
                        await self.budgetTracker.release(sequenceId: request.id)
                    }
                } catch {
                    // SAFE: pre-prefill error path, continuation is exclusively owned
                    continuation.finish(throwing: error)
                    NovaMLXLog.error("[Fused:\(request.id.uuidString.prefix(8))] internalSubmitStream prefill error — model=\(modelId): \(error) — \(type(of: error))")
                    self.lock.withLock {
                        self.activeModelCounts[modelId] = max(0, (self.activeModelCounts[modelId] ?? 1) - 1)
                        assert((self.activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow in internalSubmitStream error path")
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

    /// Atomically check admission AND reserve the concurrency slot.
    /// Returns `false` if the concurrency limit is reached or memory budget insufficient.
    /// On `true`, the caller MUST proceed — the slot is already counted.
    private func canAdmitAndReserve(_ request: InferenceRequest) async -> Bool {
        let modelId = request.model
        let concurrentLimit = await optimalConcurrency(for: modelId)

        // Atomically check limit AND reserve slot
        let reserved = lock.withLock { () -> Bool in
            let current = activeModelCounts[modelId] ?? 0
            guard current < concurrentLimit else { return false }
            activeModelCounts[modelId] = current + 1
            return true
        }
        guard reserved else {
            NovaMLXLog.info("FusedScheduler: queuing \(request.id.uuidString.prefix(8)) — model at concurrency limit")
            return false
        }

        // Check memory budget — roll back slot if insufficient
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)
        let canAdmitMemory = await budgetTracker.canAdmit(
            modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        guard canAdmitMemory else {
            // Roll back the slot we just reserved
            lock.withLock { activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1) }
            NovaMLXLog.info("FusedScheduler: queuing \(request.id.uuidString.prefix(8)) — insufficient memory")
            return false
        }

        // Reserve memory budget (slot already counted above)
        await budgetTracker.reserve(
            modelId: modelId, sequenceId: request.id,
            weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )
        return true
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
        NovaMLXLog.debug("[BudgetRelease] seq=\(victim.id.uuidString.prefix(8)) model=\(modelId) path=preempt")
        lock.withLock {
            activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
            assert((activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow after preemption release")
            totalPreemptions += 1
        }

        // Finish the victim's stream with a retryable error
        victim.safeFinish(throwing: NovaMLXError.inferenceFailed(
            "Sequence preempted due to memory pressure — please retry"
        ))

        NovaMLXLog.warning("FusedScheduler: PREEMPTED sequence \(victim.id.uuidString.prefix(8)) (model=\(modelId), \(victim.completionTokens) tokens generated) to make room for \(requestId.uuidString.prefix(8))")

        return true
    }

    // MARK: - Prefill


    /// Build the per-sequence tool-call processor from the model's declared
    /// dialect (container configuration) and the request's tool specs. Nil for
    /// models with no format — identical to the pre-tool-parsing behavior.
    nonisolated static func makeToolProcessor(
        mlxContainer: MLXLMCommon.ModelContainer,
        request: InferenceRequest
    ) async -> ActiveStreamSequence.ToolStreamState? {
        guard let format = await mlxContainer.configuration.toolCallFormat else { return nil }
        let toolSpecs: [[String: any Sendable]]? = request.tools?.map { tool in
            tool.mapValues { value in
                Self.anyToSendable(value)
            }
        }
        return ActiveStreamSequence.ToolStreamState(
            processor: ToolCallProcessor(format: format, tools: toolSpecs))
    }

    private nonisolated static func anyToSendable(_ value: Any) -> any Sendable {
        switch value {
        case let s as String: return s
        case let n as NSNumber: return n
        case let b as Bool: return b
        case let a as [Any]: return a.map(anyToSendable)
        case let d as [String: Any]:
            return d.mapValues(anyToSendable)
        case let s as any Sendable: return s
        default: return String(describing: value)
        }
    }

    /// Pre-shared-state variant of `ActiveStreamSequence.yieldProcessed` for the
    /// first-token yields in prefill (sequence not constructed yet).
    nonisolated static func emitProcessed(
        _ state: ActiveStreamSequence.ToolStreamState?,
        _ text: String,
        continuation: AsyncThrowingStream<Token, Error>.Continuation,
        logprob: (Float, [TopLogprob])? = nil
    ) {
        guard let state else {
            if !text.isEmpty {
                continuation.yield(Token(id: 0, text: text, logprob: logprob?.0, topLogprobs: logprob?.1))
            }
            return
        }
        var first = true
        for piece in state.feedPieces(text) {
            switch piece {
            case .text(let t):
                if !t.isEmpty {
                    continuation.yield(Token(id: 0, text: t,
                        logprob: first ? logprob?.0 : nil,
                        topLogprobs: first ? logprob?.1 : nil))
                }
            case .call(let call):
                state.emittedToolCall = true
                continuation.yield(Token(id: 0, text: "", toolCall: call))
            }
            first = false
        }
    }

    /// Flush a prefill-finished stream. The sequence is not entered, so the
    /// instance `drainToolProcessor` never runs.
    nonisolated static func drainProcessed(
        _ state: ActiveStreamSequence.ToolStreamState?,
        continuation: AsyncThrowingStream<Token, Error>.Continuation
    ) {
        guard let state else { return }
        for piece in state.finishPieces() {
            switch piece {
            case .text(let text):
                if !text.isEmpty { continuation.yield(Token(id: 0, text: text)) }
            case .call(let call):
                state.emittedToolCall = true
                continuation.yield(Token(id: 0, text: "", toolCall: call))
            }
        }
    }

    /// Install the `rec` tool parser on a Flint stream. Other families keep
    /// the stock processor. An omitted Flint seed is 0, and that seed is the
    /// one `seeded_tool_parser` hashes. A nil seed is only for non-Flint.
    nonisolated static func armFlintExtractor(
        _ state: ActiveStreamSequence.ToolStreamState?,
        family: ModelFamily,
        request: InferenceRequest,
        config: ModelConfig,
        promptTokens: [Int],
        maxTokens: Int
    ) {
        guard family == .flint, let state else { return }
        let identity = FlintAlignment.CallIdentity(
            seed: family == .flint
                ? FlintAlignment.resolvedSeed(requestSeed: request.seed, modelSeed: config.seed)
                : request.seed ?? config.seed,
            temperature: request.temperature ?? config.temperature,
            topP: request.topP ?? config.topP,
            topK: request.topK ?? config.topK,
            presencePenalty: Double(request.presencePenalty ?? config.presencePenalty),
            maxTokens: maxTokens,
            promptTokens: promptTokens
        )
        state.flintExtractor = FlintToolCallExtractor(tools: request.tools, identity: identity)
    }

    private func prefillSequence(
        request: InferenceRequest,
        continuation: AsyncThrowingStream<Token, Error>.Continuation
    ) async throws -> ActiveStreamSequence {
        guard let container = engine.getContainer(for: request.model),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(request.model)
        }

        let toolStream = await Self.makeToolProcessor(mlxContainer: mlxContainer, request: request)
        let hasImages = request.messages.contains { $0.images?.isEmpty == false }
        let userInput = try await engine.buildUserInput(request: request, hasImages: hasImages)
        let input = try await mlxContainer.prepare(input: userInput)
        let promptTokenCount = input.text.tokens.size
        let maxTokens = try engine.outputTokenBudget(
            request: request, config: container.config, promptTokens: Int(promptTokenCount))

        // Context window check with accurate token count (includes chat template)
        let settings = engine.settingsProvider?(request.model)
        let contextLength = settings?.maxContextWindow
            ?? container.config.contextLength
        if Int(promptTokenCount) + maxTokens > contextLength {
            let reqTag = request.id.uuidString.prefix(8)
            NovaMLXLog.warning("[ContextWindow:\(reqTag)] Prompt (\(promptTokenCount) tokens) + maxTokens (\(maxTokens)) exceeds context length (\(contextLength))")
            throw NovaMLXError.contextWindowExceeded(
                promptTokens: Int(promptTokenCount), maxTokens: maxTokens, contextLength: contextLength
            )
        }

        try await engine.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        let allTokens = input.text.tokens.asArray(Int32.self).map { Int($0) }

        let parameters = engine.buildGenerateParameters(
            request: request, config: container.config,
            settings: engine.settingsProvider?(request.model),
            outputBudget: maxTokens
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
        // Flint uses the session prefix below. The block-hash SSD cache is a
        // different mechanism and must not stand in for `select_prefix`.
        let flintFamily = container.identifier.family == .flint
        let prefixCacheManager = flintFamily
            ? nil
            : engine.getOrCreatePrefixCacheManager(modelId: request.model)
        let prefixResult = prefixCacheManager?.fetchPrefix(tokenIds: allTokens)

        var flintTrial = flintPrefixes[request.model] ?? FlintPrefixSession()
        var flintSelection = FlintPrefixEngine.Selection(cache: nil, common: 0, cached: 0)
        if flintFamily {
            let taken = flintTrial.takePrefix(ids: allTokens, trimmable: true)
            flintSelection = FlintPrefixEngine.select(prefix: taken, tokens: allTokens, trimmable: true)
            flintTrial.noteSelection(flintSelection)
            let line = FlintPrefixEngine.prefillLog(
                prompt: allTokens.count,
                common: flintSelection.common,
                reused: flintSelection.cached)
            NovaMLXLog.info("[Flint] " + line)
            flintMetrics[request.id] = (flintSelection.cached, flintSelection.common)
            if flintSelection.cached > 0,
               let snapTokens = flintKVTokens[request.model],
               snapTokens.count >= flintSelection.cached,
               let stored = flintKV[request.model]
            {
                var copies = stored.map { $0.copy() }
                let drop = snapTokens.count - flintSelection.cached
                let trimmed = drop == 0 || copies.allSatisfy { $0.trim(drop) >= drop }
                if trimmed && flintSelection.cached < allTokens.count {
                    let remaining = Array(allTokens.dropFirst(flintSelection.cached))
                    let seq = try await prefillRemaining(
                        request: request,
                        continuation: continuation,
                        container: container,
                        mlxContainer: mlxContainer,
                        parameters: parameters,
                        restoredCaches: copies,
                        remainingTokenIds: remaining,
                        cachedTokenCount: flintSelection.cached,
                        allTokens: allTokens,
                        promptTokenCount: Int(promptTokenCount),
                        maxTokens: maxTokens)
                    flintTrial.commit(tokens: allTokens)
                    flintPrefixes[request.model] = flintTrial
                    flintKVTokens[request.model] = allTokens
                    flintKV[request.model] = seq.caches.map { $0.copy() }
                    return seq
                }
            }
        }

        if let prefixResult = prefixResult,
           let restoredCaches = prefixResult.cache,
           prefixResult.cachedTokenCount > 0,
           !prefixResult.remainingTokenIds.isEmpty {

            // Validate restored caches are compatible with the current model
            // Check layer count matches and caches are non-empty
            let freshCachesBox = try await mlxContainer.perform { context in
                FusedCachesBox(caches: try context.model.newCache(parameters: parameters))
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

        let chunkSize = prefillParams.prefill.resolvedStepSize()
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
        let flintGrammarSeed = FlintChatDecode.withVocabTexts(
            FlintChatDecode.chatGrammar(for: request, family: container.identifier.family),
            decodeTokens: container.tokenizer?.decode)
        let applyFlintSampleFilter = container.identifier.family == .flint
        let flintPrefillTemperature = Float(request.temperature ?? container.config.temperature)
        let flintPrefillTopK = request.topK ?? container.config.topK

        // Run entire prefill + first token sampling inside perform to ensure
        // correct MLX lazy evaluation context.
        let cachesBox = MutableSendableBox<[KVCache]?>(nil)
        let prefillResultBox = try await mlxContainer.perform { context in
            let model = context.model
            let caches = try model.newCache(parameters: prefillParams)

            // Chunked forward pass with timeout checks between chunks
            var lastLogits: MLXArray = MLXArray()
            var offset = 0
            let prefillStart = Date()

            while offset < tokenIds.count {
                let elapsed = Date().timeIntervalSince(prefillStart)
                if elapsed > prefillTimeoutSeconds {
                    NovaMLXLog.error("[Prefill:\(reqTag)] Timeout after \(String(format: "%.1f", elapsed))s at chunk offset \(offset)/\(tokenIds.count)")
                    return SendableBox((-1, "Timeout after \(Int(elapsed))s (\(offset)/\(tokenIds.count) tokens processed)", nil as RawLogprobData?))
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
            var logitsToSample = lastLogits[0..., -1, 0...]
            let logitsF32 = logitsToSample.asType(.float32)
            let maxLogit = logitsF32.max().item(Float.self)
            let minLogit = logitsF32.min().item(Float.self)
            if maxLogit.isNaN || maxLogit.isInfinite || minLogit.isNaN || minLogit.isInfinite {
                NovaMLXLog.error("[Prefill:\(reqTag)] Logits contain NaN/Inf (max=\(maxLogit), min=\(minLogit)) — sequence too long or model unstable")
                return SendableBox((-1, "NaN/Inf", nil as RawLogprobData?))
            }
            if applyFlintSampleFilter {
                // Same mask as the filtered packet: grammar, then top-k when
                // temperature is not 0. Presence has no generated tokens yet.
                logitsToSample = FlintChatDecode.filterLogits(
                    logitsToSample,
                    grammar: flintGrammarSeed,
                    presenceSeen: [],
                    presencePenalty: 0,
                    temperature: flintPrefillTemperature,
                    topK: flintPrefillTopK)
            }
            let stats = "max=\(String(format: "%.2f", maxLogit)), min=\(String(format: "%.2f", minLogit)), shape=\(logitsToSample.shape), dtype=\(logitsToSample.dtype)"
            let prefillSampled = sampler.sample(logits: logitsToSample)
            eval(prefillSampled)
            let sampledId = prefillSampled.item(Int.self)

            let rawLogprobs: RawLogprobData? = request.includeLogprobs
                ? Self.computeRawLogprobs(from: logitsToSample, sampledTokenId: sampledId, topK: request.topLogprobsCount ?? 0)
                : nil

            cachesBox.value = caches
            return SendableBox((sampledId, stats, rawLogprobs))
        }
        let (firstTokenId, logitStats, prefillLogprobs) = prefillResultBox.value
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
        if flintFamily {
            flintTrial.commit(tokens: allTokens)
            flintPrefixes[request.model] = flintTrial
            flintKVTokens[request.model] = allTokens
            flintKV[request.model] = caches.map { $0.copy() }
        }

        // Check if first token is EOS. Flint role headers are stops even
        // when they are not the tokenizer's single eosTokenId.
        let eosId = container.tokenizer?.eosTokenId
        let isFlint = container.identifier.family == .flint
        let flintStopIds: Set<Int> = isFlint
            ? FlintChatDecode.generationFinishStopIds(
                grammarOn: flintGrammarSeed != nil,
                checkpoint: container.flintFinishStopIds,
                tokenizerEOS: eosId)
            : []

        // Resolve logprob data (token IDs → text) outside perform
        let resolvedLogprobs: (logprob: Float, topLogprobs: [TopLogprob])? = prefillLogprobs.map {
            Self.resolveLogprobs($0, tokenizer: container.tokenizer)
        }

        // Yield first token to client (unless it's EOS and we finish immediately)
        // SAFE: pre-shared-state yields — seq hasn't entered activeByModel yet
        let scrubbedFirstToken = MLXEngine.scrubControlTokens(firstTokenText)
        var flintThink = FlintChatDecode.ThinkReentryGuard()
        var flintGrammar = flintGrammarSeed
        let firstDecision: FlintChatDecode.TokenDecision
        if isFlint {
            firstDecision = FlintChatDecode.openingDecision(
                tokenId: firstTokenId, yieldText: scrubbedFirstToken,
                grammarText: firstTokenText, primaryEOS: eosId, stopIds: flintStopIds,
                grammar: &flintGrammar, guardState: &flintThink)
        } else {
            let eos = FlintChatDecode.isStopToken(firstTokenId, primaryEOS: eosId, stopIds: flintStopIds)
            firstDecision = FlintChatDecode.TokenDecision(
                yield: eos ? "" : scrubbedFirstToken, isEOS: eos, thinkReentry: false)
        }
        let finishFirst = firstDecision.isEOS || firstDecision.thinkReentry
        Self.armFlintExtractor(
            toolStream, family: container.identifier.family, request: request,
            config: container.config, promptTokens: allTokens, maxTokens: maxTokens)
        if !firstDecision.yield.isEmpty {
            Self.emitProcessed(toolStream, firstDecision.yield,
                continuation: continuation,
                logprob: resolvedLogprobs.map { ($0.logprob, $0.topLogprobs) })
        }

        let completionCount: Int
        let isFinished: Bool
        if finishFirst {
            Self.drainProcessed(toolStream, continuation: continuation)
            continuation.yield(Token(
                id: 0, text: "", finishReason: .stop, promptTokens: promptTokenCount,
                cachedTokens: flintSelection.cached))
            continuation.finish()
            completionCount = firstDecision.yield.isEmpty ? 0 : 1
            isFinished = true
        } else {
            completionCount = firstTokenText.isEmpty ? 0 : 1
            isFinished = false
        }

        var seq = ActiveStreamSequence(
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
            finishGuard: FinishGuard(),
            recentTokenIds: Array(allTokens.suffix(20)),
            frequencyPenalty: freqPenaltyValue,
            accumulatedTokenIds: initialAccumulatedIds,
            lastDecodedText: initialDecodedText,
            previousTokenWasControl: scrubbedFirstToken.isEmpty && !firstTokenText.isEmpty
        )
        seq.toolStream = toolStream
        seq.flintThink = flintThink
        seq.flintGrammar = flintGrammar
        seq.flintPresencePenalty = request.presencePenalty ?? container.config.presencePenalty
        if seq.flintPresencePenalty != 0, firstTokenId >= 0 {
            seq.flintPresenceSeen.insert(firstTokenId)
        }
        return seq
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
        let flintGrammarSeed = FlintChatDecode.withVocabTexts(
            FlintChatDecode.chatGrammar(for: request, family: container.identifier.family),
            decodeTokens: container.tokenizer?.decode)
        let applyFlintSampleFilter = container.identifier.family == .flint
        let flintPrefillTemperature = Float(request.temperature ?? container.config.temperature)
        let flintPrefillTopK = request.topK ?? container.config.topK
        let reqTag = request.id.uuidString.prefix(8)
        let remainingArray = MLXArray(remainingTokenIds.map { Int32($0) })
        let firstTokenIdBox = await mlxContainer.perform(nonSendable: (remainingArray, restoredCaches)) { context, values in
            let (remaining, caches) = values
            let logits = context.model(remaining.reshaped(1, remainingTokenIds.count), cache: caches)
            eval(logits)
            var logitsToSample = logits[0..., -1, 0...]
            // NaN/inf check — cast to float32 to avoid false positives from
            // float16 sum overflow (32K+ vocab logits easily exceed float16 max).
            let logitsF32 = logitsToSample.asType(.float32)
            let maxLogit = logitsF32.max().item(Float.self)
            let minLogit = logitsF32.min().item(Float.self)
            if maxLogit.isNaN || maxLogit.isInfinite || minLogit.isNaN || minLogit.isInfinite {
                NovaMLXLog.error("[Prefill:\(reqTag)] Remaining prefill logits contain NaN/Inf (max=\(maxLogit), min=\(minLogit))")
                return SendableBox((-1, nil as RawLogprobData?))
            }
            if applyFlintSampleFilter {
                logitsToSample = FlintChatDecode.filterLogits(
                    logitsToSample,
                    grammar: flintGrammarSeed,
                    presenceSeen: [],
                    presencePenalty: 0,
                    temperature: flintPrefillTemperature,
                    topK: flintPrefillTopK)
            }
            let sampled = sampler.sample(logits: logitsToSample)
            eval(sampled)
            let sampledId = sampled.item(Int.self)

            let rawLogprobs: RawLogprobData? = request.includeLogprobs
                ? Self.computeRawLogprobs(from: logitsToSample, sampledTokenId: sampledId, topK: request.topLogprobsCount ?? 0)
                : nil

            return SendableBox((sampledId, rawLogprobs))
        }
        let (firstTokenId, prefixRawLogprobs) = firstTokenIdBox.value

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

        // Resolve logprob data outside perform
        let prefixLogprobs: (logprob: Float, topLogprobs: [TopLogprob])? = prefixRawLogprobs.map {
            Self.resolveLogprobs($0, tokenizer: container.tokenizer)
        }

        // SAFE: pre-shared-state yield — seq hasn't entered activeByModel yet
        let toolStream = await Self.makeToolProcessor(mlxContainer: mlxContainer, request: request)
        let scrubbedPrefixToken = MLXEngine.scrubControlTokens(firstTokenText)
        let isFlint = container.identifier.family == .flint
        var flintThink = FlintChatDecode.ThinkReentryGuard()
        var flintGrammar = flintGrammarSeed
        var emitText = scrubbedPrefixToken
        var finishFirst = false
        if isFlint {
            let eosId = container.tokenizer?.eosTokenId
            let stops = FlintChatDecode.generationFinishStopIds(
                grammarOn: flintGrammarSeed != nil,
                checkpoint: container.flintFinishStopIds,
                tokenizerEOS: eosId)
            let decision = FlintChatDecode.openingDecision(
                tokenId: firstTokenId, yieldText: scrubbedPrefixToken,
                grammarText: firstTokenText, primaryEOS: eosId, stopIds: stops,
                grammar: &flintGrammar, guardState: &flintThink)
            emitText = decision.yield
            finishFirst = decision.isEOS || decision.thinkReentry
        }
        Self.armFlintExtractor(
            toolStream, family: container.identifier.family, request: request,
            config: container.config, promptTokens: allTokens, maxTokens: maxTokens)
        if !emitText.isEmpty {
            Self.emitProcessed(toolStream, emitText,
                continuation: continuation,
                logprob: prefixLogprobs.map { ($0.logprob, $0.topLogprobs) })
        }
        if finishFirst {
            Self.drainProcessed(toolStream, continuation: continuation)
            continuation.yield(Token(
                id: 0, text: "", finishReason: .stop, promptTokens: promptTokenCount,
                cachedTokens: cachedTokenCount))
            continuation.finish()
        }

        var seq = ActiveStreamSequence(
            id: request.id,
            request: request,
            caches: restoredCaches,
            lastTokenId: firstTokenId,
            generatedText: firstTokenText,
            completionTokens: finishFirst && emitText.isEmpty ? 0 : (firstTokenText.isEmpty ? 0 : 1),
            maxTokens: maxTokens,
            sampler: sampler,
            continuation: continuation,
            modelId: request.model,
            promptTokenCount: promptTokenCount,
            promptTokenIds: allTokens,
            startTime: Date(),
            admittedAt: Date(),
            isFinished: finishFirst,
            finishGuard: FinishGuard(),
            recentTokenIds: Array(allTokens.suffix(20)),
            frequencyPenalty: freqPenaltyValue,
            accumulatedTokenIds: prefixAccumulatedIds,
            lastDecodedText: prefixDecodedText
        )
        seq.toolStream = toolStream
        seq.flintThink = flintThink
        seq.flintGrammar = flintGrammar
        seq.flintPresencePenalty = request.presencePenalty ?? container.config.presencePenalty
        if seq.flintPresencePenalty != 0, firstTokenId >= 0 {
            seq.flintPresenceSeen.insert(firstTokenId)
        }
        return seq
    }

    // MARK: - Logprob Helper

    /// Compute log probabilities from logits inside a `perform` block.
    /// Returns nil when logprobs are not requested (zero overhead for the common case).
    static func computeRawLogprobs(
        from logits: MLXArray,
        sampledTokenId: Int,
        topK: Int
    ) -> RawLogprobData? {
        var l = logits
        if l.dtype == .bfloat16 { l = l.asType(.float32) }
        if l.ndim > 1 { l = l.reshaped(-1) }

        let logProbs = logSoftmax(l)
        let vocabSize = l.dim(0)
        let safeId = max(0, min(sampledTokenId, vocabSize - 1))
        let k = max(0, min(topK, vocabSize))

        // Fast path: only the sampled-token logprob is needed. Pull a single
        // scalar — MLX evaluates only what's required.
        guard k > 0 else {
            let sampledLogprob = logProbs[safeId].item(Float.self)
            return RawLogprobData(logprob: sampledLogprob, topLogprobs: [])
        }

        eval(logProbs)
        let sampledLogprob = logProbs[safeId].item(Float.self)

        // argPartition with kth=(vocab-k) places the K largest at indices
        // [vocab-k ..< vocab] in O(N) — vastly cheaper than a full O(N log N)
        // argSort over a 150K vocabulary. Sort just those K elements after.
        let pivot = vocabSize - k
        let partIdx = argPartition(logProbs, kth: pivot, axis: -1)
        eval(partIdx)

        var pairs: [(tokenId: Int, logprob: Float)] = []
        pairs.reserveCapacity(k)
        for i in pivot..<vocabSize {
            let idx = partIdx[i].item(Int.self)
            let lp = logProbs[idx].item(Float.self)
            pairs.append((tokenId: idx, logprob: lp))
        }
        pairs.sort { $0.logprob > $1.logprob }
        return RawLogprobData(logprob: sampledLogprob, topLogprobs: pairs)
    }

    /// Resolve raw logprob data to `TopLogprob` with token text from tokenizer.
    private static func resolveLogprobs(
        _ raw: RawLogprobData,
        tokenizer: Tokenizer?
    ) -> (logprob: Float, topLogprobs: [TopLogprob]) {
        let top = raw.topLogprobs.map { pair -> TopLogprob in
            let text = tokenizer.map { $0.decode([pair.tokenId]) } ?? ""
            return TopLogprob(tokenId: pair.tokenId, tokenText: text, logprob: pair.logprob)
        }
        return (raw.logprob, top)
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
                NovaMLXLog.debug("[RunLoop] iteration=\(iteration), activeByModel=\(snapshot)")
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

    private enum FastPacketApply {
        /// Leave the sequence untouched. The per-token loop handles this packet.
        case skip
        /// Tokens are committed. `finished` means max length was reached.
        case applied(finished: Bool)
    }

    /// One tokenizer decode for a clean Flint greedy packet.
    ///
    /// Returns `.skip` with no writes when the packet may stop, when the new
    /// text contains `<` (tool or think markup), or when scrubbing would
    /// change the chunk. A clean packet updates the same fields the slow
    /// loop would, in one decode, then yields through the tool parser.
    private func fastFlintPacket(
        seq: inout ActiveStreamSequence,
        tokens: [Int],
        eosIds: Set<Int>,
        tokenizer: Tokenizer?,
        controlPatterns: [String],
        processor: ChatTemplateProcessor?,
        specDecoder: SpeculativeDecoder
    ) -> FastPacketApply {
        guard !tokens.isEmpty, let tokenizer else { return .skip }
        if tokens.contains(where: { eosIds.contains($0) }) { return .skip }

        let probeIds = seq.accumulatedTokenIds + tokens
        let fullText = tokenizer.decode(probeIds)
        let prior = seq.lastDecodedText
        if let stops = seq.request.stop {
            for stop in stops where !stop.isEmpty {
                let appeared = fullText.contains(stop) && !prior.contains(stop)
                if appeared || fullText.hasSuffix(stop) { return .skip }
            }
        }
        if controlPatterns.contains(where: { pattern in
            !pattern.isEmpty && fullText.contains(pattern) && !prior.contains(pattern)
        }) {
            return .skip
        }
        let futureCount = seq.completionTokens + tokens.count
        if processor?.shouldStopForHallucination(
            generatedText: fullText, completionTokenCount: futureCount) == true
        {
            return .skip
        }
        let openedControl = (fullText.contains("<|") && !prior.contains("<|"))
            || (fullText.contains("|>") && !prior.contains("|>"))
        if openedControl { return .skip }

        let deltaLength = fullText.count - prior.count
        if deltaLength < 0 { return .skip }
        let rawDelta = deltaLength > 0 ? String(fullText.suffix(deltaLength)) : ""
        // Flint always installs the GLM4 tool parser. A `<` in this chunk can
        // be a tool tag, a think tag, or a partial tag the parser must see
        // token by token. Prose with no `<` is one decode.
        if seq.toolStream != nil && rawDelta.contains("<") { return .skip }
        var scrubbed = MLXEngine.scrubControlTokens(rawDelta)
        if let familyScrub = processor?.scrubControlTokens(scrubbed) {
            scrubbed = familyScrub
        }
        if scrubbed != rawDelta { return .skip }
        // A clean packet has no `<`. Record visible prose (or in-think text)
        // so a later `<think>` can stop the turn. A boundary falls through
        // to the per-token loop with this guard unchanged.
        guard seq.flintThink.observeFastPacket(scrubbed) else { return .skip }

        seq.accumulatedTokenIds = probeIds
        seq.lastDecodedText = fullText
        seq.generatedText = fullText
        for id in tokens {
            seq.lastTokenId = id
            seq.recentTokenIds.append(id)
            if seq.recentTokenIds.count > 128 {
                seq.recentTokenIds = Array(seq.recentTokenIds.suffix(64))
            }
            seq.completionTokens += 1
            seq.decodeTokens += 1
            seq.packetTokens += 1
            specDecoder.recordToken(id)
        }
        lock.withLock { totalTokensViaFused += UInt64(tokens.count) }
        seq.previousTokenWasControl = false
        if seq.flintPresencePenalty != 0 {
            for id in tokens {
                seq.flintPresenceSeen.insert(id)
            }
        }
        if !scrubbed.isEmpty {
            seq.yieldProcessed(scrubbed)
        }

        guard seq.completionTokens >= seq.maxTokens else {
            return .applied(finished: false)
        }
        seq.drainToolProcessor()
        if seq.harmonyInThinking {
            seq.safeYield(Token(id: 0, text: "</think>"))
            seq.harmonyInThinking = false
        }
        noteFlintPreparedPrefix(
            modelId: seq.modelId, prompt: seq.promptTokenIds, generated: seq.accumulatedTokenIds)
        seq.safeYield(Token(
            id: 0, text: "", finishReason: .length, promptTokens: seq.promptTokenCount,
            cachedTokens: flintMetrics[seq.request.id]?.cached))
        let finishedByUs = seq.safeFinish()
        seq.isFinished = true
        seq._finishedByDecodeStep = finishedByUs
        NovaMLXLog.debug(
            "[DecodeStep] flint packet seq \(seq.id.uuidString.prefix(8)) finished owned=\(finishedByUs) reason=length")
        return .applied(finished: true)
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
            let canStart = await canAdmitAndReserve(item.request)
            if canStart {
                admittedGenerate.append(item)
            } else {
                // Try preemption for high-priority items
                if item.priority == .high {
                    let preempted = await preemptNewest(for: item.request.model, toMakeRoomFor: item.request.id)
                    if preempted {
                        // Re-check admission after preemption freed some memory
                        let retryCanAdmit = await canAdmitAndReserve(item.request)
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
            // Budget already reserved atomically by canAdmitAndReserve()
            // Wrap generate as stream, collect result
            let stream = internalSubmitStream(item.request)
            Task {
                var resumed = false
                defer {
                    if !resumed {
                        item.continuation.resume(throwing: CancellationError())
                    }
                }
                var text = ""
                var completionTokens = 0
                var finishReason: FinishReason = .stop
                var logprobTokens: [Token] = []
                do {
                    for try await token in stream {
                        if let fr = token.finishReason { finishReason = fr }
                        if !token.text.isEmpty { text += token.text; completionTokens += 1 }
                        if token.logprob != nil { logprobTokens.append(token) }
                    }
                    let promptTokens = lock.withLock {
                        activeByModel[modelId]?.first?.promptTokenCount ?? 0
                    }
                    assert(!resumed, "BUG: double-resume of generate continuation for \(item.request.id)")
                    resumed = true
                    item.continuation.resume(returning: InferenceResult(
                        id: item.request.id, model: modelId, text: text,
                        tokensPerSecond: 0, promptTokens: promptTokens,
                        completionTokens: completionTokens, finishReason: finishReason,
                        tokenLogprobs: logprobTokens.isEmpty ? nil : logprobTokens
                    ))
                } catch {
                    assert(!resumed, "BUG: double-resume (throwing) of generate continuation for \(item.request.id)")
                    resumed = true
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
            let canStart = await canAdmitAndReserve(item.request)
            if canStart {
                admittedStream.append(item)
            } else {
                // Try preemption for high-priority items
                if item.priority == .high {
                    let preempted = await preemptNewest(for: item.request.model, toMakeRoomFor: item.request.id)
                    if preempted {
                        let retryCanAdmit = await canAdmitAndReserve(item.request)
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
            // Budget already reserved atomically by canAdmitAndReserve()

            do {
                let seq = try await prefillSequence(request: item.request, continuation: item.continuation)
                if !seq.isFinished {
                    lock.withLock {
                        if activeByModel[modelId] == nil { activeByModel[modelId] = [] }
                        activeByModel[modelId]?.append(seq)
                    }
                } else {
                    lock.withLock {
                        activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
                    }
                    await self.budgetTracker.release(sequenceId: item.request.id)
                }
            } catch {
                // SAFE: pre-prefill error path, continuation is exclusively owned
                item.continuation.finish(throwing: error)
                NovaMLXLog.error("[Fused:\(item.request.id.uuidString.prefix(8))] admitQueued prefill error — model=\(modelId): \(error) — \(type(of: error))")
                lock.withLock {
                    activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
                    assert((activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow in admitQueued error path")
                }
                await self.budgetTracker.release(sequenceId: item.request.id)
            }
        }
    }

    // MARK: - Fused Decode Step

    @discardableResult
    private func fusedDecodeStep() async -> Bool {
        let groups = lock.withLock { activeByModel }
        let stepNum = lock.withLock { totalDecodeSteps }
        var anyActive = false

        for (modelId, sequences) in groups {
            let active = sequences.filter { !$0.isFinished }
            guard !active.isEmpty else {
                lock.withLock { activeByModel[modelId] = [] }
                continue
            }
            NovaMLXLog.debug("[DecodeStep:\(stepNum)] entering model=\(modelId) active=\(active.count)")

            guard let container = engine.getContainer(for: modelId),
                  let mlxContainer = container.mlxContainer else {
                for seq in active {
                    if seq.safeFinish(throwing: NovaMLXError.modelNotFound(modelId)) {
                        await budgetTracker.release(sequenceId: seq.id)
                        NovaMLXLog.debug("[BudgetRelease] seq=\(seq.id.uuidString.prefix(8)) model=\(modelId) path=modelNotFound")
                    }
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

            // EOS token ID from tokenizer (primary stop condition)
            let eosId = container.tokenizer?.eosTokenId

            // Get control token patterns for turn-stop detection (cached on container)
            let controlPatterns = container.controlTokens

            // Run decode step per-sequence inside perform.
            // N-gram speculation: propose draft tokens, verify in one forward pass.

            let specDecoder = engine.specDecoder
            var specResults: [SpecDecodeResult] = []
            var firstDecodeStart: [UUID: Date] = [:]
            for seq in active {
                if seq.decodeWindowStart == nil {
                    firstDecodeStart[seq.id] = Date()
                }
                // Logprobs require per-step logits — bypass spec decoding when requested
                // so every accepted token has logprob data (avoids silent gaps in the API).
                let wantLogprobs = seq.request.includeLogprobs
                // Flint structural grammar rewrites logits every step. Drafts
                // are unmasked, so they stay off while the grammar is installed.
                // A filtered resident packet also replaces drafts: n-gram verify
                // is argmax, and it does not see the grammar or top-p mask.
                let grammarOn = seq.flintGrammar != nil
                let isFlint = container.identifier.family == .flint
                let flintStopIds: Set<Int> = isFlint
                    ? FlintChatDecode.generationFinishStopIds(
                        grammarOn: grammarOn,
                        checkpoint: container.flintFinishStopIds,
                        tokenizerEOS: eosId)
                    : []
                let flintTemperature = Float(seq.request.temperature ?? container.config.temperature)
                let flintTopP = Float(seq.request.topP ?? container.config.topP)
                let flintTopK = seq.request.topK ?? container.config.topK
                let flintMinP = seq.request.minP ?? container.config.minP
                let flintPresence = seq.flintPresencePenalty
                let flintRepetition = seq.request.repetitionPenalty ?? container.config.repeatPenalty
                let freqPenForRoute = seq.frequencyPenalty
                let flintRoute: FlintChatDecode.ResidentRoute = isFlint
                    ? FlintChatDecode.residentRoute(
                        FlintChatDecode.Policy(
                            temperature: flintTemperature,
                            topP: flintTopP,
                            topK: flintTopK,
                            minP: flintMinP,
                            frequencyPenalty: freqPenForRoute,
                            repetitionPenalty: flintRepetition,
                            presencePenalty: flintPresence,
                            wantsLogprobs: wantLogprobs,
                            batchWidth: active.count,
                            // Fused decode does not apply JSON / regex / GBNF.
                            // Greedy with that constraint keeps the device packet,
                            // matching the path from before the filtered packet.
                            usesGrammar: false,
                            usesStructuralGrammar: grammarOn))
                    : .singleStep
                // Greedy must stay step-by-step: batch verify KV vs stepwise can
                // flip near-tie argmax (max |Δ|~0.5), breaking lossless compares.
                let draftTokens = (
                    wantLogprobs || grammarOn || flintRoute != .singleStep
                        || seq.sampler.isGreedy || !specDecoder.shouldSpeculate()
                )
                    ? []
                    : specDecoder.speculate(context: seq.recentTokenIds)

                if draftTokens.isEmpty {
                    // No draft. Flint either takes one resident packet or one
                    // host sample. The device packet is greedy argmax, width
                    // 16, and only when nothing rewrites logits. Grammar,
                    // presence, and temperature / top-p / top-k use the filtered
                    // packet: the same width, one resident forward per token,
                    // mask before acceptance. A non-neutral repetition penalty is
                    // HTTP 400 before this loop; the route still returns single-step.
                    let token = MLXArray(Int32(seq.lastTokenId)).reshaped(1, 1)
                    let seqCaches = seq.caches
                    let seqSampler = seq.sampler
                    let recentIds = seq.recentTokenIds
                    let freqPen = freqPenForRoute
                    let logprobTopK = seq.request.topLogprobsCount ?? 0
                    let grammarSeed = seq.flintGrammar
                    let presenceSeed = seq.flintPresenceSeen
                    let packetCount = min(
                        FlintRecResident.width, max(1, seq.maxTokens - seq.completionTokens))
                    let packetStops = FlintChatDecode.packetStopIds(tokenizerEOS: eosId)
                    let inputToken = seq.lastTokenId
                    if flintRoute == .filtered {
                        FlintChatDecode.noteFilteredPacket(
                            temperature: flintTemperature,
                            topP: flintTopP,
                            topK: flintTopK,
                            grammar: grammarOn,
                            presence: flintPresence)
                    }
                    let tidBox = await mlxContainer.perform(nonSendable: (token, seqCaches)) { context, values in
                        let (tokenInput, caches) = values
                        if flintRoute == .greedyDevice, let flint = context.model as? FlintModel,
                            flint.recGreedyPacketReady(cache: caches)
                        {
                            do {
                                let ids = try flint.recGreedyPacket(
                                    cache: caches, token: inputToken, count: packetCount,
                                    eosTokenIds: packetStops)
                                if !ids.isEmpty {
                                    return SendableBox(FusedDecodeHit.packet(ids))
                                }
                            } catch {
                                // recGreedyPacket disables rec after a kernel failure.
                                // The single-token forward below is the eager fallback.
                            }
                        }
                        if flintRoute == .filtered {
                            var ids: [Int] = []
                            ids.reserveCapacity(packetCount)
                            var stepToken = inputToken
                            var grammar = grammarSeed
                            var seen = presenceSeed
                            var stopPacket = false
                            for _ in 0..<packetCount {
                                let step = MLXArray(Int32(stepToken)).reshaped(1, 1)
                                let stepLogits = context.model(step, cache: caches)
                                eval(stepLogits)
                                let filtered = FlintChatDecode.filterLogits(
                                    stepLogits[0..., -1, 0...],
                                    grammar: grammar,
                                    presenceSeen: seen,
                                    presencePenalty: flintPresence,
                                    temperature: flintTemperature,
                                    topK: flintTopK)
                                let sampled = seqSampler.sample(logits: filtered)
                                eval(sampled)
                                let id = sampled.item(Int.self)
                                let masked = grammar?.maskedTokenIds(vocabSize: filtered.dim(-1)) ?? []
                                switch FlintChatDecode.decideFiltered(
                                    tokenId: id, masked: masked, stopIds: flintStopIds)
                                {
                                case .rejectMasked:
                                    // Keep the id so the host can drop its text and
                                    // finish. It is not in KV yet; dropping it here
                                    // would forward `stepToken` again next step.
                                    ids.append(id)
                                    stopPacket = true
                                case .keep(_, let stop):
                                    ids.append(id)
                                    if flintPresence != 0 { seen.insert(id) }
                                    if var activeGrammar = grammar {
                                        activeGrammar.advancePhase(tokenId: id)
                                        grammar = activeGrammar
                                    }
                                    stepToken = id
                                    if stop { stopPacket = true }
                                }
                                if stopPacket { break }
                            }
                            if !ids.isEmpty {
                                return SendableBox(FusedDecodeHit.packet(ids))
                            }
                        }
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
                        if isFlint {
                            lastLogits = FlintChatDecode.filterLogits(
                                lastLogits,
                                grammar: grammarSeed,
                                presenceSeen: presenceSeed,
                                presencePenalty: flintPresence,
                                temperature: flintTemperature,
                                topK: flintTopK)
                        }

                        let sampled = seqSampler.sample(logits: lastLogits)
                        eval(sampled)
                        let sampledId = sampled.item(Int.self)

                        let rawLogprobs: RawLogprobData? = wantLogprobs
                            ? Self.computeRawLogprobs(from: lastLogits, sampledTokenId: sampledId, topK: logprobTopK)
                            : nil

                        return SendableBox(FusedDecodeHit.single(sampledId, rawLogprobs))
                    }
                    switch tidBox.value {
                    case .packet(let ids):
                        specResults.append(SpecDecodeResult(
                            acceptedTokens: ids, draftAccepted: 0, draftProposed: 0,
                            sequenceId: seq.id, logprobs: nil, fromResidentPacket: true))
                    case .single(let tid, let decodeRawLogprobs):
                        let decodedLogprobs: (logprob: Float, topLogprobs: [TopLogprob])? = decodeRawLogprobs.map {
                            Self.resolveLogprobs($0, tokenizer: container.tokenizer)
                        }
                        specResults.append(SpecDecodeResult(
                            acceptedTokens: [tid], draftAccepted: 0, draftProposed: 0, sequenceId: seq.id,
                            logprobs: decodedLogprobs
                        ))
                    }
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

                        // Speculative verify (correct semantics). logits[pos] is the
                        // next-token distribution AFTER [lastReal, d0..d_{pos-1}]:
                        //   - draft d_pos is ACCEPTED iff argmax(logits[pos]) == d_pos
                        //     (the main model, at the true position, would pick it)
                        //   - on rejection, emit argmax(logits[pos]) as the bonus and stop
                        //   - all accepted → final bonus = argmax(logits[nDraft])
                        // The old loop compared logits[pos] (conditioned on d_{pos-1}
                        // already being right) against d_{pos-1} and ALWAYS appended
                        // mainToken — the KV cache advanced through the draft while the
                        // text emitted the main model's token, desyncing text/state from
                        // the first speculated step (garbled + doubled long outputs).
                        var accepted = 0
                        var allAccepted: [Int] = []
                        for pos in 0...nDraft {
                            var posLogits = logits[0..., pos, 0...]
                            // Frequency penalty at REAL positions only (pos 0 is the
                            // true distribution; later positions assume draft prefix).
                            if pos == 0, freqPen > 0, !recentIds.isEmpty {
                                let vocabSize = posLogits.dim(-1)
                                let ids = MLXArray(recentIds.map { Int32($0) })
                                let ones = MLXArray.ones([recentIds.count], type: Float32.self)
                                let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
                                    .at[ids].add(ones)
                                posLogits = posLogits - (histogram * freqPen).reshaped(1, -1)
                            }
                            let mainToken = argMax(posLogits, axis: -1).item(Int.self)
                            if pos < nDraft {
                                if mainToken == draftTokens[pos] {
                                    accepted += 1
                                    allAccepted.append(draftTokens[pos])
                                } else {
                                    allAccepted.append(mainToken)
                                    break
                                }
                            } else {
                                // All drafts accepted — final bonus from the last position.
                                allAccepted.append(mainToken)
                            }
                        }

                        // Trim KV to exactly the emitted tokens: the forward pass
                        // processed 1 + nDraft positions; we keep allAccepted.count.
                        let excessDraft = (1 + nDraft) - allAccepted.count
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
                        sequenceId: seq.id,
                        logprobs: nil  // speculative path doesn't compute logprobs
                    ))
                }
            }

            // Apply spec decode results: yield accepted tokens per sequence.
            // Adaptive gate bookkeeping (one outcome per step, not per token).
            specDecoder.recordStepOutcome()
            var updatedActive = active
            for (i, var seq) in updatedActive.enumerated() {
                guard i < specResults.count else { break }
                let result = specResults[i]

                // Record speculation stats and update N-gram cache.
                if result.draftProposed > 0 {
                    specDecoder.recordAccepted(tokens: result.acceptedTokens, accepted: result.draftAccepted)
                }
                if seq.decodeWindowStart == nil, let start = firstDecodeStart[seq.id] {
                    seq.decodeWindowStart = start
                }

                var emitTokens = result.acceptedTokens
                if result.fromResidentPacket {
                    seq.packetLaunches += 1
                    let split = FlintChatDecode.keptPrefix(tokenIds: emitTokens, stopIds: flintStopIds)
                    if split.unused > 0 {
                        FlintChatDecode.trim(seq.caches, unused: split.unused)
                        emitTokens = split.kept
                    }
                }

                // Yield all accepted tokens.
                var sequenceFinished = false
                var consumed = 0
                var usedFast = false
                // Grammar has to see each token's text. A one-shot decode would
                // skip that and leave the next packet's mask on the old phase.
                if result.fromResidentPacket, seq.flintGrammar == nil,
                    container.identifier.family != .gptOss {
                    switch fastFlintPacket(
                        seq: &seq, tokens: emitTokens, eosIds: flintStopIds,
                        tokenizer: container.tokenizer, controlPatterns: controlPatterns,
                        processor: container.chatTemplateProcessor, specDecoder: specDecoder)
                    {
                    case .skip:
                        break
                    case .applied(let finished):
                        usedFast = true
                        sequenceFinished = finished
                    }
                }
                if !usedFast {
                for (tokenIdx, tokenId) in emitTokens.enumerated() {
                    consumed = tokenIdx + 1
                    // EOS FIRST, before any decode or yield: some tokenizers
                    // (Hy-MT2's TokenizersBackend) decode the eos token as its
                    // literal text (<｜hy_place▁holder▁no▁2｜>) instead of
                    // skipping it — yielding before the check leaked the
                    // stop marker into every streamed reply.
                    // Flint role-header ids are stops even when they are not
                    // tokenizer.eosTokenId. stopIds is empty for other families.
                    let isEOSNow = FlintChatDecode.isStopToken(
                        tokenId, primaryEOS: eosId, stopIds: flintStopIds)
                    seq.lastTokenId = tokenId
                    if seq.flintPresencePenalty != 0 {
                        seq.flintPresenceSeen.insert(tokenId)
                    }
                    if !isEOSNow {
                        seq.completionTokens += 1
                        seq.decodeTokens += 1
                    }
                    if result.fromResidentPacket {
                        seq.packetTokens += 1
                    } else if !isEOSNow {
                        seq.singleTokens += 1
                    }

                    // Update N-gram context buffer.
                    seq.recentTokenIds.append(tokenId)
                    if seq.recentTokenIds.count > 128 {
                        seq.recentTokenIds = Array(seq.recentTokenIds.suffix(64))
                    }

                    // Record into N-gram cache for future speculation.
                    specDecoder.recordToken(tokenId)

                    var decoded: String
                    var hitThinkReentry = false
                    var hitGrammarReject = false
                    if isEOSNow {
                        // EOS decodes to nothing (some tokenizers would emit its
                        // literal text) — handled entirely by the stop path below.
                        decoded = ""
                        seq.previousTokenWasControl = true
                    } else if let tok = container.tokenizer, let delta = seq.decodeNextToken(tokenId, tokenizer: tok) {
                        // Inspect the RAW delta (before scrubbing) for Harmony
                        // channel-type residue words. `MLXEngine.scrubControlTokens`
                        // scrubs `analysis` / `final` / `commentary` to "" via its
                        // `harmonyResidueWords` filter, so by the time we look at
                        // `scrubbed` the channel-type information is gone. We need
                        // the raw delta to detect "this empty scrubbed result is
                        // a channel-type residue word" vs "this empty scrubbed
                        // result is a control token".
                        let rawTrimmed = delta.trimmingCharacters(in: .whitespaces)
                        // Static scrub first (Gemma/Harmony normalization), then
                        // the FAMILY processor's scrub — it knows the family's
                        // own control markers (e.g. Hy's ｜hy_*｜ tokens) that
                        // the family-blind static pass doesn't.
                        var scrubbed = MLXEngine.scrubControlTokens(delta)
                        if let familyScrub = container.chatTemplateProcessor?.scrubControlTokens(scrubbed) {
                            scrubbed = familyScrub
                        }

                        // Harmony streaming flow (3-token boundary, split across deltas):
                        //   <|channel|>  → scrubbed=""  delta="<|channel|>"   (control token)
                        //   "analysis"   → scrubbed=""  delta="analysis"      (residue word)
                        //   <|message|>  → scrubbed=""  delta="<|message|>"   (control token)
                        //   ...thinking text streams normally...
                        //   <|end|>      → scrubbed=""
                        //   <|start|>    → scrubbed=""
                        //   "assistant"  → scrubbed=""  delta="assistant"     (residue word)
                        //   <|channel|>  → scrubbed=""
                        //   "final"      → scrubbed=""  delta="final"
                        //   <|message|>  → scrubbed=""
                        //   ...final text streams normally...
                        //   <|return|>   → scrubbed=""  (stop signal handled separately)
                        //
                        // We distinguish "control token" from "residue word" via the
                        // raw delta: control tokens contain `<|`, residue words don't.
                        let isHarmonyResidueWord =
                            scrubbed.isEmpty && !delta.isEmpty &&
                            !delta.contains("<|") && !delta.contains("|>") &&
                            MLXEngine.harmonyResidueWords.contains(rawTrimmed)
                        let isControlTokenChunk =
                            scrubbed.isEmpty && !delta.isEmpty && !isHarmonyResidueWord

                        if isHarmonyResidueWord {
                            // Only drop residue words when they immediately follow a
                            // scrubbed control token (Harmony scaffolding). Otherwise
                            // keep them — Qwen thinking text legitimately says "user".
                            if seq.previousTokenWasControl {
                                if rawTrimmed == "analysis" || rawTrimmed == "final" || rawTrimmed == "commentary" {
                                    seq.pendingHarmonyChannel = rawTrimmed
                                }
                                seq.previousTokenWasControl = false
                                decoded = ""
                            } else {
                                seq.previousTokenWasControl = false
                                decoded = delta
                            }
                        } else if isControlTokenChunk {
                            // If we have a pending Harmony channel type, this is
                            // the closing `<|message|>` of the channel header —
                            // emit the appropriate boundary.
                            if let channel = seq.pendingHarmonyChannel {
                                seq.pendingHarmonyChannel = nil
                                switch channel {
                                case "analysis":
                                    decoded = "<think>"
                                    seq.harmonyInThinking = true
                                case "final":
                                    if seq.harmonyInThinking {
                                        decoded = "</think>"
                                        seq.harmonyInThinking = false
                                    } else {
                                        // Model went straight to final without analysis.
                                        // No prior <think> to close — just strip the boundary.
                                        decoded = ""
                                    }
                                default:
                                    // commentary / unknown channel — strip entirely
                                    decoded = ""
                                }
                                seq.previousTokenWasControl = true
                            } else {
                                seq.previousTokenWasControl = true
                                decoded = ""
                            }
                        } else if seq.previousTokenWasControl && !scrubbed.isEmpty {
                            seq.previousTokenWasControl = false
                            decoded = scrubbed
                        } else {
                            seq.previousTokenWasControl = false
                            decoded = scrubbed
                        }
                    } else {
                        decoded = ""
                        seq.previousTokenWasControl = false
                    }
                    if container.identifier.family == .flint {
                        if var grammar = seq.flintGrammar {
                            if !isEOSNow {
                                let grammarText = container.tokenizer?.decode([tokenId]) ?? decoded
                                if grammar.consume(tokenId: tokenId, text: grammarText) {
                                    seq.flintGrammar = grammar
                                } else {
                                    // Mask bypass: the piece spells a marker the
                                    // grammar excludes. Drop it and end the turn.
                                    decoded = ""
                                    hitGrammarReject = true
                                }
                            }
                        } else {
                            let step = FlintChatDecode.tokenDecision(
                                tokenId: tokenId, decoded: decoded,
                                primaryEOS: eosId, stopIds: flintStopIds, state: &seq.flintThink)
                            decoded = step.yield
                            hitThinkReentry = step.thinkReentry
                        }
                    }
                    seq.generatedText = seq.lastDecodedText
                    if !decoded.isEmpty || seq.toolStream != nil {
                        // Attach logprobs to the first (real decode) token only
                        let tokenLogprobs: (logprob: Float, topLogprobs: [TopLogprob])? =
                            (tokenIdx == 0) ? result.logprobs : nil
                        seq.yieldProcessed(decoded, logprob: tokenLogprobs)
                    }

                    lock.withLock { totalTokensViaFused += 1 }

                    // Same set as isEOSNow. A scrubbed Flint stop id still finishes.
                    let isEOS = isEOSNow
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

                    // Check control token patterns (e.g. <|turn>user) — prevents infinite
                    // hallucination loops when models generate past turn boundaries
                    var hitControlToken = false
                    if !controlPatterns.isEmpty && !hitStop && !isEOS {
                        for pattern in controlPatterns {
                            if seq.generatedText.contains(pattern) {
                                hitControlToken = true
                                break
                            }
                        }
                    }

                    // Check bare multi-turn hallucination patterns — models sometimes
                    // generate `user\n...\nassistant\n` without any control tokens.
                    // Patterns come from the chat template processor (family-specific).
                    if !hitControlToken && !hitStop && !isEOS {
                        if container.chatTemplateProcessor?.shouldStopForHallucination(
                            generatedText: seq.generatedText, completionTokenCount: seq.completionTokens
                        ) == true {
                            hitControlToken = true
                        }
                    }

                    if isEOS || atMax || hitStop || hitControlToken || hitThinkReentry || hitGrammarReject {
                        // Flush any tool-call buffer the parser still holds, then
                        // mirror the ContinuousBatcher finish semantics: a stop
                        // that produced tool calls reports finish_reason=tool_calls.
                        // A think-reentry stop already dropped the new `<think>`.
                        // Other finishes release a withheld tag prefix (`<` …).
                        if container.identifier.family == .flint, !hitThinkReentry {
                            let tail = seq.flintThink.flushHeld()
                            if !tail.isEmpty { seq.yieldProcessed(tail) }
                        }
                        seq.drainToolProcessor()
                        var finishReason: FinishReason = atMax ? .length : .stop
                        // A new `<think>` after visible text is a turn stop, including
                        // when that token is also the one that reaches maxTokens.
                        if hitThinkReentry || hitGrammarReject {
                            finishReason = .stop
                        }
                        if seq.toolStream?.emittedToolCall == true && finishReason == .stop {
                            finishReason = .toolCalls
                        }
                        // Harmony: if generation ends while still in the
                        // analysis channel (no `<|channel|>final<|message|>`
                        // emitted yet), close the open `<think>` block so
                        // `ThinkingParser` can finalize cleanly. Without
                        // this, the parser sees an unclosed `<think>` and
                        // its fallback promotes the entire buffer to
                        // `content`, leaving `reasoning_content` empty.
                        if seq.harmonyInThinking {
                            seq.safeYield(Token(id: 0, text: "</think>"))
                            seq.harmonyInThinking = false
                        }
                        noteFlintPreparedPrefix(
                            modelId: seq.modelId, prompt: seq.promptTokenIds,
                            generated: seq.accumulatedTokenIds)
                        seq.safeYield(Token(
                            id: 0, text: "", finishReason: finishReason,
                            promptTokens: seq.promptTokenCount,
                            cachedTokens: flintMetrics[seq.request.id]?.cached))
                        let finishedByUs = seq.safeFinish()
                        seq.isFinished = true
                        sequenceFinished = true
                        seq._finishedByDecodeStep = finishedByUs
                        NovaMLXLog.debug("[DecodeStep:\(stepNum)] seq \(seq.id.uuidString.prefix(8)) finished owned=\(finishedByUs) reason=\(finishReason)")
                        break
                    }
                }
                if result.fromResidentPacket {
                    let unused = emitTokens.count - consumed
                    if unused > 0 {
                        FlintChatDecode.trim(seq.caches, unused: unused)
                    }
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
                // Only release budget if WE finished the sequence in this decode step.
                // Preempt/abort paths release budget themselves; skip to avoid double-release.
                guard seq._finishedByDecodeStep else { continue }
                if container.identifier.family == .flint, let start = seq.decodeWindowStart {
                    let seconds = Date().timeIntervalSince(start)
                    if let line = FlintChatDecode.line(
                        decodeTokens: seq.decodeTokens, seconds: seconds,
                        packetTokens: seq.packetTokens, singleTokens: seq.singleTokens,
                        launches: seq.packetLaunches)
                    {
                        NovaMLXLog.info(line)
                    }
                }
                await budgetTracker.release(sequenceId: seq.id)
                assert(seq._finishedByDecodeStep, "BUG: budget released for sequence not owned by decode step — double-release")
                NovaMLXLog.debug("[BudgetRelease] seq=\(seq.id.uuidString.prefix(8)) model=\(modelId) path=decodeStep")
                lock.withLock {
                    activeModelCounts[modelId] = max(0, (activeModelCounts[modelId] ?? 1) - 1)
                    assert((activeModelCounts[modelId] ?? 0) >= 0, "BUG: activeModelCounts underflow for \(modelId)")
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
            // Invariant: total active sequences must never exceed total admitted count
            let totalActive = activeByModel.values.reduce(0) { $0 + $1.count }
            let totalCounts = activeModelCounts.values.reduce(0, +)
            assert(totalActive <= totalCounts,
                "BUG: active(\(totalActive)) > admitted(\(totalCounts)) — scheduler invariant violated")
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
        let tag = requestId.uuidString.prefix(8)
        var wasInQueue = false
        var wasActive = false
        var activeModel: String?
        lock.withLock {
            let sqBefore = streamQueue.count
            streamQueue.removeAll { $0.request.id == requestId }
            wasInQueue = wasInQueue || streamQueue.count < sqBefore

            let gqBefore = generateQueue.count
            generateQueue.removeAll { $0.request.id == requestId }
            wasInQueue = wasInQueue || generateQueue.count < gqBefore

            for (modelId, var sequences) in activeByModel {
                if let idx = sequences.firstIndex(where: { $0.id == requestId }) {
                    sequences[idx].isFinished = true
                    sequences[idx].safeFinish()
                    sequences.remove(at: idx)
                    activeByModel[modelId] = sequences
                    wasActive = true
                    activeModel = modelId
                }
            }
        }

        if wasActive {
            // NOTE: abort() does not release budgetTracker or decrement activeModelCounts.
            // The decode step's _finishedByDecodeStep guard will skip cleanup for this
            // sequence (since safeFinish() already returned true from this path).
            // Budget is reclaimed when the decode step runs and sees the sequence is gone.
            NovaMLXLog.debug("[Abort:\(tag)] active sequence aborted (model=\(activeModel ?? "?")), budget release deferred to decode step")
        }
        if !wasInQueue && !wasActive {
            NovaMLXLog.debug("[Abort:\(tag)] request not found in any queue or active set")
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
