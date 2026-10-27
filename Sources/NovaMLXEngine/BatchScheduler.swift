import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

struct PendingBatchRequest {
    let id: UUID
    let request: InferenceRequest
    let continuation: CheckedContinuation<InferenceResult, Error>
}

struct ActiveBatchSequence: @unchecked Sendable {
    let id: UUID
    let request: InferenceRequest
    let caches: [KVCache]
    var generatedTokens: [Int]
    var generatedText: String
    let maxTokens: Int
    let promptTokenCount: Int
    let sampler: any LogitSampler
    let continuation: CheckedContinuation<InferenceResult, Error>
    /// Flint only. Other families leave this at the default.
    /// Used when structural grammar is off.
    var flintThink = FlintChatDecode.ThinkReentryGuard()
    /// Flint chat grammar. Nil when grammar is off.
    var flintGrammar: FlintStructuralGrammar? = nil
}

public final class BatchScheduler: @unchecked Sendable {
    private let engine: MLXEngine
    public let maxBatchSize: Int
    private let lock = NovaMLXLock()

    private var pending: [PendingBatchRequest] = []
    private var activeByModel: [String: [ActiveBatchSequence]] = [:]
    private var isRunning = false

    private var _totalFusedSteps: UInt64 = 0
    private var _totalTokensViaFused: UInt64 = 0
    private var _peakBatchWidth: Int = 0
    private var _peakFusedForwardWidth: Int = 0
    private var _totalFusedForwards: UInt64 = 0
    private var _totalBatches: UInt64 = 0

    public init(engine: MLXEngine, maxBatchSize: Int = 8) {
        self.engine = engine
        self.maxBatchSize = maxBatchSize
    }

    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock {
                pending.append(PendingBatchRequest(
                    id: request.id,
                    request: request,
                    continuation: continuation
                ))
            }
            schedule()
        }
    }

    private func schedule() {
        let shouldStart = lock.withLock {
            guard !isRunning, !pending.isEmpty else { return false }
            isRunning = true
            return true
        }
        guard shouldStart else { return }
        Task { await runLoop() }
    }

    private func runLoop() async {
        while true {
            let batch: [PendingBatchRequest] = lock.withLock {
                let take = Array(pending.prefix(maxBatchSize))
                pending.removeFirst(min(take.count, maxBatchSize))
                return take
            }

            for req in batch {
                do {
                    try await prefill(req)
                } catch {
                    req.continuation.resume(throwing: error)
                }
            }

            await fusedDecodeStep()

            let hasActive = lock.withLock { activeByModel.values.contains { !$0.isEmpty } }
            let hasPending = lock.withLock { !pending.isEmpty }

            if !hasActive && !hasPending {
                lock.withLock { isRunning = false }
                if lock.withLock({ !pending.isEmpty }) { schedule() }
                return
            }
        }
    }

    private func prefill(_ req: PendingBatchRequest) async throws {
        guard let container = engine.getContainer(for: req.request.model),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(req.request.model)
        }

        let hasImages = req.request.messages.contains { $0.images?.isEmpty == false }
        let userInput = try await engine.buildUserInput(request: req.request, hasImages: hasImages)
        let input = try await mlxContainer.prepare(input: userInput)
        let promptTokenCount = input.text.tokens.size
        let maxTokens = try engine.outputTokenBudget(
            request: req.request, config: container.config, promptTokens: Int(promptTokenCount))

        let cachesBox = MutableSendableBox<[KVCache]?>(nil)
        let prefillParams: GenerateParameters = {
            var p = GenerateParameters(maxTokens: 1, temperature: 0.6)
            p.prefill.stepSize = 512
            return p
        }()

        let stream = try await mlxContainer.perform(nonSendable: input) { context, input in
            let cache = try context.model.newCache(parameters: prefillParams)
            cachesBox.value = cache
            return try MLXLMCommon.generate(
                input: input,
                cache: cache,
                parameters: prefillParams,
                context: context
            )
        }

        var firstTokenText = ""
        for await gen in stream {
            switch gen {
            case .chunk(let text): firstTokenText = text
            case .info: break
            case .toolCall: break
            case .rejectedToolCall: break
            }
        }

        guard let caches = cachesBox.value else {
            throw NovaMLXError.inferenceFailed("Prefill produced no cache")
        }

        let parameters = engine.buildGenerateParameters(request: req.request, config: container.config, settings: engine.settingsProvider?(req.request.model))
        let firstTokenId = container.tokenizer?.encode(firstTokenText).first ?? 0

        var flintThink = FlintChatDecode.ThinkReentryGuard()
        var flintGrammar = FlintChatDecode.withVocabTexts(
            FlintChatDecode.chatGrammar(for: req.request, family: container.identifier.family),
            decodeTokens: container.tokenizer?.decode)
        if container.identifier.family == .flint, !firstTokenText.isEmpty {
            let eosId = container.tokenizer?.eosTokenId
            let stops = FlintChatDecode.generationFinishStopIds(
                grammarOn: flintGrammar != nil,
                checkpoint: container.flintFinishStopIds,
                tokenizerEOS: eosId)
            let decision = FlintChatDecode.openingDecision(
                tokenId: firstTokenId, yieldText: firstTokenText,
                grammarText: firstTokenText, primaryEOS: eosId, stopIds: stops,
                grammar: &flintGrammar, guardState: &flintThink)
            if decision.isEOS || decision.thinkReentry {
                req.continuation.resume(returning: InferenceResult(
                    id: req.id,
                    model: req.request.model,
                    text: decision.yield,
                    tokensPerSecond: 0,
                    promptTokens: promptTokenCount,
                    completionTokens: decision.yield.isEmpty ? 0 : 1,
                    finishReason: .stop
                ))
                return
            }
        }

        var seq = ActiveBatchSequence(
            id: req.id,
            request: req.request,
            caches: caches,
            generatedTokens: [firstTokenId],
            generatedText: firstTokenText,
            maxTokens: maxTokens,
            promptTokenCount: promptTokenCount,
            sampler: parameters.sampler(),
            continuation: req.continuation,
            flintThink: flintThink
        )
        seq.flintGrammar = flintGrammar

        lock.withLock {
            if activeByModel[req.request.model] == nil {
                activeByModel[req.request.model] = []
            }
            activeByModel[req.request.model]?.append(seq)
        }
    }

    private func fusedDecodeStep() async {
        let modelGroups = lock.withLock { activeByModel }

        for (modelId, var sequences) in modelGroups {
            let active = sequences
            guard !active.isEmpty else {
                lock.withLock { activeByModel[modelId] = sequences }
                continue
            }

            guard let container = engine.getContainer(for: modelId),
                  let mlxContainer = container.mlxContainer else {
                for seq in active {
                    seq.continuation.resume(throwing: NovaMLXError.modelNotFound(modelId))
                }
                lock.withLock { activeByModel[modelId] = [] }
                continue
            }

            let batchSize = active.count
            lock.withLock {
                _totalFusedSteps += 1
                _totalBatches += 1
                if batchSize > _peakBatchWidth { _peakBatchWidth = batchSize }
            }

            // Same cohort rule as FusedBatchScheduler: one forward per equal
            // KVCacheSimple offset. Unequal lengths and other cache types get
            // their own forward so RoPE stays on that sequence's offset.
            let offsets = active.map { DecodeFusePlanner.cacheOffset(of: $0.caches) }
            let lanes = DecodeFusePlanner.lanes(cacheOffsets: offsets)
            var rowLogits: [Int: MLXArray] = [:]
            for lane in lanes {
                switch lane.kind {
                case .cohort(let offset):
                    let group = lane.indices.map { active[$0] }
                    if let logits = await cohortLogits(sequences: group, mlxContainer: mlxContainer) {
                        noteFusedForward(width: group.count, offset: offset, modelId: modelId)
                        for (local, index) in lane.indices.enumerated() {
                            rowLogits[index] = logits[local][0..., -1, 0...]
                        }
                    } else {
                        for index in lane.indices {
                            if let row = await serialRowLogits(sequence: active[index], mlxContainer: mlxContainer) {
                                rowLogits[index] = row
                            }
                        }
                    }
                case .serial:
                    let index = lane.indices[0]
                    if let row = await serialRowLogits(sequence: active[index], mlxContainer: mlxContainer) {
                        rowLogits[index] = row
                    }
                }
            }

            let eosId = container.tokenizer?.eosTokenId
            let isFlint = container.identifier.family == .flint

            for (i, var seq) in active.enumerated() {
                let flintStopIds: Set<Int> = isFlint
                    ? FlintChatDecode.generationFinishStopIds(
                        grammarOn: seq.flintGrammar != nil,
                        checkpoint: container.flintFinishStopIds,
                        tokenizerEOS: eosId)
                    : []
                guard var seqLogits = rowLogits[i] else { continue }
                if let grammar = seq.flintGrammar {
                    seqLogits = FlintStructuralGrammar.suppress(
                        grammar.maskedTokenIds(vocabSize: seqLogits.dim(-1)), in: seqLogits)
                }
                let sampledToken = seq.sampler.sample(logits: seqLogits)
                let tokenId = sampledToken.item(Int.self)

                seq.generatedTokens.append(tokenId)
                let rawPiece = container.tokenizer.map { $0.decode([tokenId]) } ?? ""
                let decision: FlintChatDecode.TokenDecision
                if isFlint {
                    if seq.flintGrammar != nil {
                        decision = FlintChatDecode.openingDecision(
                            tokenId: tokenId, yieldText: rawPiece, grammarText: rawPiece,
                            primaryEOS: eosId, stopIds: flintStopIds,
                            grammar: &seq.flintGrammar, guardState: &seq.flintThink)
                    } else {
                        decision = FlintChatDecode.tokenDecision(
                            tokenId: tokenId, decoded: rawPiece,
                            primaryEOS: eosId, stopIds: flintStopIds, state: &seq.flintThink)
                    }
                    if !decision.isEOS {
                        seq.generatedText += decision.yield
                    }
                } else {
                    seq.generatedText += rawPiece
                    decision = FlintChatDecode.TokenDecision(
                        yield: rawPiece,
                        isEOS: FlintChatDecode.isStopToken(tokenId, primaryEOS: eosId, stopIds: []),
                        thinkReentry: false)
                }

                let isEOS = decision.isEOS
                let atMax = seq.generatedTokens.count >= seq.maxTokens

                if isEOS || decision.thinkReentry || atMax {
                    if isFlint && !decision.thinkReentry {
                        seq.generatedText += seq.flintThink.flushHeld()
                    }
                    let result = InferenceResult(
                        id: seq.id,
                        model: seq.request.model,
                        text: seq.generatedText,
                        tokensPerSecond: 0,
                        promptTokens: seq.promptTokenCount,
                        completionTokens: seq.generatedTokens.count,
                        finishReason: decision.thinkReentry ? .stop : (atMax ? .length : .stop)
                    )
                    seq.continuation.resume(returning: result)
                    lock.withLock { _totalTokensViaFused += UInt64(seq.generatedTokens.count) }
                    if let idx = sequences.firstIndex(where: { $0.id == seq.id }) {
                        sequences.remove(at: idx)
                    }
                } else {
                    if let idx = sequences.firstIndex(where: { $0.id == seq.id }) {
                        sequences[idx] = seq
                    }
                }
            }

            lock.withLock { activeByModel[modelId] = sequences }
        }
    }

    public var metrics: BatchSchedulerMetrics {
        lock.withLock {
            let activeCount = activeByModel.values.reduce(0) { $0 + $1.count }
            return BatchSchedulerMetrics(
                pendingCount: pending.count,
                activeSequences: activeCount,
                totalFusedSteps: _totalFusedSteps,
                totalTokensViaFused: _totalTokensViaFused,
                peakBatchWidth: _peakBatchWidth,
                totalBatches: _totalBatches,
                peakFusedForwardWidth: _peakFusedForwardWidth,
                totalFusedForwards: _totalFusedForwards
            )
        }
    }

    private func noteFusedForward(width: Int, offset: Int, modelId: String) {
        lock.withLock {
            let first = _totalFusedForwards == 0
            _totalFusedForwards += 1
            if width > _peakFusedForwardWidth {
                _peakFusedForwardWidth = width
                NovaMLXLog.info("[BatchScheduler] peak shared forward width=\(width) offset=\(offset) model=\(modelId)")
            } else if first {
                NovaMLXLog.info("[BatchScheduler] shared forward width=\(width) offset=\(offset) model=\(modelId)")
            }
        }
        NovaMLXLog.debug("[BatchScheduler] shared forward width=\(width) offset=\(offset) model=\(modelId)")
    }

    /// Stacked decode for one equal-offset cohort. Nil when the forward cannot
    /// be built; KV is untouched in that case.
    private func cohortLogits(
        sequences: [ActiveBatchSequence],
        mlxContainer: MLXLMCommon.ModelContainer
    ) async -> MLXArray? {
        guard sequences.count >= 2,
              let layerCount = sequences.first?.caches.count, layerCount > 0,
              sequences.allSatisfy({ $0.caches.count == layerCount }),
              sequences.allSatisfy({ $0.generatedTokens.last != nil })
        else { return nil }

        var fusedCaches: [KVCache] = []
        fusedCaches.reserveCapacity(layerCount)
        for layerIdx in 0..<layerCount {
            let fused = FusedBatchKVCache(maxBatchSize: max(maxBatchSize, sequences.count))
            for seq in sequences {
                guard let simple = seq.caches[layerIdx] as? KVCacheSimple else { return nil }
                _ = fused.addSequence(id: seq.id, prefillCache: simple)
            }
            guard fused.activeCount == sequences.count else { return nil }
            fusedCaches.append(fused)
        }

        let stackedTokens = stacked(
            sequences.map { MLXArray(Int32($0.generatedTokens.last!)) },
            axis: 0
        ).reshaped(sequences.count, 1)

        let logitsBox = await mlxContainer.perform(
            nonSendable: (stackedTokens, fusedCaches, sequences)
        ) { context, values in
            let (tokens, caches, seqs) = values
            let rawLogits = context.model(tokens, cache: caches)
            eval(rawLogits)
            for seq in seqs {
                eval(seq.caches)
            }
            return SendableBox(rawLogits)
        }
        return logitsBox.value
    }

    /// One sequence, its own caches. Used for unequal offsets and non-simple KV.
    private func serialRowLogits(
        sequence: ActiveBatchSequence,
        mlxContainer: MLXLMCommon.ModelContainer
    ) async -> MLXArray? {
        guard let last = sequence.generatedTokens.last, !sequence.caches.isEmpty else { return nil }
        let token = MLXArray(Int32(last)).reshaped(1, 1)
        let box = await mlxContainer.perform(nonSendable: (token, sequence.caches)) { context, values in
            let (tokens, caches) = values
            let raw = context.model(tokens, cache: caches)
            eval(raw)
            eval(caches)
            return SendableBox(raw)
        }
        return box.value[0..., -1, 0...]
    }
}

public struct BatchSchedulerMetrics: Codable, Sendable {
    public let pendingCount: Int
    public let activeSequences: Int
    public let totalFusedSteps: UInt64
    public let totalTokensViaFused: UInt64
    public let peakBatchWidth: Int
    public let totalBatches: UInt64
    public let peakFusedForwardWidth: Int
    public let totalFusedForwards: UInt64

    public init(
        pendingCount: Int,
        activeSequences: Int,
        totalFusedSteps: UInt64,
        totalTokensViaFused: UInt64,
        peakBatchWidth: Int,
        totalBatches: UInt64,
        peakFusedForwardWidth: Int = 0,
        totalFusedForwards: UInt64 = 0
    ) {
        self.pendingCount = pendingCount
        self.activeSequences = activeSequences
        self.totalFusedSteps = totalFusedSteps
        self.totalTokensViaFused = totalTokensViaFused
        self.peakBatchWidth = peakBatchWidth
        self.totalBatches = totalBatches
        self.peakFusedForwardWidth = peakFusedForwardWidth
        self.totalFusedForwards = totalFusedForwards
    }
}
