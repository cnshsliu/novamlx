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
        let maxTokens = req.request.maxTokens ?? container.config.maxTokens

        let cachesBox = MutableSendableBox<[KVCache]?>(nil)
        let prefillParams: GenerateParameters = {
            var p = GenerateParameters(maxTokens: 1, temperature: 0.6)
            p.prefillStepSize = 512
            return p
        }()

        let stream = try await mlxContainer.perform(nonSendable: input) { context, input in
            let cache = context.model.newCache(parameters: prefillParams)
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
            }
        }

        guard let caches = cachesBox.value else {
            throw NovaMLXError.inferenceFailed("Prefill produced no cache")
        }

        let parameters = engine.buildGenerateParameters(request: req.request, config: container.config, settings: engine.settingsProvider?(req.request.model))
        let firstTokenId = container.tokenizer?.encode(firstTokenText).first ?? 0

        let seq = ActiveBatchSequence(
            id: req.id,
            request: req.request,
            caches: caches,
            generatedTokens: [firstTokenId],
            generatedText: firstTokenText,
            maxTokens: maxTokens,
            promptTokenCount: promptTokenCount,
            sampler: parameters.sampler(),
            continuation: req.continuation
        )

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

            guard let firstCaches = active.first?.caches, !firstCaches.isEmpty else {
                lock.withLock { activeByModel[modelId] = sequences }
                continue
            }
            let numLayers = firstCaches.count

            let stackedTokens = stacked(
                active.map { MLXArray(Int32($0.generatedTokens.last!)) },
                axis: 0
            ).reshaped(batchSize, 1)

            let fusedCaches: [KVCache] = (0..<numLayers).map { layerIdx in
                let fbc = FusedBatchKVCache(maxBatchSize: maxBatchSize)
                for seq in active {
                    if let simpleCache = seq.caches[layerIdx] as? KVCacheSimple {
                        _ = fbc.addSequence(id: seq.id, prefillCache: simpleCache)
                    }
                }
                return fbc
            }

            let logitsBox = await mlxContainer.perform(
                nonSendable: (stackedTokens, fusedCaches)
            ) { context, values in
                let (tokens, caches) = values
                let rawLogits = context.model(tokens, cache: caches)
                eval(rawLogits)
                return SendableBox(rawLogits)
            }
            let logits = logitsBox.value

            let eosId = container.tokenizer?.eosTokenId

            for (i, var seq) in active.enumerated() {
                let seqLogits = logits[i][0..., -1, 0...]
                let sampledToken = seq.sampler.sample(logits: seqLogits)
                let tokenId = sampledToken.item(Int.self)

                seq.generatedTokens.append(tokenId)
                if let tok = container.tokenizer {
                    seq.generatedText += tok.decode([tokenId])
                }

                let isEOS = eosId == tokenId
                let atMax = seq.generatedTokens.count >= seq.maxTokens

                if isEOS || atMax {
                    let result = InferenceResult(
                        id: seq.id,
                        model: seq.request.model,
                        text: seq.generatedText,
                        tokensPerSecond: 0,
                        promptTokens: seq.promptTokenCount,
                        completionTokens: seq.generatedTokens.count,
                        finishReason: atMax ? .length : .stop
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
                totalBatches: _totalBatches
            )
        }
    }
}

public struct BatchSchedulerMetrics: Codable, Sendable {
    public let pendingCount: Int
    public let activeSequences: Int
    public let totalFusedSteps: UInt64
    public let totalTokensViaFused: UInt64
    public let peakBatchWidth: Int
    public let totalBatches: UInt64
}
