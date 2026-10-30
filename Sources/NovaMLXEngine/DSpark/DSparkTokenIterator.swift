import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore

// Speculative decoding driver for DSpark drafters (milestone 3). Mirrors
// DFlashTokenIterator's control flow — greedy accept, rollback, dense fallback —
// but feeds the drafter the target's fused hidden-state taps directly: prefill
// commits every chunk's fused rows, and each verify commits the accepted rows
// before the next block runs, so `selectBlock` is called with `pendingFused: nil`.

/// One DSpark round. `cap == 0` is a dense target step (no drafted tokens).
public struct DSparkRoundStats: Sendable {
    public let proposed: Int
    public let accepted: Int
    public let cap: Int
    public let draftMs: Double
    public let verifyMs: Double
    public let acceptMs: Double
    public let rollbackMs: Double

    /// Tokens the drafter emitted before confidence scheduling.
    public let draftLen: Int

    public init(
        proposed: Int, accepted: Int, cap: Int,
        draftMs: Double, verifyMs: Double, acceptMs: Double, rollbackMs: Double,
        draftLen: Int? = nil
    ) {
        self.proposed = proposed
        self.accepted = accepted
        self.cap = cap
        self.draftMs = draftMs
        self.verifyMs = verifyMs
        self.acceptMs = acceptMs
        self.rollbackMs = rollbackMs
        self.draftLen = draftLen ?? proposed
    }

    public var trace: String {
        "draft=\(draftLen) verified=\(proposed) accepted=\(accepted) cap=\(cap) "
            + "draftMs=\(Self.ms(draftMs)) verifyMs=\(Self.ms(verifyMs)) "
            + "acceptMs=\(Self.ms(acceptMs)) rollbackMs=\(Self.ms(rollbackMs))"
    }

    private static func ms(_ value: Double) -> String {
        String(format: "%.1f", value)
    }
}

/// Process-wide store for DSpark drafters. The vLLM-speculators checkpoints ship no
/// tokenizer files and no chat `model_type`, so they cannot go through
/// `LLMModelFactory` / the model pool — the MLXEngine dispatch sites lazy-load them
/// here, keyed by directory path. Weight reads after the load are read-only; each
/// request carries its own `DSparkCtxCache`.
public final class DSparkDrafterStore: @unchecked Sendable {
    public static let shared = DSparkDrafterStore()

    private let lock = NSLock()
    private var drafters: [String: DSparkDraftModel] = [:]

    private init() {}

    /// True when `modelId` is an on-disk DSpark drafter this store can load (config
    /// parses into a supported family). Surfaces as admin/API `companionReady`
    /// (not `loaded`) via `InferenceService.isCompanionReady`.
    public static func isLoadableDSparkDraft(modelId: String) -> Bool {
        let dir = NovaMLXPaths.directory(forModelId: modelId)
        guard DSparkConfig.isDSparkDraftConfig(at: dir) else { return false }
        guard let data = try? Data(
            contentsOf: dir.appendingPathComponent("config.json"))
        else { return false }
        return (try? DSparkConfig.fromJSON(data)) != nil
    }

    /// Load (or fetch) the drafter under `dir`. Config comes from `config.json`;
    /// weights are the merged contents of every `*.safetensors` in the directory,
    /// verified 1:1 against the module tree by `DSparkDraftModel.loadWeights`.
    public func drafter(for dir: URL) throws -> DSparkDraftModel {
        let key = dir.path
        lock.lock()
        if let cached = drafters[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let config = try DSparkConfig.fromJSON(
            Data(contentsOf: dir.appendingPathComponent("config.json")))
        let model = DSparkDraftModel(config)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(atPath: dir.path)
        where file.hasSuffix(".safetensors") {
            weights.merge(try MLX.loadArrays(url: dir.appendingPathComponent(file))) {
                _, new in new
            }
        }
        guard !weights.isEmpty else {
            throw DSparkError.invalidConfig("no .safetensors under \(dir.path)")
        }
        try model.loadWeights(weights)

        lock.lock()
        defer { lock.unlock() }
        if let cached = drafters[key] { return cached }
        drafters[key] = model
        return model
    }

    /// Test hook: drop every cached drafter.
    public func reset() {
        lock.lock()
        drafters.removeAll()
        lock.unlock()
    }
}

/// Greedy speculative decoding with a DSpark block drafter. Acceptance is the
/// DFlash greedy rule — longest matching prefix of drafts vs target argmax,
/// then the first mismatched (bonus) token. Adaptive mode drafts up to
/// `kdraft` and verifies the confidence-scheduled prefix. Fixed mode keeps
/// the vector-SDPA cap (`vectorSDPAMaxDrafts`).
public struct DSparkTokenIterator: TokenIteratorProtocol {
    /// Drafts per verify while staying on Metal vector SDPA for GQA-6 / head-256.
    /// Verify width is anchor + drafts, and that kernel requires
    /// `queryLen * (heads / kvHeads) <= 32` with `queryLen <= 8`.
    static let vectorSDPAMaxDrafts = 4

    let target: any DFlashTarget
    let drafter: DSparkDraftModel
    var mainCache: [KVCache]
    var draftCache: [DSparkCtxCache]
    var processor: LogitProcessor?
    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (DSparkRoundStats) -> Void)?
    /// Replaces the confidence head for verify-length scheduling. Tests use
    /// this to force a prefix without depending on random head weights.
    public var confidenceScheduleOverride: [Float]?

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var pending: Int = 0
    /// Committed drafter ctx rows — the rope offset of the next block or ctx append.
    private var ctxOffset = 0
    private let blockSize: Int
    private let maskId: Int
    private let tap: [Int]
    private let maxCap: Int
    private let verifyConfig: AdaptiveVerifyConfig
    private let calibrationKey: String?
    private let calibrator: AdaptiveVerifyCalibrator?

    public init(
        input: LMInput,
        target: any DFlashTarget,
        drafter: DSparkDraftModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processor: LogitProcessor? = nil,
        verifyConfig: AdaptiveVerifyConfig = .resolved(),
        calibrationKey: String? = nil,
        calibrator: AdaptiveVerifyCalibrator? = nil
    ) throws {
        self.target = target
        self.drafter = drafter
        self.mainCache = try mainCache ?? target.newCache(parameters: parameters)
        self.draftCache = drafter.makeCtxCache()
        self.processor = processor ?? parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.blockSize = drafter.config.blockSize
        self.maskId = drafter.config.maskTokenId
        self.tap = drafter.config.targetLayerIds
        self.maxCap = Swift.max(1, drafter.kdraft)
        self.verifyConfig = verifyConfig
        self.calibrationKey = calibrationKey
        self.calibrator = calibrationKey == nil ? nil : (calibrator ?? .shared)
        // This head ships no embed_tokens / lm_head — run on the target's raw tables.
        drafter.bind(
            embed: { [target] ids in target.dflashEmbed(ids) },
            lmHead: { [target] h in target.dflashLmHead(h) }
        )

        let start = Date()
        try prepare(input: input)
        self.promptPrefillTime = Date().timeIntervalSince(start)
    }

    mutating func prepare(input: LMInput) throws {
        processor?.prompt(input.text.tokens)
        var prompt = input.text.tokens
        if prompt.ndim == 1 { prompt = prompt.reshaped(1, -1) }
        let total = prompt.dim(1)
        guard total > 0 else { return }
        let chunk = DFlashLimits.prefillChunkTokens
        var logits = MLXArray.zeros([1, 1, 1])
        var start = 0
        while start < total {
            let end = Swift.min(start + chunk, total)
            let slice = prompt[0..., start ..< end]
            let step = target.dflashVerify(slice, cache: mainCache, tapLayerIds: tap)
            if start == 0 {
                // fc expects nTaps × drafter-hidden — a wrong drafter/target pairing
                // must fail loudly here, not as an MLX shape error mid-forward.
                let expected = tap.count * drafter.config.hiddenSize
                guard step.fused.dim(2) == expected else {
                    throw DSparkError.invalidConfig(
                        "fused tap width \(step.fused.dim(2)) != \(expected) "
                            + "(\(tap.count) taps × hidden \(drafter.config.hiddenSize)) "
                            + "— drafter/target mismatch")
                }
            }
            eval(step.logits, step.fused)
            drafter.updateContext(step.fused, offset: start, cache: draftCache)
            logits = step.logits
            start = end
        }
        ctxOffset = total
        var stepLogits = logits[0..., -1, 0...]
        stepLogits = processor?.process(logits: stepLogits) ?? stepLogits
        let token = argMax(stepLogits, axis: -1).item(Int.self)
        processor?.didSample(token: MLXArray([Int32(token)]))
        pending = token
        pendingTokens = [token]
    }

    mutating func speculateRound() {
        let budget = maxTokens.map { $0 - tokenCount } ?? (maxCap + 1)
        guard budget > 0 else { return }
        // A round commits the accepted drafts plus one bonus token. Leave that
        // bonus inside the remaining budget so the last token matches dense decode.
        let thisCap = Swift.min(maxCap, budget - 1)
        if thisCap <= 0 {
            denseStep()
            return
        }
        // Joint multi-token verify. At temp==0 a long window can flip near-tie
        // argmax vs width-1 decode when it leaves vector SDPA (query length > 5
        // for head 256 / GQA 6). Fixed mode stays inside that window. Adaptive
        // mode may cross it only when the confidence head says the extra drafts
        // are worth the slower kernel. Acceptance is still the greedy prefix.
        let tDraft = Date()
        let blockIds = [Int32(pending)] + Array(repeating: Int32(maskId), count: blockSize - 1)
        let block = MLXArray(blockIds, [1, blockSize])
        let kernelCap = Swift.min(thisCap, Self.vectorSDPAMaxDrafts)
        let draftCap: Int
        if verifyConfig.mode == .adaptive {
            let ceiling = verifyConfig.maxVerify.map { Swift.min(thisCap, $0) } ?? thisCap
            draftCap = Swift.max(0, ceiling)
        } else {
            draftCap = kernelCap
        }
        let draft = drafter.selectBlock(
            block: block, pendingFused: nil, cache: draftCache,
            cap: draftCap, anchorId: pending)
        let confidences = confidenceScheduleOverride ?? draft.confidences
        let verifyLen = verifyConfig.verifyLength(
            confidences: confidences, draftCount: draft.tokens.count, fixedLength: kernelCap,
            positionTemperatures: adaptivePositionTemperatures(
                config: verifyConfig, calibrator: calibrator, key: calibrationKey))
        let drafted = Array(draft.tokens.prefix(verifyLen))
        let draftMs = Date().timeIntervalSince(tDraft) * 1000

        let tVerify = Date()
        var verifyList = [Int32(pending)]
        verifyList.append(contentsOf: drafted.map { Int32($0) })
        let verifyIds = MLXArray(verifyList, [1, verifyList.count])
        let (vLogits, vFused) = target.dflashVerify(
            verifyIds, cache: mainCache, tapLayerIds: tap)
        let argmax = argmaxRows(vLogits)
        eval(vFused)
        let verifyMs = Date().timeIntervalSince(tVerify) * 1000
        finishRound(
            drafted: drafted, argmax: argmax, fused: vFused,
            draftMs: draftMs, verifyMs: verifyMs, draftLen: draft.tokens.count,
            rawConfidences: confidences)
    }

    /// One target token. Still a tapped `dflashVerify` — the anchor's fused row must
    /// reach the drafter ctx or the next draft attends over a hole.
    private mutating func denseStep() {
        let tVerify = Date()
        let verifyIds = MLXArray([Int32(pending)], [1, 1])
        let step = target.dflashVerify(verifyIds, cache: mainCache, tapLayerIds: tap)
        var row = step.logits[0..., -1, 0...]
        row = processor?.process(logits: row) ?? row
        let token = argMax(row, axis: -1).item(Int.self)
        drafter.updateContext(step.fused, offset: ctxOffset, cache: draftCache)
        ctxOffset += 1
        asyncEval(mainCache.flatMap { $0.state })
        let verifyMs = Date().timeIntervalSince(tVerify) * 1000

        pendingTokens.append(token)
        pending = token
        processor?.didSample(token: MLXArray([Int32(token)]))
        onSpeculationRound?(
            DSparkRoundStats(
                proposed: 0, accepted: 0, cap: 0,
                draftMs: 0, verifyMs: verifyMs,
                acceptMs: 0, rollbackMs: 0))
    }


    private func argmaxRows(_ logits: MLXArray) -> [Int] {
        var argmax = [Int]()
        let seq = logits.dim(1)
        for i in 0 ..< seq {
            var row = logits[0..., i, 0...]
            row = processor?.process(logits: row) ?? row
            argmax.append(argMax(row, axis: -1).item(Int.self))
        }
        return argmax
    }


    private mutating func finishRound(
        drafted: [Int], argmax: [Int], fused: MLXArray,
        draftMs: Double, verifyMs: Double, draftLen: Int,
        rawConfidences: [Float]
    ) {
        let tAccept = Date()
        let result = DFlashAccept.greedy(drafted: drafted, targetArgmax: argmax)
        let acceptMs = Date().timeIntervalSince(tAccept) * 1000
        let n = result.accepted
        totalProposed += drafted.count
        totalAccepted += n
        recordAdaptiveVerifyOutcome(
            calibrator: calibrator, key: calibrationKey,
            rawConfidences: rawConfidences, verified: drafted.count,
            accepted: n, prior: verifyConfig.confidenceTemperature)
        let tRoll = Date()
        target.dflashRollback(
            cache: mainCache, nRejected: drafted.count - n, specWidth: 1 + drafted.count)
        let rollbackMs = Date().timeIntervalSince(tRoll) * 1000

        // Commit the fused rows of the anchor plus the accepted drafts — the bonus
        // token's fused row arrives as next round's anchor row.
        let keep = Swift.min(n + 1, fused.dim(1))
        if keep > 0 {
            drafter.updateContext(fused[0..., ..<keep, 0...], offset: ctxOffset, cache: draftCache)
            ctxOffset += keep
        }
        pendingTokens.append(contentsOf: result.committed)
        if let last = result.committed.last {
            pending = last
            for tid in result.committed {
                processor?.didSample(token: MLXArray([Int32(tid)]))
            }
        }
        onSpeculationRound?(
            DSparkRoundStats(
                proposed: drafted.count, accepted: n, cap: drafted.count,
                draftMs: draftMs, verifyMs: verifyMs,
                acceptMs: acceptMs, rollbackMs: rollbackMs,
                draftLen: draftLen))
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        if pendingIndex >= pendingTokens.count {
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            speculateRound()
            if pendingTokens.isEmpty { return nil }
        }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return t
    }
}
