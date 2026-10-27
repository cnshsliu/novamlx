import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NovaMLXCore
import NovaMLXUtils

// MARK: - Compiled decode reporting

/// Models that publish how many single-token decode segments `mlx-swift-lm` compiled.
public protocol CompiledDecodeReporting: AnyObject {
    var compiledDecodeSegmentCount: Int { get }
}

// MARK: - DFlash target

/// Same contract as `MLXLMCommon.DFlashTarget` — Qwen3/Qwen3.5/etc. conform there.
/// Keep a single protocol identity so `model.value as? any DFlashTarget` succeeds.
public typealias DFlashTarget = MLXLMCommon.DFlashTarget

public struct DFlashRoundStats: Sendable {
    public let proposed: Int
    public let accepted: Int
    public let cap: Int
    public let draftMs: Double
    public let verifyMs: Double
    public let acceptMs: Double
    public let rollbackMs: Double

    /// Tokens the drafter emitted before confidence scheduling. Equals `proposed`
    /// when the whole draft is verified.
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

public enum DFlashError: Error, Sendable {
    case invalidConfig(String)
}

/// Greedy accept: matching draft prefix, then the first target token that disagrees.
public enum DFlashAccept {
    public struct Result: Sendable, Equatable {
        public let accepted: Int
        public let committed: [Int]
    }

    public static func greedy(drafted: [Int], targetArgmax: [Int]) -> Result {
        var accepted = 0
        while accepted < drafted.count, accepted < targetArgmax.count,
              drafted[accepted] == targetArgmax[accepted]
        {
            accepted += 1
        }
        var committed = Array(drafted.prefix(accepted))
        if accepted < targetArgmax.count {
            committed.append(targetArgmax[accepted])
        }
        return Result(accepted: accepted, committed: committed)
    }
}

/// Bounds for DFlash verify capture and chunked prefill.
public enum DFlashLimits {
    public static let prefillChunkTokens = 4096

    /// GDN spec-capture is only valid for the short verify window.
    public static func shouldRecordSpec(sequenceLength: Int) -> Bool {
        (2...8192).contains(sequenceLength)
    }
}

public struct DFlashConfig: Sendable {
    public var hiddenSize: Int
    public var vocabSize: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var blockSize: Int
    public var convKernelSize: Int
    public var maskTokenId: Int
    public var selectorRank: Int
    public var selectorTopK: Int
    public var targetLayerIds: [Int]
    /// Group width of the local conv. Checkpoints that omit it use 16.
    public var convGroupSize: Int = 16
    public var numAttentionHeads: Int = 1
    public var headDim: Int = 0
    public var intermediateSize: Int = 0
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 1_000_000
    public var maxPositionEmbeddings: Int = 262_144
    public var layerTypes: [String] = []
    public var slidingWindow: Int?
    public var isDFlash2: Bool

    public static func fromJSON(_ data: Data) throws -> DFlashConfig {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DFlashError.invalidConfig("not a JSON object")
        }
        let nested = json["dflash_config"] as? [String: Any] ?? [:]
        func value(_ key: String) -> Any? {
            json[key] ?? nested[key]
        }
        let architectures = json["architectures"] as? [String] ?? []
        let isDFlash2Arch = architectures.contains { $0.localizedCaseInsensitiveContains("dflash2") }
        let selectorRank = jsonInt(value("selector_rank"))
        let selectorTopK = jsonInt(value("selector_top_k"))
        if isDFlash2Arch && (selectorRank == nil || selectorTopK == nil) {
            throw DFlashError.invalidConfig("DFlash2 config is missing selector_rank or selector_top_k")
        }
        let rank = selectorRank ?? 0
        let topK = selectorTopK ?? 0
        var cfg = DFlashConfig(
            hiddenSize: jsonInt(value("hidden_size")) ?? 0,
            vocabSize: jsonInt(value("vocab_size")) ?? 0,
            numHiddenLayers: jsonInt(value("num_hidden_layers")) ?? 1,
            numKeyValueHeads: jsonInt(value("num_key_value_heads")) ?? 1,
            blockSize: jsonInt(value("block_size")) ?? 8,
            convKernelSize: jsonInt(value("conv_kernel_size")) ?? 0,
            maskTokenId: jsonInt(value("mask_token_id")) ?? 0,
            selectorRank: rank,
            selectorTopK: topK,
            targetLayerIds: jsonInts(value("target_layer_ids")),
            isDFlash2: isDFlash2Arch || (rank > 0 && topK > 0)
        )
        cfg.convGroupSize = jsonInt(value("conv_group_size")) ?? 16
        let hidden = cfg.hiddenSize
        let heads = jsonInt(value("num_attention_heads")) ?? max(cfg.numKeyValueHeads, 1)
        cfg.numAttentionHeads = max(heads, 1)
        cfg.headDim = jsonInt(value("head_dim"))
            ?? (hidden > 0 ? max(hidden / cfg.numAttentionHeads, 1) : 1)
        cfg.intermediateSize = jsonInt(value("intermediate_size")) ?? max(hidden, 1)
        cfg.rmsNormEps = jsonFloat(value("rms_norm_eps")) ?? 1e-6
        let rope = json["rope_parameters"] as? [String: Any] ?? [:]
        cfg.ropeTheta = jsonFloat(value("rope_theta")) ?? jsonFloat(rope["rope_theta"]) ?? 1_000_000
        cfg.maxPositionEmbeddings = jsonInt(value("max_position_embeddings")) ?? 262_144
        cfg.layerTypes = json["layer_types"] as? [String] ?? []
        cfg.slidingWindow = jsonInt(value("sliding_window"))
        return cfg
    }
}

/// The pool model is a DFlash draft, or the directory is a DFlash checkpoint
/// loaded as some other class. Weights are read once from disk.
/// Returns the drafter even when the selector is missing, so the caller can
/// fall back to plain decode instead of another speculative iterator.
func dflashDrafterReady(_ model: Any, draftModelId: String) -> DFlashDraftModel? {
    let dir = NovaMLXPaths.directory(forModelId: draftModelId)
    if let drafter = model as? DFlashDraftModel {
        if !drafter.hasSpeculativeWeights {
            drafter.loadSpeculativeWeightsIfPresent(at: dir)
        }
        return drafter
    }
    guard NovaMLXEngine.isDFlashDraftConfig(at: dir) else { return nil }
    return DFlashDiskDrafters.shared.drafter(at: dir)
}

/// Disk-built DFlash drafts for checkpoints the model pool did not instantiate
/// as ``DFlashDraftModel``. One instance per directory.
private final class DFlashDiskDrafters: @unchecked Sendable {
    static let shared = DFlashDiskDrafters()
    private let lock = NovaMLXLock()
    private var models: [String: DFlashDraftModel] = [:]

    func drafter(at dir: URL) -> DFlashDraftModel {
        let key = dir.path
        if let cached = lock.withLock({ models[key] }) { return cached }
        let model: DFlashDraftModel
        if let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
           let config = try? DFlashConfig.fromJSON(data)
        {
            let built = DFlashDraftModel(config)
            built.loadSpeculativeWeightsIfPresent(at: dir)
            model = built
        } else {
            model = DFlashDraftModel(DFlashConfig(
                hiddenSize: 1, vocabSize: 1, numHiddenLayers: 1, numKeyValueHeads: 1,
                blockSize: 8, convKernelSize: 0, maskTokenId: 0,
                selectorRank: 0, selectorTopK: 0, targetLayerIds: [],
                isDFlash2: false
            ))
        }
        lock.withLock { models[key] = model }
        return model
    }
}

/// True when a DFlash checkpoint is on disk with weights the drafter can read.
/// Config alone is not enough: a failed download must not look loaded.
public func dflashCompanionReady(modelId: String) -> Bool {
    let dir = NovaMLXPaths.directory(forModelId: modelId)
    guard NovaMLXEngine.isDFlashDraftConfig(at: dir) else { return false }
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
          (try? DFlashConfig.fromJSON(data)) != nil
    else { return false }
    if let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path),
       files.contains(where: { $0.hasSuffix(".safetensors") })
    {
        return true
    }
    return FileManager.default.fileExists(
        atPath: dir.appendingPathComponent("model.safetensors.index.json").path)
}

func adaptiveVerifyConfig(for request: InferenceRequest) -> AdaptiveVerifyConfig {
    AdaptiveVerifyConfig.resolved(
        modeOverride: request.adaptiveVerify,
        minOverride: request.adaptiveVerifyMin,
        maxOverride: request.adaptiveVerifyMax
    )
}

/// Running accept/reject totals for one generation. The HTTP response reads
/// these after the iterator finishes.
final class SpeculationMeter: @unchecked Sendable {
    private let lock = NSLock()
    private var proposed = 0
    private var accepted = 0

    func record(proposed: Int, accepted: Int) {
        lock.lock()
        self.proposed += max(proposed, 0)
        self.accepted += max(accepted, 0)
        lock.unlock()
    }

    var totals: (proposed: Int, accepted: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (proposed, accepted)
    }
}

func adaptivePositionTemperatures(
    config: AdaptiveVerifyConfig,
    calibrator: AdaptiveVerifyCalibrator?,
    key: String?
) -> [Float]? {
    guard config.mode == .adaptive, let calibrator, let key else { return nil }
    let temperatures = calibrator.temperatures(for: key)
    return temperatures.isEmpty ? nil : temperatures
}

func recordAdaptiveVerifyOutcome(
    calibrator: AdaptiveVerifyCalibrator?,
    key: String?,
    rawConfidences: [Float],
    verified: Int,
    accepted: Int,
    prior: Float
) {
    guard let calibrator, let key, verified > 0 else { return }
    let slice = Array(rawConfidences.prefix(verified))
    guard !slice.isEmpty else { return }
    calibrator.observe(
        confidences: slice,
        accepted: min(max(accepted, 0), slice.count),
        key: key,
        prior: prior
    )
}

/// Fit rounds a previous request left in memory, before this request reads temperatures.
func beginAdaptiveVerifyRequest(key: String) {
    AdaptiveVerifyCalibrator.shared.flush(pairKey: key)
}

/// Fit this request's rounds after decode timing has already been captured.
func finishAdaptiveVerifyRequest(key: String?) {
    guard let key else { return }
    AdaptiveVerifyCalibrator.shared.flush(pairKey: key)
}

/// Start a DFlash2 iterator when the checkpoint is a real drafter and the
/// request is greedy. Temperature above 0 stays on the ordinary sampler:
/// block accept is argmax. Returns nil when this request should not speculate.
func openDFlashIterator(
    model: any LanguageModel,
    draftModelId: String,
    input: LMInput,
    cache: [KVCache]?,
    parameters: GenerateParameters,
    processor: LogitProcessor?,
    promptTokenCount: Int,
    logTag: String,
    verifyConfig: AdaptiveVerifyConfig = .resolved(),
    calibrationKey: String? = nil,
    calibrator: AdaptiveVerifyCalibrator? = nil
) throws -> DFlashTokenIterator? {
    let drafter = dflashDrafterReady(model, draftModelId: draftModelId)
    let canTap = model is any DFlashTarget || DFlashHooks.make(model: model) != nil
    let promptOK = promptTokenCount <= ResourceLimits.dflashMaxPromptTokens
    let greedy = parameters.temperature <= 0
    guard greedy, promptOK, canTap, let drafter, drafter.hasSpeculativeWeights else {
        if !greedy {
            NovaMLXLog.info(
                "[\(logTag)] DFlash2 '\(draftModelId)' idle — temperature \(parameters.temperature) > 0; one-token decode"
            )
        } else if drafter?.hasSpeculativeWeights != true {
            NovaMLXLog.info(
                "[\(logTag)] DFlash2 '\(draftModelId)' has no path-selector weights — plain decode"
            )
        } else if !canTap {
            NovaMLXLog.info(
                "[\(logTag)] DFlash2 '\(draftModelId)' target has no hidden-state taps — plain decode"
            )
        } else {
            NovaMLXLog.info(
                "[\(logTag)] Skipping DFlash2 — prompt \(promptTokenCount) > \(ResourceLimits.dflashMaxPromptTokens) tok"
            )
        }
        return nil
    }
    let iterator = try DFlashTokenIterator(
        input: input,
        model: model,
        drafter: drafter,
        mainCache: cache,
        parameters: parameters,
        processor: processor,
        verifyConfig: verifyConfig,
        calibrationKey: calibrationKey,
        calibrator: calibrator
    )
    let kind = drafter.hasBlockWeights ? "block" : "selector"
    let activeCalibrator = calibrationKey == nil ? nil : (calibrator ?? .shared)
    let calibrationNote = activeCalibrator.map { " " + $0.summary(for: calibrationKey ?? "") } ?? ""
    NovaMLXLog.info(
        "[\(logTag)] DFlash2 draft=\(draftModelId) block=\(drafter.config.blockSize) weights=\(kind) verify=\(verifyConfig.summary)\(calibrationNote)"
    )
    return iterator
}

/// DFlash2 iterator. Refuses to start unless the path selector and local conv
/// are installed, so a weightless stub cannot log a boost and then run plain decode.
public struct DFlashTokenIterator: TokenIteratorProtocol {
    let target: (any DFlashTarget)?
    let hooks: DFlashHooks?
    let drafter: DFlashDraftModel
    var mainCache: [KVCache]
    var processor: LogitProcessor?
    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (DFlashRoundStats) -> Void)?
    public var state: LMOutput.State? { nil }

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var pending: Int = 0
    private var lastLogits: MLXArray
    private var lastHidden: MLXArray
    private var capPolicy: DFlashCapPolicy
    private let verifyConfig: AdaptiveVerifyConfig
    private let calibrationKey: String?
    private let calibrator: AdaptiveVerifyCalibrator?
    private var draftCache: [DFlashCtxCache]
    private var cacheBackup: [KVCache]
    private let tap: [Int]

    public init(
        input: LMInput,
        target: any DFlashTarget,
        drafter: DFlashDraftModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processor: LogitProcessor? = nil,
        verifyConfig: AdaptiveVerifyConfig = .resolved(),
        calibrationKey: String? = nil,
        calibrator: AdaptiveVerifyCalibrator? = nil
    ) throws {
        try self.init(
            input: input,
            target: target,
            hooks: nil,
            drafter: drafter,
            mainCache: mainCache,
            parameters: parameters,
            processor: processor,
            cacheModel: target,
            verifyConfig: verifyConfig,
            calibrationKey: calibrationKey,
            calibrator: calibrator
        )
    }

    /// Same loop as the `DFlashTarget` init. Uses installed `novamlxDFlash*` taps when
    /// the backbone (upstream Qwen3.5, or the VLM) does not conform to `DFlashTarget`.
    public init(
        input: LMInput,
        model: any LanguageModel,
        drafter: DFlashDraftModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processor: LogitProcessor? = nil,
        verifyConfig: AdaptiveVerifyConfig = .resolved(),
        calibrationKey: String? = nil,
        calibrator: AdaptiveVerifyCalibrator? = nil
    ) throws {
        if let target = model as? any DFlashTarget {
            try self.init(
                input: input, target: target, drafter: drafter, mainCache: mainCache,
                parameters: parameters, processor: processor, verifyConfig: verifyConfig,
                calibrationKey: calibrationKey, calibrator: calibrator)
            return
        }
        guard let hooks = DFlashHooks.make(model: model) else {
            throw DFlashError.invalidConfig("target has no DFlash hidden-state taps")
        }
        try self.init(
            input: input,
            target: nil,
            hooks: hooks,
            drafter: drafter,
            mainCache: mainCache,
            parameters: parameters,
            processor: processor,
            cacheModel: model,
            verifyConfig: verifyConfig,
            calibrationKey: calibrationKey,
            calibrator: calibrator
        )
    }

    /// Both public inits delegate here. A delegating init cannot assign `let`
    /// properties on another path, which is what the Apple Silicon compiler rejected.
    private init(
        input: LMInput,
        target: (any DFlashTarget)?,
        hooks: DFlashHooks?,
        drafter: DFlashDraftModel,
        mainCache: [KVCache]?,
        parameters: GenerateParameters,
        processor: LogitProcessor?,
        cacheModel: any LanguageModel,
        verifyConfig: AdaptiveVerifyConfig,
        calibrationKey: String?,
        calibrator: AdaptiveVerifyCalibrator?
    ) throws {
        guard drafter.hasSpeculativeWeights else {
            throw DFlashError.invalidConfig(
                "DFlash2 path selector and local conv are not loaded; refusing to claim a boost for plain decode"
            )
        }
        if drafter.hasBlockWeights {
            if let target {
                drafter.bind(
                    embed: target.dflashEmbedding,
                    lmHead: { target.dflashLmHead($0) }
                )
            } else if let hooks {
                drafter.bind(embed: hooks.embedding, lmHead: hooks.lmHead)
            }
        }
        self.target = target
        self.hooks = hooks
        self.drafter = drafter
        self.mainCache = try mainCache ?? cacheModel.newCache(parameters: parameters)
        self.processor = processor ?? parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.tap = drafter.config.targetLayerIds.isEmpty ? [0] : drafter.config.targetLayerIds
        let maxCap = Swift.max(1, drafter.config.blockSize - 1)
        self.capPolicy = DFlashCapPolicy(maxCap: maxCap)
        self.verifyConfig = verifyConfig
        self.calibrationKey = calibrationKey
        self.calibrator = calibrationKey == nil ? nil : (calibrator ?? .shared)
        self.draftCache = []
        self.cacheBackup = []
        self.lastLogits = MLXArray.zeros([1, 1])
        self.lastHidden = MLXArray.zeros([1, 1, 1])
        let start = Date()
        try prepare(input: input)
        self.promptPrefillTime = Date().timeIntervalSince(start)
    }

    private func verify(_ tokens: MLXArray) -> (logits: MLXArray, fused: MLXArray) {
        if let target {
            return target.dflashVerify(tokens, cache: mainCache, tapLayerIds: tap)
        }
        return hooks!.verify(tokens, mainCache, tap)
    }

    mutating func prepare(input: LMInput) throws {
        processor?.prompt(input.text.tokens)
        var prompt = input.text.tokens
        if prompt.ndim == 1 { prompt = prompt.reshaped(1, -1) }
        let total = prompt.dim(1)
        guard total > 0 else { return }
        let chunk = DFlashLimits.prefillChunkTokens
        var logits = MLXArray.zeros([1, 1, 1])
        var fused = MLXArray.zeros([1, 1, 1])
        var start = 0
        while start < total {
            let end = Swift.min(start + chunk, total)
            let slice = prompt[0..., start ..< end]
            let step = verify(slice)
            eval(step.logits, step.fused)
            logits = step.logits
            fused = step.fused
            start = end
        }
        lastLogits = logits
        lastHidden = fused
        var stepLogits = logits[0..., -1, 0...]
        stepLogits = processor?.process(logits: stepLogits) ?? stepLogits
        let token = argMax(stepLogits, axis: -1).item(Int.self)
        processor?.didSample(token: MLXArray([Int32(token)]))
        pending = token
        pendingTokens = [token]
    }

    mutating func speculateRound() {
        let budget = maxTokens.map { $0 - tokenCount } ?? (capPolicy.maxCap + 1)
        guard budget > 0 else { return }
        let probed = capPolicy.cap == 0 && capPolicy.shouldProbe
        let policyCap = Swift.min(probed ? capPolicy.maxCap : capPolicy.cap, budget - 1)
        if policyCap <= 0 {
            denseStep()
            capPolicy.observe(accepted: 0, capUsed: 0)
            return
        }
        // Fixed mode keeps the acceptance-rate cap. Adaptive drafts the full
        // block (while the policy is not parked) and then keeps a confidence
        // prefix. Parking still skips the draft entirely.
        let draftBudget = verifyConfig.mode == .adaptive
            ? Swift.min(capPolicy.maxCap, budget - 1)
            : policyCap

        let tDraft = Date()
        let anchor = MLXArray([Int32(pending)]).reshaped(1, 1)
        let fullIDs: [Int]
        let confidences: [Float]
        if drafter.hasBlockWeights {
            if draftCache.isEmpty { draftCache = drafter.makeCache() }
            let proposal = drafter.draftBlock(
                anchorId: pending, targetHidden: lastHidden, cache: draftCache,
                cap: draftBudget, marginScale: verifyConfig.marginScale)
            fullIDs = proposal.tokens
            confidences = proposal.confidences
        } else {
            let proposal = drafter.proposeGreedy(
                anchorLogits: lastLogits, hidden: lastHidden, cap: draftBudget,
                marginScale: verifyConfig.marginScale)
            fullIDs = proposal.ids
            confidences = proposal.confidences
        }
        let draftMs = Date().timeIntervalSince(tDraft) * 1000
        if fullIDs.isEmpty {
            denseStep()
            capPolicy.observe(accepted: 0, capUsed: 0)
            return
        }
        let verifyLen = verifyConfig.verifyLength(
            confidences: confidences, draftCount: fullIDs.count, fixedLength: policyCap,
            positionTemperatures: adaptivePositionTemperatures(
                config: verifyConfig, calibrator: calibrator, key: calibrationKey))
        let drafted = Array(fullIDs.prefix(verifyLen))
        if drafted.isEmpty {
            denseStep()
            capPolicy.observe(accepted: 0, capUsed: 0)
            return
        }
        let verifyIds = concatenated(
            [anchor, MLXArray(drafted.map { Int32($0) }).reshaped(1, -1)], axis: 1)
        if hooks != nil { cacheBackup = mainCache.map { $0.copy() } }
        let tVerify = Date()
        let verified = verify(verifyIds)
        eval(verified.logits, verified.fused)
        let verifyMs = Date().timeIntervalSince(tVerify) * 1000
        finishRound(
            drafted: drafted, step: verified, draftMs: draftMs, verifyMs: verifyMs,
            draftLen: fullIDs.count, rawConfidences: confidences)
    }

    /// Accept a verified block. The `DFlashTarget` path uses `dflashRollback`.
    /// Hook taps have no rollback, so a partial accept restores the pre-verify
    /// snapshot and replays only the kept prefix. The bonus token stays out of
    /// the cache. A length mismatch restores and takes one target token.
    private mutating func finishRound(
        drafted: [Int],
        step: (logits: MLXArray, fused: MLXArray),
        draftMs: Double,
        verifyMs: Double,
        draftLen: Int,
        rawConfidences: [Float]
    ) {
        let tAccept = Date()
        let argmax = argmaxRows(step.logits)
        let result = DFlashAccept.greedy(drafted: drafted, targetArgmax: argmax)
        let acceptMs = Date().timeIntervalSince(tAccept) * 1000
        let decision = DFlashBlockCheck.decide(
            draftCount: drafted.count,
            logitLength: argmax.count,
            fusedLength: step.fused.ndim >= 2 ? step.fused.dim(1) : 0,
            accepted: result.accepted
        )
        let tRoll = Date()
        if decision.acceptNormally, hooks != nil, decision.nRejected > 0 {
            // The verify already wrote rejected drafts into the cache. Restore
            // the snapshot and replay only the kept prefix so the bonus stays out.
            restoreMainCache()
            let kept = [pending] + Array(drafted.prefix(result.accepted))
            let replayed = replay(kept)
            rememberBonus(from: replayed, accepted: result.accepted)
        } else if decision.acceptNormally {
            if let target, decision.nRejected > 0 {
                target.dflashRollback(
                    cache: mainCache, nRejected: decision.nRejected, specWidth: decision.specWidth)
            }
            rememberBonus(from: step, accepted: result.accepted)
        } else if hooks != nil {
            restoreMainCache()
        } else if let target {
            target.dflashRollback(
                cache: mainCache, nRejected: decision.nRejected, specWidth: decision.specWidth)
            let anchorLeft = decision.specWidth - decision.nRejected
            if anchorLeft > 0 {
                target.dflashRollback(cache: mainCache, nRejected: anchorLeft, specWidth: anchorLeft)
            }
        }
        let rollbackMs = Date().timeIntervalSince(tRoll) * 1000
        let acceptedCount = decision.acceptNormally ? result.accepted : 0
        if decision.acceptNormally {
            recordAdaptiveVerifyOutcome(
                calibrator: calibrator, key: calibrationKey,
                rawConfidences: rawConfidences, verified: drafted.count,
                accepted: acceptedCount, prior: verifyConfig.confidenceTemperature)
        }
        capPolicy.observe(
            accepted: acceptedCount, capUsed: drafted.count,
            scaleToMaxCap: verifyConfig.mode == .adaptive)
        totalProposed += drafted.count
        if decision.acceptNormally, !result.committed.isEmpty {
            totalAccepted += result.accepted
            pendingTokens.append(contentsOf: result.committed)
            if let last = result.committed.last {
                pending = last
                for tid in result.committed {
                    processor?.didSample(token: MLXArray([Int32(tid)]))
                }
            }
            onSpeculationRound?(
                DFlashRoundStats(
                    proposed: drafted.count, accepted: result.accepted, cap: drafted.count,
                    draftMs: draftMs, verifyMs: verifyMs,
                    acceptMs: acceptMs, rollbackMs: rollbackMs,
                    draftLen: draftLen
                )
            )
            return
        }
        if !decision.acceptNormally {
            NovaMLXLog.info("[DFlash] Block length mismatch — dropped the draft and took one target token")
            onSpeculationRound?(
                DFlashRoundStats(
                    proposed: drafted.count, accepted: 0, cap: drafted.count,
                    draftMs: draftMs, verifyMs: verifyMs,
                    acceptMs: acceptMs, rollbackMs: rollbackMs,
                    draftLen: draftLen
                )
            )
        }
        denseStep()
    }

    private mutating func rememberBonus(
        from step: (logits: MLXArray, fused: MLXArray), accepted: Int
    ) {
        let bonus = Swift.min(accepted, step.logits.dim(1) - 1)
        lastLogits = step.logits[0..., bonus ..< (bonus + 1), 0...]
        if step.fused.ndim >= 2, step.fused.dim(1) > bonus {
            lastHidden = step.fused[0..., bonus ..< (bonus + 1), 0...]
        }
    }

    /// Drop rejected drafts. Target models use the installed rollback, which
    /// also rewinds GDN spec capture. Hook taps restore the snapshot instead.
    private mutating func undoVerify(specWidth: Int, nRejected: Int) {
        if hooks != nil {
            restoreMainCache()
            return
        }
        target?.dflashRollback(cache: mainCache, nRejected: nRejected, specWidth: specWidth)
    }

    private mutating func restoreMainCache() {
        guard cacheBackup.count == mainCache.count else { return }
        for index in mainCache.indices {
            var live = mainCache[index]
            let saved = cacheBackup[index]
            live.state = saved.state
            mainCache[index] = live
            if let liveBase = mainCache[index] as? BaseKVCache,
               let savedBase = saved as? BaseKVCache
            {
                liveBase.offset = savedBase.offset
            }
        }
    }

    private mutating func replay(_ tokens: [Int]) -> (logits: MLXArray, fused: MLXArray) {
        guard !tokens.isEmpty else { return (lastLogits, lastHidden) }
        let ids = MLXArray(tokens.map { Int32($0) }, [1, tokens.count])
        let step = verify(ids)
        eval(step.logits, step.fused)
        return step
    }

    private mutating func denseStep() {
        let verifyIds = MLXArray([Int32(pending)], [1, 1])
        let step = verify(verifyIds)
        var row = step.logits[0..., -1, 0...]
        row = processor?.process(logits: row) ?? row
        let token = argMax(row, axis: -1).item(Int.self)
        eval(step.fused)
        lastLogits = step.logits
        lastHidden = step.fused
        pendingTokens.append(token)
        pending = token
        processor?.didSample(token: MLXArray([Int32(token)]))
        onSpeculationRound?(
            DFlashRoundStats(
                proposed: 0, accepted: 0, cap: 0,
                draftMs: 0, verifyMs: 0, acceptMs: 0, rollbackMs: 0
            )
        )
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

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        if pendingIndex >= pendingTokens.count {
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            speculateRound()
            if pendingTokens.isEmpty { return nil }
        }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

// MARK: - MTP

/// Target model that can embed, project, and score an MTP verify window.
public protocol MtpTarget: LanguageModel {
    func mtpEmbed(_ tokens: MLXArray) -> MLXArray
    func mtpLmHead(_ hidden: MLXArray) -> MLXArray
    func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (
        hidden: MLXArray, logits: MLXArray
    )
}

/// Drafter head. DeepSeek V4 is both ``MtpTarget`` and ``MtpDrafter``.
///
/// These protocols are not MLXLMCommon's ``MTPDrafterModel``. Qwen greedy
/// verify uses that type and ``MTPSpeculativeTokenIterator``. Porting the
/// DeepSeek head onto ``MTPDrafterModel`` is a follow-up; until then Engine
/// must not run ``MtpTokenIterator``.
public protocol MtpDrafter: AnyObject {
    var mtpBlockSize: Int { get }
    func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray,
        lmHead: @escaping (MLXArray) -> MLXArray
    )
    func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray
    func mtpNewCache(parameters: GenerateParameters?) -> [KVCache]
}

public enum MtpVerifyError: Error, Sendable, CustomStringConvertible {
    /// The old iterator sampled with ``TokenIterator`` and reported MTP.
    case engineStubDisabled

    public var description: String {
        "Nova MtpTokenIterator is disabled. It did not verify drafts. Qwen greedy uses MLXLMCommon.MTPSpeculativeTokenIterator."
    }
}

/// Disabled. Constructing it throws ``MtpVerifyError/engineStubDisabled``.
/// The previous body ignored the drafter and ran ordinary decode.
public struct MtpTokenIterator: TokenIteratorProtocol {
    public var maxTokens: Int? { nil }
    public var tokenCount: Int { 0 }
    public var promptPrefillTime: TimeInterval { 0 }
    public var state: LMOutput.State? { nil }

    public init(
        input: LMInput,
        target: any MtpTarget,
        drafter: any MtpDrafter,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processor: LogitProcessor? = nil
    ) throws {
        _ = input
        _ = target
        _ = drafter
        _ = mainCache
        _ = parameters
        _ = numDraftTokens
        _ = processor
        throw MtpVerifyError.engineStubDisabled
    }

    public mutating func next() -> Int? { nil }
}

// MARK: - Companion discovery

public func isDSparkDraftConfig(at modelDir: URL) -> Bool {
    DSparkConfig.isDSparkDraftConfig(at: modelDir)
}

public func isDFlashDraftConfig(at modelDir: URL) -> Bool {
    guard let json = modelConfigJSON(at: modelDir) else { return false }
    if DSparkConfig.isDSparkDraftConfig(jsonObject: json) { return false }
    let modelType = (json["model_type"] as? String)?.lowercased() ?? ""
    if modelType.contains("dflash") { return true }
    let architectures = json["architectures"] as? [String] ?? []
    if architectures.contains(where: { $0.lowercased().contains("dflash") }) { return true }
    return json["dflash_config"] != nil
}

public func isMtpDraftConfig(at modelDir: URL) -> Bool {
    guard let json = modelConfigJSON(at: modelDir) else { return false }
    let modelType = (json["model_type"] as? String)?.lowercased() ?? ""
    if modelType == "qwen3_5_mtp" || modelType.hasSuffix("_mtp") { return true }
    let architectures = json["architectures"] as? [String] ?? []
    return architectures.contains { $0.lowercased().contains("mtp") }
}

public func checkpointHasMtpWeights(at modelDir: URL) -> Bool {
    if NovaMLXEngine.isMtpDraftConfig(at: modelDir) { return true }
    return weightMapKeys(at: modelDir).contains(where: isMtpWeightKey)
}

public func dsparkDraftCandidates(forMainId mainId: String) -> [String] {
    let name = modelLeaf(mainId)
    if name.contains("dspark") || name.contains("dflash") || name.contains("-mtp") {
        return []
    }
    if name.contains("deepseek-v4.1-flash") {
        return ["mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"]
    }
    return []
}

public func dflashDraftCandidates(forMainId mainId: String) -> [String] {
    let name = modelLeaf(mainId)
    if name.contains("dflash") || name.contains("-mtp") || name.contains("dspark") {
        return []
    }
    if name.contains("qwen3.8-27b") || name.contains("qwen3_8-27b") {
        return ["incoai/Qwen3.8-27B-DFlash2"]
    }
    return []
}

public func mtpDraftCandidates(forMainId mainId: String) -> [String] {
    let name = modelLeaf(mainId)
    if name.contains("-mtp") || name.contains("dflash") || name.contains("dspark") {
        return []
    }
    if name.contains("qwen3.8-27b") || name.contains("qwen3_8-27b") {
        return [
            "mlx-community/Qwen3.8-27B-MTP-4bit",
            "mlx-community/Qwen3.8-27B-MTP-8bit",
        ]
    }
    return []
}

private func modelLeaf(_ modelId: String) -> String {
    (modelId.split(separator: "/").last.map(String.init) ?? modelId).lowercased()
}

private func modelConfigJSON(at dir: URL) -> [String: Any]? {
    let url = dir.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return json
}

private func weightMapKeys(at dir: URL) -> [String] {
    let url = dir.appendingPathComponent("model.safetensors.index.json")
    guard let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let map = json["weight_map"] as? [String: Any]
    else { return [] }
    return Array(map.keys)
}

private func isMtpWeightKey(_ key: String) -> Bool {
    let lower = key.lowercased()
    return lower.contains("mtp_layers") || lower.contains("mtp.") || lower.contains(".mtp")
        || lower.hasPrefix("mtp")
}

private func jsonInt(_ value: Any?) -> Int? {
    if let int = value as? Int { return int }
    if let number = value as? NSNumber { return number.intValue }
    return nil
}

private func jsonInts(_ value: Any?) -> [Int] {
    guard let array = value as? [Any] else { return [] }
    return array.compactMap { jsonInt($0) }
}

private func jsonFloat(_ value: Any?) -> Float? {
    if let value = value as? Float { return value }
    if let value = value as? Double { return Float(value) }
    if let value = value as? Int { return Float(value) }
    if let value = value as? NSNumber { return value.floatValue }
    return nil
}
