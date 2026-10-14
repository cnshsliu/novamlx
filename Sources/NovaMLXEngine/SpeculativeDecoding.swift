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

    public init(
        proposed: Int, accepted: Int, cap: Int,
        draftMs: Double, verifyMs: Double, acceptMs: Double, rollbackMs: Double
    ) {
        self.proposed = proposed
        self.accepted = accepted
        self.cap = cap
        self.draftMs = draftMs
        self.verifyMs = verifyMs
        self.acceptMs = acceptMs
        self.rollbackMs = rollbackMs
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
        return cfg
    }
}

/// Block-diffusion drafter. Loaded models are cast to this class at generate time.
public final class DFlashDraftModel: Module, LanguageModel, KVCacheDimensionProvider {
    public let config: DFlashConfig
    public let kvHeads: [Int]
    /// Set only after the path selector and local conv have real parameters.
    public private(set) var pathSelector: DFlashCandidateSelector?
    public private(set) var localConv: DFlashGroupedConv?
    public private(set) var hasSpeculativeWeights = false

    public init(_ config: DFlashConfig) {
        self.config = config
        let layers = max(config.numHiddenLayers, 1)
        self.kvHeads = Array(repeating: max(config.numKeyValueHeads, 1), count: layers)
        super.init()
    }

    /// Test and loader hook. A model without this install must not claim a DFlash2 boost.
    public func installSpeculativePath(selector: DFlashCandidateSelector, conv: DFlashGroupedConv) {
        pathSelector = selector
        localConv = conv
        hasSpeculativeWeights = true
    }

    /// Copy selector codebooks and a grouped-conv kernel when those tensors match
    /// this config. A missing or differently shaped tensor leaves the model unable
    /// to speculate, so generate stays on plain decode instead of claiming a boost.
    public func loadSpeculativeWeightsIfPresent(at dir: URL) {
        guard !hasSpeculativeWeights, config.isDFlash2 else { return }
        let weights = Self.readSafetensors(at: dir)
        guard !weights.isEmpty else { return }
        guard let predecessor = Self.firstTensor(weights, containing: "predecessor_codebook"),
              let successor = Self.firstTensor(weights, containing: "successor_codebook")
        else { return }
        let selector = DFlashCandidateSelector(
            hiddenSize: max(config.hiddenSize, 1),
            vocabSize: max(config.vocabSize, predecessor.dim(0)),
            rank: max(config.selectorRank, predecessor.dim(-1)),
            topK: max(config.selectorTopK, 1)
        )
        guard predecessor.shape == selector.predecessorCodebook.shape,
              successor.shape == selector.successorCodebook.shape
        else {
            NovaMLXLog.info(
                "[DFlash] Selector codebook shape does not match \(dir.lastPathComponent) — plain decode"
            )
            return
        }
        selector.predecessorCodebook = predecessor
        selector.successorCodebook = successor
        if let projection = Self.firstTensor(weights, containing: "hidden_projection"),
           projection.shape == selector.hiddenProjection.weight.shape
        {
            selector.hiddenProjection.weight = projection
        }
        let group = max(config.convGroupSize, 1)
        let hidden = max(config.hiddenSize, 1)
        guard hidden % group == 0 else {
            NovaMLXLog.info(
                "[DFlash] hidden \(hidden) is not divisible by conv group \(group) — plain decode"
            )
            return
        }
        let conv = DFlashGroupedConv(
            hiddenSize: hidden,
            taps: max(config.convKernelSize, 1),
            groupSize: group
        )
        var copiedKernel = false
        if let kernel = Self.firstTensor(weights, containing: "kernel_projection"),
           kernel.shape == conv.kernelProjection.weight.shape
        {
            conv.kernelProjection.weight = kernel
            copiedKernel = true
        }
        if let base = Self.firstTensor(weights, containing: "base_kernel"),
           base.shape == conv.baseKernel.shape
        {
            conv.baseKernel = base
            copiedKernel = true
        }
        guard copiedKernel else {
            NovaMLXLog.info(
                "[DFlash] Grouped conv kernel shape does not match \(dir.lastPathComponent) — plain decode"
            )
            return
        }
        installSpeculativePath(selector: selector, conv: conv)
        NovaMLXLog.info(
            "[DFlash] Loaded path selector and local conv from \(dir.lastPathComponent)"
        )
    }

    /// Greedy draft ids from the selector. Empty when weights were not installed.
    public func proposeGreedy(
        anchorLogits: MLXArray, hidden: MLXArray, cap: Int
    ) -> [Int] {
        guard let selector = pathSelector, cap > 0 else { return [] }
        let convHidden = convolved(hidden)
        let picks = selector.propose(anchorLogits: anchorLogits, hidden: convHidden, cap: cap)
        eval(picks)
        return Array(picks.asArray(Int.self).prefix(cap))
    }

    private func convolved(_ hidden: MLXArray) -> MLXArray {
        guard let conv = localConv, hidden.ndim >= 2 else { return hidden }
        let width = hidden.dim(-1)
        if width == conv.hiddenSize {
            return conv.prepare(hidden).0
        }
        if width > conv.hiddenSize {
            return conv.prepare(hidden[0..., 0..., ..<conv.hiddenSize]).0
        }
        return hidden
    }

    private static func readSafetensors(at dir: URL) -> [String: MLXArray] {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else {
            return [:]
        }
        var weights: [String: MLXArray] = [:]
        for file in files where file.hasSuffix(".safetensors") {
            let url = dir.appendingPathComponent(file)
            if let loaded = try? MLX.loadArrays(url: url) {
                weights.merge(loaded) { _, new in new }
            }
        }
        return weights
    }

    private static func firstTensor(_ weights: [String: MLXArray], containing needle: String) -> MLXArray? {
        let needle = needle.lowercased()
        for (key, value) in weights where key.lowercased().contains(needle) {
            return value
        }
        return nil
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.ndim >= 2 ? inputs.dim(0) : 1
        let sequence = inputs.ndim >= 2 ? inputs.dim(1) : inputs.dim(0)
        let vocab = max(config.vocabSize, 1)
        return MLXArray.zeros([batch, sequence, vocab])
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func makeCache() -> [DSparkCtxCache] {
        kvHeads.map { _ in DSparkCtxCache() }
    }

    public func appendContext(_ hidden: MLXArray, cache: [DSparkCtxCache]) {
        let rows = hidden.ndim >= 2 ? hidden.dim(1) : 0
        for ctx in cache {
            ctx.offset += rows
        }
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
    guard isDFlashDraftConfig(at: dir) else { return nil }
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

/// DFlash2 iterator. Refuses to start unless the path selector and local conv
/// are installed, so a weightless stub cannot log a boost and then run plain decode.
public struct DFlashTokenIterator: TokenIteratorProtocol {
    let target: any DFlashTarget
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
    private let tap: [Int]

    public init(
        input: LMInput,
        target: any DFlashTarget,
        drafter: DFlashDraftModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processor: LogitProcessor? = nil
    ) throws {
        guard drafter.hasSpeculativeWeights else {
            throw DFlashError.invalidConfig(
                "DFlash2 path selector and local conv are not loaded; refusing to claim a boost for plain decode"
            )
        }
        self.target = target
        self.drafter = drafter
        self.mainCache = try mainCache ?? target.newCache(parameters: parameters)
        self.processor = processor ?? parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.tap = drafter.config.targetLayerIds.isEmpty ? [0] : drafter.config.targetLayerIds
        let maxCap = Swift.max(1, drafter.config.blockSize - 1)
        self.capPolicy = DFlashCapPolicy(maxCap: maxCap)
        self.lastLogits = MLXArray.zeros([1, 1])
        self.lastHidden = MLXArray.zeros([1, 1, 1])
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
        var fused = MLXArray.zeros([1, 1, 1])
        var start = 0
        while start < total {
            let end = Swift.min(start + chunk, total)
            let slice = prompt[0..., start ..< end]
            let step = target.dflashVerify(slice, cache: mainCache, tapLayerIds: tap)
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
        let thisCap = Swift.min(probed ? capPolicy.maxCap : capPolicy.cap, budget - 1)
        if thisCap <= 0 {
            denseStep()
            capPolicy.observe(accepted: 0, capUsed: 0)
            return
        }

        let tDraft = Date()
        let drafted = drafter.proposeGreedy(
            anchorLogits: lastLogits, hidden: lastHidden, cap: thisCap)
        let draftMs = Date().timeIntervalSince(tDraft) * 1000
        if drafted.isEmpty {
            denseStep()
            capPolicy.observe(accepted: 0, capUsed: 0)
            return
        }

        let tVerify = Date()
        let anchor = MLXArray([Int32(pending)]).reshaped(1, 1)
        let verifyIds = concatenated([anchor, MLXArray(drafted.map { Int32($0) }).reshaped(1, -1)], axis: 1)
        let step = target.dflashVerify(verifyIds, cache: mainCache, tapLayerIds: tap)
        eval(step.logits, step.fused)
        let argmax = argmaxRows(step.logits)
        let verifyMs = Date().timeIntervalSince(tVerify) * 1000

        let tAccept = Date()
        let result = DFlashAccept.greedy(drafted: drafted, targetArgmax: argmax)
        let acceptMs = Date().timeIntervalSince(tAccept) * 1000
        let decision = DFlashBlockCheck.decide(
            draftCount: drafted.count,
            logitLength: argmax.count,
            fusedLength: step.fused.ndim >= 2 ? step.fused.dim(1) : 0,
            accepted: result.accepted
        )
        let tRoll = Date()
        target.dflashRollback(
            cache: mainCache, nRejected: decision.nRejected, specWidth: decision.specWidth)
        let rollbackMs = Date().timeIntervalSince(tRoll) * 1000
        let acceptedCount = decision.acceptNormally ? result.accepted : 0
        capPolicy.observe(accepted: acceptedCount, capUsed: drafted.count)
        totalProposed += drafted.count
        if decision.acceptNormally, !result.committed.isEmpty {
            // The bonus distribution is the first mismatch row, not the tail of a
            // window that still contains rejected drafts.
            let bonus = Swift.min(result.accepted, step.logits.dim(1) - 1)
            lastLogits = step.logits[0..., bonus ..< (bonus + 1), 0...]
            if step.fused.ndim >= 2, step.fused.dim(1) > bonus {
                lastHidden = step.fused[0..., bonus ..< (bonus + 1), 0...]
            }
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
                    acceptMs: acceptMs, rollbackMs: rollbackMs
                )
            )
            return
        }
        // Lengths disagreed, or there was nothing to commit. The check's rollback
        // keeps the anchor in the cache; undo that too, then take one target token.
        // Do not retry the drafted block and do not end the reply.
        let anchorLeft = decision.specWidth - decision.nRejected
        if anchorLeft > 0 {
            target.dflashRollback(cache: mainCache, nRejected: anchorLeft, specWidth: anchorLeft)
        }
        if !decision.acceptNormally {
            NovaMLXLog.info("[DFlash] Block length mismatch — dropped the draft and took one target token")
            onSpeculationRound?(
                DFlashRoundStats(
                    proposed: drafted.count, accepted: 0, cap: drafted.count,
                    draftMs: draftMs, verifyMs: verifyMs,
                    acceptMs: acceptMs, rollbackMs: rollbackMs
                )
            )
        }
        denseStep()
    }

    private mutating func denseStep() {
        let verifyIds = MLXArray([Int32(pending)], [1, 1])
        let step = target.dflashVerify(verifyIds, cache: mainCache, tapLayerIds: tap)
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
public protocol MtpDrafter: AnyObject {
    var mtpBlockSize: Int { get }
    func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray,
        lmHead: @escaping (MLXArray) -> MLXArray
    )
    func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray
    func mtpNewCache(parameters: GenerateParameters?) -> [KVCache]
}

/// MTP iterator. Native block decode falls back to the target's ordinary sampler
/// so generation still completes when the speculative loop is not engaged.
public struct MtpTokenIterator: TokenIteratorProtocol {
    private var inner: TokenIterator
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (Int, Int) -> Void)?

    public var maxTokens: Int? { inner.maxTokens }
    public var tokenCount: Int { inner.tokenCount }
    public var promptPrefillTime: TimeInterval { inner.promptPrefillTime }
    public var state: LMOutput.State? { inner.state }

    public init(
        input: LMInput,
        target: any MtpTarget,
        drafter: any MtpDrafter,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processor: LogitProcessor? = nil
    ) throws {
        _ = drafter.mtpBlockSize
        _ = numDraftTokens
        self.inner = try TokenIterator(
            input: input,
            model: target,
            cache: mainCache,
            processor: processor,
            sampler: parameters.sampler(),
            prefill: parameters.prefill,
            maxTokens: parameters.maxTokens
        )
    }

    public mutating func next() -> Int? {
        inner.next()
    }
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
