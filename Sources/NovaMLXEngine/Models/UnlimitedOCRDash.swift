import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Shared decode state

/// Prompt metadata plus the in-graph page cursor. Every layer cache points at
/// the same instance so a generated `<PAGE>` advances once per token.
final class UnlimitedOCRDashRuntime {
    let params: UnlimitedOCRDashParams
    let imageTokenID: Int
    let pageTokenID: Int
    let perPageTokens: Int
    var pages: UnlimitedOCRPromptPages?
    var promptLength = 0
    var tokenInPage: MLXArray?
    var page: MLXArray?
    var markers: MLXArray?
    var prefillCount: MLXArray?
    var pageCount = 1
    private var blockTableCap = 0
    private var blockTableMin: MLXArray?
    private var blockTableMax: MLXArray?

    init(_ config: UnlimitedOCRConfiguration) {
        params = config.dash
        imageTokenID = config.imageTokenID
        pageTokenID = config.pageTokenID
        perPageTokens = config.perPageImageTokens
    }

    func start(_ ids: [Int]) {
        let pages = UnlimitedOCRPromptPages(
            ids: ids,
            blockSize: params.blockSize,
            imageToken: imageTokenID,
            pageToken: pageTokenID,
            perPage: perPageTokens)
        self.pages = pages
        promptLength = pages.length
        if let tok = pages.tokenInPage, let pg = pages.pageIndex, let last = tok.indices.last {
            tokenInPage = MLXArray(Int32(tok[last]))
            page = MLXArray(Int32(pg[last]))
            pageCount = (pg.max() ?? 0) + 1
            prefillCount = MLXArray(pages.prefillCountByPage.map(Int32.init))
        } else {
            tokenInPage = nil
            page = nil
            prefillCount = nil
            pageCount = 1
        }
        markers = MLXArray(Int32(pages.markerCount))
    }

    func copyState() -> UnlimitedOCRDashRuntime {
        let copy = UnlimitedOCRDashRuntime.bare(params, imageTokenID, pageTokenID, perPageTokens)
        copy.pages = pages
        copy.promptLength = promptLength
        copy.tokenInPage = tokenInPage
        copy.page = page
        copy.markers = markers
        copy.prefillCount = prefillCount
        copy.pageCount = pageCount
        copy.blockTableCap = blockTableCap
        copy.blockTableMin = blockTableMin
        copy.blockTableMax = blockTableMax
        return copy
    }

    /// Prompt block bounds, padded to the decode capacity. Built once per
    /// capacity so a decode step does not upload a new table per layer.
    func blockTables(cap: Int) -> (MLXArray, MLXArray)? {
        guard let pages else { return nil }
        if blockTableCap == cap, let blockTableMin, let blockTableMax {
            return (blockTableMin, blockTableMax)
        }
        let filled = min(cap, pages.decodeBlockMin.count)
        let pad = Array(repeating: Int32(-1), count: cap - filled)
        let lower = MLXArray(pages.decodeBlockMin.prefix(filled).map(Int32.init) + pad)
        let upper = MLXArray(pages.decodeBlockMax.prefix(filled).map(Int32.init) + pad)
        blockTableCap = cap
        blockTableMin = lower
        blockTableMax = upper
        return (lower, upper)
    }

    private static func bare(
        _ params: UnlimitedOCRDashParams, _ image: Int, _ page: Int, _ perPage: Int
    ) -> UnlimitedOCRDashRuntime {
        let config = UnlimitedOCRConfiguration.makeBare(
            params: params, image: image, page: page, perPage: perPage)
        return UnlimitedOCRDashRuntime(config)
    }
}

// MARK: - Cache

/// Slab-backed KV. Appends land in spare capacity so a decode step does not
/// copy the whole document. `step` is the growth quantum.
///
/// Conforms to ``KVCache`` directly. ``BaseKVCache`` is not subclassable from
/// this module: its initializer is internal.
class UnlimitedOCRKVBuffer: KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var step = 1024
    var offset: Int = 0

    init() {}

    var maxSize: Int? { nil }

    func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let grew = self.keys == nil || previous + keys.dim(2) > self.keys!.dim(2)
        if grew {
            let batches = keys.dim(0)
            let heads = keys.dim(1)
            let slabs = (step + keys.dim(2) - 1) / step
            let extraK = MLXArray.zeros(
                [batches, heads, slabs * step, keys.dim(3)], dtype: keys.dtype)
            let extraV = MLXArray.zeros(
                [batches, heads, slabs * step, values.dim(3)], dtype: values.dtype)
            if var currentK = self.keys, var currentV = self.values {
                // Drop unused slab padding before concatenating. A multiple of
                // `step` is already tight, matching `KVCacheSimple`.
                if previous % step != 0 {
                    currentK = currentK[.ellipsis, ..<previous, 0...]
                    currentV = currentV[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentK, extraK], axis: 2)
                self.values = concatenated([currentV, extraV], axis: 2)
            } else {
                self.keys = extraK
                self.values = extraV
            }
        }
        offset += keys.dim(2)
        self.keys?[.ellipsis, previous..<offset, 0...] = keys
        self.values?[.ellipsis, previous..<offset, 0...] = values
        return (
            self.keys![.ellipsis, ..<offset, 0...],
            self.values![.ellipsis, ..<offset, 0...]
        )
    }

    func copy() -> any KVCache {
        let cache = UnlimitedOCRKVBuffer()
        cache.step = step
        let saved = state
        if !saved.isEmpty { cache.state = saved.map { $0[.ellipsis] } }
        return cache
    }

    var isTrimmable: Bool { false }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    var metaState: [String] {
        get { [String(offset), String(step)] }
        set {
            if newValue.count >= 1, let value = Int(newValue[0]) { offset = value }
            if newValue.count >= 2, let value = Int(newValue[1]), value > 0 { step = value }
        }
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            if offset == keys.dim(2) { return [keys, values] }
            return [keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...]]
        }
        set {
            guard newValue.count == 2 else { return }
            keys = newValue[0]
            values = newValue[1]
            offset = keys?.dim(2) ?? 0
        }
    }
}

/// Full KV plus the router's completed-block summaries.
///
/// Attention never scores the stored sequence. Decode gathers
/// `params.attendedKeys` positions (512 on the released config).
final class UnlimitedOCRDashCache: UnlimitedOCRKVBuffer {
    let runtime: UnlimitedOCRDashRuntime
    var pending: MLXArray?
    var summaries: MLXArray?
    var summaryCount = 0

    init(runtime: UnlimitedOCRDashRuntime) {
        self.runtime = runtime
        super.init()
    }

    override var isTrimmable: Bool { false }

    @discardableResult
    override func trim(_ n: Int) -> Int { 0 }

    override func copy() -> any KVCache {
        let cache = UnlimitedOCRDashCache(runtime: runtime.copyState())
        cache.step = step
        let saved = state
        if !saved.isEmpty { cache.state = saved.map { $0[.ellipsis] } }
        cache.pending = pending
        cache.summaries = summaries
        cache.summaryCount = summaryCount
        return cache
    }

    func addIndexKeys(_ keys: MLXArray, head: MLXArray) {
        let block = runtime.params.blockSize
        let joined = pending == nil ? keys : concatenated([pending!, keys], axis: 0)
        let full = joined.dim(0) / block
        if full > 0 {
            let blocks = joined[..<(full * block)].reshaped(full, block, joined.dim(1))
            appendSummaries(Self.summarize(blocks, head: head))
        }
        let rest = joined.dim(0) - full * block
        pending = rest > 0 ? joined[(full * block)...] : nil
    }

    func summaryRows(capacity: Int, dim: Int) -> MLXArray {
        if summaries == nil || summaries!.dim(0) < capacity {
            var grown = MLXArray.zeros([capacity, dim], dtype: .float32)
            if summaryCount > 0, let summaries {
                grown[0..<summaryCount] = summaries[0..<summaryCount]
            }
            summaries = grown
        }
        return summaries![0..<capacity]
    }

    func completedSummaries() -> MLXArray? {
        guard summaryCount > 0, let summaries else { return nil }
        return summaries[0..<summaryCount]
    }

    private func appendSummaries(_ rows: MLXArray) {
        let n = rows.dim(0)
        let need = summaryCount + n
        if summaries == nil || need > summaries!.dim(0) {
            let cap = UnlimitedOCRDashMath.blockCapacity(need)
            var grown = MLXArray.zeros([cap, rows.dim(-1)], dtype: rows.dtype)
            if summaryCount > 0, let summaries {
                grown[0..<summaryCount] = summaries[0..<summaryCount]
            }
            summaries = grown
        }
        summaries![summaryCount..<need] = rows
        summaryCount = need
    }

    /// head_cls attention inside each completed block. One summary key per block.
    private static func summarize(_ blocks: MLXArray, head: MLXArray) -> MLXArray {
        let width = blocks.dim(-1)
        let scale = 1 / Float(width).squareRoot()
        let scores = (blocks * head.asType(.float32)).sum(axis: -1) * scale
        let weights = softmax(
            MLX.clip(scores, min: MLXArray(Float(-1e4)), max: MLXArray(Float(1e4))),
            axis: -1, precise: true)
        return (weights.expandedDimensions(axis: -1) * blocks).sum(axis: 1)
    }
}

/// Gen-1 R-SWA: the prompt stays, generated keys occupy a fixed ring.
final class UnlimitedOCRRingCache: UnlimitedOCRKVBuffer {
    let window: Int
    var prefillLength: Int?
    var ringPos = 0

    init(window: Int) {
        self.window = window
        super.init()
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let n = keys.dim(2)
        if prefillLength == nil {
            if n > 1 {
                return super.update(keys: keys, values: values)
            }
            prefillLength = offset
        }
        guard let prefill = prefillLength else {
            return super.update(keys: keys, values: values)
        }
        if offset < prefill + window {
            let pair = super.update(keys: keys, values: values)
            if offset >= prefill + window { ringPos = 0 }
            return pair
        }
        ensureCapacity(keys: keys, values: values, minimum: prefill + window)
        for index in 0..<n {
            let slot = prefill + ringPos
            self.keys?[.ellipsis, slot..<(slot + 1), 0...] = keys[.ellipsis, index..<(index + 1), 0...]
            self.values?[.ellipsis, slot..<(slot + 1), 0...] = values[.ellipsis, index..<(index + 1), 0...]
            ringPos = (ringPos + 1) % window
        }
        offset += n
        let end = prefill + window
        return (
            self.keys![.ellipsis, ..<end, 0...],
            self.values![.ellipsis, ..<end, 0...]
        )
    }

    private func ensureCapacity(keys: MLXArray, values: MLXArray, minimum: Int) {
        guard self.keys == nil || self.keys!.dim(2) < minimum else { return }
        let batches = keys.dim(0)
        let heads = keys.dim(1)
        let extraK = MLXArray.zeros([batches, heads, minimum, keys.dim(3)], dtype: keys.dtype)
        let extraV = MLXArray.zeros([batches, heads, minimum, values.dim(3)], dtype: values.dtype)
        if let currentK = self.keys, let currentV = self.values, offset > 0 {
            extraK[.ellipsis, ..<offset, 0...] = currentK[.ellipsis, ..<offset, 0...]
            extraV[.ellipsis, ..<offset, 0...] = currentV[.ellipsis, ..<offset, 0...]
        }
        self.keys = extraK
        self.values = extraV
    }

    override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let prefillLength, offset >= prefillLength + window, n == 1, !returnArray {
            return .none
        }
        return super.makeMask(n: n, windowSize: nil, returnArray: returnArray)
    }

    override func copy() -> any KVCache {
        let cache = UnlimitedOCRRingCache(window: window)
        cache.prefillLength = prefillLength
        cache.ringPos = ringPos
        cache.step = step
        let saved = state
        if !saved.isEmpty { cache.state = saved.map { $0[.ellipsis] } }
        return cache
    }
}

// MARK: - Math

enum UnlimitedOCRDashMath {
    static let scoreNegative: Float = -1e9
    static let activationClamp: Float = 1e4
    static let entmaxNegative: Float = -1e4
    static let blockCapacityStep = 64

    static func blockCapacity(_ blocks: Int) -> Int {
        ((blocks + blockCapacityStep - 1) / blockCapacityStep) * blockCapacityStep
    }

    static func finite(_ x: MLXArray) -> MLXArray {
        let ok = (x .== x) .&& (x.abs() .< MLXArray(Float(1e30)))
        return which(ok, x, MLXArray(x.dtype == .bfloat16 ? Float(0) : Float(0)).asType(x.dtype))
    }

    static func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, ..<half]
        let x2 = x[.ellipsis, half...]
        let rotated = concatenated([-x2, x1], axis: -1)
        return x * cos + rotated * sin
    }

    static func cosSin(_ angles: MLXArray, dtype: DType, scale: Float = 1) -> (MLXArray, MLXArray) {
        let emb = concatenated([angles, angles], axis: -1)
        var c = MLX.cos(emb)
        var s = MLX.sin(emb)
        if scale != 1 {
            c = c * scale
            s = s * scale
        }
        return (
            c.asType(dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0),
            s.asType(dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        )
    }

    /// 1.5-entmax. Rows must already be sorted descending.
    /// Host form used by `UnlimitedOCRDashPlan.reference` and mirrored in the Metal plan.
    static func entmax15(_ x: [Float]) -> [Float] {
        guard let origin = x.first else { return [] }
        let shifted = x.map { ($0 - origin) / 2 }
        let m = shifted.count
        var cumulative: Float = 0
        var cumulativeSquare: Float = 0
        var tau = [Float](repeating: 0, count: m)
        for index in 0..<m {
            let rho = Float(index + 1)
            cumulative += shifted[index]
            cumulativeSquare += shifted[index] * shifted[index]
            let mean = cumulative / rho
            let meanSquare = cumulativeSquare / rho
            let variance = rho * (meanSquare - mean * mean)
            let delta = max((1 - variance) / rho, 0)
            tau[index] = mean - delta.squareRoot()
        }
        var support = 0
        for index in 0..<m where tau[index] <= shifted[index] {
            support += 1
        }
        if support < 1 { support = 1 }
        let tauStar = tau[support - 1]
        return shifted.map { gap in
            let positive = max(gap - tauStar, 0)
            return positive * positive
        }
    }

    static func entmax15(_ x: MLXArray, width: Int) -> MLXArray {
        let shifted = (x - x[.ellipsis, ..<1]) / 2
        let m = width
        let rho = MLXArray(Int32(1)..<Int32(m + 1)).asType(shifted.dtype)
        let mean = shifted.cumsum(axis: -1) / rho
        let meanSq = (shifted * shifted).cumsum(axis: -1) / rho
        let ss = rho * (meanSq - mean * mean)
        let delta = (1 - ss) / rho
        let tau = mean - maximum(delta, MLXArray(Float(0)).asType(shifted.dtype)).sqrt()
        let support = maximum((tau .<= shifted).sum(axis: -1, keepDims: true), MLXArray(Int32(1)))
        let tauStar = takeAlong(tau, (support - 1).asType(.int32), axis: -1)
        let positive = maximum(shifted - tauStar, MLXArray(Float(0)).asType(shifted.dtype))
        return positive * positive
    }

    /// One decode query: top-m entmax, top-k blocks, then the 512 gather plan.
    /// Traced once per concrete `cap`. Not shapeless: a symbolic slice cannot
    /// infer its output shape.
    static func decodePlan(
        query: MLXArray,
        summaries: MLXArray,
        candidates: MLXArray,
        position: MLXArray,
        indexScale: Float,
        floor: Float,
        params: UnlimitedOCRDashParams,
        cap: Int,
        indexDim: Int
    ) -> (MLXArray, MLXArray) {
        UnlimitedOCRCompiledDecode.plan(
            query: query, summaries: summaries, candidates: candidates, position: position,
            indexScale: indexScale, floor: floor, params: params, cap: cap, indexDim: indexDim)
    }

    static func planBody(
        query: MLXArray,
        summaries: MLXArray,
        candidates: MLXArray,
        position: MLXArray,
        indexScale: Float,
        floor: Float,
        params: UnlimitedOCRDashParams,
        cap: Int,
        indexDim: Int
    ) -> (MLXArray, MLXArray) {
        let block = params.blockSize
        let local = params.localWindow
        let width = indexDim
        var scores = matmul(query, summaries.T) * (1 / Float(width).squareRoot())
        scores = MLX.clip(
            scores, min: MLXArray(-activationClamp), max: MLXArray(activationClamp))
        let remote = MLXArray(Int32(0)..<Int32(cap))
            .<= (position.floorDivide(MLXArray(Int32(block))) - Int32(params.localBlocks))
        var selected = remote.expandedDimensions(axis: 0) .&& candidates
        let any = selected.sum(axis: -1, keepDims: true) .> MLXArray(Int32(0))
        selected = which(any, selected, remote.expandedDimensions(axis: 0))
        let masked = which(
            selected, scores * indexScale, MLXArray(entmaxNegative))
        let m = max(1, min(params.topM, cap))
        let order = argSort(-masked, axis: -1)[.ellipsis, ..<m]
        let prior = entmax15(takeAlong(masked, order, axis: -1), width: m)
        let k = max(1, min(params.topK, m))
        let topi = order[0, ..<k]
        let valid = takeAlong(selected, order[.ellipsis, ..<k], axis: -1)[0]
        let chosen = prior[0, ..<k]
        let logp = maximum(chosen, MLXArray(floor)).log()
        let count = maximum(valid.asType(.float32).sum(), MLXArray(Float(1)))
        let mu = which(valid, logp, MLXArray(Float(0))).sum() / count
        let bias = which(valid, (logp - mu) * (1 / params.sigma), MLXArray(Float(0)))
        let blockIndex = topi.expandedDimensions(axis: 1) * Int32(block)
            + MLXArray(Int32(0)..<Int32(block)).expandedDimensions(axis: 0)
        let blockOK = valid.expandedDimensions(axis: 1)
            .&& (blockIndex .<= (position - Int32(local)))
        let blockMask = which(blockOK, bias.expandedDimensions(axis: 1), MLXArray(-Float.infinity))
            .reshaped([k * block])
        let localIndex = position - Int32(local) + 1 + MLXArray(Int32(0)..<Int32(local))
        let localMask = which(localIndex .>= MLXArray(Int32(0)), MLXArray(Float(0)), MLXArray(-Float.infinity))
        let flatBlocks = blockIndex.reshaped([k * block])
        let idx = concatenated(
            [minimum(flatBlocks, position), maximum(localIndex, MLXArray(Int32(0)))], axis: 0)
        return (idx.asType(.int32), concatenated([blockMask, localMask], axis: 0))
    }
}

/// Static-shape traces of the decode plan and the 512-key gather.
/// `shapeless` is off. Each trace is used only at the `cap` or slab it was
/// built with, and no slice uses `-1`.
private enum UnlimitedOCRCompiledDecode {
    private struct PlanKey: Hashable {
        var cap: Int
        var indexDim: Int
        var block: Int
        var local: Int
        var localBlocks: Int
        var topM: Int
        var topK: Int
        var sigmaBits: UInt32
        var scaleBits: UInt32
        var floorBits: UInt32
    }

    private struct AttendKey: Hashable {
        var slab: Int
        var width: Int
        var scaleBits: UInt32
    }

    private final class Store: @unchecked Sendable {
        static let shared = Store()
        let lock = NSLock()
        var plans: [PlanKey: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
        var attend: [AttendKey: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
    }

    static func plan(
        query: MLXArray,
        summaries: MLXArray,
        candidates: MLXArray,
        position: MLXArray,
        indexScale: Float,
        floor: Float,
        params: UnlimitedOCRDashParams,
        cap: Int,
        indexDim: Int
    ) -> (MLXArray, MLXArray) {
        let key = PlanKey(
            cap: cap,
            indexDim: indexDim,
            block: params.blockSize,
            local: params.localWindow,
            localBlocks: params.localBlocks,
            topM: params.topM,
            topK: params.topK,
            sigmaBits: params.sigma.bitPattern,
            scaleBits: indexScale.bitPattern,
            floorBits: floor.bitPattern)
        let function = planFunction(key, indexScale: indexScale, floor: floor, params: params, cap: cap, indexDim: indexDim)
        let out = function([query, summaries, candidates, position])
        return (out[0], out[1])
    }

    static func attend(
        q: MLXArray, keys: MLXArray, values: MLXArray, index: MLXArray, mask: MLXArray,
        scale: Float, slab: Int, width: Int
    ) -> MLXArray {
        attendFunction(scale: scale, slab: slab, width: width)([q, keys, values, index, mask])[0]
    }

    private static func planFunction(
        _ key: PlanKey, indexScale: Float, floor: Float, params: UnlimitedOCRDashParams,
        cap: Int, indexDim: Int
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let store = Store.shared
        store.lock.lock()
        defer { store.lock.unlock() }
        if let cached = store.plans[key] { return cached }
        let created = compile { (args: [MLXArray]) -> [MLXArray] in
            let pair = UnlimitedOCRDashMath.planBody(
                query: args[0], summaries: args[1], candidates: args[2], position: args[3],
                indexScale: indexScale, floor: floor, params: params, cap: cap, indexDim: indexDim)
            return [pair.0, pair.1]
        }
        store.plans[key] = created
        return created
    }

    private static func attendFunction(scale: Float, slab: Int, width: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let store = Store.shared
        let key = AttendKey(slab: slab, width: width, scaleBits: scale.bitPattern)
        store.lock.lock()
        defer { store.lock.unlock() }
        if let cached = store.attend[key] { return cached }
        let created = compile { (args: [MLXArray]) -> [MLXArray] in
            let queries = args[0]
            let gatheredK = args[1].take(args[3], axis: 2)
            let gatheredV = args[2].take(args[3], axis: 2)
            let additive = args[4].asType(queries.dtype).reshaped(1, 1, 1, width)
            let out = MLXFast.scaledDotProductAttention(
                queries: queries, keys: gatheredK, values: gatheredV,
                scale: scale, mask: .array(additive))
            return [out]
        }
        store.attend[key] = created
        return created
    }
}

// MARK: - RoPE

final class UnlimitedOCRRope {
    let params: UnlimitedOCRDashParams
    let invToken: MLXArray
    let invPage: MLXArray
    let invLinear: MLXArray
    let linearScale: Float

    init(_ config: UnlimitedOCRConfiguration) {
        params = config.dash
        let dim = config.headDim
        if params.page2D {
            let half = dim / 2
            let nTok = half - params.ropePagePairs
            var token: [Float] = []
            for i in 0..<max(nTok, 0) {
                let expo: Float = params.resampleTokenFrequencies
                    ? Float(i) / Float(max(nTok, 1))
                    : Float(i) * 2 / Float(dim)
                token.append(1 / pow(params.ropeTokTheta, expo))
            }
            var page: [Float] = []
            let pairs = max(params.ropePagePairs, 1)
            for i in 0..<params.ropePagePairs {
                page.append(1 / pow(params.ropePageTheta, Float(i) / Float(pairs)))
            }
            invToken = MLXArray(token)
            invPage = MLXArray(page)
        } else {
            invToken = MLXArray([Float(1)])
            invPage = MLXArray([Float(1)])
        }
        let yarn = UnlimitedOCRRope.yarn(
            dim: dim, base: config.ropeTheta, scaling: config.ropeScaling,
            maxPosition: config.maxPositionEmbeddings)
        invLinear = yarn.0
        linearScale = yarn.1
    }

    func page2D(token: MLXArray, page: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        var page = page
        if params.ropePageMod > 0 {
            let mod = MLXArray(Int32(params.ropePageMod))
            page = page - page.floorDivide(mod) * mod
        }
        let angles = concatenated(
            [
                token.asType(.float32).expandedDimensions(axis: 1) * invToken.expandedDimensions(axis: 0),
                page.asType(.float32).expandedDimensions(axis: 1) * invPage.expandedDimensions(axis: 0),
            ],
            axis: -1)
        return UnlimitedOCRDashMath.cosSin(angles, dtype: dtype)
    }

    func linear(positions: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let angles = positions.asType(.float32).expandedDimensions(axis: 1)
            * invLinear.expandedDimensions(axis: 0)
        return UnlimitedOCRDashMath.cosSin(angles, dtype: dtype, scale: linearScale)
    }

    private static func yarn(
        dim: Int, base: Float, scaling: UnlimitedOCRRopeScaling?, maxPosition: Int
    ) -> (MLXArray, Float) {
        var inv = [Float]()
        for i in stride(from: 0, to: dim, by: 2) {
            inv.append(1 / pow(base, Float(i) / Float(dim)))
        }
        guard scaling?.ropeType == "yarn", let factor = scaling?.factor, factor > 0 else {
            return (MLXArray(inv), 1)
        }
        let original = Float(scaling?.originalMax ?? maxPosition)
        let betaFast = scaling?.betaFast ?? 32
        let betaSlow = scaling?.betaSlow ?? 1
        let attention = scaling?.attentionFactor ?? (0.1 * log(factor) + 1)
        func correction(_ rotations: Float) -> Float {
            (Float(dim) * log(original / (rotations * 2 * Float.pi))) / (2 * log(base))
        }
        var low = floor(correction(betaFast))
        var high = ceil(correction(betaSlow))
        low = max(low, 0)
        high = min(high, Float(dim - 1))
        if low == high { high += 0.001 }
        var mixed = [Float]()
        let pos = (0..<dim).filter { $0 % 2 == 0 }.map { pow(base, Float($0) / Float(dim)) }
        for (index, freq) in pos.enumerated() {
            let ramp = min(max((Float(index) - low) / (high - low), 0), 1)
            let extra = 1 - ramp
            let inter = 1 / (factor * freq)
            let baseInv = 1 / freq
            mixed.append(inter * (1 - extra) + baseInv * extra)
        }
        return (MLXArray(mixed), attention)
    }
}

struct UnlimitedOCRDashContext {
    var cos: MLXArray
    var sin: MLXArray
    var offset: Int
    var length: Int
    var decode: Bool
    var candidates: MLXArray?
    /// Width of `candidates`. Kept on the host so decode does not read a shape.
    var candidateCount: Int
    var position: MLXArray?
    /// Decode block eligibility as uint32, one value per summary row.
    /// Built once per token and reused by every layer's plan kernel.
    var candidateBits: MLXArray?
}

func unlimitedOCRContext(
    runtime: UnlimitedOCRDashRuntime,
    rope: UnlimitedOCRRope,
    tokens: MLXArray?,
    offset: Int,
    length: Int,
    dtype: DType
) -> UnlimitedOCRDashContext {
    let params = runtime.params
    let total = offset + length
    let blocks = (total + params.blockSize - 1) / params.blockSize
    let pages = runtime.pages
    if offset + length <= runtime.promptLength, let pages {
        let positions = Array(offset..<total)
        let angles: (MLXArray, MLXArray)
        if let tok = pages.tokenInPage, let pg = pages.pageIndex {
            angles = rope.page2D(
                token: MLXArray(positions.map { Int32(tok[$0]) }),
                page: MLXArray(positions.map { Int32(pg[$0]) }),
                dtype: dtype)
        } else {
            angles = rope.linear(positions: MLXArray(positions.map { Int32($0) }), dtype: dtype)
        }
        let allow = pages.prefillAllow(positions: positions, blockCount: blocks)
        let flat = allow.flatMap { row in row.map { $0 ? Int32(1) : Int32(0) } }
        let cand = MLXArray(flat).reshaped(length, blocks) .== MLXArray(Int32(1))
        return UnlimitedOCRDashContext(
            cos: angles.0, sin: angles.1, offset: offset, length: length,
            decode: false, candidates: cand, candidateCount: blocks, position: nil)
    }

    let token = tokens!.reshaped(-1)[-1].asType(.int32)
    let isPage = token .== MLXArray(Int32(runtime.pageTokenID))
    runtime.markers = (runtime.markers ?? MLXArray(Int32(0))) + isPage.asType(.int32)
    let angles: (MLXArray, MLXArray)
    if runtime.tokenInPage != nil, let counts = runtime.prefillCount {
        let nextPage = MLX.minimum(
            (runtime.page ?? MLXArray(Int32(0))) + 1, MLXArray(Int32(runtime.pageCount - 1)))
        let page = which(isPage, nextPage, runtime.page ?? MLXArray(Int32(0)))
        let restarted = counts.take(page.reshaped([1]), axis: 0)[0]
        let tok = which(isPage, restarted, (runtime.tokenInPage ?? MLXArray(Int32(0))) + 1)
        runtime.page = page
        runtime.tokenInPage = tok
        angles = rope.page2D(token: tok.reshaped(1), page: page.reshaped(1), dtype: dtype)
    } else {
        angles = rope.linear(positions: MLXArray([Int32(offset)]), dtype: dtype)
    }

    let cap = UnlimitedOCRDashMath.blockCapacity(blocks)
    var candidates: MLXArray
    if params.imageOnly, let pages = runtime.pages {
        let qp = (runtime.markers ?? MLXArray(Int32(0))) - 1 - Int32(pages.promptPages)
        let query = which(qp .>= MLXArray(Int32(0)), qp, MLXArray(Int32(-1)))
        // The page test stays on device. The block table is prompt-fixed.
        let bmin: MLXArray
        let bmax: MLXArray
        if let tables = runtime.blockTables(cap: cap) {
            bmin = tables.0
            bmax = tables.1
        } else {
            let missing = MLXArray(Array(repeating: Int32(-1), count: cap))
            bmin = missing
            bmax = missing
        }
        let same = (bmin .>= MLXArray(Int32(0))) .&& (bmin .<= query) .&& (bmax .>= query)
            .&& (query .>= MLXArray(Int32(0)))
        let open = logicalNot(same.sum() .> MLXArray(Int32(0))) .|| (query .< MLXArray(Int32(0)))
        candidates = (same .|| open).expandedDimensions(axis: 0)
    } else {
        candidates = (MLXArray.zeros([1, cap], dtype: .int32) + 1) .== MLXArray(Int32(1))
    }
    var context = UnlimitedOCRDashContext(
        cos: angles.0, sin: angles.1, offset: offset, length: length,
        decode: true, candidates: candidates, candidateCount: cap,
        position: MLXArray(Int32(offset)))
    context.candidateBits = candidates.asType(.uint32).reshaped([cap])
    return context
}

// MARK: - Router and attention

final class UnlimitedOCRRouterNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var value = UnlimitedOCRDashMath.finite(x.asType(.float32))
        value = value * (value * value).mean(axis: -1, keepDims: true).rsqrt()
        value = MLX.clip(value, min: MLXArray(-UnlimitedOCRDashMath.activationClamp), max: MLXArray(UnlimitedOCRDashMath.activationClamp))
        let scaled = value * (1 + weight.asType(.float32))
        return MLX.clip(
            scaled,
            min: MLXArray(-UnlimitedOCRDashMath.activationClamp),
            max: MLXArray(UnlimitedOCRDashMath.activationClamp)
        ).asType(x.dtype)
    }
}

final class UnlimitedOCRDashRouter: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: UnlimitedOCRRouterNorm
    @ModuleInfo(key: "k_norm") var kNorm: UnlimitedOCRRouterNorm
    @ParameterInfo(key: "head_cls") var headCls: MLXArray

    init(hidden: Int, indexDim: Int, eps: Float) {
        _qProj.wrappedValue = Linear(hidden, indexDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, indexDim, bias: false)
        _qNorm.wrappedValue = UnlimitedOCRRouterNorm(dimensions: indexDim, eps: eps)
        _kNorm.wrappedValue = UnlimitedOCRRouterNorm(dimensions: indexDim, eps: eps)
        _headCls.wrappedValue = MLXArray.zeros([indexDim])
        super.init()
    }

    func indexQK(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let hidden = UnlimitedOCRDashMath.finite(x)
        var q = qNorm(qProj(hidden)).expandedDimensions(axis: 1)
        var k = kNorm(kProj(hidden)).expandedDimensions(axis: 1)
        q = UnlimitedOCRDashMath.applyRope(q, cos: cos, sin: sin)
        k = UnlimitedOCRDashMath.applyRope(k, cos: cos, sin: sin)
        return (
            UnlimitedOCRDashMath.finite(q[0, 0].asType(.float32)),
            UnlimitedOCRDashMath.finite(k[0, 0].asType(.float32))
        )
    }
}

final class UnlimitedOCRDashAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let params: UnlimitedOCRDashParams
    let indexScale: Float
    let floor: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "dash_router") var router: UnlimitedOCRDashRouter

    init(_ config: UnlimitedOCRConfiguration, layer: Int) {
        heads = config.attentionHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        params = config.dash
        indexScale = config.dash.indexScale[layer]
        floor = config.dash.floor[layer]
        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, config.kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, config.kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
        _router.wrappedValue = UnlimitedOCRDashRouter(
            hidden: dim, indexDim: config.dash.indexDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, _ ctx: UnlimitedOCRDashContext, _ cache: UnlimitedOCRDashCache
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        precondition(batch == 1, "Unlimited OCR DASH attends one sequence.")
        var q = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        q = UnlimitedOCRDashMath.applyRope(q, cos: ctx.cos, sin: ctx.sin)
        k = UnlimitedOCRDashMath.applyRope(k, cos: ctx.cos, sin: ctx.sin)
        let (keys, values) = cache.update(keys: k, values: v)
        let (idxQ, idxK) = router.indexQK(x, cos: ctx.cos, sin: ctx.sin)
        cache.addIndexKeys(idxK, head: router.headCls)
        let out: MLXArray
        if ctx.decode {
            let cap = ctx.candidateCount
            let summaries = cache.summaryRows(capacity: cap, dim: router.headCls.dim(0))
            let idx: MLXArray
            let mask: MLXArray
            if UnlimitedOCRDashPlan.supports(params, cap: cap), let bits = ctx.candidateBits,
                let position = ctx.position
            {
                (idx, mask) = UnlimitedOCRDashPlan.evaluate(
                    query: idxQ.reshaped([params.indexDim]),
                    summaries: summaries,
                    candidates: bits,
                    position: position,
                    indexScale: indexScale,
                    floor: floor,
                    params: params,
                    cap: cap)
            } else {
                (idx, mask) = UnlimitedOCRDashMath.decodePlan(
                    query: idxQ, summaries: summaries, candidates: ctx.candidates!,
                    position: ctx.position!, indexScale: indexScale, floor: floor, params: params,
                    cap: cap, indexDim: params.indexDim)
            }
            // Slab length is the allocation, fixed until the next growth.
            // The compiled gather is keyed by that length and uses the
            // constant 512-wide mask. No shapeless trace, no `-1` slice.
            let sourceK = cache.keys ?? keys
            let sourceV = cache.values ?? values
            out = UnlimitedOCRCompiledDecode.attend(
                q: q, keys: sourceK, values: sourceV, index: idx, mask: mask,
                scale: scale, slab: sourceK.dim(2), width: params.attendedKeys)
        } else {
            out = prefill(q, keys: keys, values: values, query: idxQ, ctx: ctx, cache: cache)
        }
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    /// Selected blocks carry their log-prior bias. The local window is unbiased.
    /// Queries are chunked so the mask stays a slice of the prompt, not a second copy of it.
    private func prefill(
        _ q: MLXArray, keys: MLXArray, values: MLXArray, query: MLXArray,
        ctx: UnlimitedOCRDashContext, cache: UnlimitedOCRDashCache
    ) -> MLXArray {
        let length = q.dim(2)
        let total = ctx.offset + length
        let blocks = (total + params.blockSize - 1) / params.blockSize
        let positions = MLXArray(Int32(ctx.offset)..<Int32(total))
        let scores = blockScores(query, summaries: cache.completedSummaries(), positions: positions, blocks: blocks)
        var cand = ctx.candidates
        if let current = cand, current.dim(-1) != blocks {
            cand = current[.ellipsis, ..<blocks]
        }
        let (topi, bias, valid) = route(scores, candidates: cand)
        let keyPos = MLXArray(Int32(0)..<Int32(total))
        let keyBlock = keyPos.floorDivide(MLXArray(Int32(params.blockSize)))
        var blockBias = MLXArray.zeros([length, blocks], dtype: .float32) + (-Float.infinity)
        blockBias = putAlong(
            blockBias, topi,
            values: which(valid, bias, MLXArray(-Float.infinity)),
            axis: -1)
        var parts: [MLXArray] = []
        let chunk = 2048
        var start = 0
        while start < length {
            let end = min(start + chunk, length)
            let qPos = MLXArray(Int32(ctx.offset + start)..<Int32(ctx.offset + end)).expandedDimensions(axis: 1)
            let base = blockBias[start..<end][0..., keyBlock]
            let causal = keyPos.expandedDimensions(axis: 0) .<= qPos
            let local = causal .&& (keyPos.expandedDimensions(axis: 0) .> (qPos - Int32(params.localWindow)))
            let mask = which(local, MLXArray(Float(0)), which(causal, base, MLXArray(-Float.infinity)))
            let additive = mask.asType(q.dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            parts.append(
                MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., start..<end, 0...],
                    keys: keys, values: values, scale: scale, mask: .array(additive)))
            start = end
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }

    private func blockScores(
        _ query: MLXArray, summaries: MLXArray?, positions: MLXArray, blocks: Int
    ) -> MLXArray {
        let length = query.dim(0)
        guard let summaries else {
            return MLXArray.zeros([length, blocks], dtype: .float32)
                + UnlimitedOCRDashMath.scoreNegative
        }
        let width = query.dim(-1)
        var scores = matmul(query, summaries.T) * (1 / Float(width).squareRoot())
        scores = MLX.clip(
            scores,
            min: MLXArray(-UnlimitedOCRDashMath.activationClamp),
            max: MLXArray(UnlimitedOCRDashMath.activationClamp))
        let queryBlock = positions.expandedDimensions(axis: 1).floorDivide(MLXArray(Int32(params.blockSize)))
        let keep = MLXArray(Int32(0)..<Int32(summaries.dim(0))).expandedDimensions(axis: 0)
            .<= (queryBlock - Int32(params.localBlocks))
        scores = which(keep, scores, MLXArray(UnlimitedOCRDashMath.scoreNegative))
        if summaries.dim(0) < blocks {
            scores = padded(
                scores,
                widths: [IntOrPair(0), IntOrPair((0, blocks - summaries.dim(0)))],
                value: MLXArray(UnlimitedOCRDashMath.scoreNegative))
        }
        return scores
    }

    private func route(_ scores: MLXArray, candidates: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        let blocks = scores.dim(-1)
        let remote = scores .> MLXArray(UnlimitedOCRDashMath.scoreNegative * 0.5)
        var cand = remote
        if let candidates {
            cand = remote .&& candidates
            let any = cand.sum(axis: -1, keepDims: true) .> MLXArray(Int32(0))
            cand = which(any, cand, remote)
        }
        let masked = which(cand, scores * indexScale, MLXArray(UnlimitedOCRDashMath.entmaxNegative))
        let m = max(1, min(params.topM, blocks))
        let order = argSort(-masked, axis: -1)[.ellipsis, ..<m]
        let prior = UnlimitedOCRDashMath.entmax15(takeAlong(masked, order, axis: -1), width: m)
        let k = max(1, min(params.topK, m))
        let topi = order[.ellipsis, ..<k]
        let valid = takeAlong(cand, topi, axis: -1)
        let logp = maximum(prior[.ellipsis, ..<k], MLXArray(floor)).log()
        let count = maximum(valid.asType(.float32).sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let mu = which(valid, logp, MLXArray(Float(0))).sum(axis: -1, keepDims: true) / count
        let bias = which(valid, (logp - mu) * (1 / params.sigma), MLXArray(Float(0)))
        return (topi, bias, valid)
    }
}

extension UnlimitedOCRConfiguration {
    /// Placeholder so a copied runtime can be built without re-decoding JSON.
    static func makeBare(
        params: UnlimitedOCRDashParams, image: Int, page: Int, perPage: Int
    ) -> UnlimitedOCRConfiguration {
        let data = """
        {"model_type":"unlimited-ocr","language_config":{"hidden_size":1280,"num_hidden_layers":\(max(params.indexScale.count, 1))},"sparse_attention_config":{"training_stage":"dash","use_sparse_attention":true,"sparse_block_size":\(params.blockSize),"sparse_local_block":\(params.localBlocks),"dash_topk_blocks":\(params.topK),"sparse_topk_blocks":\(params.topM),"image_token_id":\(image),"page_token_id":\(page),"per_page_image_tokens":\(perPage)}}
        """.data(using: .utf8)!
        return try! JSONDecoder().decode(UnlimitedOCRConfiguration.self, from: data)
    }
}
