import Foundation
import MLX
import MLXNN
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine

// MARK: - ShardableModel

/// Provides layer-level access to a ``LanguageModel`` for pipeline-parallel sharding.
///
/// Uses ``Module/children()`` reflection to dynamically discover model structure
/// (embed, layers, norm, head) without modifying vendor model code.
///
/// Works with any model that follows the standard transformer pattern:
/// ```
/// TopModel
///   ├── model: InnerModel
///   │     ├── embed_tokens: Embedding
///   │     ├── layers: [TransformerBlock]    // N blocks
///   │     └── norm: NormLayer
///   └── lm_head: Linear?                    // or tieWordEmbeddings
/// ```
public final class ShardableModel: @unchecked Sendable {

    /// The wrapped model.
    public let model: any LanguageModel

    /// Number of transformer layers discovered via reflection.
    public private(set) var layerCount: Int = 0

    // Discovered components
    private var embedModule: Module?
    private var layerModules: [Module] = []
    private var normModule: Module?
    private var headModule: Module?

    public init(model: any LanguageModel) {
        self.model = model
        discoverStructure()
    }

    // MARK: - Structure Discovery

    /// Walk the model tree using Module.children() reflection to find
    /// embed, layers, norm, and head. Recurses up to 4 levels deep to handle
    /// multi-tier nesting (e.g. Qwen35MoE: Model → language_model → model → layers).
    private func discoverStructure() {
        // Recursive search for the first module containing "layers" children
        if let result = searchForLayers(in: model, depth: 0, maxDepth: 4) {
            layerModules = result.layers
            layerCount = result.layers.count
            embedModule = result.embed
            normModule = result.norm

            // Search for lm_head at any level (may be sibling to the inner model)
            headModule = searchForModule(named: "lm_head", in: model, maxDepth: 4)
                ?? searchForModule(named: "head", in: model, maxDepth: 4)

            NovaMLXLog.info("[ShardableModel] Discovered \(result.layers.count) layers at depth \(result.depth), embed=\(embedModule != nil), norm=\(normModule != nil), head=\(headModule != nil)")
            return
        }

        NovaMLXLog.error("[ShardableModel] Failed to discover layers via reflection (searched 4 levels)")
    }

    /// Recursive search for the first module containing "layers" as a direct child.
    private func searchForLayers(in module: Module, depth: Int, maxDepth: Int) -> (layers: [Module], embed: Module?, norm: Module?, depth: Int)? {
        let children = module.children()

        // Found the layer container
        let layers = extractLayers(from: children["layers"])
        if !layers.isEmpty {
            return (
                layers: layers,
                embed: unwrapModule(children["embed_tokens"]) ?? unwrapModule(children["embed"]),
                norm: unwrapModule(children["norm"]),
                depth: depth
            )
        }

        // Recurse into child modules (but not into arrays of modules — those ARE the layers)
        guard depth < maxDepth else { return nil }
        for (_, item) in children {
            if let child = unwrapModule(item) {
                if let result = searchForLayers(in: child, depth: depth + 1, maxDepth: maxDepth) {
                    return result
                }
            }
        }
        return nil
    }

    /// Search for a named module at any depth (e.g. "lm_head" or "head").
    private func searchForModule(named name: String, in module: Module, maxDepth: Int) -> Module? {
        let children = module.children()
        if let found = unwrapModule(children[name]) {
            return found
        }
        guard maxDepth > 0 else { return nil }
        for (_, item) in children {
            if let child = unwrapModule(item) {
                if let found = searchForModule(named: name, in: child, maxDepth: maxDepth - 1) {
                    return found
                }
            }
        }
        return nil
    }

    // MARK: - Forward Pass Slicing

    /// Run embedding only (first shard pre-step).
    func embed(_ tokens: MLXArray) -> MLXArray? {
        guard let embed = embedModule else { return nil }
        // Embedding modules conform to callAsFunction via @objc or direct invocation.
        // Use performNumericClosure to call the embed module.
        return (embed as? any UnaryLayer)?.callAsFunction(tokens)
    }

    /// Run the output head (norm + lm_head) — last shard post-step.
    func head(_ hidden: MLXArray) -> MLXArray? {
        var h = hidden
        // Apply final norm
        if let norm = normModule as? any UnaryLayer {
            h = norm(h)
        }
        // Apply lm_head
        if let head = headModule as? any UnaryLayer {
            return head(h)
        }
        // Fallback: if no head, try embed.asLinear (tieWordEmbeddings)
        if let embed = embedModule {
            return (embed as? any EmbeddingAsLinearProtocol)?.asLinear(h)
        }
        return nil
    }

    /// Run forward pass through a specific layer by index.
    /// Tries protocol conformance in order: 4-param (hybrid), 3-param (standard), 2-param (simple).
    func forwardLayer(_ index: Int, input: MLXArray, cache: KVCache?, attentionMask: MLXFast.ScaledDotProductAttentionMaskMode, ssmMask: MLXArray?) -> MLXArray? {
        guard index >= 0 && index < layerModules.count else { return nil }
        let layer = layerModules[index]

        // Try 4-param signature: (x, attentionMask, ssmMask, cache) — Qwen35, Qwen3Next, etc
        if let block = layer as? any DistributedLayerProtocol4 {
            return block.callAsFunction(input, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
        }
        // Try 3-param signature: (x, mask, cache) — Qwen3, Qwen3MoE, etc
        if let block = layer as? any DistributedLayerProtocol3 {
            return block.callAsFunction(input, mask: attentionMask, cache: cache)
        }
        // Fallback: 2-param signature (x, cache) — older/simpler architectures
        if let block = layer as? any TransformerBlockProtocol {
            return block.callAsFunction(input, cache: cache)
        }
        return nil
    }

    /// Run forward pass through a range of layers with proper mask construction.
    /// For hybrid models (e.g. Qwen35), builds separate masks for attention vs SSM layers
    /// based on cache type: KVCacheSimple → attention, MambaCache → linear/SSM.
    func forwardLayers(_ range: Range<Int>, input: MLXArray, caches: [KVCache]) -> MLXArray {
        var h = input
        // Build masks: find an attention cache for faMask, a MambaCache for ssmMask
        let attnCache = caches.first { !($0 is MambaCache) }
        let mambaCache = caches.first { $0 is MambaCache }
        let faMask = createAttentionMask(h: h, cache: attnCache)
        let ssmMask = createSSMMask(h: h, cache: mambaCache as? MambaCache)

        for i in range {
            let cacheIdx = i - range.lowerBound
            let cache = cacheIdx < caches.count ? caches[cacheIdx] : nil
            // Select mask per layer based on cache type (matches Qwen35 model logic)
            let isSSM = cache is MambaCache
            let layerAttnMask: MLXFast.ScaledDotProductAttentionMaskMode = isSSM ? .none : faMask
            let layerSSMMask: MLXArray? = isSSM ? ssmMask : nil
            if let output = forwardLayer(i, input: h, cache: cache, attentionMask: layerAttnMask, ssmMask: layerSSMMask) {
                h = output
            }
        }
        return h
    }

    /// Total number of layers.
    public var count: Int { layerCount }

    // MARK: - Helpers

    private func unwrapModule(_ item: NestedItem<String, Module>?) -> Module? {
        guard let item = item else { return nil }
        switch item {
        case .value(let module): return module
        default: return nil
        }
    }

    private func extractLayers(from item: NestedItem<String, Module>?) -> [Module] {
        guard let item = item else { return [] }
        switch item {
        case .array(let items):
            return items.compactMap { subItem in
                if case .value(let module) = subItem { return module }
                return nil
            }
        case .value(let module):
            return [module]
        default:
            return []
        }
    }
}

// MARK: - SlicedForwardPolicy

/// A ``ComputePolicy`` that runs only a range of model layers, enabling pipeline-parallel
/// distribution across multiple nodes.
///
/// Uses ``ShardableModel`` reflection to discover and execute specific layers without
/// modifying vendor model code.
public final class SlicedForwardPolicy: ComputePolicy, @unchecked Sendable {

    public let assignment: ShardAssignment
    public private(set) var isReady: Bool = false

    private weak var engine: MLXEngine?
    private let modelId: String
    private var shardableModel: ShardableModel? = nil
    private var kvCaches: [KVCache] = []

    /// Which layers this policy is responsible for.
    public let layerRange: Range<Int>

    /// Whether this shard includes the embedding step (first shard).
    public let isFirst: Bool

    /// Whether this shard includes norm + head (last shard).
    public let isLast: Bool

    /// Whether any of this shard's caches are MambaCache (hybrid model with SSM/linear layers).
    /// MambaCache uses fixed-size state that can't be trimmed — double-buffer decode
    /// would corrupt it on rollback.
    public var hasMambaCache: Bool {
        kvCaches.contains { $0 is MambaCache }
    }

    public init(
        assignment: ShardAssignment,
        engine: MLXEngine,
        modelId: String,
        isFirst: Bool,
        isLast: Bool
    ) {
        self.assignment = assignment
        self.engine = engine
        self.modelId = modelId
        self.layerRange = assignment.startLayer..<assignment.endLayer
        self.isFirst = isFirst
        self.isLast = isLast
    }

    public func bindWeights() async throws {
        // Ensure model is loaded in engine (may not be when workerMode=true,
        // since models load in the Worker subprocess, not in the main engine)
        if engine?.getContainer(for: modelId) == nil {
            NovaMLXLog.info("[SlicedForwardPolicy] Model \(modelId) not in engine, loading via admin API...")
            guard let url = URL(string: "http://127.0.0.1:6591/admin/models/load") else {
                throw ShardEngineError.modelNotAvailable(modelId)
            }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if let body = try? JSONEncoder().encode(["modelId": modelId]) {
                request.httpBody = body
            }
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResp = response as? HTTPURLResponse, !(200...299).contains(httpResp.statusCode) {
                throw ShardEngineError.modelNotAvailable(modelId)
            }
            NovaMLXLog.info("[SlicedForwardPolicy] Model \(modelId) loaded via admin API")
        }

        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else {
            throw ShardEngineError.modelNotAvailable(modelId)
        }

        let shardableBox = await mlxContainer.perform { context in
            let shardable = ShardableModel(model: context.model)
            // Create caches for the assigned layer range (clamped to actual layer count)
            let allCaches = context.model.newCache(parameters: nil)
            let clampedRange = Swift.min(layerRange.lowerBound, allCaches.count)..<Swift.min(layerRange.upperBound, allCaches.count)
            let slicedCaches = Array(allCaches[clampedRange])
            return (SendableBox(shardable), KVCacheBox(slicedCaches))
        }

        self.shardableModel = shardableBox.0.value
        self.kvCaches = shardableBox.1.caches
        isReady = true

        NovaMLXLog.info("[SlicedForwardPolicy] Bound layers \(layerRange) (\(isFirst ? "first" : "mid")/\(isLast ? "last" : "mid")), \(kvCaches.count) caches")
    }

    /// Reset KV caches for a new inference request.
    /// Creates fresh caches — old state is discarded.
    /// Must be called before each new conversation to prevent cross-request contamination.
    public func resetCaches() async throws {
        guard isReady else { return }
        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else { return }

        let range = self.layerRange
        let cacheBox = await mlxContainer.perform { context in
            let allCaches = context.model.newCache(parameters: nil)
            let clampedRange = Swift.min(range.lowerBound, allCaches.count)..<Swift.min(range.upperBound, allCaches.count)
            return KVCacheBox(Array(allCaches[clampedRange]))
        }
        self.kvCaches = cacheBox.caches
    }

    public func compute(input: MLXArray) async throws -> MLXArray {
        return try await computeInternal(input: input, runHead: isLast)
    }

    /// Compute embedding + layers only, skipping norm+head.
    /// Used during prefill when coordinator owns head but needs to send hidden state to worker.
    func computeLayersOnly(input: MLXArray) async throws -> MLXArray {
        return try await computeInternal(input: input, runHead: false)
    }

    private func computeInternal(input: MLXArray, runHead: Bool) async throws -> MLXArray {
        guard isReady, let shardable = shardableModel else {
            throw ShardEngineError.notReady
        }
        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else {
            throw ShardEngineError.modelNotAvailable(modelId)
        }

        let inputBox = SendableBox(input)
        let cacheBox = KVCacheBox(kvCaches)
        let range = self.layerRange
        let isFirst = self.isFirst

        let resultBox = await mlxContainer.perform { context in
            var h = inputBox.value

            // Step 1: Embed (first shard only)
            if isFirst {
                // Normalize input to 3D: [batch=1, seq_len, hidden_dim]
                // Prefill: [seq_len] → [1, seq_len]; Decode: scalar → [1] → [1, 1]
                if h.ndim == 0 {
                    h = h.reshaped([1, 1])
                } else if h.ndim == 1 {
                    h = h.expandedDimensions(axis: 0) // [seq_len] → [1, seq_len]
                }
                if let embedded = shardable.embed(h) {
                    h = embedded // [1, seq_len, hidden_dim]
                }
            } else {
                // Hidden states from previous shard: ensure 3D for layer processing
                if h.ndim == 2 {
                    h = h.expandedDimensions(axis: 0) // [seq_len, dim] → [1, seq_len, dim]
                }
            }

            // Step 2: Run assigned layers
            let layerCaches = cacheBox.caches
            h = shardable.forwardLayers(range, input: h, caches: layerCaches)

            // Step 3: Norm + Head (only when runHead=true)
            if runHead {
                if let output = shardable.head(h) {
                    h = output
                }
                // Head returns [batch, seq_len, vocab] — squeeze batch for sampling
                if h.ndim == 3 && h.dim(0) == 1 {
                    h = h.squeezed(axis: 0)
                }
            }

            // asyncEval: GPU starts computing while CPU prepares send/sampling
            // Downstream (sendTensor.asData / argmax) will sync when needed
            MLX.asyncEval(h)
            return SendableBox(h)
        }

        return resultBox.value
    }

    /// Run norm + lm_head + argmax on a hidden state tensor.
    /// Used by the coordinator after receiving the worker's output.
    /// Returns (sampledTokenId, logits).
    func computeHeadOnly(_ hidden: MLXArray) async -> (tokenId: Int, logits: MLXArray)? {
        guard isReady, let shardable = shardableModel else { return nil }
        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else { return nil }

        let hiddenBox = SendableBox(hidden)
        let resultBox = await mlxContainer.perform { _ in
            guard let logits = shardable.head(hiddenBox.value) else {
                return nil as SendableBox<(Int, MLXArray)>?
            }
            MLX.eval(logits)
            let tokenId = argmaxToken(logits)
            return SendableBox((tokenId, logits))
        }
        guard let result = resultBox?.value else { return nil }
        return (tokenId: result.0, logits: result.1)
    }

    public func releaseWeights() {
        shardableModel = nil
        kvCaches = []
        isReady = false
    }

    /// Snapshot of MambaCache states for safe rollback during double-buffer decode.
    /// MambaCache uses fixed-size arrays updated in-place — trim() corrupts them.
    /// Instead, snapshot before prediction, restore if prediction was wrong.
    public struct MambaSnapshot: @unchecked Sendable {
        let states: [[MLXArray]]  // One [MLXArray] per MambaCache in kvCaches order
        let offsets: [Int]
    }

    /// Snapshot all MambaCache states. Non-Mamba caches are skipped (trim handles them).
    /// Forces eval on copied arrays to ensure snapshot is materialized (not lazy views).
    public func snapshotMambaStates() -> MambaSnapshot? {
        guard isReady else { return nil }
        var states: [[MLXArray]] = []
        var offsets: [Int] = []
        for cache in kvCaches {
            if let mamba = cache as? MambaCache {
                let copied = mamba.state.map { $0[.ellipsis] }
                for arr in copied { MLX.eval(arr) }
                states.append(copied)
                offsets.append(mamba.offset)
            }
        }
        guard !states.isEmpty else { return nil }
        return MambaSnapshot(states: states, offsets: offsets)
    }

    /// Restore MambaCache states from a previous snapshot.
    /// KVCacheSimple entries are handled by trim in rollbackCache.
    public func restoreMambaStates(from snapshot: MambaSnapshot) {
        guard isReady else { return }
        var snapIdx = 0
        for cache in kvCaches {
            if let mamba = cache as? MambaCache, snapIdx < snapshot.states.count {
                mamba.state = snapshot.states[snapIdx]
                mamba.offset = snapshot.offsets[snapIdx]
                snapIdx += 1
            }
        }
    }

    /// Rollback caches after wrong predictions.
    /// KVCacheSimple: trim excess entries. MambaCache: restore from snapshot.
    /// - Parameters:
    ///   - position: Cache offset to restore to (confirmed position)
    ///   - speculatedCount: How many tokens were speculated (for logging)
    ///   - mambaSnapshot: Snapshot taken before speculation (restores MambaCache)
    public func rollbackCache(position: Int, speculatedCount: Int = 1, mambaSnapshot: MambaSnapshot? = nil) async throws {
        guard isReady else { return }
        for cache in kvCaches {
            if cache is MambaCache { continue }
            let currentOffset = cache.offset
            let excess = currentOffset - position
            if excess > 0 {
                cache.trim(excess)
            }
        }
        if let snapshot = mambaSnapshot {
            restoreMambaStates(from: snapshot)
        }
    }

    // MARK: - Double-Buffer Prediction

    /// Result of speculative multi-token draft: predicted tokens + their precomputed hidden states.
    public struct DraftBundle: @unchecked Sendable {
        public let predictedTokens: [Int]
        public let precomputedStates: [MLXArray]
        public let mambaSnapshot: MambaSnapshot?
        public var count: Int { predictedTokens.count }
    }

    /// Draft K tokens from a hidden state, precomputing hidden states for each.
    ///
    /// Pipeline per draft token:
    /// 1. Apply norm + head on current hidden → argmax → predicted token
    /// 2. Embed predicted token, run coord layers → precomputed hidden (updates caches)
    ///
    /// The first token also runs N extra worker layers (throwaway caches) for better accuracy.
    /// MambaCache snapshot is taken before any cache modifications for safe rollback.
    public func draftTokens(
        from activation: MLXArray,
        count: Int = 2,
        extraPredictionLayers: Int = 0
    ) async throws -> DraftBundle? {
        guard isReady, let shardable = shardableModel else { return nil }
        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else { return nil }

        let mambaSnapshot = snapshotMambaStates()

        let hiddenBox = SendableBox(activation)
        let cacheBox = KVCacheBox(kvCaches)
        let range = self.layerRange
        let extraLayers = extraPredictionLayers

        let result = await mlxContainer.perform { context in
            var h = hiddenBox.value

            // Step 1: Run extra worker layers for better first-token prediction
            if extraLayers > 0 {
                let workerStart = range.upperBound
                let extraEnd = min(workerStart + extraLayers, shardable.count)
                if workerStart < extraEnd {
                    let allCaches = context.model.newCache(parameters: nil)
                    let extraRange = workerStart..<extraEnd
                    let clamped = Swift.min(extraRange.lowerBound, allCaches.count)..<Swift.min(extraRange.upperBound, allCaches.count)
                    let tempCaches = Array(allCaches[clamped])
                    h = shardable.forwardLayers(extraRange, input: h, caches: tempCaches)
                }
            }

            // Step 2: Draft K tokens — predict, embed, run coord layers for each
            var tokens: [Int] = []
            var states: [SendableBox<MLXArray>] = []

            for _ in 0..<count {
                guard let logits = shardable.head(h) else { break }
                MLX.eval(logits)
                let predicted = argmaxToken(logits)
                tokens.append(predicted)

                var embedded = MLXArray(Int32(predicted))
                if embedded.ndim == 0 { embedded = embedded.reshaped([1, 1]) }
                else if embedded.ndim == 1 { embedded = embedded.expandedDimensions(axis: 0) }
                if let emb = shardable.embed(embedded) { embedded = emb }
                h = shardable.forwardLayers(range, input: embedded, caches: cacheBox.caches)
                MLX.asyncEval(h)
                states.append(SendableBox(h))
            }

            guard !tokens.isEmpty else { return nil as ([Int], [SendableBox<MLXArray>])? }
            return (tokens, states)
        }

        guard let result = result else { return nil }
        return DraftBundle(
            predictedTokens: result.0,
            precomputedStates: result.1.map { $0.value },
            mambaSnapshot: mambaSnapshot
        )
    }
}
