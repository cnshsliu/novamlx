import Foundation
import MLX
import MLXNN
import MLXLMCommon
import NovaMLXUtils

// Local typealias to avoid MLXNN import path ambiguity in installator body.
typealias TIEQuantizedLinear = QuantizedLinear

private extension String {
    /// Returns `self` minus `prefix` if it starts with `prefix`, else nil.
    func strippedPrefix(_ prefix: String) -> String? {
        guard hasPrefix(prefix) else { return nil }
        return String(dropFirst(prefix.count))
    }
}

// NovaMLX-TIE: TierHookInstallator
//
// One-shot installer that wires MLXNN's TierHooks.switchLinearHook and
// TierHooks.linearHook to forward calls into TierHookCoordinator. The hooks
// fire on every SwitchLinear/Linear callAsFunction from any model.
//
// Hot-path discipline: hooks must be cheap because they fire per-token per-layer.
// We deliberately do NOT call `indices.asArray()` synchronously (it forces
// GPU→CPU eval and would be O(num_layers) sync stalls per token). Instead
// we capture ObjectIdentifier (cheap) and defer materialization to Phase 3
// when the SSD streaming consumer actually needs the expert IDs.

public enum TierHookInstallator {
    private nonisolated(unsafe) static var installed = false
    private nonisolated(unsafe) static var installLock = NSLock()

    /// Toggle for Phase 3: when true, the SwitchLinear hook materializes
    /// indices and forwards expert IDs to the heat map. False (default) =
    /// record layer activity only, zero sync eval overhead.
    public nonisolated(unsafe) static var detailedExpertTracking = false

    public static func installIfNeeded() {
        installLock.lock(); defer { installLock.unlock() }
        guard !installed else { return }
        installed = true

        // Async hooks used to spawn a Task per Linear/SwitchLinear call
        // (hundreds per token). Heat-map + prefetch now run in the sync hook
        // from IDs we already materialized — no extra GPU sync, no Task.
        TierHooks.switchLinearHook = nil
        TierHooks.linearHook = nil
        // Sync hook: lazy weight load BEFORE matmul. First Linear in a layer
        // pays SSD; the rest hit WeightTierManager's layer-file cache.
        TierHooks.linearSyncHook = { instance in
            let id = ObjectIdentifier(instance)
            if TierContextStore.shared.isLoaded(id) {
                TierContextStore.shared.touch(id)
                if let ctx = TierContextStore.shared.get(id, kind: .linear) {
                    Self.prefetchAhead(ctx: ctx, expertIDs: [])
                }
                return
            }
            guard let ctx = TierContextStore.shared.get(id, kind: .linear) else { return }
            guard ctx.layerIdx >= 0 else { return }
            guard let linear = instance as? Linear else { return }
            ctx.policy.heatMap.record(layer: ctx.layerIdx, experts: [])
            do {
                let tensors = try ctx.policy.weightManager.readLayer(layer: ctx.layerIdx)
                func lookup(_ suffix: String) -> MLXArray? {
                    if let v = tensors[ctx.path + suffix] { return v }
                    if ctx.path.hasPrefix("layers."),
                       let v = tensors["model." + ctx.path + suffix] { return v }
                    if ctx.path.hasPrefix("model.layers."),
                       let stripped = ctx.path.strippedPrefix("model."),
                       let v = tensors[stripped + suffix] { return v }
                    return nil
                }
                if let weight = lookup(".weight") { linear.weight = weight }
                else { NovaMLXLog.warning("[TIE-SYNC] no weight key for path=\(ctx.path)") }
                if let bias = lookup(".bias") { linear.bias = bias }
                if let q = linear as? TIEQuantizedLinear {
                    if let scales = lookup(".scales") { q.scales = scales }
                    else { NovaMLXLog.warning("[TIE-SYNC] no scales key for path=\(ctx.path)") }
                    if let biases = lookup(".biases") { q.biases = biases }
                    NovaMLXLog.debug("[TIE-SYNC] QuantizedLinear loaded path=\(ctx.path)")
                }
                let shape = linear.weight.shape
                let bytes = Int64(shape.reduce(1, *) * linear.weight.dtype.size)
                TierContextStore.shared.markLoaded(id, bytes: bytes) { [weak linear] in
                    linear?.weight = MLXArray.zeros(shape)
                }
                Self.prefetchAhead(ctx: ctx, expertIDs: [])
            } catch {
                // SSD read failed — leave weight as-is (eager value).
            }
        }
        TierHooks.switchLinearSyncHook = { instance, indices in
            let id = ObjectIdentifier(instance)
            guard let ctx = TierContextStore.shared.get(id, kind: .switchLinear) else { return }
            guard ctx.layerIdx >= 0 else { return }
            guard let sw = instance as? SwitchLinear else { return }

            if ctx.policy.perExpertStreaming {
                Self.applyPerExpertStreaming(id: id, ctx: ctx, sw: sw, indices: indices)
                return
            }

            if TierContextStore.shared.isLoaded(id) {
                TierContextStore.shared.touch(id)
                Self.prefetchAhead(ctx: ctx, expertIDs: [])
                return
            }
            do {
                let tensors = try ctx.policy.weightManager.readLayerStacked(layer: ctx.layerIdx)
                let weightKey = ctx.path + ".weight"
                if let weight = tensors[weightKey] { sw.weight = weight }
                let biasKey = ctx.path + ".bias"
                if let bias = tensors[biasKey] { sw.bias = bias }
                if let q = sw as? QuantizedSwitchLinear {
                    let scalesKey = ctx.path + ".scales"
                    if let scales = tensors[scalesKey] { q.scales = scales }
                    let qBiasesKey = ctx.path + ".biases"
                    if let qBiases = tensors[qBiasesKey] { q.biases = qBiases }
                }
                let shape = sw.weight.shape
                let bytes = Int64(shape.reduce(1, *) * sw.weight.dtype.size)
                TierContextStore.shared.markLoaded(id, bytes: bytes) { [weak sw] in
                    sw?.weight = MLXArray.zeros(shape)
                }
                Self.prefetchAhead(ctx: ctx, expertIDs: [])
            } catch { }
        }
        NovaMLXLog.info("[TIE] global hooks installed (SwitchLinear + Linear)")
    }

    /// Per-expert MoE path. SwitchGLU hits this 3× with the same indices
    /// tensor — reuse the first asArray + remapping. Same expert set on
    /// the next token skips restack.
    private static func applyPerExpertStreaming(
        id: ObjectIdentifier,
        ctx: TierHookContext,
        sw: SwitchLinear,
        indices: MLXArray
    ) {
        let indicesId = ObjectIdentifier(indices)
        let cache = TierHotPathCache.shared

        var rawIndices: [Int]?
        var expertIDs: [Int]
        var localArray: MLXArray?

        if let hit = cache.reuseForward(layer: ctx.layerIdx, indicesId: indicesId) {
            expertIDs = hit.expertIDs
            localArray = hit.localIndices
        } else {
            let raw = materializeIndices(indices)
            if raw.isEmpty { return }
            rawIndices = raw
            expertIDs = Array(Set(raw)).sorted()
        }
        if expertIDs.isEmpty { return }

        ctx.policy.heatMap.record(layer: ctx.layerIdx, experts: expertIDs)

        if cache.stackedMatches(id: id, uniqueSorted: expertIDs) {
            let remapped: MLXArray
            if let local = localArray {
                remapped = local
            } else {
                let original = rawIndices ?? materializeIndices(indices)
                let globalToLocal = Dictionary(uniqueKeysWithValues: expertIDs.enumerated().map { ($1, $0) })
                remapped = MLXArray(original.map { globalToLocal[$0] ?? 0 })
                    .asType(.int32)
                    .reshaped(indices.shape)
                cache.rememberForward(
                    layer: ctx.layerIdx, indicesId: indicesId,
                    expertIDs: expertIDs, localIndices: remapped
                )
            }
            sw.tieLocalIndices = remapped
            Self.prefetchAhead(ctx: ctx, expertIDs: expertIDs)
            return
        }

        do {
            let loaded = try ctx.policy.weightManager.loadActivatedExperts(
                layer: ctx.layerIdx, expertIDs: expertIDs)

            var weights: [MLXArray] = []
            var biasesList: [MLXArray] = []
            var hasBias = false
            var scalesList: [MLXArray] = []
            var qBiasesList: [MLXArray] = []
            var hasScales = false
            for e in expertIDs {
                guard let tensors = loaded[e],
                      let w = tensors[ctx.path + ".weight"] else { continue }
                weights.append(w)
                if let b = tensors[ctx.path + ".bias"] {
                    biasesList.append(b)
                    hasBias = true
                }
                if let s = tensors[ctx.path + ".scales"] {
                    scalesList.append(s)
                    hasScales = true
                }
                if let qb = tensors[ctx.path + ".biases"] {
                    qBiasesList.append(qb)
                }
            }
            guard !weights.isEmpty else { return }

            let stackedWeight = MLX.stacked(weights, axis: 0)
            MLX.eval(stackedWeight)
            sw.weight = stackedWeight
            if hasBias && biasesList.count == weights.count {
                let stackedBias = MLX.stacked(biasesList, axis: 0)
                MLX.eval(stackedBias)
                sw.bias = stackedBias
            }
            if let q = sw as? QuantizedSwitchLinear, hasScales {
                let stackedScales = MLX.stacked(scalesList, axis: 0)
                MLX.eval(stackedScales)
                q.scales = stackedScales
                if !qBiasesList.isEmpty && qBiasesList.count == weights.count {
                    let stackedQBiases = MLX.stacked(qBiasesList, axis: 0)
                    MLX.eval(stackedQBiases)
                    q.biases = stackedQBiases
                }
            }

            let remapped: MLXArray
            if let local = localArray {
                remapped = local
            } else {
                let original = rawIndices ?? materializeIndices(indices)
                let globalToLocal = Dictionary(uniqueKeysWithValues: expertIDs.enumerated().map { ($1, $0) })
                let localInts = original.map { globalToLocal[$0] ?? 0 }
                remapped = MLXArray(localInts).asType(.int32).reshaped(indices.shape)
            }
            sw.tieLocalIndices = remapped
            cache.rememberForward(
                layer: ctx.layerIdx, indicesId: indicesId,
                expertIDs: expertIDs, localIndices: remapped
            )
            cache.rememberStacked(id: id, uniqueSorted: expertIDs)
            Self.prefetchAhead(ctx: ctx, expertIDs: expertIDs)
        } catch {
            NovaMLXLog.warning("[TIE-PE] per-expert load failed: \(error.localizedDescription)")
        }
    }

    /// Overlap next-layer I/O with this layer's GEMM. Same expert IDs are a
    /// cheap heuristic (MoE layers are correlated); heat-map top-k fills gaps.
    private static func prefetchAhead(ctx: TierHookContext, expertIDs: [Int]) {
        let wm = ctx.policy.weightManager
        let current = ctx.layerIdx
        guard current >= 0 else { return }
        var next = current + 1
        let maxLayer = wm.maxLayerIndex
        if maxLayer >= 0 && next > maxLayer { next = 0 }
        wm.markHotLayers([current, next])

        if ctx.policy.perExpertStreaming || ctx.shardKind == .expert {
            var ids = expertIDs
            if ids.count > 16 {
                ids = Array(ids.prefix(16))
            }
            let hot = ctx.policy.heatMap.topExperts(layer: next, k: 8)
            if !hot.isEmpty {
                ids = Array(Set(ids + hot))
            }
            if !ids.isEmpty {
                wm.prefetchExperts(layer: next, expertIDs: ids)
            }
        } else {
            wm.prefetchLayer(next)
        }
    }

    private static func materializeIndices(_ indices: MLXArray) -> [Int] {
        if indices.dtype == .int64 {
            return indices.asArray(Int.self)
        }
        return indices.asArray(Int32.self).map { Int($0) }
    }
}
