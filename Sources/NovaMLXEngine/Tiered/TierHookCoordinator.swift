import Foundation
import MLX
import MLXNN
import MLXLMCommon
import NovaMLXUtils

// NovaMLX-TIE: TierHookCoordinator
//
// Universal registry that connects a loaded TieredOffloadPolicy to every
// SwitchLinear + Linear instance in a model. The patched SwitchLinear /
// Linear callAsFunction consult this coordinator at the top of each call;
// hooks fire heat-map.record / prefetcher.notify / acquire when a context
// is registered for that instance.
//
// This decouples TIE from any specific model class. Any model that uses
// SwitchLinear (all MoE: Bailing, Qwen3-Next, DeepseekV3, V4, PhiMoE, ...)
// or Linear (all dense: Llama, Mistral, Gemma, Phi, Qwen-dense, ...) gets
// TIE automatically — zero per-model code.
//
// Lifecycle:
//   - On model load: MLXEngine.applyTierPolicyToModel calls register(model:policy:)
//   - register walks Module.visit to find all SwitchLinear + Linear instances,
//     derives layerIdx from path regex `model\.layers\.(\d+)`, stores context
//   - Patched primitives call `contextFor(instance)` at top of callAsFunction
//   - On model unload: unregister(policy:) clears all entries

/// Per-instance context stored in the registry. Looked up by ObjectIdentifier
/// from inside SwitchLinear/Linear callAsFunction.
public struct TierHookContext: Sendable {
    public let policy: TieredOffloadPolicy
    public let layerIdx: Int
    public let shardKind: ShardKind
    /// Optional path (e.g. "model.layers.5.mlp.switch_mlp.gate_proj") for debugging.
    public let path: String
}

public enum ShardKind: Sendable {
    case expert    // SwitchLinear inside MoE
    case layer     // Linear in a dense model
}

public actor TierHookCoordinator {
    public static let shared = TierHookCoordinator()

    /// Track all contexts registered for a given policy (for unregister).
    /// The actual context dicts live in TierContextStore for sync access.
    private var contextsByPolicy: [ObjectIdentifier: [ObjectIdentifier]] = [:]

    public init() {}

    /// Walk the model's module tree, register a context for every SwitchLinear
    /// and Linear instance. Layer index is derived from the path prefix
    /// `model.layers.{N}.*`; modules without a layer prefix are skipped
    /// (embeddings, final norm, lm_head, etc. — these stay in tier 0).
    public func register(model: Module, policy: TieredOffloadPolicy, strategy: TierStrategy) -> (experts: Int, linears: Int) {
        var expertCount = 0
        var linearCount = 0
        let policyId = ObjectIdentifier(policy)

        // Determine shardKind per-model based on strategy
        let expertKind: ShardKind = .expert
        let linearKind: ShardKind = (strategy == .layer) ? .layer : .expert

        model.visit { name, module in
            // Derive layer index from path
            let layerIdx = Self.extractLayerIndex(from: name)

            if let sw = module as? SwitchLinear {
                let id = ObjectIdentifier(sw)
                let ctx = TierHookContext(policy: policy, layerIdx: layerIdx ?? -1,
                                          shardKind: expertKind, path: name)
                TierContextStore.shared.setSwitchLinear(id, ctx)
                contextsByPolicy[policyId, default: []].append(id)
                expertCount += 1
            } else if let li = module as? Linear {
                let id = ObjectIdentifier(li)
                let ctx = TierHookContext(policy: policy, layerIdx: layerIdx ?? -1,
                                          shardKind: linearKind, path: name)
                TierContextStore.shared.setLinear(id, ctx)
                contextsByPolicy[policyId, default: []].append(id)
                linearCount += 1
            }
        }

        NovaMLXLog.info("[TIE] registered \(expertCount) SwitchLinear + \(linearCount) Linear instances, strategy=\(strategy.rawValue)")
        return (expertCount, linearCount)
    }

    /// Remove all contexts associated with the given policy.
    public func unregister(policy: TieredOffloadPolicy) {
        let policyId = ObjectIdentifier(policy)
        guard let ids = contextsByPolicy[policyId] else { return }
        TierContextStore.shared.clearForPolicy(policyId, ids: ids)
        contextsByPolicy.removeValue(forKey: policyId)
        TierHotPathCache.shared.reset()
        NovaMLXLog.info("[TIE] unregistered \(ids.count) hook contexts")
    }

    // MARK: - Lookup (called from patched primitives)

    /// Sendable-safe lookup: takes ObjectIdentifier instead of the actual
    /// non-Sendable module instance. Used by tests + external callers.
    public func context(id: ObjectIdentifier, kind: TierContextStore.HookKind) -> TierHookContext? {
        TierContextStore.shared.get(id, kind: kind)
    }

    public func context(forSwitchLinear instance: SwitchLinear) -> TierHookContext? {
        TierContextStore.shared.get(ObjectIdentifier(instance), kind: .switchLinear)
    }

    public func context(forLinear instance: Linear) -> TierHookContext? {
        TierContextStore.shared.get(ObjectIdentifier(instance), kind: .linear)
    }

    // MARK: - Hook entry points (called from TierHookInstallator)

    /// Called from the patched SwitchLinear.callAsFunction via the global hook.
    /// `id` is the ObjectIdentifier of the SwitchLinear instance (AnyObject-
    /// erased because TierHooks lives in MLXNN which can't name SwitchLinear).
    /// `expertIDs` is materialized synchronously by the installator before the
    /// Task to keep Sendable invariants.
    public func handleSwitchLinearCall(id: ObjectIdentifier, expertIDs: [Int]) async {
        guard let ctx = TierContextStore.shared.get(id, kind: .switchLinear) else { return }
        ctx.policy.heatMap.record(layer: ctx.layerIdx, experts: expertIDs)
    }

    /// Called from the patched Linear.callAsFunction via the global hook.
    public func handleLinearCall(id: ObjectIdentifier) async {
        guard let ctx = TierContextStore.shared.get(id, kind: .linear) else { return }
        ctx.policy.heatMap.record(layer: ctx.layerIdx, experts: [])
    }

    // MARK: - Internals

    /// Parse "model.layers.5.mlp.switch_mlp.gate_proj" → 5.
    /// Returns nil for paths without a layer index (embed_tokens, norm, etc.).
    private static func extractLayerIndex(from path: String) -> Int? {
        // Match `layers.N` and extract digits AFTER the dot.
        guard let range = path.range(of: #"layers\.(\d+)"#, options: .regularExpression) else {
            return nil
        }
        let matchStr = String(path[range])  // e.g. "layers.5"
        // Everything after "layers." is the digit run.
        guard matchStr.hasPrefix("layers.") else { return nil }
        let digits = String(matchStr.dropFirst("layers.".count))
        return Int(digits)
    }
}
