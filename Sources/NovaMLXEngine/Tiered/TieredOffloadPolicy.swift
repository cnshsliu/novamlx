import Foundation
import NovaMLXUtils

// NovaMLX-TIE: TieredOffloadPolicy
//
// Per-model coordinator for the Tiered Inference Engine. Owns the
// WeightTierManager + ExpertHeatMap + ExpertPrefetcher (Phase 2+) and is
// attached to ModelContainer.tierPolicy when a tier-manifest.json is present.
//
// Phase 1 scope:
//   - bind() loads the manifest, updates MetricsStore with tier sizes
//   - unbind() clears state
//   - No inference-time behavior change yet. The eager MLX load path still
//     loads all weights. This is intentional: Phase 1 validates the plumbing
//     and instrumentation without changing runtime semantics.
//
// Phase 2 will:
//   - Replace eager load with Tier 0-only load (via WeightTierManager.bind)
//   - Add ExpertHeatMap + tier-1 LRU
//   - Wire prefetcher hooks into the layer loop (BailingHybridModel.swift:711)
//
// Phase 3+:
//   - Router-logit prefetch
//   - Tiered quantization
//   - (optional) NPU shared-expert offload

public final class TieredOffloadPolicy: @unchecked Sendable {
    public let manifest: TierManifest
    public let modelDir: URL
    private let metrics: MetricsStore?
    public let weightManager: WeightTierManager
    public let heatMap: ExpertHeatMap
    public let prefetcher: ExpertPrefetcher

    /// Phase 6: when true, SwitchLinear sync hook does per-expert streaming
    /// (load ONLY activated experts, not whole layer). Auto-enabled for MoE
    /// models (.expert strategy) — savings are largest here.
    public var perExpertStreaming: Bool = false

    /// Background Task that triggers periodic LRU eviction. Cancelled on unbind.
    private var evictionTask: Task<Void, Never>?

    /// True iff this policy is actually changing runtime behavior.
    /// Phase 1: always false (manifest detected + logged, but no streaming yet).
    /// Phase 2: true when WeightTierManager.bind completes mlock + Tier 2 setup.
    public var isActive: Bool { weightManager.isBound }

    public init(manifest: TierManifest, modelDir: URL, metrics: MetricsStore? = nil) {
        self.manifest = manifest
        self.modelDir = modelDir
        self.metrics = metrics
        self.weightManager = WeightTierManager()
        // Phase 6: auto-enable per-expert streaming for MoE models.
        // 256 experts × ~8 activated per token = ~32× memory reduction.
        self.perExpertStreaming = (manifest.strategy == .expert)
        // Estimate layer + expert counts from manifest.
        let layerCount = Set(manifest.experts.map { $0.layer }).count
        let expertCount = manifest.expertCount / max(1, layerCount)
        self.heatMap = ExpertHeatMap(layerCount: layerCount, expertCount: expertCount)
        self.prefetcher = ExpertPrefetcher(
            manifest: manifest,
            heatMap: heatMap,
            weightManager: weightManager,
            metrics: metrics
        )
    }

    /// Bind Tier 0 weights + open Tier 2 file handles. Phase 1: WeightTierManager
    /// just records state without doing the mlock dance (deferred to Phase 2).
    public func bind() async {
        NovaMLXLog.info("[TIE] TieredOffloadPolicy.bind: \(modelDir.lastPathComponent) layout=\(manifest.layout) experts=\(manifest.expertCount)")
        do {
            try await weightManager.bind(modelDir: modelDir, manifest: manifest)
            await prefetcher.setWeightManager(weightManager)
        } catch {
            NovaMLXLog.warning("[TIE] bind failed: \(error.localizedDescription). Falling back to eager load.")
            metrics?.recordMlockFailure()
            return
        }
        let (t0, t1, t2) = weightManager.tierSizes()
        metrics?.setTierSizes(tier0: t0, tier1: t1, tier2: t2)
        NovaMLXLog.info("[TIE] bound: tier0=\(t0 / 1_048_576)MB tier2_total=\(t2 / 1_048_576)MB experts=\(manifest.expertCount)")

        // Start periodic eviction task (every 5s). Decoupled from scheduler —
        // runs as long as the policy is bound.
        startEvictionTask()
    }

    private func startEvictionTask() {
        evictionTask?.cancel()
        evictionTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)  // 5s
                if Task.isCancelled { break }
                self?.endOfRequest()
            }
        }
    }

    public func unbind() {
        NovaMLXLog.info("[TIE] TieredOffloadPolicy.unbind: \(modelDir.lastPathComponent)")
        evictionTask?.cancel()
        evictionTask = nil
        weightManager.unbind()
        heatMap.reset()
        metrics?.setTierSizes(tier0: 0, tier1: 0, tier2: 0)
    }

    /// Phase 4: trigger LRU eviction at request boundary (called by scheduler
    /// when a batch drains). Releases Tier 1 weights that haven't been touched
    /// recently, freeing unified memory for the next request's working set.
    public func endOfRequest() {
        let budget = weightManager.tier1BudgetBytes
        // Only drop weights idle for 10s. Evicting the working set mid-prefill
        // or mid-decode forces a full SSD reload on the next token.
        let evicted = TierContextStore.shared.evictToFit(byteBudget: budget, minIdleSeconds: 10)
        if evicted > 0 {
            NovaMLXLog.info("[TIE] endOfRequest: evicted \(evicted) entries to fit \(budget / 1_048_576)MB budget")
        }
        let (t0, _, t2) = weightManager.tierSizes()
        metrics?.setTierSizes(tier0: t0, tier1: TierContextStore.shared.loadedBytes, tier2: t2)
    }

    /// Convenience: detect + bind in one call. Returns the policy if the model
    /// dir has a tier-manifest.json, else nil (caller uses eager load path).
    public static func bindIfTiered(
        modelDir: URL,
        metrics: MetricsStore? = nil
    ) async -> TieredOffloadPolicy? {
        guard TierManifestLoader.isTiered(modelDir) else { return nil }
        do {
            guard let manifest = try TierManifestLoader.loadIfPresent(modelDir: modelDir) else {
                return nil
            }
            let policy = TieredOffloadPolicy(manifest: manifest, modelDir: modelDir, metrics: metrics)
            await policy.bind()
            return policy
        } catch {
            NovaMLXLog.warning("[TIE] manifest present but unreadable: \(error.localizedDescription). Falling back to eager load.")
            return nil
        }
    }
}
