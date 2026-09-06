import Foundation
import MLX
import NovaMLXUtils

// NovaMLX-TIE: ExpertHeatMap
//
// Records per-token expert activations to identify hot experts (Tier 1
// candidates) vs cold experts (Tier 2 / SSD-resident).
//
// Data flow:
//   Layer loop calls record(layer:N, expertIndices:[...]) per token.
//   At request boundary (or batch drain), promote/demote pass runs:
//     - Hot experts (count > promoteThreshold) get loaded into Tier 1
//     - Cold experts (count < demoteThreshold) in Tier 1 get evicted
//
// Cold-start: per-model seed profile ships alongside the model (built offline
// from reference prompt corpus). For the first `seedWindow` tokens we use seed
// weights, decaying into empirical counts as we observe actual traffic.

public final class ExpertHeatMap: @unchecked Sendable {
    /// Per-layer expert activation counts.
    /// layer -> expert -> count
    private struct LayerCounts {
        var counts: [Int: UInt64] = [:]
    }

    private let lock = NSLock()
    private var layerCounts: [Int: LayerCounts] = [:]
    private let layerCount: Int
    private let expertCount: Int

    /// Ring buffer of the last N tokens' activations, used to compute recent
    /// promotion/demotion thresholds (workload shift detection).
    private let ringWindow: Int
    private var ring: [(layer: Int, experts: [Int])] = []
    private var ringIndex: Int = 0

    /// Promotion/demotion thresholds. Defaults are conservative; tune via
    /// MetricsStore.expertActivationEntropy.
    public var promoteThreshold: UInt64 = 8
    public var demoteThreshold: UInt64 = 2

    /// Seed profile: layer -> expert -> base weight (0..1). Used during cold-start.
    /// Loaded from `tier-manifest.json` companion file `tier-seed.json` if present.
    private var seed: [Int: [Int: Double]] = [:]
    private let seedWindow: Int = 256
    private var tokensObserved: UInt64 = 0

    public init(layerCount: Int, expertCount: Int, ringWindow: Int = 512) {
        self.layerCount = layerCount
        self.expertCount = expertCount
        self.ringWindow = ringWindow
        self.ring.reserveCapacity(ringWindow)
    }

    public func loadSeed(_ seed: [Int: [Int: Double]]) {
        lock.lock(); defer { lock.unlock() }
        self.seed = seed
    }

    /// Record expert activations for one token at one layer.
    /// `layer` must be >= 0 (negative = unknown/unset, skipped).
    /// `expertCount` upper bound is enforced softly — extras are ignored but
    /// activity still recorded.
    public func record(layer: Int, experts: [Int]) {
        guard layer >= 0 else { return }
        lock.lock(); defer { lock.unlock() }
        var lc = layerCounts[layer] ?? LayerCounts()
        for e in experts {
            guard e >= 0, e < expertCount else { continue }
            lc.counts[e, default: 0] &+= 1
        }
        layerCounts[layer] = lc
        tokensObserved &+= 1

        // Ring buffer maintenance
        if ring.count < ringWindow {
            ring.append((layer: layer, experts: experts))
        } else {
            ring[ringIndex] = (layer: layer, experts: experts)
            ringIndex = (ringIndex + 1) % ringWindow
        }
    }

    /// Run promotion/demotion pass. Returns (toPromote, toDemote) lists of
    /// expert IDs that should change tiers. Caller (WeightTierManager) executes
    /// the actual load/evict.
    public func promoteDemotePass() -> (toPromote: [ExpertID], toDemote: [ExpertID]) {
        lock.lock(); defer { lock.unlock() }

        var promote: [ExpertID] = []
        var demote: [ExpertID] = []

        // Apply seed decay if still in cold-start window
        let seedWeight = max(0.0, 1.0 - Double(tokensObserved) / Double(seedWindow))

        for (layer, lc) in layerCounts {
            // Seed-suggested experts get a virtual count boost during cold-start
            var effective: [Int: Double] = [:]
            for (e, c) in lc.counts { effective[e] = Double(c) }
            if seedWeight > 0, let seedLayer = seed[layer] {
                for (e, w) in seedLayer {
                    effective[e, default: 0] += w * seedWeight * Double(promoteThreshold)
                }
            }
            for (e, score) in effective {
                if score >= Double(promoteThreshold) {
                    promote.append(ExpertID(layer: layer, expert: e))
                } else if score <= Double(demoteThreshold) {
                    demote.append(ExpertID(layer: layer, expert: e))
                }
            }
        }
        return (promote, demote)
    }

    /// Reset all counts. Called on model unload.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        layerCounts.removeAll(keepingCapacity: true)
        ring.removeAll(keepingCapacity: true)
        ringIndex = 0
        tokensObserved = 0
    }

    /// Snapshot for MetricsStore.expertActivationEntropy (Phase 3+).
    public func snapshotEntropy() -> [Int: Double] {
        lock.lock(); defer { lock.unlock() }
        var out: [Int: Double] = [:]
        for (layer, lc) in layerCounts {
            let total = lc.counts.values.reduce(0, +)
            guard total > 0 else { continue }
            let p = lc.counts.values.map { Double($0) / Double(total) }
            let entropy = -p.reduce(0.0) { $0 + ($1 > 0 ? $1 * log2($1) : 0) }
            out[layer] = entropy
        }
        return out
    }

    public var layersTracked: Int { layerCount }
    public var expertsPerLayer: Int { expertCount }
    public var totalObservations: UInt64 {
        lock.withLock { tokensObserved }
    }

    /// Hottest experts in `layer`, highest count first. Used to prefetch the
    /// next layer's likely set without an extra GPU sync.
    public func topExperts(layer: Int, k: Int) -> [Int] {
        guard k > 0 else { return [] }
        lock.lock(); defer { lock.unlock() }
        guard let lc = layerCounts[layer], !lc.counts.isEmpty else { return [] }
        return lc.counts.sorted { lhs, rhs in
            if lhs.value != rhs.value { return lhs.value > rhs.value }
            return lhs.key < rhs.key
        }.prefix(k).map(\.key)
    }
}
