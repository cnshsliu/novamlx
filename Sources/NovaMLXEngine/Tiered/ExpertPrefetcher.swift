import Foundation
import MLX
import NovaMLXUtils

// NovaMLX-TIE: ExpertPrefetcher
//
// Async actor that consumes router logits from layer N and issues prefetch
// hints for layer N+1's top-k experts, running concurrently with layer N's
// GEMM. Mis-speculation cost is one wasted prefetch call — cheap.
//
// Failure modes (per plan):
//   - Kernel panic risk (llama.cpp #19825): rate-limit outstanding prefetches
//     via byte-counting gate capped at config.maxSsdMbps.
//   - Watchdog: if fault latency > 250ms, switch to synchronous mode.
//
// Phase 2: real prefetch via WeightTierManager.madvise(). Rate-limiting is
// advisory (per-second budget); not a hard semaphore yet.

public actor ExpertPrefetcher {
    /// Configuration. maxSsdMbps protects against kernel panics from
    /// uncontrolled SSD pressure (llama.cpp issue #19825).
    public struct Config: Sendable {
        public var maxSsdMbps: Int = 1024          // cap on outstanding prefetch bytes
        public var prefetchDepth: Int = 1           // layers ahead to prefetch (1 = N -> N+1)
        public var topK: Int = 8                    // experts to prefetch per layer
        public var faultLatencyWatchdogMs: Int = 250

        public init() {}
    }

    public let config: Config
    private let manifest: TierManifest
    private let heatMap: ExpertHeatMap
    private weak var weightManager: WeightTierManager?
    private weak var metrics: MetricsStore?

    /// Outstanding prefetch bytes in the current 1-second window.
    private var outstandingBytesThisSecond: Int64 = 0
    private var windowStart: Date = Date()

    /// Total prefetches issued + average fault latency (for metrics).
    private var totalPrefetches: UInt64 = 0
    private var totalPrefetchBytes: Int64 = 0
    private var recentFaultLatenciesMs: [Int] = []  // ring of last 64 samples
    private let recentFaultCap = 64

    public init(manifest: TierManifest,
                heatMap: ExpertHeatMap,
                weightManager: WeightTierManager? = nil,
                config: Config = Config(),
                metrics: MetricsStore? = nil) {
        self.manifest = manifest
        self.heatMap = heatMap
        self.weightManager = weightManager
        self.config = config
        self.metrics = metrics
    }

    public func setWeightManager(_ wm: WeightTierManager) {
        self.weightManager = wm
    }

    /// Called from the layer loop after layer N's router runs.
    /// `nextLayerRouterLogits` is the gate output from layer N. We pick top-k
    /// experts by score and issue prefetch hints for layer N+1's experts.
    public func notify(layerN: Int, nextLayerRouterLogits: MLXArray?) async {
        guard let logits = nextLayerRouterLogits else { return }
        let topK = pickTopK(logits: logits, k: config.topK)
        guard !topK.isEmpty, let weightManager else { return }

        // Reset 1-second budget window if needed
        let now = Date()
        if now.timeIntervalSince(windowStart) >= 1.0 {
            windowStart = now
            outstandingBytesThisSecond = 0
        }

        // Estimate bytes
        var bytesToPrefetch: Int64 = 0
        for e in topK {
            if let entry = manifest.experts.first(where: { $0.layer == layerN + 1 && $0.expert == e }) {
                bytesToPrefetch &+= entry.bytes
            }
        }

        // Rate-limit: skip if we'd exceed the per-second budget
        let budgetBytes = Int64(config.maxSsdMbps) * 1_048_576
        if outstandingBytesThisSecond + bytesToPrefetch > budgetBytes {
            metrics?.updateTierMetrics { m in
                m.prefetchMisses &+= UInt64(topK.count)
            }
            return
        }
        outstandingBytesThisSecond &+= bytesToPrefetch

        // Warm next-layer experts into the MLX cache (not a 4KB hint).
        weightManager.prefetchExperts(layer: layerN + 1, expertIDs: topK)

        totalPrefetches &+= UInt64(topK.count)
        totalPrefetchBytes &+= bytesToPrefetch
        metrics?.updateTierMetrics { m in
            m.prefetchHits &+= UInt64(topK.count)
        }
    }

    /// Called by WeightTierManager when an actual fault-in completes.
    /// Updates p99 latency watchdog.
    public func recordFaultCompletion(latencyMs: Int) async {
        if recentFaultLatenciesMs.count >= recentFaultCap {
            recentFaultLatenciesMs.removeFirst()
        }
        recentFaultLatenciesMs.append(latencyMs)
        let sorted = recentFaultLatenciesMs.sorted()
        let p99 = sorted.isEmpty ? 0 : sorted[(sorted.count * 99) / 100]
        metrics?.setFaultLatencyP99(ms: Double(p99))
        if latencyMs > config.faultLatencyWatchdogMs {
            // Phase 3 will switch to synchronous prefetch mode under sustained watchdog trips
            NovaMLXLog.warning("[TIE] fault latency \(latencyMs)ms > watchdog \(config.faultLatencyWatchdogMs)ms — throttling soon")
        }
    }

    /// Snapshot for metrics.
    public func snapshot() -> (prefetches: UInt64, bytes: Int64, p99Ms: Int) {
        let sorted = recentFaultLatenciesMs.sorted()
        let p99 = sorted.isEmpty ? 0 : sorted[(sorted.count * 99) / 100]
        return (totalPrefetches, totalPrefetchBytes, p99)
    }

    /// Pick top-k expert indices from router logits.
    /// `logits` shape: [batch * seq, numExperts] or [numExperts].
    private nonisolated func pickTopK(logits: MLXArray, k: Int) -> [Int] {
        let flat = logits.flattened()
        let count = flat.size
        guard count > 0 else { return [] }

        // Materialize small arrays only
        let vals = flat.asArray(Float.self)
        guard !vals.isEmpty else { return [] }

        let actualK = min(k, vals.count)
        let indices = (0..<vals.count).sorted { vals[$0] > vals[$1] }.prefix(actualK)
        return Array(indices)
    }
}
