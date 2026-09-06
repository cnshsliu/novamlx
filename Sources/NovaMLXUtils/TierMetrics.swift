import Foundation

// NovaMLX-TIE: live (non-persistent) tier inference metrics.
//
// These counters reset every worker session. They track SSD streaming behavior
// so the status panel can show "TIE is saving you X" in real time. Persistent
// lifetime counters live in PersistentMetrics; TierMetrics is intentionally
// ephemeral to avoid DB write churn on every token.

public struct TierMetrics: Sendable, Codable {
    public var tier0Bytes: Int64 = 0
    public var tier1Bytes: Int64 = 0
    public var tier2Bytes: Int64 = 0

    public var ssdBytesReadTotal: Int64 = 0
    public var ssdReadOpsTotal: UInt64 = 0

    public var prefetchHits: UInt64 = 0
    public var prefetchMisses: UInt64 = 0
    public var prefetchMissStallMsTotal: UInt64 = 0

    public var tier1Evictions: UInt64 = 0
    public var tier1Promotions: UInt64 = 0

    /// Fraction of Tier 2 mmap pages currently resident according to mincore().
    /// 0.0..1.0. Zero when Tier 2 inactive.
    public var mincoreResidentRatio: Double = 0

    public var mlockFailures: UInt64 = 0
    public var faultLatencyP99Ms: Double = 0

    public var prefetchHitRate: Double {
        let total = prefetchHits + prefetchMisses
        return total > 0 ? Double(prefetchHits) / Double(total) : 0
    }

    public var ssdReadMbps: Double {
        // Best-effort snapshot; ssdBytesReadTotal is a monotonic counter so
        // callers must diff over time to get a rate.
        return 0
    }

    public init() {}
}

public extension MetricsStore {
    /// Snapshot of live TIE counters. Resets each session.
    var tierMetrics: TierMetrics {
        tierLock.withLock { _tierMetrics }
    }

    func updateTierMetrics(_ mutate: (inout TierMetrics) -> Void) {
        tierLock.withLock { mutate(&_tierMetrics) }
    }

    func recordSsdRead(bytes: Int64) {
        tierLock.withLock {
            _tierMetrics.ssdBytesReadTotal &+= bytes
            _tierMetrics.ssdReadOpsTotal &+= 1
        }
    }

    func recordPrefetchHit() {
        tierLock.withLock { _tierMetrics.prefetchHits &+= 1 }
    }

    func recordPrefetchMiss(stallMs: UInt64) {
        tierLock.withLock {
            _tierMetrics.prefetchMisses &+= 1
            _tierMetrics.prefetchMissStallMsTotal &+= stallMs
        }
    }

    func recordTier1Eviction() {
        tierLock.withLock { _tierMetrics.tier1Evictions &+= 1 }
    }

    func recordTier1Promotion() {
        tierLock.withLock { _tierMetrics.tier1Promotions &+= 1 }
    }

    func recordMlockFailure() {
        tierLock.withLock { _tierMetrics.mlockFailures &+= 1 }
    }

    func setTierSizes(tier0: Int64, tier1: Int64, tier2: Int64) {
        tierLock.withLock {
            _tierMetrics.tier0Bytes = tier0
            _tierMetrics.tier1Bytes = tier1
            _tierMetrics.tier2Bytes = tier2
        }
    }

    func setMincoreResidentRatio(_ ratio: Double) {
        tierLock.withLock { _tierMetrics.mincoreResidentRatio = max(0, min(1, ratio)) }
    }

    func setFaultLatencyP99(ms: Double) {
        tierLock.withLock { _tierMetrics.faultLatencyP99Ms = ms }
    }

    func resetTierMetrics() {
        tierLock.withLock { _tierMetrics = TierMetrics() }
    }
}
