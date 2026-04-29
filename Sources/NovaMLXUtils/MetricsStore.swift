import Foundation
import NovaMLXCore

public struct PersistentMetrics: Codable, Sendable {
    public var totalRequestsAllTime: UInt64
    public var totalTokensAllTime: UInt64
    public var totalInferenceTimeAllTime: Double
    public var totalRequestsByModel: [String: UInt64]
    public var totalTokensByModel: [String: UInt64]
    public var cacheHits: UInt64
    public var cacheMisses: UInt64
    public var modelsLoaded: UInt64
    public var modelsUnloaded: UInt64
    public var evictions: UInt64
    public var ttlEvictions: UInt64
    public var memoryPressureEvictions: UInt64
    public var lastUpdated: Date

    public init(
        totalRequestsAllTime: UInt64 = 0,
        totalTokensAllTime: UInt64 = 0,
        totalInferenceTimeAllTime: Double = 0,
        totalRequestsByModel: [String: UInt64] = [:],
        totalTokensByModel: [String: UInt64] = [:],
        cacheHits: UInt64 = 0,
        cacheMisses: UInt64 = 0,
        modelsLoaded: UInt64 = 0,
        modelsUnloaded: UInt64 = 0,
        evictions: UInt64 = 0,
        ttlEvictions: UInt64 = 0,
        memoryPressureEvictions: UInt64 = 0
    ) {
        self.totalRequestsAllTime = totalRequestsAllTime
        self.totalTokensAllTime = totalTokensAllTime
        self.totalInferenceTimeAllTime = totalInferenceTimeAllTime
        self.totalRequestsByModel = totalRequestsByModel
        self.totalTokensByModel = totalTokensByModel
        self.cacheHits = cacheHits
        self.cacheMisses = cacheMisses
        self.modelsLoaded = modelsLoaded
        self.modelsUnloaded = modelsUnloaded
        self.evictions = evictions
        self.ttlEvictions = ttlEvictions
        self.memoryPressureEvictions = memoryPressureEvictions
        self.lastUpdated = Date()
    }

    public var averageTokensPerSecond: Double {
        totalInferenceTimeAllTime > 0 ? Double(totalTokensAllTime) / totalInferenceTimeAllTime : 0
    }

    public var cacheHitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0
    }
}

public final class MetricsStore: @unchecked Sendable {
    private let metricsFile: URL
    private var _metrics: PersistentMetrics
    private let lock = NovaMLXLock()
    private var saveCounter: Int = 0
    private var _recentTps: Double = 0
    private var _lastTpsUpdate: Date = Date.distantPast

    /// Seconds after which a cached TPS value is considered stale and returned as 0
    private let tpsStaleThreshold: TimeInterval = 5

    public init(baseDirectory: URL) {
        self.metricsFile = baseDirectory.appendingPathComponent("metrics.json")
        self._metrics = PersistentMetrics()
        try? FileManager.default.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        load()
    }

    public var metrics: PersistentMetrics {
        lock.withLock { _metrics }
    }

    public var recentTokensPerSecond: Double {
        lock.withLock {
            let stale = Date().timeIntervalSince(_lastTpsUpdate) > tpsStaleThreshold
            if stale { return 0 }
            return _recentTps
        }
    }

    public func recordRequest(model: String, tokens: UInt64, inferenceTime: Double) {
        lock.withLock {
            if inferenceTime > 0 {
                _recentTps = Double(tokens) / inferenceTime
                _lastTpsUpdate = Date()
            }
            _metrics.totalRequestsAllTime += 1
            _metrics.totalTokensAllTime += tokens
            _metrics.totalInferenceTimeAllTime += inferenceTime
            _metrics.totalRequestsByModel[model, default: 0] += 1
            _metrics.totalTokensByModel[model, default: 0] += tokens
            _metrics.lastUpdated = Date()
        }
        maybeSave()
    }

    /// Update live TPS without affecting cumulative counters — called during active generation
    /// to keep the TPS value fresh for the status chart.
    public func updateLiveTps(_ tps: Double) {
        lock.withLock {
            _recentTps = tps
            _lastTpsUpdate = Date()
        }
    }

    public func recordCacheHit() {
        lock.withLock { _metrics.cacheHits += 1 }
    }

    public func recordCacheMiss() {
        lock.withLock { _metrics.cacheMisses += 1 }
    }

    public func recordModelLoad() {
        lock.withLock {
            _metrics.modelsLoaded += 1
            _metrics.lastUpdated = Date()
        }
        maybeSave()
    }

    public func recordModelUnload() {
        lock.withLock {
            _metrics.modelsUnloaded += 1
            _metrics.lastUpdated = Date()
        }
        maybeSave()
    }

    public func recordEviction(reason: String = "lru") {
        lock.withLock {
            _metrics.evictions += 1
            if reason == "ttl" { _metrics.ttlEvictions += 1 }
            else if reason == "memory_pressure" { _metrics.memoryPressureEvictions += 1 }
            _metrics.lastUpdated = Date()
        }
        maybeSave()
    }

    private func maybeSave() {
        saveCounter += 1
        if saveCounter % 10 == 0 {
            save()
        }
    }

    public func forceSave() {
        save()
    }

    public func clearAllTime() {
        lock.withLock {
            _metrics = PersistentMetrics()
            _metrics.lastUpdated = Date()
        }
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: metricsFile),
              let decoded = try? JSONDecoder().decode(PersistentMetrics.self, from: data) else { return }
        lock.withLock { _metrics = decoded }
        NovaMLXLog.info("Loaded persistent metrics: \(_metrics.totalRequestsAllTime) requests, \(_metrics.totalTokensAllTime) tokens all-time")
    }

    private func save() {
        let data = lock.withLock {
            (try? JSONEncoder().encode(_metrics)) ?? Data()
        }
        try? data.write(to: metricsFile, options: .atomic)
    }
}
