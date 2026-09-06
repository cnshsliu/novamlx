import Foundation
import NovaMLXCore
import NovaMLXDB

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

/// The kind of inference being performed. Drives which unit/label the UI shows
/// alongside the live speed number.
public enum InferenceKind: String, Sendable {
    case llm          // text generation (chat / completions)
    case vlm          // vision-language (image understanding)
    case asr          // speech-to-text / transcription
    case tts          // text-to-speech / speech synthesis
    case image        // image generation / edit / variation

    /// Short human label for UI badges.
    public var label: String {
        switch self {
        case .llm: return "LLM"
        case .vlm: return "VLM"
        case .asr: return "ASR"
        case .tts: return "TTS"
        case .image: return "Image"
        }
    }
}

/// A single live inference-in-progress record. Stored in MetricsStore so the
/// status panel can show "what is running right now" regardless of model type.
public struct LiveActivity: Sendable, Equatable {
    public let model: String
    public let kind: InferenceKind
    /// Speed in kind-appropriate units (tok/s, sec/s, img/s).
    public let speed: Double
    /// Human unit string, e.g. "tok/s", "×RT", "img/s".
    public let unit: String
    public let startedAt: Date
    public let updatedAt: Date

    public init(model: String, kind: InferenceKind, speed: Double, unit: String,
                startedAt: Date = Date(), updatedAt: Date = Date()) {
        self.model = model
        self.kind = kind
        self.speed = speed
        self.unit = unit
        self.startedAt = startedAt
        self.updatedAt = updatedAt
    }
}

public final class MetricsStore: @unchecked Sendable {
    private let metricsFile: URL
    private var _metrics: PersistentMetrics
    private let lock = NovaMLXLock()
    private var saveCounter: Int = 0
    private var _recentTps: Double = 0
    private var _lastTpsUpdate: Date = Date.distantPast
    private var _liveActivity: LiveActivity? = nil
    private var _lastActivityUpdate: Date = Date.distantPast

    // NovaMLX-TIE: live tier-inference metrics. Separate lock so high-frequency
    // SSD/prefetch updates don't contend with the main metrics lock.
    internal var _tierMetrics = TierMetrics()
    internal let tierLock = NovaMLXLock()

    /// Seconds after which a cached TPS value is considered stale and returned as 0
    private let tpsStaleThreshold: TimeInterval = 5

    public init(baseDirectory: URL) {
        // Kept for one-shot legacy import; no longer used for ongoing writes.
        self.metricsFile = baseDirectory.appendingPathComponent("metrics.json")
        self._metrics = PersistentMetrics()
        try? FileManager.default.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        importLegacyJSONIfNeeded()
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

    /// Compute the real-time-factor for streaming-style media: generated-output
    /// seconds per wall-clock second. ASR and TTS both use this ("×RT").
    public static func realTimeFactor(outputSeconds: Double, wallSeconds: Double) -> Double {
        wallSeconds > 0 ? outputSeconds / wallSeconds : 0
    }

    /// Record (or refresh) a live, in-progress inference operation. Called by every
    /// backend (LLM, VLM, ASR, TTS, image-gen) so the status panel always reflects
    /// what is running right now, not just the LLM tok/s path.
    public func reportActivity(model: String, kind: InferenceKind, speed: Double, unit: String) {
        lock.withLock {
            if let existing = _liveActivity, existing.model == model {
                _liveActivity = LiveActivity(
                    model: model, kind: kind, speed: speed, unit: unit,
                    startedAt: existing.startedAt, updatedAt: Date())
            } else {
                let now = Date()
                _liveActivity = LiveActivity(model: model, kind: kind, speed: speed, unit: unit, startedAt: now, updatedAt: now)
            }
            _lastActivityUpdate = Date()
        }
    }

    /// Clear a single finished operation. Only clears if the finishing model
    /// matches the current activity, so a concurrent request isn't wiped early.
    public func clearActivity(forModel model: String) {
        lock.withLock {
            if _liveActivity?.model == model {
                _liveActivity = nil
            }
        }
    }

    /// Live activity if one is in progress and not stale (5s). Returns nil when idle.
    public var liveActivity: LiveActivity? {
        lock.withLock {
            guard let activity = _liveActivity else { return nil }
            let stale = Date().timeIntervalSince(_lastActivityUpdate) > tpsStaleThreshold
            return stale ? nil : activity
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

    // MARK: - SQLite-backed load/save (Phase E)

    private func load() {
        guard let record = try? NovaDB.shared.metricsStore.get() else { return }
        lock.withLock { _metrics = Self.recordToMetrics(record) }
        NovaMLXLog.info("Loaded persistent metrics: \(_metrics.totalRequestsAllTime) requests, \(_metrics.totalTokensAllTime) tokens all-time")
    }

    private func save() {
        let snapshot = lock.withLock { _metrics }
        let record = Self.metricsToRecord(snapshot)
        do {
            try NovaDB.shared.metricsStore.replaceAll(record)
        } catch {
            NovaMLXLog.warning("[MetricsStore] Failed to persist: \(error.localizedDescription)")
        }
    }

    /// One-shot import of legacy `~/.nova/metrics.json`. Idempotent: if the
    /// DB store already has non-zero counters, the file is left alone;
    /// otherwise we decode and upsert, then rename the file to .migrated.
    private func importLegacyJSONIfNeeded() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: metricsFile.path) else { return }

        // Skip if store already populated — SQLite is source of truth.
        if let existing = try? NovaDB.shared.metricsStore.get(),
           existing.totalRequests > 0 || existing.totalTokens > 0 || existing.cacheHits > 0 {
            return
        }

        guard let data = try? Data(contentsOf: metricsFile),
              let decoded = try? JSONDecoder().decode(PersistentMetrics.self, from: data) else {
            NovaMLXLog.warning("[MetricsStore] Failed to parse legacy metrics.json; leaving file in place")
            return
        }

        let record = Self.metricsToRecord(decoded)
        do {
            try NovaDB.shared.metricsStore.replaceAll(record)
            NovaMLXLog.info("[MetricsStore] Imported legacy metrics: \(decoded.totalRequestsAllTime) requests, \(decoded.totalTokensAllTime) tokens")
        } catch {
            NovaMLXLog.error("[MetricsStore] Failed to import legacy metrics: \(error.localizedDescription)")
            return
        }

        let migrated = metricsFile.appendingPathExtension("migrated")
        if fm.fileExists(atPath: migrated.path) {
            try? fm.removeItem(at: metricsFile)
        } else {
            try? fm.moveItem(at: metricsFile, to: migrated)
        }
    }

    // MARK: - Mapping

    private static func metricsToRecord(_ m: PersistentMetrics) -> MetricsRecord {
        // perModelStats is reused to round-trip the full per-model dicts
        // (requests + tokens + inference time) the legacy JSON carried.
        // Schema only has per_model_stats as JSON, so we pack both
        // totalRequestsByModel and totalTokensByModel into a single dict.
        var stats: [String: [String: Int64]] = [:]
        for (model, reqs) in m.totalRequestsByModel {
            var entry = stats[model] ?? [:]
            entry["requests"] = Int64(reqs)
            stats[model] = entry
        }
        for (model, toks) in m.totalTokensByModel {
            var entry = stats[model] ?? [:]
            entry["tokens"] = Int64(toks)
            stats[model] = entry
        }
        let statsJSON = (try? String(data: JSONEncoder().encode(stats), encoding: .utf8)) ?? "{}"

        return MetricsRecord(
            id: 1,
            totalRequests: Int64(m.totalRequestsAllTime),
            totalTokens: Int64(m.totalTokensAllTime),
            totalInferenceTimeMs: Int64(m.totalInferenceTimeAllTime * 1000),
            cacheHits: Int64(m.cacheHits),
            cacheMisses: Int64(m.cacheMisses),
            evictions: Int64(m.evictions),
            perModelStats: statsJSON,
            perModelCache: "{}",
            updatedAt: m.lastUpdated,
            modelsLoaded: Int64(m.modelsLoaded),
            modelsUnloaded: Int64(m.modelsUnloaded),
            ttlEvictions: Int64(m.ttlEvictions),
            memoryPressureEvictions: Int64(m.memoryPressureEvictions)
        )
    }

    private static func recordToMetrics(_ r: MetricsRecord) -> PersistentMetrics {
        var requests: [String: UInt64] = [:]
        var tokens: [String: UInt64] = [:]
        if let json = r.perModelStats,
           let data = json.data(using: .utf8),
           let stats = try? JSONDecoder().decode([String: [String: Int64]].self, from: data) {
            for (model, entry) in stats {
                if let v = entry["requests"] { requests[model] = UInt64(v) }
                if let v = entry["tokens"] { tokens[model] = UInt64(v) }
            }
        }
        return PersistentMetrics(
            totalRequestsAllTime: UInt64(max(0, r.totalRequests)),
            totalTokensAllTime: UInt64(max(0, r.totalTokens)),
            totalInferenceTimeAllTime: Double(max(0, r.totalInferenceTimeMs)) / 1000.0,
            totalRequestsByModel: requests,
            totalTokensByModel: tokens,
            cacheHits: UInt64(max(0, r.cacheHits)),
            cacheMisses: UInt64(max(0, r.cacheMisses)),
            modelsLoaded: UInt64(max(0, r.modelsLoaded)),
            modelsUnloaded: UInt64(max(0, r.modelsUnloaded)),
            evictions: UInt64(max(0, r.evictions)),
            ttlEvictions: UInt64(max(0, r.ttlEvictions)),
            memoryPressureEvictions: UInt64(max(0, r.memoryPressureEvictions))
        )
    }
}
