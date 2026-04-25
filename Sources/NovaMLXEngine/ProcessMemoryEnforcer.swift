import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils

public actor ProcessMemoryEnforcer {

    public struct Status: Sendable {
        public let enabled: Bool
        public let softLimitBytes: UInt64
        public let hardLimitBytes: UInt64
        public let currentBytes: UInt64
        public let utilization: Double
        public let totalEvictions: UInt64
        public let lastEvictedModel: String?
        public let lastEvictionTime: Date?
        public let physicalRAM: UInt64
    }

    private let pool: EnginePool
    private let settingsProvider: (@Sendable (String) -> ModelSettings?)?
    private var enforcerSettingsProvider: (@Sendable (String) -> ModelSettings?)?
    private let softLimitBytes: UInt64
    private let hardLimitBytes: UInt64
    private let enabled: Bool
    private let physicalRAM: UInt64
    private let pollInterval: Duration

    private var timerTask: Task<Void, Never>?
    private var totalEvictions: UInt64 = 0
    private var lastEvictedModel: String?
    private var lastEvictionTime: Date?
    private var abortAllHandler: (() async -> Void)?

    public init(
        pool: EnginePool,
        settingsProvider: (@Sendable (String) -> ModelSettings?)?,
        limit: ProcessMemoryLimit,
        physicalRAM: UInt64 = UInt64(ProcessInfo.processInfo.physicalMemory),
        pollIntervalSeconds: Double = 1.0
    ) {
        self.pool = pool
        self.settingsProvider = settingsProvider
        self.enforcerSettingsProvider = settingsProvider
        self.physicalRAM = physicalRAM
        self.pollInterval = .milliseconds(Int(pollIntervalSeconds * 1000))

        if case .disabled = limit {
            self.enabled = false
            self.softLimitBytes = 0
            self.hardLimitBytes = 0
        } else {
            self.enabled = true
            self.softLimitBytes = limit.resolveBytes(physicalRAM: physicalRAM) ?? 0
            self.hardLimitBytes = limit.resolveHardLimit(physicalRAM: physicalRAM)
        }
    }

    public func start() {
        guard enabled, softLimitBytes > 0 else { return }
        guard timerTask == nil else { return }

        let interval = pollInterval
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard !Task.isCancelled else { return }
                await self?.poll()
            }
        }
        NovaMLXLog.info("[MemoryEnforcer] started (soft: \(softLimitBytes / 1024 / 1024)MB, hard: \(hardLimitBytes / 1024 / 1024)MB)")
    }

    public func stop() {
        timerTask?.cancel()
        timerTask = nil
        NovaMLXLog.info("[MemoryEnforcer] stopped")
    }

    public func setAbortAllHandler(_ handler: @escaping () async -> Void) {
        self.abortAllHandler = handler
    }

    public func updateSettingsProvider(_ provider: @escaping @Sendable (String) -> ModelSettings?) {
        self.enforcerSettingsProvider = provider
    }

    public var status: Status {
        let current = currentMemoryBytes
        let util = softLimitBytes > 0 ? Double(current) / Double(softLimitBytes) : 0
        return Status(
            enabled: enabled,
            softLimitBytes: softLimitBytes,
            hardLimitBytes: hardLimitBytes,
            currentBytes: current,
            utilization: min(util, 1.0),
            totalEvictions: totalEvictions,
            lastEvictedModel: lastEvictedModel,
            lastEvictionTime: lastEvictionTime,
            physicalRAM: physicalRAM
        )
    }

    public var currentMemoryBytes: UInt64 {
        UInt64(MLX.Memory.activeMemory)
    }

    // MARK: - Internal

    private func poll() async {
        let current = currentMemoryBytes
        guard current > softLimitBytes else {
            await checkTTLExpirations()
            return
        }

        NovaMLXLog.warning("[MemoryEnforcer] soft limit exceeded: \(current / 1024 / 1024)MB > \(softLimitBytes / 1024 / 1024)MB")

        await enforceSoftLimit(currentBytes: current)
        await checkTTLExpirations()
    }

    private func enforceSoftLimit(currentBytes: UInt64) async {
        let models = pool.allModelInfo()
        let unpinned = models.filter { !$0.pinned }

        // Single loaded model — abort requests but keep weights
        if models.count == 1 {
            NovaMLXLog.warning("[MemoryEnforcer] single model loaded, aborting active requests to free KV cache")
            await abortAllHandler?()
            MLX.Memory.clearCache()
            return
        }

        // Multiple models — evict LRU unpinned until under limit
        let sorted = unpinned.sorted { $0.lastAccessed < $1.lastAccessed }
        var current = currentBytes

        for victim in sorted {
            guard current > softLimitBytes else { break }

            NovaMLXLog.info("[MemoryEnforcer] evicting LRU model: \(victim.id) (\(victim.sizeMB)MB)")
            let freedMB = pool.evictLRU()
            if let freedMB {
                totalEvictions += 1
                lastEvictedModel = victim.id
                lastEvictionTime = Date()
                current = currentMemoryBytes
                NovaMLXLog.info("[MemoryEnforcer] evicted \(victim.id), freed \(freedMB)MB, now \(current / 1024 / 1024)MB")
            } else {
                break
            }
        }

        // Wait for Metal to actually reclaim buffers
        await waitForMemorySettle(targetBytes: softLimitBytes)
    }

    private func waitForMemorySettle(targetBytes: UInt64) async {
        MLX.Memory.clearCache()
        var current = currentMemoryBytes

        for _ in 0..<5 {
            if current <= targetBytes { return }
            try? await Task.sleep(for: .milliseconds(200))
            current = currentMemoryBytes
        }

        if current > targetBytes {
            NovaMLXLog.warning("[MemoryEnforcer] memory did not settle after eviction: \(current / 1024 / 1024)MB > \(targetBytes / 1024 / 1024)MB")
        }
    }

    private func checkTTLExpirations() async {
        guard let settingsProvider = enforcerSettingsProvider ?? settingsProvider else { return }
        let models = pool.allModelInfo()
        let now = Date()

        for model in models {
            guard !model.pinned,
                  let settings = settingsProvider(model.id),
                  let ttl = settings.ttlSeconds,
                  ttl > 0 else { continue }

            let idleSeconds = now.timeIntervalSince(model.lastAccessed)
            if idleSeconds > Double(ttl) {
                NovaMLXLog.info("[MemoryEnforcer] TTL expired for \(model.id) (idle \(Int(idleSeconds))s > \(ttl)s)")
                pool.evictLRU()
                totalEvictions += 1
                lastEvictedModel = model.id
                lastEvictionTime = now
            }
        }
    }
}
