import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXModelManager

public final class MemoryPressureHandler: @unchecked Sendable {
    private let engine: MLXEngine
    private let settingsManager: ModelSettingsManager
    private var source: DispatchSourceMemoryPressure?
    private var timer: DispatchSourceTimer?
    private let lock = NovaMLXLock()

    public init(engine: MLXEngine, settingsManager: ModelSettingsManager) {
        self.engine = engine
        self.settingsManager = settingsManager
    }

    public func start() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: DispatchQueue.global(qos: .utility))
        source.setEventHandler { [weak self] in
            self?.handleMemoryPressure()
        }
        source.resume()
        lock.withLock { self.source = source }

        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + 60, repeating: 60)
        timer.setEventHandler { [weak self] in
            self?.periodicCheck()
        }
        timer.resume()
        lock.withLock { self.timer = timer }

        NovaMLXLog.info("Memory pressure handler started")
    }

    public func stop() {
        lock.withLock {
            source?.cancel()
            source = nil
            timer?.cancel()
            timer = nil
        }
    }

    private func handleMemoryPressure() {
        let rawFlags = lock.withLock { source?.data ?? .normal }
        let pressure: MemoryPressureLevel
        switch rawFlags {
        case .critical: pressure = .critical
        case .warning: pressure = .warning
        default: pressure = .normal
        }

        NovaMLXLog.warning("OS memory pressure: \(pressure.rawValue)")

        switch pressure {
        case .critical:
            evictUnpinnedModels(keepLast: 0)
        case .warning:
            evictUnpinnedModels(keepLast: 1)
        case .normal:
            break
        }

        checkTTLExpirations()
    }

    private func periodicCheck() {
        checkTTLExpirations()

        let activeMB = engine.pool.totalMemoryMB()
        let totalRAM = UInt64(ProcessInfo.processInfo.physicalMemory / 1_048_576)
        let usagePercent = totalRAM > 0 ? Double(activeMB) / Double(totalRAM) * 100 : 0

        if usagePercent > 80 {
            NovaMLXLog.warning("GPU memory usage high: \(activeMB)MB / \(totalRAM)MB (\(String(format: "%.1f", usagePercent))%)")
            evictUnpinnedModels(keepLast: 1)
        }
    }

    private func checkTTLExpirations() {
        let allInfo = engine.pool.allModelInfo()
        let now = Date()
        for info in allInfo {
            if info.pinned { continue }
            let settings = settingsManager.getSettings(info.id)
            guard let ttl = settings.ttlSeconds, ttl > 0 else { continue }
            let idleTime = now.timeIntervalSince(info.lastAccessed)
            if idleTime >= Double(ttl) {
                let identifier = ModelIdentifier(id: info.id, family: .other)
                engine.unloadModel(identifier)
                NovaMLXLog.info("TTL expired for \(info.id), unloaded after \(ttl)s idle")
            }
        }
    }

    private func evictUnpinnedModels(keepLast: Int) {
        let allInfo = engine.pool.allModelInfo()
            .filter { !$0.pinned }
            .sorted { $0.lastAccessed < $1.lastAccessed }

        let toEvict = keepLast > 0 ? max(0, allInfo.count - keepLast) : allInfo.count

        for i in 0..<toEvict {
            let modelId = allInfo[i].id
            let identifier = ModelIdentifier(id: modelId, family: .other)
            engine.unloadModel(identifier)
            NovaMLXLog.info("Memory pressure: evicted \(modelId)")
        }

        if toEvict > 0 {
            MLX.Memory.clearCache()
        }
    }
}

private enum MemoryPressureLevel: String {
    case normal = "normal"
    case warning = "warning"
    case critical = "critical"
}
