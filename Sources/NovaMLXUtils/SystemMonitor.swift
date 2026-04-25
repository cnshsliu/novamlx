import Foundation
import NovaMLXCore

public final class SystemMonitor: @unchecked Sendable {
    public static let shared = SystemMonitor()

    private var startTime: Date = Date()

    private init() {}

    public var uptime: TimeInterval {
        Date().timeIntervalSince(startTime)
    }

    public func currentStats(activeRequests: Int = 0, tokensPerSecond: Double = 0, gpuMemoryUsed: UInt64 = 0) -> SystemStats {
        var totalMemory: UInt64 = 0
        var usedMemory: UInt64 = 0

        let info = ProcessInfo.processInfo
        #if os(macOS)
        totalMemory = UInt64(info.physicalMemory)
        usedMemory = usedMemoryBytes()
        #endif

        return SystemStats(
            cpuUsage: cpuUsage(),
            memoryUsed: usedMemory,
            memoryTotal: totalMemory,
            gpuMemoryUsed: gpuMemoryUsed,
            activeRequests: activeRequests,
            tokensPerSecond: tokensPerSecond,
            uptime: uptime
        )
    }

    private func cpuUsage() -> Double {
        var totalUsage: Double = 0
        var threads: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0

        guard task_threads(mach_task_self_, &threads, &threadCount) == KERN_SUCCESS,
              let threads = threads else { return 0 }

        defer { vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threads), vm_size_t(threadCount) * vm_size_t(MemoryLayout<thread_t>.size)) }

        for i in 0..<Int(threadCount) {
            var threadInfo = thread_basic_info()
            var threadInfoCount = mach_msg_type_number_t(THREAD_INFO_MAX)
            let threadInfoPtr = withUnsafeMutablePointer(to: &threadInfo) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(threadInfoCount)) { $0 }
            }
            guard thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), threadInfoPtr, &threadInfoCount) == KERN_SUCCESS else { continue }

            if !(threadInfo.flags & TH_FLAGS_IDLE != 0) {
                totalUsage += Double(threadInfo.cpu_usage) / Double(TH_USAGE_SCALE) * 100.0
            }
        }

        return min(totalUsage, 100.0)
    }

    private func usedMemoryBytes() -> UInt64 {
        var taskInfo = task_basic_info_64()
        var count = mach_msg_type_number_t(MemoryLayout<task_basic_info_64>.size) / 4
        let ptr = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { $0 }
        }
        let result = task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO_64), ptr, &count)
        return result == KERN_SUCCESS ? taskInfo.resident_size : 0
    }
}
