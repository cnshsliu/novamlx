import Foundation
import NovaMLXCore

public struct DeviceInfo: Codable, Sendable {
    public let chipName: String
    public let chipVariant: String
    public let memoryGB: UInt64
    public let gpuCores: Int
    public let cpuCores: Int
    public let osVersion: String
    public let novaMLXVersion: String

    public init(
        chipName: String = "",
        chipVariant: String = "",
        memoryGB: UInt64 = 0,
        gpuCores: Int = 0,
        cpuCores: Int = 0,
        osVersion: String = "",
        novaMLXVersion: String = ""
    ) {
        self.chipName = chipName
        self.chipVariant = chipVariant
        self.memoryGB = memoryGB
        self.gpuCores = gpuCores
        self.cpuCores = cpuCores
        self.osVersion = osVersion
        self.novaMLXVersion = novaMLXVersion
    }

    public static func current() -> DeviceInfo {
        let processInfo = ProcessInfo.processInfo
        let totalMemory = UInt64(processInfo.physicalMemory) / (1024 * 1024 * 1024)
        let cpuCores = processInfo.processorCount
        let osVersion = processInfo.operatingSystemVersionString

        let chipName = readSysctl("machdep.cpu.brand_string")
        let gpuCoresStr = readSysctl("hw.gpucores")
        let gpuCores = Int(gpuCoresStr) ?? 0

        let chipVariant = detectChipVariant(chipName: chipName, gpuCores: gpuCores, memoryGB: totalMemory)

        return DeviceInfo(
            chipName: chipName,
            chipVariant: chipVariant,
            memoryGB: totalMemory,
            gpuCores: gpuCores,
            cpuCores: cpuCores,
            osVersion: osVersion,
            novaMLXVersion: NovaMLXCore.version
        )
    }

    private static func readSysctl(_ name: String) -> String {
        var size = 0
        sysctlbyname(name, nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname(name, &buffer, &size, nil, 0)
        return buffer.withUnsafeBufferPointer { ptr in
            String(cString: ptr.baseAddress!)
        }
    }

    private static func detectChipVariant(chipName: String, gpuCores: Int, memoryGB: UInt64) -> String {
        if chipName.contains("M4 Max") {
            if gpuCores >= 40 { return "16-core" }
            return "14-core"
        } else if chipName.contains("M4 Pro") {
            if gpuCores >= 20 { return "10-core" }
            return "8-core"
        } else if chipName.contains("M4") {
            return "10-core"
        } else if chipName.contains("M3 Max") {
            if gpuCores >= 30 { return "14-core" }
            return "12-core"
        } else if chipName.contains("M3 Pro") {
            if gpuCores >= 18 { return "6-core" }
            return "5-core"
        } else if chipName.contains("M3") {
            return "8-core"
        } else if chipName.contains("M2 Ultra") {
            return "24-core"
        } else if chipName.contains("M2 Max") {
            if gpuCores >= 30 { return "12-core" }
            return "8-core"
        } else if chipName.contains("M2 Pro") {
            if gpuCores >= 16 { return "6-core" }
            return "5-core"
        } else if chipName.contains("M2") {
            return "8-core"
        } else if chipName.contains("M1 Ultra") {
            return "16-core"
        } else if chipName.contains("M1 Max") {
            return "8-core"
        } else if chipName.contains("M1 Pro") {
            return "5-core"
        } else if chipName.contains("M1") {
            return "4-core"
        }
        return "\(gpuCores)-core"
    }
}
