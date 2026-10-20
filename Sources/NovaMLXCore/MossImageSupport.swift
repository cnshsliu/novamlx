import Foundation

/// Catalog and load policy for OpenMOSS MOSS-Image text-to-image.
///
/// The download is the bf16 diffusers snapshot. Nova quantizes the DiT and the
/// Qwen3.5 text encoder to int8 on load when unified memory is under 48 GiB,
/// or when the caller asks for int8. Sampling defaults match the upstream
/// Euler schedule: 50 steps, CFG 3, shift 5, empty negative, skip layer 22.
public enum MossImageQuantTier: String, Sendable, Equatable {
    case bf16
    case int8
}

public enum MossImageSupport: Sendable {
    public static let sftId = "OpenMOSS-Team/MOSS-Image-SFT"
    public static let baseId = "OpenMOSS-Team/MOSS-Image-Base"
    public static let defaultSteps = 50
    public static let defaultGuidance: Float = 3
    public static let defaultShift: Float = 5
    public static let negativeSkipLayer = 22
    public static let minSide = 256
    /// Upstream large example. Sides stay multiples of 16.
    public static let maxOutputPixels = 1456 * 1456
    /// At or above this unified-memory size, keep the bf16 snapshot as bf16.
    public static let bf16MemoryFloorBytes: UInt64 = 48 * 1024 * 1024 * 1024
    public static let quantEnvironmentKey = "NOVA_MOSS_IMAGE_QUANT"

    public static func matches(id: String) -> Bool {
        let lower = id.lowercased()
        return lower.contains("moss-image")
            || lower.contains("moss_image")
            || lower.contains("mossimage")
    }

    /// `environment` is the process environment. Tests pass a dictionary.
    /// `NOVA_MOSS_IMAGE_QUANT=bf16` or `int8` wins. Otherwise an id that
    /// names an int8 quant uses int8. Otherwise machines under 48 GiB use int8
    /// so the bf16 download still fits. Unknown memory (`0`) stays bf16.
    public static func quantTier(
        id: String,
        physicalBytes: UInt64,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MossImageQuantTier {
        if let raw = environment[quantEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
           !raw.isEmpty
        {
            switch raw {
            case "int8", "8", "8bit", "8-bit": return .int8
            case "bf16", "16", "bfloat16": return .bf16
            default: break
            }
        }
        let lower = id.lowercased()
        if lower.contains("int8") || lower.contains("8bit") || lower.contains("8-bit") {
            return .int8
        }
        if physicalBytes > 0, physicalBytes < bf16MemoryFloorBytes {
            return .int8
        }
        return .bf16
    }

    /// Floor each side to a multiple of 16. `nil` when the result is below
    /// 256 px on a side or the area exceeds 1456×1456.
    public static func outputSize(width: Int, height: Int) -> (width: Int, height: Int)? {
        guard width > 0, height > 0 else { return nil }
        let resolvedWidth = width / 16 * 16
        let resolvedHeight = height / 16 * 16
        guard resolvedWidth >= minSide, resolvedHeight >= minSide,
              resolvedWidth * resolvedHeight <= maxOutputPixels
        else { return nil }
        return (resolvedWidth, resolvedHeight)
    }

    public static func sizeRefusal(width: Int, height: Int) -> String {
        "MOSS-Image needs each side at least \(minSide) px, a multiple of 16, and an area of at most 1456×1456. Got \(width)×\(height)."
    }
}
