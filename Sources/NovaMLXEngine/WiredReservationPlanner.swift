import Foundation

/// Decide whether a newly loaded model should take a wired-memory reservation.
///
/// `WiredSumPolicy.canAdmit` suspends forever when the projected reservation
/// exceeds `GPU.maxRecommendedWorkingSetBytes()`. Model load used to wait on
/// that admission with no timeout, so a second 27B load (on top of an already
/// reserved 8-bit + MTP pair) hung the UI indefinitely.
public enum WiredReservationDecision: Equatable, Sendable {
    case skip(String)
    case reserve(Int)
}

public enum WiredReservationPlanner: Sendable {
    public static let headroomBytes: UInt64 = 1_024 * 1_024 * 1_024

    /// Size the ticket to *this* model, not `MLX.Memory.activeMemory` (which
    /// includes every model already resident).
    public static func thisModelWeightBytes(
        activeBefore: Int,
        activeAfter: Int,
        estimated: UInt64?
    ) -> Int {
        let delta = activeAfter - activeBefore
        if delta > 0 { return delta }
        if let estimated, estimated > 0 {
            return Int(clamping: estimated)
        }
        return 0
    }

    public static func decide(
        thisModelBytes: Int,
        alreadyReservedBytes: Int,
        availablePhysicalBytes: UInt64,
        recommendedWorkingSetBytes: Int?
    ) -> WiredReservationDecision {
        guard thisModelBytes > 0 else {
            return .skip("this model added no measurable weight bytes")
        }

        let needed = UInt64(thisModelBytes) + headroomBytes
        if availablePhysicalBytes <= needed {
            let freeMB = availablePhysicalBytes / 1_048_576
            let needMB = UInt64(thisModelBytes) / 1_048_576
            return .skip("only \(freeMB)MB free, need \(needMB)MB + 1GB headroom")
        }

        if let cap = recommendedWorkingSetBytes, cap > 0 {
            let projected = alreadyReservedBytes + thisModelBytes
            if projected > cap {
                let capMB = cap / 1_048_576
                let reservedMB = alreadyReservedBytes / 1_048_576
                let thisMB = thisModelBytes / 1_048_576
                return .skip(
                    "would exceed Metal working set (\(capMB)MB); existing reservations \(reservedMB)MB + this model \(thisMB)MB"
                )
            }
        }

        return .reserve(thisModelBytes)
    }
}
