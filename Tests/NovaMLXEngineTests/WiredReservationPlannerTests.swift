import Testing
import Foundation
@testable import NovaMLXEngine

@Suite("WiredReservationPlanner")
struct WiredReservationPlannerTests {
    @Test("Uses positive activeMemory delta as this model's size")
    func thisModelBytesFromDelta() {
        let bytes = WiredReservationPlanner.thisModelWeightBytes(
            activeBefore: 34_000_000_000,
            activeAfter: 53_000_000_000,
            estimated: 18_000_000_000
        )
        #expect(bytes == 19_000_000_000)
    }

    @Test("Falls back to estimate when delta is not positive")
    func thisModelBytesFallsBackToEstimate() {
        let bytes = WiredReservationPlanner.thisModelWeightBytes(
            activeBefore: 34_000_000_000,
            activeAfter: 34_000_000_000,
            estimated: 18_900_000_000
        )
        #expect(bytes == 18_900_000_000)
    }

    @Test("Reserves when this model fits in free RAM and Metal cap")
    func reservesWhenFits() {
        let decision = WiredReservationPlanner.decide(
            thisModelBytes: 19 * 1_024 * 1_024 * 1_024,
            alreadyReservedBytes: 0,
            availablePhysicalBytes: 80 * 1_024 * 1_024 * 1_024,
            recommendedWorkingSetBytes: 100 * 1_024 * 1_024 * 1_024
        )
        let expected = 19 * 1_024 * 1_024 * 1_024
        #expect(decision == .reserve(expected))
    }

    @Test("Skips when free RAM cannot cover this model plus headroom")
    func skipsWhenPhysicalMemoryTight() {
        let thisModel = 19 * 1_024 * 1_024 * 1_024
        let decision = WiredReservationPlanner.decide(
            thisModelBytes: thisModel,
            alreadyReservedBytes: 0,
            availablePhysicalBytes: UInt64(thisModel),
            recommendedWorkingSetBytes: 100 * 1_024 * 1_024 * 1_024
        )
        guard case .skip(let reason) = decision else {
            Issue.record("expected skip, got \(decision)")
            return
        }
        #expect(reason.contains("free"))
    }

    @Test("Skips when existing reservations plus this model exceed Metal cap")
    func skipsWhenAdmissionWouldDeadlock() {
        // Reproduces the Qwen3.8-27B-8bit (34GB) + MTP (34GB) + OptiQ load hang:
        // WiredSumPolicy.canAdmit waits forever once projected > maxRecommendedWorkingSet.
        let gb = 1_024 * 1_024 * 1_024
        let decision = WiredReservationPlanner.decide(
            thisModelBytes: 19 * gb,
            alreadyReservedBytes: 68 * gb,
            availablePhysicalBytes: UInt64(80 * gb),
            recommendedWorkingSetBytes: 80 * gb
        )
        guard case .skip(let reason) = decision else {
            Issue.record("expected skip, got \(decision)")
            return
        }
        #expect(reason.contains("working set") || reason.contains("cap"))
    }

    @Test("Does not size the ticket to total MLX active memory of every loaded model")
    func doesNotUseTotalActiveMemory() {
        let totalActive = 53 * 1_024 * 1_024 * 1_024
        let thisModel = WiredReservationPlanner.thisModelWeightBytes(
            activeBefore: 34 * 1_024 * 1_024 * 1_024,
            activeAfter: totalActive,
            estimated: Optional<UInt64>.none
        )
        let decision = WiredReservationPlanner.decide(
            thisModelBytes: thisModel,
            alreadyReservedBytes: 34 * 1_024 * 1_024 * 1_024,
            availablePhysicalBytes: 90 * 1_024 * 1_024 * 1_024,
            recommendedWorkingSetBytes: 96 * 1_024 * 1_024 * 1_024
        )
        let expected = 19 * 1_024 * 1_024 * 1_024
        #expect(decision == .reserve(expected))
    }
}
