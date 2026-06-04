import Foundation
import NovaMLXDistributed

/// Self-contained simulation benchmark for distributed speculative decoding.
/// Runs entirely in one process with no real models — only mocks that simulate
/// realistic coordinator (fast) + worker (slow) latency.
///
/// This lets us measure the concrete benefit of the K>1 speculative loop we just implemented
/// (round-trips saved, effective tokens per round, simulated tok/s) without requiring
/// the two-machine Thunderbolt cluster.
public enum DistributedSpeculativeBench {

    /// Result of one simulation run.
    public struct Result: Sendable {
        public let mode: String
        public let tokensGenerated: Int
        public let workerRoundTrips: Int
        public let effectiveTokPerRound: Double
        public let simulatedTokPerSec: Double
        public let avgAcceptanceRate: Double?
        public let elapsedMs: Double
    }

    /// Run both baseline (no speculation) and speculative (numDraft=4) simulations
    /// with the same "workload" and print a clear before/after comparison.
    public static func runComparison(targetTokens: Int = 80, numDraft: Int = 4) {
        print("═══════════════════════════════════════════════════════════════")
        print("  Distributed Speculative Decoding — Local Simulation Benchmark")
        print("  (2-node pipeline: fast M4 Max coord + slower M4 worker)")
        print("═══════════════════════════════════════════════════════════════\n")

        // Simulate realistic per-token times observed on the real cluster
        // Coordinator (58 layers on 40-core M4 Max): ~70ms
        // Worker (8 layers + head on 10-core M4):      ~36ms
        // One round-trip baseline:                   ~106ms → ~9.4 tok/s raw
        let baselineCoordMs: UInt32 = 70
        let baselineWorkerMs: UInt32 = 36

        let baseline = runSimulation(
            mode: "Baseline (numDraft=0)",
            targetTokens: targetTokens,
            numDraft: 0,
            coordLatencyMs: baselineCoordMs,
            workerLatencyMs: baselineWorkerMs,
            simulatedAcceptanceRate: 0.0   // no speculation
        )

        // With speculation we still pay the full worker cost per round,
        // but we get multiple tokens per round on average.
        let spec = runSimulation(
            mode: "Speculative (numDraft=\(numDraft))",
            targetTokens: targetTokens,
            numDraft: numDraft,
            coordLatencyMs: baselineCoordMs,
            workerLatencyMs: baselineWorkerMs,
            simulatedAcceptanceRate: 0.62   // realistic n-gram acceptance on mixed text
        )

        print("\n──────────────── Comparison ────────────────")
        let roundTripReduction = Double(baseline.workerRoundTrips) / Double(spec.workerRoundTrips)
        let speedup = spec.simulatedTokPerSec / baseline.simulatedTokPerSec

        print(String(format: "Worker round-trips:  %3d  →  %3d   (%.2fx fewer calls)",
                     baseline.workerRoundTrips, spec.workerRoundTrips, roundTripReduction))
        print(String(format: "Effective tok/round:  %.2f  →  %.2f",
                     baseline.effectiveTokPerRound, spec.effectiveTokPerRound))
        print(String(format: "Simulated tok/s:     %.2f  →  %.2f   (%.2fx speedup)",
                     baseline.simulatedTokPerSec, spec.simulatedTokPerSec, speedup))
        if let acc = spec.avgAcceptanceRate {
            print(String(format: "Average acceptance:  %.1f%%", acc * 100))
        }
        print("─────────────────────────────────────────────\n")
    }

    /// Core simulation loop that mirrors the logic in DistributedInferenceRunner.
    private static func runSimulation(
        mode: String,
        targetTokens: Int,
        numDraft: Int,
        coordLatencyMs: UInt32,
        workerLatencyMs: UInt32,
        simulatedAcceptanceRate: Double
    ) -> Result {

        var tokens = 0
        var roundTrips = 0
        var totalAccepted: Int = 0
        var totalProposed: Int = 0
        var workerCacheOffset = 0

        let start = CFAbsoluteTimeGetCurrent()

        // Simple deterministic "draft" generator for the simulation.
        // In reality this comes from the n-gram speculator.
        func proposeDrafts(_ k: Int) -> [Int] {
            // Just generate plausible token IDs (the actual values don't matter for timing).
            return Array(1000..<1000 + k)
        }

        // Simulate one coordinator forward (fast)
        func coordForward(tokens: Int) {
            usleep(coordLatencyMs * 1000)
        }

        // Simulate one worker forward (slow) — this is the expensive part we want to amortize.
        func workerForward(tokens: Int) {
            usleep(workerLatencyMs * 1000)
            roundTrips += 1
        }

        // Simulate the speculativeVerify call (one worker forward for K tokens)
        func speculativeVerify(k: Int) -> [Int] {
            workerForward(tokens: k + 1)   // one batched forward

            // Simulate realistic acceptance: we accept on average `simulatedAcceptanceRate`
            var accepted = 0
            for _ in 0..<k {
                if Double.random(in: 0...1) < simulatedAcceptanceRate {
                    accepted += 1
                } else {
                    break
                }
            }
            totalProposed += k
            totalAccepted += accepted
            return Array(0..<k + 1)   // pretend verified tokens
        }

        while tokens < targetTokens {
            if numDraft > 0 {
                // Speculative round
                let drafts = proposeDrafts(numDraft)
                let k = min(numDraft, drafts.count)

                if k == 0 {
                    // fallback single
                    coordForward(tokens: 1)
                    workerForward(tokens: 1)
                    tokens += 1
                    continue
                }

                // 1. Coordinator runs on the whole speculated sequence (batched)
                coordForward(tokens: k + 1)

                // 2. One worker call verifies everything (result ignored in pure timing sim)
                _ = speculativeVerify(k: k)

                // 3. Compute acceptance (simulated)
                var accepted = 0
                for _ in 0..<k {
                    if Double.random(in: 0...1) < simulatedAcceptanceRate {
                        accepted += 1
                    } else {
                        break
                    }
                }

                // 4. "Rollback" simulation (just accounting)
                let rejected = k - accepted
                if rejected > 0 {
                    workerCacheOffset -= rejected   // pretend we trimmed
                }

                // 5. Advance by accepted + possible bonus
                let gained = accepted + (rejected > 0 ? 1 : 0)
                tokens += min(gained, targetTokens - tokens)

                // Record for stats
                if k > 0 {
                    // In real code: specDecoder.recordAccepted(...)
                }

            } else {
                // Baseline single-token pipeline (exactly what we had before speculation)
                coordForward(tokens: 1)
                workerForward(tokens: 1)
                tokens += 1
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let elapsedMs = elapsed * 1000.0
        let tokPerSec = elapsed > 0 ? Double(tokens) / elapsed : 0.0
        let tokPerRound = roundTrips > 0 ? Double(tokens) / Double(roundTrips) : 0.0
        let accRate: Double? = totalProposed > 0 ? Double(totalAccepted) / Double(totalProposed) : nil

        print(String(format: "%-28s %3d tokens, %3d worker calls, %.2f tok/round, %.2f tok/s",
                     mode, tokens, roundTrips, tokPerRound, tokPerSec))
        if let acc = accRate {
            print(String(format: "    Acceptance rate: %.1f%%", acc * 100))
        }

        return Result(
            mode: mode,
            tokensGenerated: tokens,
            workerRoundTrips: roundTrips,
            effectiveTokPerRound: tokPerRound,
            simulatedTokPerSec: tokPerSec,
            avgAcceptanceRate: accRate,
            elapsedMs: elapsedMs
        )
    }
}
