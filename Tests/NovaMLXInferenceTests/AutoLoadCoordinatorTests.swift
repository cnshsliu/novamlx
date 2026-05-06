import Testing
import Foundation
@testable import NovaMLXInference
@testable import NovaMLXCore
@testable import NovaMLXEngine
@testable import NovaMLXModelManager

@Suite("AutoLoadCoordinator Tests")
struct AutoLoadCoordinatorTests {

    @Test("modelNotLoaded error has correct description")
    func modelNotLoadedError() async throws {
        let error = NovaMLXError.modelNotLoaded("test-model")
        #expect(error.errorDescription?.contains("test-model") == true)
        #expect(error.retryAfter == nil)
    }

    @Test("modelLoadInProgress error has correct description and Retry-After")
    func modelLoadInProgressError() async throws {
        let error = NovaMLXError.modelLoadInProgress(modelId: "test-model", etaSeconds: 45)
        #expect(error.errorDescription?.contains("test-model") == true)
        #expect(error.retryAfter == 45)
    }

    @Test("modelLoadInProgress with nil eta defaults to 60")
    func modelLoadInProgressNilEta() {
        let error = NovaMLXError.modelLoadInProgress(modelId: "test-model", etaSeconds: nil)
        #expect(error.retryAfter == 60)
    }

    @Test("LoadPhase enum has all expected cases")
    func loadPhaseCases() {
        let phases: [LoadPhase] = [
            .queued, .feasibilityChecking, .evicting,
            .loadingWeights, .warmingUp, .ready, .failed
        ]
        #expect(phases.count == 7)
        #expect(LoadPhase(rawValue: "loadingWeights") == .loadingWeights)
        #expect(LoadPhase(rawValue: "nonexistent") == nil)
    }

    @Test("AutoLoadConfig defaults are correct")
    func autoLoadConfigDefaults() {
        let cfg = AutoLoadConfig()
        #expect(cfg.enabled == true)
        #expect(cfg.evictOnConflict == true)
        #expect(cfg.allowDownload == false)
        #expect(cfg.coldLoadTimeoutSeconds == 180)
        #expect(cfg.coldLoadTimeoutMaxSeconds == 600)
        #expect(cfg.coldLoadTimeoutMultiplier == 3.0)
        #expect(cfg.defaultTTLSecondsAfterAutoLoad == 600)
        #expect(cfg.emitProgressEvents == false)
    }

    @Test("ServerConfig backward compat: decoding without autoLoad key uses defaults")
    func serverConfigBackwardCompat() throws {
        let oldConfigJSON = """
        {
            "host": "127.0.0.1",
            "port": 6590,
            "adminPort": 6591
        }
        """
        let data = oldConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(ServerConfig.self, from: data)
        #expect(config.autoLoad.enabled == true)
        #expect(config.autoLoad.evictOnConflict == true)
        #expect(config.autoLoad.allowDownload == false)
    }

    @Test("AutoLoadConfig round-trip encoding")
    func autoLoadConfigRoundTrip() throws {
        var cfg = AutoLoadConfig()
        cfg.enabled = false
        cfg.coldLoadTimeoutSeconds = 300
        cfg.emitProgressEvents = true
        let data = try JSONEncoder().encode(cfg)
        let decoded = try JSONDecoder().decode(AutoLoadConfig.self, from: data)
        #expect(decoded.enabled == false)
        #expect(decoded.coldLoadTimeoutSeconds == 300)
        #expect(decoded.emitProgressEvents == true)
    }

    @Test("MemoryFeasibility.evaluateSafetyMargin returns correct values")
    func safetyMarginValues() {
        // Small model (< 20GB): 20% margin
        let small = MemoryFeasibility.evaluateSafetyMargin(estimatedBytes: 10 * 1_073_741_824)
        #expect(small == UInt64(Double(10 * 1_073_741_824) * 0.2))

        // Medium model (25GB): 30% margin
        let medium = MemoryFeasibility.evaluateSafetyMargin(estimatedBytes: 25 * 1_073_741_824)
        #expect(medium == UInt64(Double(25 * 1_073_741_824) * 0.3))

        // Large model (> 30GB): fixed 5GB
        let large = MemoryFeasibility.evaluateSafetyMargin(estimatedBytes: 40 * 1_073_741_824)
        #expect(large == 5 * 1_073_741_824)
    }

    @Test("MemoryFeasibility.evaluate returns canLoad=false when exceeds budget")
    func feasibilityEvaluateExceedsBudget() {
        let result = MemoryFeasibility.evaluate(
            modelId: "test",
            modelSizeBytes: 50 * 1_073_741_824,
            currentlyAvailableBytes: 10 * 1_073_741_824,
            gpuBudgetBytes: 20 * 1_073_741_824
        )
        #expect(result.canLoad == false)
        #expect(result.reason != nil)
    }

    @Test("MemoryFeasibility.evaluate returns canLoad=true when fits")
    func feasibilityEvaluateFits() {
        let result = MemoryFeasibility.evaluate(
            modelId: "test",
            modelSizeBytes: 5 * 1_073_741_824,
            currentlyAvailableBytes: 30 * 1_073_741_824,
            gpuBudgetBytes: 64 * 1_073_741_824
        )
        #expect(result.canLoad == true)
        #expect(result.reason == nil)
    }

    // MARK: - Behavioral: LoadDedup primitive

    @Test("LoadDedup: concurrent calls for same modelId run work exactly once")
    func dedupSameModelRunsOnce() async throws {
        let dedup = LoadDedup()
        let counter = Counter()
        let started = Date()

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    try await dedup.ensureSingle(modelId: "M") {
                        await counter.bump()
                        try await Task.sleep(for: .milliseconds(100))
                    }
                }
            }
            try await group.waitForAll()
        }

        #expect(await counter.value == 1, "work block must run exactly once across 10 concurrent calls")
        // All 10 callers must have awaited the single in-flight task (≥100ms total)
        #expect(Date().timeIntervalSince(started) >= 0.09)
    }

    @Test("LoadDedup: different modelIds run work independently")
    func dedupDifferentModelsIndependent() async throws {
        let dedup = LoadDedup()
        let counter = Counter()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for id in ["A", "B", "C"] {
                group.addTask {
                    try await dedup.ensureSingle(modelId: id) {
                        await counter.bump()
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(await counter.value == 3, "each distinct modelId triggers its own work")
    }

    @Test("LoadDedup: failed load releases slot — retry runs work again")
    func dedupFailedLoadReleasesSlot() async throws {
        struct Boom: Error {}
        let dedup = LoadDedup()
        let counter = Counter()

        // First attempt fails
        do {
            try await dedup.ensureSingle(modelId: "X") {
                await counter.bump()
                throw Boom()
            }
            #expect(Bool(false), "should have thrown")
        } catch is Boom {
            // expected
        }

        // Slot must be released — second attempt re-runs work
        try await dedup.ensureSingle(modelId: "X") {
            await counter.bump()
        }

        #expect(await counter.value == 2, "failed load must release slot for retry")
        #expect(await dedup.inFlightCount == 0, "no leaked inFlight entries after success/failure")
    }

    @Test("LoadDedup: concurrent waiters on a failing load all see the error")
    func dedupConcurrentWaitersSeeError() async throws {
        struct Boom: Error {}
        let dedup = LoadDedup()
        let errorCount = Counter()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<5 {
                group.addTask {
                    do {
                        try await dedup.ensureSingle(modelId: "F") {
                            try await Task.sleep(for: .milliseconds(50))
                            throw Boom()
                        }
                    } catch {
                        await errorCount.bump()
                    }
                }
            }
            await group.waitForAll()
        }
        #expect(await errorCount.value == 5, "all 5 concurrent callers must observe the failure")
        #expect(await dedup.inFlightCount == 0)
    }

    // MARK: - Behavioral: AsyncSemaphore ordering

    @Test("AsyncSemaphore(1) serializes concurrent waiters")
    func semaphoreSerializes() async throws {
        let sem = AsyncSemaphore(value: 1)
        let log = OrderedLog()

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<5 {
                group.addTask {
                    await sem.wait()
                    await log.append("enter-\(i)")
                    try? await Task.sleep(for: .milliseconds(10))
                    await log.append("leave-\(i)")
                    sem.signal()
                }
            }
            await group.waitForAll()
        }

        // Pairs of enter/leave must alternate — no overlap
        let entries = await log.entries
        for i in stride(from: 0, to: entries.count, by: 2) {
            let enter = entries[i]
            let leave = entries[i + 1]
            #expect(enter.hasPrefix("enter-"))
            #expect(leave.hasPrefix("leave-"))
            // enter-N must be followed by leave-N (same i)
            let enterIdx = enter.dropFirst("enter-".count)
            let leaveIdx = leave.dropFirst("leave-".count)
            #expect(enterIdx == leaveIdx, "entries should pair: \(enter) then \(leave)")
        }
    }
}

// MARK: - Test helpers

private actor Counter {
    private(set) var value: Int = 0
    func bump() { value += 1 }
}

private actor OrderedLog {
    private(set) var entries: [String] = []
    func append(_ s: String) { entries.append(s) }
}
