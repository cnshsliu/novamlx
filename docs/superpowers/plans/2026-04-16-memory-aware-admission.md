# Phase 1: Memory-Aware Admission — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace count-based request admission with memory-aware admission, so multiple concurrent requests (like Claude Code's dual-request pattern) are queued when GPU memory is insufficient instead of crashing.

**Architecture:** A new `MemoryBudgetTracker` actor maintains a global memory budget, tracking KV memory committed per active sequence. The `ContinuousBatcher` queries this tracker before admitting requests—admitting if budget allows, queuing if not. TurboQuant compression ratios are factored into estimates so 4-bit KV caches use accurate (not FP16) memory projections. All request paths (including grammar, session, JSON schema) route through the batcher for unified admission control.

**Tech Stack:** Swift 5.9+, MLX framework (Apple Silicon), os_unfair_lock (NovaMLXLock), swift-log

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `Sources/NovaMLXEngine/MemoryBudgetTracker.swift` | **Create** | Global memory budget accounting: reserve/release per sequence, TurboQuant-aware bytesPerToken, available budget calculation |
| `Sources/NovaMLXEngine/ContinuousBatcher.swift` | **Modify** | Replace count-based admission with memory-aware admission using MemoryBudgetTracker |
| `Sources/NovaMLXEngine/MLXEngine.swift` | **Modify** | Expose `effectiveBytesPerToken(modelId:)` method factoring in TurboQuant; wire MemoryBudgetTracker; release budget on generation complete |
| `Sources/NovaMLXInference/InferenceService.swift` | **Modify** | Route session/grammar/JSON schema requests through batcher for unified admission |
| `Sources/NovaMLXEngine/EnginePool.swift` | **Modify** | Fix `evictIfNeeded` bug where `pool[evicted]` returns nil |
| `Tests/NovaMLXEngineTests/MemoryBudgetTrackerTests.swift` | **Create** | Unit tests for budget tracker: reserve, release, canAdmit, TurboQuant ratio |
| `Tests/NovaMLXEngineTests/BatcherAdmissionTests.swift` | **Create** | Unit tests for memory-aware batcher admission |

---

### Task 1: Create MemoryBudgetTracker

**Files:**
- Create: `Sources/NovaMLXEngine/MemoryBudgetTracker.swift`
- Test: `Tests/NovaMLXEngineTests/MemoryBudgetTrackerTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// Tests/NovaMLXEngineTests/MemoryBudgetTrackerTests.swift
import XCTest
@testable import NovaMLXEngine
import NovaMLXCore

final class MemoryBudgetTrackerTests: XCTestCase {

    // MARK: - Basic Reserve/Release

    func testCanAdmitWhenBudgetAvailable() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000_000) // 1 GB
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 1000,
            bytesPerToken: 256
        )
        XCTAssertTrue(canAdmit)
    }

    func testCannotAdmitWhenBudgetExceeded() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB
        // Reserve 900 KB (90% of budget)
        await tracker.reserve(
            modelId: "test-model",
            sequenceId: UUID(),
            weightsBytes: 0,
            estimatedTokens: 3600,
            bytesPerToken: 256 // 3600 * 256 = 921,600 bytes ≈ 900 KB
        )
        // Try to admit another 900 KB request — should fail (10% headroom rule)
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 3600,
            bytesPerToken: 256
        )
        XCTAssertFalse(canAdmit)
    }

    func testReleaseFreesBudget() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB
        let seqId = UUID()
        await tracker.reserve(
            modelId: "test-model",
            sequenceId: seqId,
            weightsBytes: 0,
            estimatedTokens: 3600,
            bytesPerToken: 256
        )
        await tracker.release(sequenceId: seqId)
        // After release, should be able to admit again
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 3600,
            bytesPerToken: 256
        )
        XCTAssertTrue(canAdmit)
    }

    // MARK: - Headroom

    func testHeadroomPreventsFullBudgetUsage() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB
        // Try to use 95% of budget (only 5% headroom, default is 10%)
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 3700,
            bytesPerToken: 256 // 3700 * 256 = 947,200 bytes ≈ 94.7% of budget
        )
        XCTAssertFalse(canAdmit) // Rejected because it exceeds 90% (10% headroom)
    }

    // MARK: - Multiple Sequences

    func testMultipleSequencesTrackCorrectly() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000) // 10 MB
        let seq1 = UUID()
        let seq2 = UUID()

        // Reserve seq1: 2500 tokens * 256 bytes = 640,000 bytes
        await tracker.reserve(modelId: "model", sequenceId: seq1, weightsBytes: 0, estimatedTokens: 2500, bytesPerToken: 256)
        // Reserve seq2: 2500 tokens * 256 bytes = 640,000 bytes
        await tracker.reserve(modelId: "model", sequenceId: seq2, weightsBytes: 0, estimatedTokens: 2500, bytesPerToken: 256)

        // Total committed: 1,280,000 bytes. Available: 10,000,000 - 1,280,000 = 8,720,000
        // With 10% headroom: usable = 9,000,000. Remaining: 9,000,000 - 1,280,000 = 7,720,000
        // Should be able to admit another 2500 tokens (640,000 bytes)
        let canAdmit = await tracker.canAdmit(modelId: "model", estimatedTokens: 2500, bytesPerToken: 256)
        XCTAssertTrue(canAdmit)

        // Release seq1
        await tracker.release(sequenceId: seq1)
        // Now committed: 640,000. Should definitely be able to admit
        let canAdmit2 = await tracker.canAdmit(modelId: "model", estimatedTokens: 2500, bytesPerToken: 256)
        XCTAssertTrue(canAdmit2)
    }

    // MARK: - Available Budget Query

    func testAvailableKVBudget() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000)
        let available = await tracker.availableKVBudget
        // No weights, no committed: available = 10,000,000
        XCTAssertEqual(available, 10_000_000)
    }

    func testAvailableKVBudgetAfterReserve() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000)
        await tracker.reserve(modelId: "model", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)
        let available = await tracker.availableKVBudget
        XCTAssertEqual(available, 10_000_000 - 256_000)
    }

    // MARK: - Metrics

    func testMetricsTracking() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000)
        let seq1 = UUID()
        await tracker.reserve(modelId: "model", sequenceId: seq1, weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)

        let metrics = await tracker.metrics
        XCTAssertEqual(metrics.committedKVBytes, 256_000)
        XCTAssertEqual(metrics.activeSequenceCount, 1)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter MemoryBudgetTrackerTests 2>&1 | tail -20`
Expected: FAIL — `MemoryBudgetTracker` does not exist yet.

- [ ] **Step 3: Implement MemoryBudgetTracker**

```swift
// Sources/NovaMLXEngine/MemoryBudgetTracker.swift
import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Tracks GPU memory budget across all active inference sequences.
/// Provides memory-aware admission control: canAdmit() checks whether
/// a new sequence's estimated KV cache fits within the remaining budget.
public actor MemoryBudgetTracker {

    // MARK: - Types

    public struct Metrics: Sendable {
        public let gpuLimitBytes: UInt64
        public let weightsMemoryBytes: UInt64
        public let committedKVBytes: UInt64
        public let availableBudgetBytes: UInt64
        public let activeSequenceCount: Int
        public let modelCount: Int
    }

    private struct SequenceEntry {
        let modelId: String
        let estimatedBytes: UInt64
    }

    private struct ModelBudget {
        var weightsBytes: UInt64
        var sequences: [UUID: SequenceEntry]
    }

    // MARK: - State

    private let gpuLimitBytes: UInt64
    private var weightsMemoryBytes: UInt64 = 0
    private var committedKVBytes: UInt64 = 0
    private var models: [String: ModelBudget] = [:]
    private let headroomPercent: Int  // e.g. 10 means keep 10% free

    // MARK: - Init

    public init(gpuLimitBytes: UInt64, headroomPercent: Int = 10) {
        self.gpuLimitBytes = gpuLimitBytes
        self.headroomPercent = headroomPercent
    }

    // MARK: - Queries

    /// Available bytes for new KV cache allocations (after weights and committed KV).
    public var availableKVBudget: UInt64 {
        gpuLimitBytes >= (weightsMemoryBytes + committedKVBytes)
            ? gpuLimitBytes - weightsMemoryBytes - committedKVBytes
            : 0
    }

    /// Check whether a new sequence can be admitted without exceeding budget.
    /// Factors in the headroom percentage (default 10%).
    public func canAdmit(
        modelId: String,
        estimatedTokens: Int,
        bytesPerToken: Int
    ) -> Bool {
        let needed = UInt64(estimatedTokens) * UInt64(bytesPerToken)
        let usableBudget = availableKVBudget * UInt64(100 - headroomPercent) / 100
        return needed <= usableBudget
    }

    /// Current metrics snapshot.
    public var metrics: Metrics {
        Metrics(
            gpuLimitBytes: gpuLimitBytes,
            weightsMemoryBytes: weightsMemoryBytes,
            committedKVBytes: committedKVBytes,
            availableBudgetBytes: availableKVBudget,
            activeSequenceCount: models.values.reduce(0) { $0 + $1.sequences.count },
            modelCount: models.count
        )
    }

    // MARK: - Mutation

    /// Reserve memory for a new sequence. Called at admission time.
    public func reserve(
        modelId: String,
        sequenceId: UUID,
        weightsBytes: UInt64,
        estimatedTokens: Int,
        bytesPerToken: Int
    ) {
        let kvBytes = UInt64(estimatedTokens) * UInt64(bytesPerToken)

        if models[modelId] == nil {
            models[modelId] = ModelBudget(weightsBytes: weightsBytes, sequences: [:])
            self.weightsMemoryBytes += weightsBytes
            NovaMLXLog.info("MemoryBudget: registered model \(modelId), weights=\(weightsBytes / 1024 / 1024)MB")
        }

        models[modelId]?.sequences[sequenceId] = SequenceEntry(
            modelId: modelId,
            estimatedBytes: kvBytes
        )
        committedKVBytes += kvBytes

        NovaMLXLog.info("MemoryBudget: reserved \(kvBytes / 1024 / 1024)MB for sequence \(sequenceId.uuidString.prefix(8)), committed=\(committedKVBytes / 1024 / 1024)MB, available=\(availableKVBudget / 1024 / 1024)MB")
    }

    /// Release memory when a sequence completes or is preempted.
    public func release(sequenceId: UUID) {
        for (modelId, budget) in models {
            if let entry = budget.sequences.removeValue(forKey: sequenceId) {
                committedKVBytes = committedKVBytes >= entry.estimatedBytes
                    ? committedKVBytes - entry.estimatedBytes
                    : 0
                NovaMLXLog.info("MemoryBudget: released \(entry.estimatedBytes / 1024 / 1024)MB for sequence \(sequenceId.uuidString.prefix(8)), committed=\(committedKVBytes / 1024 / 1024)MB, available=\(availableKVBudget / 1024 / 1024)MB")

                // Clean up empty model entries (but keep weights tracked)
                if budget.sequences.isEmpty {
                    // Keep the model entry for weights tracking
                }
                return
            }
        }
        NovaMLXLog.warning("MemoryBudget: release called for unknown sequence \(sequenceId.uuidString.prefix(8))")
    }

    /// Update the actual memory usage for a sequence (feedback loop).
    /// If actual differs from estimate, adjust the committed total.
    public func updateActual(modelId: String, sequenceId: UUID, actualBytes: UInt64) {
        guard var budget = models[modelId],
              let entry = budget.sequences[sequenceId] else { return }
        let delta = Int64(actualBytes) - Int64(entry.estimatedBytes)
        if delta != 0 {
            committedKVBytes = UInt64(Int64(committedKVBytes) + delta)
            budget.sequences[sequenceId] = SequenceEntry(
                modelId: modelId,
                estimatedBytes: actualBytes
            )
            models[modelId] = budget
            NovaMLXLog.info("MemoryBudget: adjusted sequence \(sequenceId.uuidString.prefix(8)) by \(delta / 1024)KB")
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter MemoryBudgetTrackerTests 2>&1 | tail -20`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXEngine/MemoryBudgetTracker.swift Tests/NovaMLXEngineTests/MemoryBudgetTrackerTests.swift
git commit -m "feat: add MemoryBudgetTracker for memory-aware admission control"
```

---

### Task 2: Add TurboQuant-Aware bytesPerToken to MLXEngine

**Files:**
- Modify: `Sources/NovaMLXEngine/MLXEngine.swift`

This adds a method to compute effective bytesPerToken that factors in TurboQuant compression. It also exposes the budget tracker and wires it into the engine.

- [ ] **Step 1: Add budget tracker property and effectiveBytesPerToken method to MLXEngine**

In `Sources/NovaMLXEngine/MLXEngine.swift`, add after the existing property declarations (around line 75):

```swift
// Memory budget tracker for admission control
public let budgetTracker: MemoryBudgetTracker
```

In `init()`, after `self.wiredPolicy = MLXLMCommon.WiredSumPolicy()` (around line 91), add:

```swift
// Initialize budget tracker with actual GPU limit
let gpuLimit = GPU.maxRecommendedWorkingSetBytes() ?? 0
self.budgetTracker = MemoryBudgetTracker(gpuLimitBytes: UInt64(gpuLimit))
```

Add the `effectiveBytesPerToken` method (after `preflightCheck`, around line 292):

```swift
/// Compute effective bytesPerToken for a model, factoring in TurboQuant compression.
/// Returns raw FP16 bytesPerToken when TurboQuant is not configured.
public func effectiveBytesPerToken(modelId: String) -> Int {
    guard let container = getContainer(for: modelId) else { return 262_144 }
    let raw = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144

    // Factor in TurboQuant compression
    if let turboConfig = turboQuantService.getConfig(forModel: modelId) {
        // FP16 = 16 bits, TurboQuant uses N bits → compression = 16/N
        let compressionRatio = Double(16) / Double(turboConfig.bits)
        return max(1, Int(Double(raw) / compressionRatio))
    }

    // Also check per-model config kvBits
    if let kvBits = container.config.kvBits, kvBits < 16 {
        let compressionRatio = Double(16) / Double(kvBits)
        return max(1, Int(Double(raw) / compressionRatio))
    }

    return raw
}

/// Estimate total tokens for a request (prompt + max output).
/// Uses conservative defaults when maxTokens is not specified.
public func estimateRequestTokens(modelId: String, request: InferenceRequest) -> Int {
    let maxTokens = request.maxTokens ?? getContainer(for: modelId)?.config.maxTokens ?? 4096
    // Conservative prompt estimate: assume ~500 tokens if not yet tokenized.
    // Actual count will be checked during preflight inside engine.generate/stream.
    let estimatedPromptTokens = 500
    return estimatedPromptTokens + maxTokens
}
```

- [ ] **Step 2: Wire budget release into generate completion**

In `MLXEngine.generate()`, after the generation completes (around line 406, after `var generatedText = ""`), add budget release. Find the defer block or completion point and add:

In the `generate()` method, after `defer { lock.withLock { activeCount -= 1 } }` (line 362), add:

```swift
defer { await budgetTracker.release(sequenceId: request.id) }
```

And at the beginning of `generate()`, before the preflight check (around line 379), add reservation:

```swift
// Reserve memory in budget tracker before preflight
let bpt = effectiveBytesPerToken(modelId: request.model)
// We'll refine token count after tokenization, but reserve conservatively now
```

**Important**: Since `generate()` is not async everywhere we need it, we'll handle the reserve/release in the `ContinuousBatcher` instead (Task 3), which wraps `generate()`. The engine's `preflightCheck` remains as a safety net.

Remove this step's engine-level budget wiring and instead keep `effectiveBytesPerToken` and `estimateRequestTokens` as utility methods. The batcher will use them.

- [ ] **Step 3: Build to verify compilation**

Run: `swift build -c release 2>&1 | tail -10`
Expected: Build complete.

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXEngine/MLXEngine.swift
git commit -m "feat: add effectiveBytesPerToken and budget tracker to MLXEngine"
```

---

### Task 3: Modify ContinuousBatcher for Memory-Aware Admission

**Files:**
- Modify: `Sources/NovaMLXEngine/ContinuousBatcher.swift`

This is the core change: replace `activeRequests < maxBatchSize` with a memory-budget check.

- [ ] **Step 1: Add budget tracker reference and memory-aware admission logic**

In `ContinuousBatcher.swift`, modify the init and add the memory check:

Change the init (line 89) to accept the budget tracker:

```swift
private let budgetTracker: MemoryBudgetTracker
private let engine: MLXEngine
public let maxBatchSize: Int
private let maxConcurrentPerModel: Int
private let lock = NovaMLXLock()

public init(engine: MLXEngine, maxBatchSize: Int = 8, maxConcurrentPerModel: Int = 4) {
    self.engine = engine
    self.budgetTracker = engine.budgetTracker
    self.maxBatchSize = maxBatchSize
    self.maxConcurrentPerModel = maxConcurrentPerModel
    self.generateQueue = []
    self.streamQueue = []
    self._activeCount = 0
    self._totalQueued = 0
    self._totalCompleted = 0
    self._totalQueueWaitTime = 0
    self._totalPreempted = 0
    self._peakActiveCount = 0
    self._startTime = Date()
    self.activeTasks = [:]
}
```

Add helper methods for admission check:

```swift
/// Check whether a request can be admitted based on memory budget and per-model concurrency.
private func canAdmitRequest(_ request: InferenceRequest) async -> Bool {
    let modelId = request.model

    // Per-model concurrency check (like OLLAMA_NUM_PARALLEL)
    let activeForModel = lock.withLock {
        _activeModelCounts[modelId] ?? 0
    }
    if activeForModel >= maxConcurrentPerModel {
        NovaMLXLog.info("Scheduler: queuing \(request.id.uuidString.prefix(8)) — model \(modelId) at concurrency limit (\(activeForModel)/\(maxConcurrentPerModel))")
        return false
    }

    // Memory budget check
    let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
    let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)
    let canAdmit = await budgetTracker.canAdmit(
        modelId: modelId,
        estimatedTokens: estimatedTokens,
        bytesPerToken: bytesPerToken
    )

    if !canAdmit {
        NovaMLXLog.info("Scheduler: queuing \(request.id.uuidString.prefix(8)) — insufficient memory (need \(estimatedTokens * bytesPerToken / 1024 / 1024)MB, available \(await budgetTracker.availableKVBudget / 1024 / 1024)MB)")
    }

    return canAdmit
}
```

Add `_activeModelCounts` tracking dictionary alongside existing state:

```swift
private var _activeModelCounts: [String: Int] = [:]
```

- [ ] **Step 2: Modify submit() to use memory-aware admission**

Replace the `submit()` method (lines 142-158):

```swift
public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
    let priority: RequestPriority = .normal
    let modelId = request.model

    // Memory-aware admission check
    let canStart = await canAdmitRequest(request)

    guard canStart else {
        return try await enqueueAndWait(request: request, priority: priority)
    }

    // Admit: increment counts and reserve memory
    let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
    let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

    lock.withLock {
        _activeCount += 1
        _totalQueued += 1
        if _activeCount > _peakActiveCount { _peakActiveCount = _activeCount }
        _activeModelCounts[modelId] = (_activeModelCounts[modelId] ?? 0) + 1
    }

    await budgetTracker.reserve(
        modelId: modelId,
        sequenceId: request.id,
        weightsBytes: 0, // weights already tracked at model load
        estimatedTokens: estimatedTokens,
        bytesPerToken: bytesPerToken
    )

    defer {
        lock.withLock {
            _activeCount -= 1
            _totalCompleted += 1
            _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
        }
        Task { await budgetTracker.release(sequenceId: request.id) }
    }

    NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing generate (active: \(activeRequests), model=\(modelId))")
    return try await engine.generate(request)
}
```

- [ ] **Step 3: Modify submitStream() to use memory-aware admission**

Replace the `submitStream()` method (lines 160-210):

```swift
public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
    return AsyncThrowingStream { continuation in
        let priority: RequestPriority = .normal
        let modelId = request.model

        // Create a Task for the async admission check
        let admissionTask = Task {
            await self.canAdmitRequest(request)
        }

        Task {
            let canStart = await admissionTask.value

            if canStart {
                let bytesPerToken = self.engine.effectiveBytesPerToken(modelId: modelId)
                let estimatedTokens = self.engine.estimateRequestTokens(modelId: modelId, request: request)

                self.lock.withLock {
                    self._activeCount += 1
                    self._totalQueued += 1
                    if self._activeCount > self._peakActiveCount { self._peakActiveCount = self._activeCount }
                    self._activeModelCounts[modelId] = (self._activeModelCounts[modelId] ?? 0) + 1
                }

                await self.budgetTracker.reserve(
                    modelId: modelId,
                    sequenceId: request.id,
                    weightsBytes: 0,
                    estimatedTokens: estimatedTokens,
                    bytesPerToken: bytesPerToken
                )

                NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing stream (active: \(self.activeRequests), model=\(modelId))")

                do {
                    let stream = self.engine.stream(request)
                    for try await token in stream {
                        if Task.isCancelled { break }
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }

                self.lock.withLock {
                    self._activeCount -= 1
                    self._totalCompleted += 1
                    self._activeModelCounts[modelId] = max(0, (self._activeModelCounts[modelId] ?? 1) - 1)
                }
                await self.budgetTracker.release(sequenceId: request.id)

                // Check if queued requests can now be admitted
                self.processQueuedStream()
                self.processQueuedGenerate()
            } else {
                // Queue the request
                self.lock.withLock {
                    self.streamQueue.append(QueuedStreamRequest(
                        request: request, priority: priority, enqueuedAt: Date(), continuation: continuation
                    ))
                    self.streamQueue.sort { $0.priority > $1.priority }
                    self._totalQueued += 1
                }
                NovaMLXLog.info("Stream request queued: \(request.id.uuidString.prefix(8)), queue depth: \(self.queueDepth)")
            }
        }
    }
}
```

- [ ] **Step 4: Update processQueuedGenerate and processQueuedStream to be memory-aware**

Replace `processQueuedGenerate()` (lines 239-269):

```swift
private func processQueuedGenerate() {
    let item = lock.withLock { () -> QueuedRequest? in
        guard _activeCount < maxBatchSize, !generateQueue.isEmpty else { return nil }
        let item = generateQueue.removeFirst()

        // Check per-model concurrency (synchronous check; memory will be checked async)
        let currentForModel = _activeModelCounts[item.request.model] ?? 0
        guard currentForModel < maxConcurrentPerModel else { return nil }

        _activeCount += 1
        _activeModelCounts[item.request.model] = currentForModel + 1
        return item
    }

    guard let item = item else { return }

    let waitTime = Date().timeIntervalSince(item.enqueuedAt)
    lock.withLock { _totalQueueWaitTime += waitTime }

    let request = item.request
    let continuation = item.continuation
    let modelId = request.model

    Task {
        // Async memory budget check and reserve
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

        let canAdmit = await budgetTracker.canAdmit(
            modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        if !canAdmit {
            // Put back at front of queue and return — will retry on next completion
            lock.withLock {
                _activeCount -= 1
                _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
                generateQueue.insert(item, at: 0)
                _totalQueued -= 1 // don't double-count
            }
            NovaMLXLog.info("Scheduler: generate dequeue rejected — insufficient memory, re-queuing")
            return
        }

        await budgetTracker.reserve(
            modelId: modelId, sequenceId: request.id,
            weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        do {
            let result = try await engine.generate(request)
            continuation.resume(returning: result)
        } catch {
            continuation.resume(throwing: error)
        }

        lock.withLock {
            _activeCount -= 1
            _totalCompleted += 1
            _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
        }
        await budgetTracker.release(sequenceId: request.id)

        processQueuedGenerate()
        processQueuedStream()
    }
}
```

Replace `processQueuedStream()` (lines 271-307) with the same pattern:

```swift
private func processQueuedStream() {
    let item = lock.withLock { () -> QueuedStreamRequest? in
        guard _activeCount < maxBatchSize, !streamQueue.isEmpty else { return nil }
        let item = streamQueue.removeFirst()

        let currentForModel = _activeModelCounts[item.request.model] ?? 0
        guard currentForModel < maxConcurrentPerModel else { return nil }

        _activeCount += 1
        _activeModelCounts[item.request.model] = currentForModel + 1
        return item
    }

    guard let item = item else { return }

    let waitTime = Date().timeIntervalSince(item.enqueuedAt)
    lock.withLock { _totalQueueWaitTime += waitTime }

    let request = item.request
    let continuation = item.continuation
    let modelId = request.model

    let task = Task {
        let bytesPerToken = engine.effectiveBytesPerToken(modelId: modelId)
        let estimatedTokens = engine.estimateRequestTokens(modelId: modelId, request: request)

        let canAdmit = await budgetTracker.canAdmit(
            modelId: modelId, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        if !canAdmit {
            lock.withLock {
                _activeCount -= 1
                _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
                streamQueue.insert(item, at: 0)
                _totalQueued -= 1
            }
            NovaMLXLog.info("Scheduler: stream dequeue rejected — insufficient memory, re-queuing")
            return
        }

        await budgetTracker.reserve(
            modelId: modelId, sequenceId: request.id,
            weightsBytes: 0, estimatedTokens: estimatedTokens, bytesPerToken: bytesPerToken
        )

        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Processing dequeued stream")
        do {
            let stream = engine.stream(request)
            for try await token in stream {
                if Task.isCancelled { break }
                continuation.yield(token)
            }
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }

        lock.withLock {
            _activeCount -= 1
            _totalCompleted += 1
            _activeModelCounts[modelId] = max(0, (_activeModelCounts[modelId] ?? 1) - 1)
        }
        await budgetTracker.release(sequenceId: request.id)

        processQueuedStream()
        processQueuedGenerate()
    }
    lock.withLock { activeTasks[request.id] = task }
}
```

- [ ] **Step 5: Build to verify compilation**

Run: `swift build -c release 2>&1 | tail -10`
Expected: Build complete.

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXEngine/ContinuousBatcher.swift
git commit -m "feat: memory-aware admission in ContinuousBatcher — queue when budget insufficient"
```

---

### Task 4: Route All InferenceService Paths Through Batcher

**Files:**
- Modify: `Sources/NovaMLXInference/InferenceService.swift`

Currently, session, grammar, JSON schema, and regex requests bypass the batcher entirely. These must go through the admission gate for memory tracking.

- [ ] **Step 1: Refactor generate() to route all requests through batcher**

In `Sources/NovaMLXInference/InferenceService.swift`, the `generate()` method has multiple bypass paths. Wrap them all through the batcher:

Find the `generate()` method and change it to route ALL requests through `batcher.submit()`:

```swift
public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
    let resolved = settingsManager.resolveModelId(request.model)
    let settings = settingsManager.settings(for: resolved)
    var finalRequest = settings.applySamplingOverrides(to: request)
    finalRequest = InferenceRequest(
        id: finalRequest.id, model: resolved, messages: finalRequest.messages,
        temperature: finalRequest.temperature, maxTokens: finalRequest.maxTokens,
        topP: finalRequest.topP, topK: finalRequest.topK, minP: finalRequest.minP,
        frequencyPenalty: finalRequest.frequencyPenalty, presencePenalty: finalRequest.presencePenalty,
        repetitionPenalty: finalRequest.repetitionPenalty, seed: finalRequest.seed,
        stream: finalRequest.stream, stop: finalRequest.stop,
        sessionId: finalRequest.sessionId,
        responseFormat: finalRequest.responseFormat,
        jsonSchemaDef: finalRequest.jsonSchemaDef,
        regexPattern: finalRequest.regexPattern,
        gbnfGrammar: finalRequest.gbnfGrammar,
        thinkingBudget: finalRequest.thinkingBudget
    )

    // ALL requests go through the batcher for unified memory-aware admission
    // Session, grammar, and JSON schema paths are handled inside engine.generate()
    // which dispatches to the appropriate method based on request fields.
    return try await batcher.submit(finalRequest)
}
```

Do the same for `stream()`:

```swift
public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
    let resolved = settingsManager.resolveModelId(request.model)
    let settings = settingsManager.settings(for: resolved)
    var finalRequest = settings.applySamplingOverrides(to: request)
    finalRequest = InferenceRequest(
        id: finalRequest.id, model: resolved, messages: finalRequest.messages,
        temperature: finalRequest.temperature, maxTokens: finalRequest.maxTokens,
        topP: finalRequest.topP, topK: finalRequest.topK, minP: finalRequest.minP,
        frequencyPenalty: finalRequest.frequencyPenalty, presencePenalty: finalRequest.presencePenalty,
        repetitionPenalty: finalRequest.repetitionPenalty, seed: finalRequest.seed,
        stream: true, stop: finalRequest.stop,
        sessionId: finalRequest.sessionId,
        responseFormat: finalRequest.responseFormat,
        jsonSchemaDef: finalRequest.jsonSchemaDef,
        regexPattern: finalRequest.regexPattern,
        gbnfGrammar: finalRequest.gbnfGrammar,
        thinkingBudget: finalRequest.thinkingBudget
    )

    return batcher.submitStream(finalRequest)
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build -c release 2>&1 | tail -10`
Expected: Build complete.

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXInference/InferenceService.swift
git commit -m "feat: unified admission — all request paths route through batcher"
```

---

### Task 5: Fix EnginePool evictIfNeeded Bug

**Files:**
- Modify: `Sources/NovaMLXEngine/EnginePool.swift`

The `evictIfNeeded` method has a bug where it looks up `pool[evicted]` after `evictLRU()` already removed the entry, so `freed` is always 0 and all models get evicted.

- [ ] **Step 1: Read the current evictIfNeeded method**

Read `Sources/NovaMLXEngine/EnginePool.swift` lines 94-115 to understand the current code.

- [ ] **Step 2: Fix the bug**

The fix: `evictLRU()` should return the size of the evicted model, not just its ID.

Change `evictLRU()` to return the freed size:

```swift
@discardableResult
func evictLRU() -> UInt64? {
    let candidates = pool.values
        .filter { !$0.isPinned }
        .sorted { $0.lastAccessed < $1.lastAccessed }

    guard let victim = candidates.first else { return nil }

    let freedMB = UInt64(victim.estimatedSizeMB)
    let modelId = victim.container.identifier.id
    victim.container.unload()
    pool.removeValue(forKey: modelId)
    NovaMLXLog.info("EnginePool: evicted \(modelId) (\(freedMB)MB)")
    return freedMB
}
```

Update `evictIfNeeded`:

```swift
func evictIfNeeded(forNewModelSizeMB newModelSizeMB: UInt64) {
    let currentMB = totalMemoryMB()
    guard currentMB + newModelSizeMB > maxMemoryMB else { return }

    var freed: UInt64 = 0
    let needed = currentMB + newModelSizeMB - maxMemoryMB

    while freed < needed {
        guard let freedMB = evictLRU() else { break }
        freed += freedMB
    }

    if freed < needed {
        NovaMLXLog.warning("EnginePool: could only free \(freed)MB of \(needed)MB needed")
    }
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build -c release 2>&1 | tail -10`
Expected: Build complete.

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXEngine/EnginePool.swift
git commit -m "fix: EnginePool evictIfNeeded now correctly tracks freed memory"
```

---

### Task 6: Integration Test — Dual Concurrent Requests

**Files:**
- Create: `Tests/NovaMLXEngineTests/BatcherAdmissionTests.swift`

- [ ] **Step 1: Write integration test for Claude Code's dual-request pattern**

```swift
// Tests/NovaMLXEngineTests/BatcherAdmissionTests.swift
import XCTest
@testable import NovaMLXEngine
@testable import NovaMLXInference
@testable import NovaMLXCore

final class BatcherAdmissionTests: XCTestCase {

    /// Simulate Claude Code's pattern: two concurrent requests to the same model.
    /// Both should complete without crash or "I/O on closed channel" error.
    func testDualConcurrentRequestsAreQueuedCorrectly() async throws {
        // This test verifies the admission logic with a real MemoryBudgetTracker.
        // We use a small budget to force queuing.

        let tracker = MemoryBudgetTracker(gpuLimitBytes: 2_000_000) // 2 MB

        // Request 1: 1000 tokens * 256 bytes = 256 KB — should admit
        let canAdmit1 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 1000, bytesPerToken: 256)
        XCTAssertTrue(canAdmit1)

        await tracker.reserve(modelId: "test-model", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)

        // After reserving 256 KB, remaining = 2MB - 256KB = 1.75MB
        // With 10% headroom: usable = 1.8MB, remaining after reserve = 1.8MB - 256KB = 1.544MB
        // Request 2: 1000 tokens * 256 bytes = 256 KB — should still admit
        let canAdmit2 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 1000, bytesPerToken: 256)
        XCTAssertTrue(canAdmit2)
    }

    /// When budget is tight, second request should be rejected for admission
    /// (it will be queued by the batcher).
    func testSecondRequestRejectedWhenBudgetTight() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 600_000) // 600 KB total

        // Request 1: 2000 tokens * 256 bytes = 512 KB — should admit (under 540KB = 90% of 600KB)
        let canAdmit1 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 2000, bytesPerToken: 256)
        XCTAssertTrue(canAdmit1)

        await tracker.reserve(modelId: "test-model", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 2000, bytesPerToken: 256)

        // After reserving 512KB: remaining = 600KB - 512KB = 88KB
        // With 10% headroom: usable = 540KB, remaining usable = 540KB - 512KB = 28KB
        // Request 2: 200 tokens * 256 bytes = 51.2 KB — too much for 28KB remaining
        let canAdmit2 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 200, bytesPerToken: 256)
        XCTAssertFalse(canAdmit2)
    }

    /// TurboQuant compression should increase effective capacity.
    /// 4-bit quantization = 4x compression → can admit 4x more sequences.
    func testTurboQuantDoublesAdmissionCapacity() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB

        // Without TurboQuant: 256 bytes/token
        // With 4-bit TurboQuant: 256 / 4 = 64 bytes/token
        let rawBytesPerToken = 256
        let turboBytesPerToken = rawBytesPerToken / 4  // 64

        // Without TurboQuant: can admit ~3500 tokens
        let canAdmitRaw = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 3500, bytesPerToken: rawBytesPerToken)
        XCTAssertFalse(canAdmitRaw) // 3500 * 256 = 896,000 > 900,000 (90% of 1MB)

        // With TurboQuant: can admit 3500 tokens easily
        let canAdmitTurbo = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 3500, bytesPerToken: turboBytesPerToken)
        XCTAssertTrue(canAdmitTurbo) // 3500 * 64 = 224,000 < 900,000
    }
}
```

- [ ] **Step 2: Run all tests**

Run: `swift test --filter "MemoryBudgetTracker|BatcherAdmission" 2>&1 | tail -30`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add Tests/NovaMLXEngineTests/BatcherAdmissionTests.swift
git commit -m "test: add admission tests for dual-request and TurboQuant scenarios"
```

---

### Task 7: Full Build and Smoke Test

**Files:**
- No new files — integration verification

- [ ] **Step 1: Full release build**

Run: `bash build.sh 2>&1 | tail -10`
Expected: Build complete.

- [ ] **Step 2: Run complete test suite**

Run: `swift test 2>&1 | tail -30`
Expected: All tests pass (existing tests may need minor updates if interfaces changed).

- [ ] **Step 3: Smoke test with NovaMLX running**

Start NovaMLX and send two concurrent curl requests to simulate Claude Code's pattern:

```bash
# Terminal 1: Start NovaMLX
.build/release/NovaMLX

# Terminal 2: Send two concurrent requests
curl -s http://127.0.0.1:6590/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: abcd1234" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"mlx-community/Phi-3.5-mini-instruct-4bit","max_tokens":30,"messages":[{"role":"user","content":"Say hi"}]}' &

curl -s http://127.0.0.1:6590/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: abcd1234" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"mlx-community/Phi-3.5-mini-instruct-4bit","max_tokens":30,"messages":[{"role":"user","content":"Say hello"}]}' &

wait
```

Expected: Both requests return successfully. NovaMLX logs show:
- "Processing generate (active: 1)" for first request
- "Processing generate (active: 2)" for second (or queued then dequeued)
- No "I/O on closed channel" error

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: Phase 1 complete — memory-aware admission with budget tracking"
```

---

## Self-Review

### Spec Coverage

| Design Spec Section | Covered by Task |
|---|---|
| MemoryBudgetTracker (Section 3.2) | Task 1 |
| Scheduler admission (Section 3.3) | Task 3 (modifies ContinuousBatcher) |
| Admission Controller (Section 3.4) | Task 3 (canAdmitRequest method) |
| TurboQuant-aware estimation (Section 3.5) | Task 2 (effectiveBytesPerToken) |
| Unified routing (Section 3.3 "route all through batcher") | Task 4 |
| EnginePool eviction bug (Appendix C.4) | Task 5 |
| Integration test for Claude Code pattern | Task 6 |

**Not covered in Phase 1** (deferred to Phase 2-4 per design spec):
- Fused decode production path (Phase 2)
- Prefix cache read-path (Phase 3)
- Preemption/dynamic quantization (Phase 4)

### Placeholder Scan

No TBDs, TODOs, or "implement later" patterns found. All steps contain actual code.

### Type Consistency

- `MemoryBudgetTracker.canAdmit(modelId:estimatedTokens:bytesPerToken:)` — used consistently in Task 1, 3, 6
- `MemoryBudgetTracker.reserve(modelId:sequenceId:weightsBytes:estimatedTokens:bytesPerToken:)` — used consistently in Task 1, 3
- `MemoryBudgetTracker.release(sequenceId:)` — used consistently in Task 1, 3
- `engine.effectiveBytesPerToken(modelId:)` — defined in Task 2, used in Task 3
- `engine.estimateRequestTokens(modelId:request:)` — defined in Task 2, used in Task 3
- `_activeModelCounts: [String: Int]` — added in Task 3, used throughout Task 3
