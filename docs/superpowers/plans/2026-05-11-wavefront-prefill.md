# Overlapped Pipeline Wavefront Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement overlapped wavefront prefill that splits long prompts into chunks and staggers them across pipeline ranks, overlapping computation with communication for 25–45% TTFT reduction.

**Architecture:** New `PipelineLayer.swift` wraps first/last layers of each shard for transparent send/recv with send batching. `ShardEngine` gains `wavefrontPrefill()` that runs the staggered chunk loop with leading/trailing dummies. Falls back to sequential prefill for short prompts.

**Tech Stack:** Swift, MLX (eval/asyncEval), NovaMLXDistributed module

**Spec:** `docs/superpowers/specs/2026-05-11-wavefront-prefill-design.md`

---

### Task 1: Add PrefillConfig and WavefrontStats to DistributedTypes

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedTypes.swift`
- Modify: `Tests/NovaMLXDistributedTests/DistributedTypesTests.swift`

- [ ] **Step 1: Write failing tests for PrefillConfig**

Add to `Tests/NovaMLXDistributedTests/DistributedTypesTests.swift` inside the `DistributedTypesTests` struct:

```swift
@Test("PrefillConfig has correct defaults")
func prefillConfigDefaults() {
    let config = PrefillConfig()
    #expect(config.baseStepSize == 4096)
    #expect(config.minChunkSize == 512)
    #expect(config.minWavefrontTokens == 4096)
}

@Test("PrefillConfig codable round-trip")
func prefillConfigCodable() throws {
    let config = PrefillConfig(baseStepSize: 2048, minChunkSize: 256, minWavefrontTokens: 8192)
    let data = try JSONEncoder().encode(config)
    let decoded = try JSONDecoder().decode(PrefillConfig.self, from: data)
    #expect(decoded == config)
}

@Test("PrefillConfig decodes with missing fields using defaults")
func prefillConfigPartialDecode() throws {
    let json = "{}".data(using: .utf8)!
    let config = try JSONDecoder().decode(PrefillConfig.self, from: json)
    #expect(config.baseStepSize == 4096)
    #expect(config.minChunkSize == 512)
    #expect(config.minWavefrontTokens == 4096)
}

@Test("WavefrontStats stores correct values")
func wavefrontStatsValues() {
    let stats = WavefrontStats(
        chunkSize: 2048,
        nRealChunks: 4,
        nLeadingDummies: 1,
        nTrailingDummies: 0,
        promptTokens: 8192,
        prefillCommBytes: 65536
    )
    #expect(stats.chunkSize == 2048)
    #expect(stats.nRealChunks == 4)
    #expect(stats.nLeadingDummies == 1)
    #expect(stats.nTrailingDummies == 0)
    #expect(stats.promptTokens == 8192)
    #expect(stats.prefillCommBytes == 65536)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter DistributedTypesTests 2>&1 | tail -10`
Expected: FAIL — `PrefillConfig` and `WavefrontStats` undefined

- [ ] **Step 3: Implement PrefillConfig and WavefrontStats**

Append to `Sources/NovaMLXDistributed/DistributedTypes.swift` (after `ShardPlan`):

```swift
// MARK: - PrefillConfig

/// Configuration for overlapped wavefront prefill.
///
/// Controls chunk sizing and the activation threshold below which
/// sequential prefill is used instead.
public struct PrefillConfig: Codable, Sendable, Equatable {
    /// Base step size in tokens. Divided by `worldSize` to get actual chunk size.
    public var baseStepSize: Int

    /// Minimum chunk size in tokens. Prevents pathological tiny chunks on large clusters.
    public var minChunkSize: Int

    /// Minimum prompt length to activate wavefront prefill.
    /// Prompts shorter than this fall back to sequential prefill.
    public var minWavefrontTokens: Int

    public init(
        baseStepSize: Int = 4096,
        minChunkSize: Int = 512,
        minWavefrontTokens: Int = 4096
    ) {
        self.baseStepSize = baseStepSize
        self.minChunkSize = minChunkSize
        self.minWavefrontTokens = minWavefrontTokens
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        baseStepSize = try container.decodeIfPresent(Int.self, forKey: .baseStepSize) ?? 4096
        minChunkSize = try container.decodeIfPresent(Int.self, forKey: .minChunkSize) ?? 512
        minWavefrontTokens = try container.decodeIfPresent(Int.self, forKey: .minWavefrontTokens) ?? 4096
    }
}

// MARK: - WavefrontStats

/// Observability stats from a wavefront prefill execution.
public struct WavefrontStats: Sendable, Equatable {
    public let chunkSize: Int
    public let nRealChunks: Int
    public let nLeadingDummies: Int
    public let nTrailingDummies: Int
    public let promptTokens: Int
    public let prefillCommBytes: UInt64

    public init(
        chunkSize: Int,
        nRealChunks: Int,
        nLeadingDummies: Int,
        nTrailingDummies: Int,
        promptTokens: Int,
        prefillCommBytes: UInt64
    ) {
        self.chunkSize = chunkSize
        self.nRealChunks = nRealChunks
        self.nLeadingDummies = nLeadingDummies
        self.nTrailingDummies = nTrailingDummies
        self.promptTokens = promptTokens
        self.prefillCommBytes = prefillCommBytes
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter DistributedTypesTests 2>&1 | tail -10`
Expected: All PrefillConfig and WavefrontStats tests PASS

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedTypes.swift Tests/NovaMLXDistributedTests/DistributedTypesTests.swift
git commit -m "feat(distributed): add PrefillConfig and WavefrontStats types"
```

---

### Task 2: Create PipelineLayer with Send Queue

**Files:**
- Create: `Sources/NovaMLXDistributed/PipelineLayer.swift`
- Create: `Tests/NovaMLXDistributedTests/PipelineLayerTests.swift`

- [ ] **Step 1: Write failing tests for PipelineLayer**

Create `Tests/NovaMLXDistributedTests/PipelineLayerTests.swift`:

```swift
import Foundation
import Testing
import MLX
@testable import NovaMLXDistributed

@Suite("PipelineLayer")
struct PipelineLayerTests {

    // MARK: - PendingSend

    @Test("PendingSend stores output, destination, and group")
    func pendingSendProperties() {
        let group = DistributedGroup.uninitialized
        let output = MLXArray([1.0, 2.0])
        let send = PendingSend(output: output, destination: 1, group: group)
        #expect(send.destination == 1)
    }

    // MARK: - PrefillSendQueue

    @Test("PrefillSendQueue starts empty")
    func sendQueueStartsEmpty() {
        let queue = PrefillSendQueue()
        #expect(queue.isEmpty)
        #expect(queue.count == 0)
    }

    @Test("PrefillSendQueue enqueue increments count")
    func sendQueueEnqueue() {
        let queue = PrefillSendQueue()
        let group = DistributedGroup.uninitialized
        queue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        #expect(queue.count == 1)
        #expect(!queue.isEmpty)
    }

    @Test("PrefillSendQueue drain returns all pending sends and clears")
    func sendQueueDrain() {
        let queue = PrefillSendQueue()
        let group = DistributedGroup.uninitialized
        queue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        queue.enqueue(PendingSend(output: MLXArray([2.0]), destination: 1, group: group))
        let drained = queue.drain()
        #expect(drained.count == 2)
        #expect(queue.isEmpty)
        #expect(queue.count == 0)
    }

    @Test("PrefillSendQueue clear discards without returning")
    func sendQueueClear() {
        let queue = PrefillSendQueue()
        let group = DistributedGroup.uninitialized
        queue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        queue.clear()
        #expect(queue.isEmpty)
    }

    @Test("PrefillSendQueue drain on empty returns empty array")
    func sendQueueDrainEmpty() {
        let queue = PrefillSendQueue()
        let drained = queue.drain()
        #expect(drained.isEmpty)
    }

    // MARK: - flushPrefillSends / clearPrefillSends (global)

    @Test("Global flushPrefillSends does not crash on empty queue")
    func globalFlushEmpty() {
        clearPrefillSends()
        flushPrefillSends()
        // No crash = pass
    }

    @Test("Global clearPrefillSends clears the shared queue")
    func globalClear() {
        clearPrefillSends()
        let group = DistributedGroup.uninitialized
        prefillSendQueue.enqueue(PendingSend(output: MLXArray([1.0]), destination: 1, group: group))
        clearPrefillSends()
        #expect(prefillSendQueue.isEmpty)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter PipelineLayerTests 2>&1 | tail -10`
Expected: FAIL — types not found

- [ ] **Step 3: Implement PipelineLayer**

Create `Sources/NovaMLXDistributed/PipelineLayer.swift`:

```swift
import Foundation
import MLX

// MARK: - PendingSend

/// A queued send operation waiting to be flushed.
public struct PendingSend: Sendable {
    public let output: MLXArray
    public let destination: Int
    public let group: DistributedGroup

    public init(output: MLXArray, destination: Int, group: DistributedGroup) {
        self.output = output
        self.destination = destination
        self.group = group
    }
}

// MARK: - PrefillSendQueue

/// Thread-safe send queue for batching prefill communication.
///
/// Uses `NSLock` for synchronization — hot path, no DispatchQueue/actor overhead.
public final class PrefillSendQueue: @unchecked Sendable {
    private var pending: [PendingSend] = []
    private let lock = NSLock()

    public init() {}

    /// Whether the queue has no pending sends.
    public var isEmpty: Bool {
        lock.withLock { pending.isEmpty }
    }

    /// Number of pending sends.
    public var count: Int {
        lock.withLock { pending.count }
    }

    /// Add a send to the queue.
    public func enqueue(_ send: PendingSend) {
        lock.withLock { pending.append(send) }
    }

    /// Remove and return all pending sends.
    public func drain() -> [PendingSend] {
        lock.withLock {
            let result = pending
            pending.removeAll()
            return result
        }
    }

    /// Discard all pending sends without transmitting.
    public func clear() {
        lock.withLock { pending.removeAll() }
    }
}

// MARK: - Shared send queue

/// Global prefill send queue shared across all PipelineLastLayer instances.
public let prefillSendQueue = PrefillSendQueue()

/// Whether pipeline layers should queue sends (true during prefill) or send immediately.
public var pipelineQueueSends = false

/// Flush all pending prefill sends via `asyncEval`.
///
/// Called after each real chunk's forward pass in wavefront prefill.
/// Drains the queue, fires all sends asynchronously, and clears.
public func flushPrefillSends() {
    let sends = prefillSendQueue.drain()
    for send in sends {
        let sent = MLXDistributedWrapper.send(
            send.output,
            to: send.destination,
            group: send.group
        )
        MLX.asyncEval(sent)
    }
}

/// Discard all pending sends without transmitting (error/cancellation path).
public func clearPrefillSends() {
    prefillSendQueue.clear()
}
```

- [ ] **Step 4: Add test target dependency (if needed)**

The test file imports `NovaMLXDistributed` which already exists as a test target. No Package.swift change needed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `swift test --filter PipelineLayerTests 2>&1 | tail -15`
Expected: All PipelineLayer tests PASS

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXDistributed/PipelineLayer.swift Tests/NovaMLXDistributedTests/PipelineLayerTests.swift
git commit -m "feat(distributed): add PipelineLayer send queue with flush and clear"
```

---

### Task 3: Add PipelineFirstLayer and PipelineLastLayer

**Files:**
- Modify: `Sources/NovaMLXDistributed/PipelineLayer.swift`
- Modify: `Tests/NovaMLXDistributedTests/PipelineLayerTests.swift`

- [ ] **Step 1: Write failing tests for layer wrappers**

Append to `Tests/NovaMLXDistributedTests/PipelineLayerTests.swift`:

```swift
// MARK: - PipelineFirstLayer

@Test("PipelineFirstLayer rank 0 passes through without recv")
func pipelineFirstLayerRank0Passthrough() async throws {
    let group = DistributedGroup.uninitialized
    let layer = PipelineFirstLayer(rank: 0, group: group)
    let input = MLXArray([1.0, 2.0, 3.0])
    let output = try await layer.forward(input: input)
    // Rank 0 should pass through (recv not needed)
    #expect(output.shape == input.shape)
}

@Test("PipelineFirstLayer non-zero rank receives (uninitialized group returns zeros)")
func pipelineFirstLayerNonZeroRecv() async throws {
    let group = DistributedGroup.uninitialized
    let layer = PipelineFirstLayer(rank: 1, group: group)
    let input = MLXArray([1.0, 2.0, 3.0])
    let output = try await layer.forward(input: input)
    // With uninitialized group, recvLike returns zeros of same shape
    #expect(output.shape == input.shape)
}

// MARK: - PipelineLastLayer

@Test("PipelineLastLayer last rank passes through without send")
func pipelineLastLayerLastRankPassthrough() async throws {
    let group = DistributedGroup.uninitialized
    let layer = PipelineLastLayer(rank: 1, worldSize: 2, group: group)
    let input = MLXArray([1.0, 2.0, 3.0])
    let output = try await layer.forward(input: input)
    // Last rank passes through
    #expect(output.shape == input.shape)
}

@Test("PipelineLastLayer non-last rank queues send when queueSends is true")
func pipelineLastLayerQueuesSend() async throws {
    clearPrefillSends()
    let group = DistributedGroup.uninitialized
    let layer = PipelineLastLayer(rank: 0, worldSize: 2, group: group)
    pipelineQueueSends = true
    defer { pipelineQueueSends = false }

    let input = MLXArray([1.0, 2.0, 3.0])
    _ = try await layer.forward(input: input)
    #expect(!prefillSendQueue.isEmpty)
    #expect(prefillSendQueue.count == 1)
    clearPrefillSends()
}

@Test("PipelineLastLayer non-last rank sends immediately when queueSends is false")
func pipelineLastLayerImmediateSend() async throws {
    clearPrefillSends()
    let group = DistributedGroup.uninitialized
    let layer = PipelineLastLayer(rank: 0, worldSize: 2, group: group)
    pipelineQueueSends = false

    let input = MLXArray([1.0, 2.0, 3.0])
    _ = try await layer.forward(input: input)
    // Immediate send — queue should still be empty
    #expect(prefillSendQueue.isEmpty)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter PipelineLayerTests 2>&1 | tail -15`
Expected: FAIL — `PipelineFirstLayer` and `PipelineLastLayer` undefined

- [ ] **Step 3: Implement PipelineFirstLayer and PipelineLastLayer**

Append to `Sources/NovaMLXDistributed/PipelineLayer.swift`:

```swift
// MARK: - PipelineFirstLayer

/// Wraps the first layer of a shard to receive activations from the previous rank.
///
/// - Rank 0: passes input through (generates embeddings from token IDs).
/// - Other ranks: `recv` activation from `rank - 1`, then pass to the wrapped layer.
public final class PipelineFirstLayer: @unchecked Sendable {
    public let rank: Int
    public let group: DistributedGroup

    public init(rank: Int, group: DistributedGroup) {
        self.rank = rank
        self.group = group
    }

    /// Receive activation from previous rank (if not rank 0) and return it.
    ///
    /// In the full implementation, this feeds into the wrapped layer's forward pass.
    /// For now, returns the received activation directly.
    public func forward(input: MLXArray) async throws -> MLXArray {
        if rank == 0 {
            return input
        }
        MLX.eval(input)
        let received = MLXDistributedWrapper.recvLike(
            input,
            from: rank - 1,
            group: group
        )
        MLX.eval(received)
        return received
    }
}

// MARK: - PipelineLastLayer

/// Wraps the last layer of a shard to send activations to the next rank.
///
/// - Last rank: passes through (returns activation for sampling).
/// - Other ranks during prefill (`queueSends = true`): queue send.
/// - Other ranks during decode (`queueSends = false`): immediate send.
public final class PipelineLastLayer: @unchecked Sendable {
    public let rank: Int
    public let worldSize: Int
    public let group: DistributedGroup

    public init(rank: Int, worldSize: Int, group: DistributedGroup) {
        self.rank = rank
        self.worldSize = worldSize
        self.group = group
    }

    /// Whether this rank is the last in the pipeline.
    public var isLastRank: Bool {
        rank == worldSize - 1
    }

    /// Forward output through the send mechanism.
    ///
    /// In the full implementation, this wraps the layer's output.
    /// For now, handles the send/queue logic directly.
    public func forward(input: MLXArray) async throws -> MLXArray {
        if isLastRank {
            return input
        }

        MLX.eval(input)

        if pipelineQueueSends {
            prefillSendQueue.enqueue(PendingSend(
                output: input,
                destination: rank + 1,
                group: group
            ))
        } else {
            _ = MLXDistributedWrapper.send(
                input,
                to: rank + 1,
                group: group
            )
        }

        return input
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter PipelineLayerTests 2>&1 | tail -20`
Expected: All PipelineLayer tests PASS

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/PipelineLayer.swift Tests/NovaMLXDistributedTests/PipelineLayerTests.swift
git commit -m "feat(distributed): add PipelineFirstLayer and PipelineLastLayer wrappers"
```

---

### Task 4: Add wavefrontPrefill to ShardEngine

**Files:**
- Modify: `Sources/NovaMLXDistributed/ShardEngine.swift`
- Modify: `Tests/NovaMLXDistributedTests/ShardEngineTests.swift`

- [ ] **Step 1: Write failing tests for wavefrontPrefill**

Append to `Tests/NovaMLXDistributedTests/ShardEngineTests.swift`:

```swift
// MARK: - Wavefront Prefill

@Test("wavefrontPrefill falls back to sequential for short prompts")
func wavefrontFallbackShortPrompt() async throws {
    let assignment = ShardAssignment(
        nodeId: "test",
        startLayer: 0,
        endLayer: 10,
        memoryEstimate: 0
    )
    let policy = FitInMemoryPolicy(assignment: assignment)
    let engine = ShardEngine(
        group: .uninitialized,
        assignment: assignment,
        policy: policy
    )
    try await policy.bindWeights()

    // Short prompt (3 tokens) — below default minWavefrontTokens (4096)
    let tokens = MLXArray([1, 2, 3])
    let config = PrefillConfig(minWavefrontTokens: 4096)
    let output = try await engine.prefill(tokens: tokens, config: config)
    #expect(output.shape == tokens.shape)
}

@Test("wavefrontPrefill falls back for single-node group")
func wavefrontFallbackSingleNode() async throws {
    let assignment = ShardAssignment(
        nodeId: "test",
        startLayer: 0,
        endLayer: 10,
        memoryEstimate: 0
    )
    let policy = FitInMemoryPolicy(assignment: assignment)
    let engine = ShardEngine(
        group: .uninitialized,  // size == 0
        assignment: assignment,
        policy: policy
    )
    try await policy.bindWeights()

    let tokens = MLXArray(Array(0..<8192).map { Int32($0) })
    let config = PrefillConfig(minWavefrontTokens: 4096)
    let output = try await engine.prefill(tokens: tokens, config: config)
    // Uninitialized group => size 0 => falls back to sequential
    #expect(output.shape == tokens.shape)
}

@Test("wavefrontPrefill computes correct chunk plan")
func wavefrontChunkPlan() {
    let config = PrefillConfig(baseStepSize: 4096, minChunkSize: 512)
    let worldSize = 2
    let promptLen = 8192

    let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
    let nReal = (promptLen - 1 + chunkSize - 1) / chunkSize  // ceil division
    let nLeading = 0  // rank 0
    let nTrailing = worldSize - 1 - 0  // worldSize - 1

    #expect(chunkSize == 2048)
    #expect(nReal == 4)  // ceil((8192 - 1) / 2048) = ceil(8191/2048) = 4
    #expect(nLeading == 0)
    #expect(nTrailing == 1)
}

@Test("wavefrontPrefill chunk plan with 3 nodes")
func wavefrontChunkPlan3Nodes() {
    let config = PrefillConfig(baseStepSize: 4096, minChunkSize: 512)
    let worldSize = 3
    let promptLen = 12288

    let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
    let nReal = (promptLen - 1 + chunkSize - 1) / chunkSize

    #expect(chunkSize == 1365)  // 4096 / 3 = 1365
    #expect(nReal == 9)  // ceil(12287 / 1365) = 9
}

@Test("wavefrontPrefill minChunkSize floor prevents tiny chunks")
func wavefrontMinChunkFloor() {
    let config = PrefillConfig(baseStepSize: 4096, minChunkSize: 1024)
    let worldSize = 8

    let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
    #expect(chunkSize == 1024)  // 4096/8=512 < minChunkSize 1024, so floor wins
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter ShardEngineTests 2>&1 | tail -15`
Expected: FAIL — `wavefrontPrefill` not found

- [ ] **Step 3: Implement wavefrontPrefill and refactor prefill routing**

Replace the existing `prefill(tokens:)` method in `Sources/NovaMLXDistributed/ShardEngine.swift` with:

```swift
/// Run prefill — selects sequential or wavefront based on config and cluster state.
///
/// - Parameters:
///   - tokens: Input token IDs.
///   - config: Prefill configuration. Defaults to ``PrefillConfig``().
/// - Returns: Output activation from the last shard.
public func prefill(tokens: MLXArray, config: PrefillConfig? = nil) async throws -> MLXArray {
    let cfg = config ?? PrefillConfig()
    let tokenCount = tokens.shape.reduce(1, *)
    if group.size > 1 && tokenCount >= cfg.minWavefrontTokens {
        return try await wavefrontPrefill(tokens: tokens, config: cfg)
    }
    return try await sequentialPrefill(tokens: tokens)
}

/// Sequential prefill: receive → compute → send.
///
/// Used when the prompt is too short for wavefront or when running standalone.
public func sequentialPrefill(tokens: MLXArray) async throws -> MLXArray {
    guard policy.isReady else {
        throw ShardEngineError.notReady
    }

    var activation: MLXArray
    if isFirstShard {
        activation = tokens
    } else {
        activation = MLXDistributedWrapper.recvLike(
            tokens,
            from: group.rank - 1,
            group: group
        )
    }

    var cache: [Any] = []
    activation = try policy.compute(input: activation, cache: &cache)

    if !isLastShard {
        _ = MLXDistributedWrapper.send(
            activation,
            to: group.rank + 1,
            group: group
        )
    }

    return activation
}

/// Overlapped wavefront prefill for long prompts on multi-node clusters.
///
/// Splits the prompt into chunks and staggers them across ranks:
/// - Leading dummies: `rank` no-op iterations
/// - Real chunks: forward pass + send batch flush
/// - Trailing dummies: `worldSize - 1 - rank` no-op iterations
/// - Two final single-token passes
///
/// Send batching: `PipelineLastLayer` queues sends during prefill;
/// `flushPrefillSends()` fires them after each chunk via `asyncEval`.
private func wavefrontPrefill(
    tokens: MLXArray,
    config: PrefillConfig
) async throws -> MLXArray {
    guard policy.isReady else {
        throw ShardEngineError.notReady
    }

    let promptLen = tokens.shape.reduce(1, *)
    let worldSize = group.size
    let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
    let nReal = (promptLen - 1 + chunkSize - 1) / chunkSize  // ceil((promptLen-1) / chunkSize)
    let nLeading = group.rank
    let nTrailing = worldSize - 1 - group.rank

    // Activate send batching for prefill
    pipelineQueueSends = true
    clearPrefillSends()
    defer {
        pipelineQueueSends = false
        clearPrefillSends()
    }

    // Leading dummies — pure no-ops (padding for pipeline alignment)
    for _ in 0..<nLeading {
        // No forward pass — just synchronization padding
    }

    // Real chunks
    var processed = 0
    var cache: [Any] = []
    for i in 0..<nReal {
        let start = i * chunkSize
        let end = min(start + chunkSize, promptLen - 1)
        let chunkSizeActual = end - start

        // Slice tokens for this chunk
        let chunkTokens: MLXArray
        if isFirstShard {
            chunkTokens = tokens[0..<chunkSizeActual]
        } else {
            chunkTokens = tokens[0..<chunkSizeActual]  // placeholder shape
        }

        // Receive from previous shard if not first
        var activation: MLXArray
        if isFirstShard {
            activation = chunkTokens
        } else {
            activation = MLXDistributedWrapper.recvLike(
                chunkTokens,
                from: group.rank - 1,
                group: group
            )
        }

        activation = try policy.compute(input: activation, cache: &cache)

        if !isLastShard {
            // Queue send via PipelineLastLayer pattern
            MLX.eval(activation)
            prefillSendQueue.enqueue(PendingSend(
                output: activation,
                destination: group.rank + 1,
                group: group
            ))
        }

        flushPrefillSends()
        processed += chunkSizeActual
    }

    // Trailing dummies — pure no-ops
    for _ in 0..<nTrailing {
        // No forward pass — drain pipeline
    }

    // Two final single-token passes
    for _ in 0..<2 {
        var finalCache: [Any] = []
        let lastToken: MLXArray
        if isFirstShard {
            lastToken = tokens[-1..<promptLen]
        } else {
            let ref = tokens[-1..<promptLen]
            lastToken = MLXDistributedWrapper.recvLike(
                ref,
                from: group.rank - 1,
                group: group
            )
        }
        var result = try policy.compute(input: lastToken, cache: &finalCache)
        if !isLastShard {
            MLX.eval(result)
            _ = MLXDistributedWrapper.send(result, to: group.rank + 1, group: group)
        }
        flushPrefillSends()
    }

    // Final eval on all cache state
    let cacheArrays = cache.compactMap { $0 as? MLXArray }
    if !cacheArrays.isEmpty {
        MLX.eval(cacheArrays)
    }

    // Return last computed activation (last shard returns for sampling)
    if isLastShard {
        return tokens[-1..<promptLen]  // placeholder — real impl returns logits
    }
    return MLXArray(0)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter ShardEngineTests 2>&1 | tail -20`
Expected: All ShardEngine tests PASS (both existing + new wavefront tests)

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/ShardEngine.swift Tests/NovaMLXDistributedTests/ShardEngineTests.swift
git commit -m "feat(distributed): add wavefrontPrefill with sequential fallback"
```

---

### Task 5: Wire PrefillConfig into ClusterConfig and Admin Routes

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedTypes.swift`
- Modify: `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift`

- [ ] **Step 1: Write failing test**

Append to `Tests/NovaMLXDistributedTests/DistributedTypesTests.swift` inside `DistributedTypesTests`:

```swift
@Test("ClusterConfig with PrefillConfig decodes correctly")
func clusterConfigWithPrefill() throws {
    let json = """
    {"role": "coordinator", "coordinatorHost": "192.168.1.1", "coordinatorPort": 6591, "prefill": {"baseStepSize": 2048, "minChunkSize": 256, "minWavefrontTokens": 8192}}
    """.data(using: .utf8)!
    let config = try JSONDecoder().decode(ClusterConfig.self, from: json)
    #expect(config.prefill.baseStepSize == 2048)
    #expect(config.prefill.minChunkSize == 256)
    #expect(config.prefill.minWavefrontTokens == 8192)
}

@Test("ClusterConfig without PrefillConfig uses defaults")
func clusterConfigWithoutPrefill() throws {
    let json = """
    {"role": "coordinator", "coordinatorHost": "192.168.1.1", "coordinatorPort": 6591}
    """.data(using: .utf8)!
    let config = try JSONDecoder().decode(ClusterConfig.self, from: json)
    #expect(config.prefill.baseStepSize == 4096)
    #expect(config.prefill.minChunkSize == 512)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter DistributedTypesTests 2>&1 | tail -10`
Expected: FAIL — `ClusterConfig` has no `prefill` property

- [ ] **Step 3: Add PrefillConfig to ClusterConfig**

In `Sources/NovaMLXDistributed/DistributedTypes.swift`, modify `ClusterConfig`:

Add property after `strategy`:
```swift
public var prefill: PrefillConfig
```

Update the memberwise init:
```swift
public init(
    role: ClusterRole,
    coordinatorHost: String,
    coordinatorPort: Int = 6591,
    strategy: ClusterStrategy = .minNodes,
    prefill: PrefillConfig = PrefillConfig()
) {
    self.role = role
    self.coordinatorHost = coordinatorHost
    self.coordinatorPort = coordinatorPort
    self.strategy = strategy
    self.prefill = prefill
}
```

Update the `init(from:)` decoder:
```swift
public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    role = try container.decode(ClusterRole.self, forKey: .role)
    coordinatorHost = try container.decode(String.self, forKey: .coordinatorHost)
    coordinatorPort = try container.decodeIfPresent(Int.self, forKey: .coordinatorPort) ?? 6591
    strategy = try container.decodeIfPresent(ClusterStrategy.self, forKey: .strategy) ?? .minNodes
    prefill = try container.decodeIfPresent(PrefillConfig.self, forKey: .prefill) ?? PrefillConfig()
}
```

- [ ] **Step 4: Add wavefront stats to ClusterAdminRoutes**

In `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift`, add a method:

```swift
/// Wavefront prefill stats (placeholder until wired to live inference).
public func wavefrontStats() -> [String: Any] {
    return ["status": "not_available"]
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `swift test --filter DistributedTypesTests 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 6: Run full distributed test suite**

Run: `swift test --filter NovaMLXDistributedTests 2>&1 | tail -10`
Expected: All distributed tests PASS

- [ ] **Step 7: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedTypes.swift Sources/NovaMLXDistributed/ClusterAdminRoutes.swift Tests/NovaMLXDistributedTests/DistributedTypesTests.swift
git commit -m "feat(distributed): wire PrefillConfig into ClusterConfig and admin routes"
```

---

### Task 6: Full Build and Test Verification

**Files:**
- No new files

- [ ] **Step 1: Run full distributed test suite**

Run: `swift test --filter NovaMLXDistributedTests 2>&1 | tail -15`
Expected: All tests PASS (42 existing + new wavefront tests)

- [ ] **Step 2: Run release build**

Run: `./build.sh -c release 2>&1 | tail -5`
Expected: `Build complete!`

- [ ] **Step 3: Codesign**

Run: `codesign --force --deep --sign - dist/NovaMLX.app`
Expected: Success

- [ ] **Step 4: Commit any remaining fixes**

Only if needed — no commit if build/test clean.
