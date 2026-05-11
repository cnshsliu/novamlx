# NovaMLX Distributed Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable NovaMLX to distribute LLM inference across multiple Macs via pipeline parallelism, transparently serving models too large for a single machine.

**Architecture:** Pipeline parallelism over MLX Distributed (JACCL RDMA auto / Ring TCP fallback). Coordinator node handles API + first N layers; worker nodes handle remaining layers. Last shard samples directly, returns token ID to coordinator. All distributed code dormant when no cluster configured — zero single-machine overhead.

**Tech Stack:** Swift, MLX/Cmlx distributed C API, Network.framework (Bonjour/NWBrowser), Hummingbird HTTP, safetensors header parsing.

**Spec:** `docs/superpowers/specs/2026-05-10-distributed-inference-design.md`

---

## File Structure

### New files to create:
```
Sources/NovaMLXDistributed/
├── DistributedTypes.swift         # ClusterConfig, ClusterRole, ClusterStrategy, ShardPlan, ShardAssignment, NodeSpec, LayerProfile, LayerType
├── DistributedGroup.swift         # Swift wrapper for mlx_distributed_group + init/send/recv/all_gather/all_sum
├── ModelAnalyzer.swift            # Safetensors header parsing → LayerProfile → ShardPlan computation
├── ClusterManager.swift           # Coordinator-side: Bonjour advertise, worker registration, heartbeat, fault detection
├── WorkerService.swift            # Worker-side: Bonjour discover, registration, heartbeat, shard execution
├── ShardEngine.swift              # Per-node inference: ComputePolicy, prefill, decode, send/recv activations
├── WeightDistributor.swift        # Model file sync: local check + auto-download from coordinator
├── FaultRecovery.swift            # L1 (transient 30s), L2 (spare swap), L3a (auto-reshard), L3b (hard fail / manual)
├── ClusterAdminRoutes.swift       # Admin API endpoints for cluster management
Tests/NovaMLXDistributedTests/
├── DistributedTypesTests.swift
├── ModelAnalyzerTests.swift
├── ShardEngineTests.swift
├── FaultRecoveryTests.swift
```

### Files to modify:
```
Package.swift                                                    # Add NovaMLXDistributed target + test target
Sources/NovaMLXCore/Types.swift                                  # Add cluster field to ServerConfig
Sources/NovaMLXCore/Configuration.swift                          # Add cluster config loading/saving
Sources/NovaMLXAPI/APIServer.swift                               # Add cluster admin routes + routing hook
Sources/NovaMLXInference/InferenceService.swift                  # Add cluster dispatch path
```

---

## Task 1: Add NovaMLXDistributed Module to Package.swift

**Files:**
- Modify: `Package.swift`

- [ ] **Step 1: Add library product**

Add after the existing `.library(name: "NovaMLXMCP", ...)` line in the products array:

```swift
.library(name: "NovaMLXDistributed", targets: ["NovaMLXDistributed"]),
```

- [ ] **Step 2: Add target definition**

Add the target in the targets array, after the NovaMLXWorker target:

```swift
.target(
    name: "NovaMLXDistributed",
    dependencies: [
        "NovaMLXCore",
        "NovaMLXUtils",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "Cmlx", package: "mlx-swift"),
    ],
    swiftSettings: swiftSettings
),
```

- [ ] **Step 3: Add test target**

```swift
.testTarget(
    name: "NovaMLXDistributedTests",
    dependencies: ["NovaMLXDistributed"],
    swiftSettings: swiftSettings
),
```

- [ ] **Step 4: Add NovaMLXDistributed dependency to NovaMLXAPI target**

Add `"NovaMLXDistributed"` to the `NovaMLXAPI` target's dependencies array.

- [ ] **Step 5: Verify build compiles**

Run: `./build.sh -c debug 2>&1 | tail -5`

The build will fail because the module has no source files yet. Create a placeholder:

```bash
mkdir -p Sources/NovaMLXDistributed Tests/NovaMLXDistributedTests
echo "// Placeholder" > Sources/NovaMLXDistributed/Placeholder.swift
echo "// Placeholder" > Tests/NovaMLXDistributedTests/PlaceholderTests.swift
```

Run: `./build.sh -c debug 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add Package.swift Sources/NovaMLXDistributed/ Tests/NovaMLXDistributedTests/
git commit -m "feat(distributed): add NovaMLXDistributed module skeleton"
```

---

## Task 2: Distributed Types — Config, ShardPlan, NodeSpec

**Files:**
- Create: `Sources/NovaMLXDistributed/DistributedTypes.swift`
- Create: `Tests/NovaMLXDistributedTests/DistributedTypesTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// Tests/NovaMLXDistributedTests/DistributedTypesTests.swift
import Testing
@testable import NovaMLXDistributed

@Suite("Distributed Types")
struct DistributedTypesTests {

    @Test("ClusterConfig decodes with defaults")
    func clusterConfigDefaults() throws {
        let json = """
        {"role": "worker", "coordinatorHost": "192.168.1.1", "coordinatorPort": 6591}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ClusterConfig.self, from: json)
        #expect(config.role == .worker)
        #expect(config.coordinatorHost == "192.168.1.1")
        #expect(config.coordinatorPort == 6591)
        #expect(config.strategy == .minNodes)
    }

    @Test("ShardPlan computes correct layer ranges for 2 nodes")
    func shardPlanTwoNodes() {
        let profiles = (0..<40).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let nodes = [
            NodeSpec(nodeId: "mac-a", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0),
            NodeSpec(nodeId: "mac-b", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 0.6),
        ]
        let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .minNodes)
        #expect(plan.assignments.count == 2)
        let totalCovered = plan.assignments.reduce(0) { $0 + ($1.endLayer - $1.startLayer) }
        #expect(totalCovered == 40)
    }

    @Test("ClusterRole codable round-trip")
    func clusterRoleRoundTrip() throws {
        for role in [ClusterRole.coordinator, .worker] {
            let encoded = try JSONEncoder().encode(role)
            let decoded = try JSONDecoder().decode(ClusterRole.self, from: encoded)
            #expect(decoded == role)
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter DistributedTypesTests 2>&1 | tail -10`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement DistributedTypes**

```swift
// Sources/NovaMLXDistributed/DistributedTypes.swift
import Foundation

// MARK: - Cluster Configuration

public enum ClusterRole: String, Codable, Sendable {
    case coordinator
    case worker
}

public enum ClusterStrategy: String, Codable, Sendable {
    case minNodes
    case spread
}

public struct ClusterConfig: Codable, Sendable, Equatable {
    public let role: ClusterRole
    public let coordinatorHost: String
    public let coordinatorPort: Int
    public let strategy: ClusterStrategy

    private enum CodingKeys: String, CodingKey {
        case role, coordinatorHost, coordinatorPort, strategy
    }

    public init(
        role: ClusterRole,
        coordinatorHost: String,
        coordinatorPort: Int = 6591,
        strategy: ClusterStrategy = .minNodes
    ) {
        self.role = role
        self.coordinatorHost = coordinatorHost
        self.coordinatorPort = coordinatorPort
        self.strategy = strategy
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        role = try c.decode(ClusterRole.self, forKey: .role)
        coordinatorHost = try c.decode(String.self, forKey: .coordinatorHost)
        coordinatorPort = try c.decodeIfPresent(Int.self, forKey: .coordinatorPort) ?? 6591
        strategy = try c.decodeIfPresent(ClusterStrategy.self, forKey: .strategy) ?? .minNodes
    }
}

// MARK: - Node Specification

public struct NodeSpec: Codable, Sendable, Equatable {
    public let nodeId: String
    public let totalMemoryBytes: UInt64
    public let computeCapability: Double
    public let hostname: String
    public let port: Int

    public init(nodeId: String, totalMemoryBytes: UInt64, computeCapability: Double, hostname: String, port: Int) {
        self.nodeId = nodeId
        self.totalMemoryBytes = totalMemoryBytes
        self.computeCapability = computeCapability
        self.hostname = hostname
        self.port = port
    }
}

// MARK: - Model Layer Analysis

public enum LayerType: String, Codable, Sendable {
    case embedding
    case transformer
    case output
    case moe
}

public struct LayerProfile: Codable, Sendable, Equatable {
    public let layerIndex: Int
    public let parameterCount: UInt64
    public let estimatedMemoryBytes: UInt64
    public let layerType: LayerType

    public init(layerIndex: Int, parameterCount: UInt64, estimatedMemoryBytes: UInt64, layerType: LayerType) {
        self.layerIndex = layerIndex
        self.parameterCount = parameterCount
        self.estimatedMemoryBytes = estimatedMemoryBytes
        self.layerType = layerType
    }
}

// MARK: - Shard Plan

public struct ShardAssignment: Codable, Sendable, Equatable {
    public let nodeId: String
    public let startLayer: Int
    public let endLayer: Int
    public let memoryEstimate: UInt64

    public init(nodeId: String, startLayer: Int, endLayer: Int, memoryEstimate: UInt64) {
        self.nodeId = nodeId
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.memoryEstimate = memoryEstimate
    }

    public var layerCount: Int { endLayer - startLayer }
}

public struct ShardPlan: Codable, Sendable, Equatable {
    public let assignments: [ShardAssignment]
    public let totalLayers: Int
    public let strategy: ClusterStrategy

    public init(assignments: [ShardAssignment], totalLayers: Int, strategy: ClusterStrategy) {
        self.assignments = assignments
        self.totalLayers = totalLayers
        self.strategy = strategy
    }

    public init(profiles: [LayerProfile], nodes: [NodeSpec], strategy: ClusterStrategy) {
        self.strategy = strategy
        self.totalLayers = profiles.count

        let totalMemory = nodes.reduce(UInt64(0)) { $0 + $1.totalMemoryBytes }
        let totalParams = profiles.reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }

        var assignments: [ShardAssignment] = []
        var layerOffset = 0

        for (i, node) in nodes.enumerated() {
            let isLast = (i == nodes.count - 1)
            let fraction = Double(node.totalMemoryBytes) / Double(totalMemory)
            var layerCount: Int
            if isLast {
                layerCount = profiles.count - layerOffset
            } else {
                layerCount = max(1, Int(Double(profiles.count) * fraction))
            }
            let memEstimate = profiles[layerOffset..<layerOffset+layerCount].reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }
            assignments.append(ShardAssignment(
                nodeId: node.nodeId,
                startLayer: layerOffset,
                endLayer: layerOffset + layerCount,
                memoryEstimate: memEstimate
            ))
            layerOffset += layerCount
        }
        self.assignments = assignments
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter DistributedTypesTests 2>&1 | tail -5`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedTypes.swift Tests/NovaMLXDistributedTests/DistributedTypesTests.swift
git rm Sources/NovaMLXDistributed/Placeholder.swift Tests/NovaMLXDistributedTests/PlaceholderTests.swift 2>/dev/null; true
git commit -m "feat(distributed): add cluster config, shard plan, and node spec types"
```

---

## Task 3: MLX Distributed Swift Wrappers

**Files:**
- Create: `Sources/NovaMLXDistributed/DistributedGroup.swift`
- Modify: `Tests/NovaMLXDistributedTests/DistributedTypesTests.swift` (add wrapper tests)

- [ ] **Step 1: Write the failing test**

Add to `DistributedTypesTests.swift`:

```swift
@Suite("Distributed Group Wrappers")
struct DistributedGroupTests {

    @Test("Backend availability check does not crash")
    func backendAvailabilityCheck() {
        // This should not crash even without a cluster
        let ringAvailable = MLXDistributedWrapper.isBackendAvailable("ring")
        #expect(type(of: ringAvailable) == Bool.self)
    }

    @Test("DistributedGroup wraps C handle")
    func groupWrapsHandle() {
        // Can't create a real group without a cluster, but we can test the type exists
        let group = DistributedGroup.uninitialized
        #expect(group.rank == -1)
        #expect(group.size == 0)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter DistributedGroupTests 2>&1 | tail -5`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement DistributedGroup.swift**

```swift
// Sources/NovaMLXDistributed/DistributedGroup.swift
import Cmlx
import MLX

/// Swift wrapper around mlx_distributed_group C handle
public final class DistributedGroup: @unchecked Sendable {
    var ctx: mlx_distributed_group

    /// Sentinel for "no group initialized"
    public static let uninitialized = DistributedGroup(ctx: mlx_distributed_group(ctx: nil))

    public var rank: Int {
        guard ctx.ctx != nil else { return -1 }
        return mlx_distributed_group_rank(ctx)
    }

    public var size: Int {
        guard ctx.ctx != nil else { return 0 }
        return mlx_distributed_group_size(ctx)
    }

    public var isValid: Bool { ctx.ctx != nil }

    init(ctx: mlx_distributed_group) {
        self.ctx = ctx
    }

    deinit {
        // mlx_distributed_group is a value type with ARC-managed ctx;
        // no explicit free needed — the Cmlx layer handles it
    }
}

/// Swift wrappers for MLX distributed operations
public enum MLXDistributedWrapper {

    /// Check if a distributed backend is available
    public static func isBackendAvailable(_ backend: String) -> Bool {
        return backend.withCString { ptr in
            mlx_distributed_is_available(ptr)
        }
    }

    /// Select best available backend: JACCL (RDMA) if available, else ring (TCP)
    public static func bestAvailableBackend() -> String {
        if isBackendAvailable("jaccl") {
            return "jaccl"
        }
        return "ring"
    }

    /// Initialize a distributed group with the given backend
    public static func initialize(strict: Bool = false, backend: String? = nil) -> DistributedGroup {
        let bk = backend ?? bestAvailableBackend()
        let group = bk.withCString { bkPtr in
            mlx_distributed_init(strict, bkPtr)
        }
        return DistributedGroup(ctx: group)
    }

    /// Send an array to a destination rank
    public static func send(_ array: MLXArray, to dst: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_send(&result, array.ctx, dst, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array from a source rank
    public static func recv(shape: [Int], dtype: Dtype = .float16, from src: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        let shapeInt32 = shape.map { Int32($0) }
        var dtypeC = dtype.cmlxDtype
        mlx_distributed_recv(&result, shapeInt32, shapeInt32.count, dtypeC, src, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array with the same shape/dtype as a reference array
    public static func recvLike(_ reference: MLXArray, from src: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_recv_like(&result, reference.ctx, src, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// All-gather: collect arrays from all ranks
    public static func allGather(_ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_gather(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// All-sum: sum-reduce across all ranks
    public static func allSum(_ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_sum(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }
}
```

- [ ] **Step 4: Build to check compilation**

Run: `./build.sh -c debug 2>&1 | tail -5`
Expected: Build succeeds. The wrappers compile against Cmlx headers.

- [ ] **Step 5: Run tests**

Run: `swift test --filter DistributedGroupTests 2>&1 | tail -5`
Expected: PASS (availability check returns false on single machine, uninitialized group returns -1/0)

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedGroup.swift Tests/NovaMLXDistributedTests/DistributedTypesTests.swift
git commit -m "feat(distributed): add MLX distributed Swift wrappers for send/recv/allGather"
```

---

## Task 4: ModelAnalyzer — Safetensors Header Parsing

**Files:**
- Create: `Sources/NovaMLXDistributed/ModelAnalyzer.swift`
- Create: `Tests/NovaMLXDistributedTests/ModelAnalyzerTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// Tests/NovaMLXDistributedTests/ModelAnalyzerTests.swift
import Testing
import Foundation
@testable import NovaMLXDistributed

@Suite("Model Analyzer")
struct ModelAnalyzerTests {

    @Test("Parse safetensors header returns layer map")
    func parseSafetensorsHeader() async throws {
        // Create a minimal safetensors file for testing
        let headerJSON: [String: Any] = [
            "__metadata__": ["model_type": "qwen2"],
            "model.layers.0.self_attn.q_proj.weight": [
                "dtype": "F16",
                "shape": [4096, 4096],
                "data_offsets": [0, 33554432]
            ],
            "model.layers.0.self_attn.k_proj.weight": [
                "dtype": "F16",
                "shape": [1024, 4096],
                "data_offsets": [33554432, 36962304]
            ],
            "model.layers.1.self_attn.q_proj.weight": [
                "dtype": "F16",
                "shape": [4096, 4096],
                "data_offsets": [36962304, 70516736]
            ],
            "model.embed_tokens.weight": [
                "dtype": "F16",
                "shape": [152064, 4096],
                "data_offsets": [70516736, 195035136]
            ],
            "lm_head.weight": [
                "dtype": "F16",
                "shape": [152064, 4096],
                "data_offsets": [195035136, 319553536]
            ]
        ]
        let headerData = try JSONSerialization.data(withJSONObject: headerJSON)
        let headerLen = UInt64(headerData.count).littleEndian
        var lenBytes = headerLen
        let tmpFile = FileManager.default.temporaryDirectory.appendingPathComponent("test_model.safetensors")
        let output = NSMutableData()
        output.append(&lenBytes, length: 8)
        output.append(headerData)
        // Padding data for the actual tensor content
        output.append(Data(count: 319553536))
        try output.write(to: tmpFile, options: .atomic)

        defer { try? FileManager.default.removeItem(at: tmpFile) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: tmpFile.deletingLastPathComponent().path)
        #expect(profiles.count >= 2)  // At least embedding + 2 transformer layers + output
        // Transformer layers should be identified
        let transformerLayers = profiles.filter { $0.layerType == .transformer }
        #expect(transformerLayers.count == 2)
    }

    @Test("ShardPlan respects memory ratios for unequal nodes")
    func shardPlanMemoryRatio() {
        let profiles = (0..<40).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let nodes = [
            NodeSpec(nodeId: "big", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "big.local", port: 6591),
            NodeSpec(nodeId: "small", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 0.6, hostname: "small.local", port: 6591),
        ]
        let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .minNodes)
        #expect(plan.assignments.count == 2)
        // big node (128/192 = 66%) should get ~26-27 layers
        let bigAssignment = plan.assignments.first { $0.nodeId == "big" }
        #expect(bigAssignment!.layerCount > 20)
        #expect(bigAssignment!.layerCount < 30)
        let total = plan.assignments.reduce(0) { $0 + $1.layerCount }
        #expect(total == 40)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelAnalyzerTests 2>&1 | tail -5`
Expected: FAIL — ModelAnalyzer not defined

- [ ] **Step 3: Implement ModelAnalyzer**

```swift
// Sources/NovaMLXDistributed/ModelAnalyzer.swift
import Foundation

public final class ModelAnalyzer: @unchecked Sendable {
    public static let shared = ModelAnalyzer()

    private init() {}

    /// Parse all safetensors files in a model directory and return per-layer profiles
    public func analyze(modelPath: String) async throws -> [LayerProfile] {
        let dirURL = URL(fileURLWithPath: modelPath)
        let files = try FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }

        var allTensorInfo: [(name: String, shape: [Int], byteSize: Int)] = []

        for file in files {
            let tensorMap = try readSafetensorsTensorMap(url: file)
            for (name, info) in tensorMap {
                allTensorInfo.append((name, info.shape, info.byteSize))
            }
        }

        return buildLayerProfiles(from: allTensorInfo)
    }

    // MARK: - Private

    private struct TensorInfo {
        let shape: [Int]
        let byteSize: Int
    }

    private func readSafetensorsTensorMap(url: URL) throws -> [String: TensorInfo] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else {
            throw ModelAnalyzerError.invalidHeader
        }
        let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        guard headerLen > 0, headerLen < 64 * 1024 * 1024 else {
            throw ModelAnalyzerError.invalidHeader
        }
        guard let jsonData = try handle.read(upToCount: Int(headerLen)),
              jsonData.count == Int(headerLen) else {
            throw ModelAnalyzerError.invalidHeader
        }

        guard let parsed = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw ModelAnalyzerError.invalidJSON
        }

        var result: [String: TensorInfo] = [:]
        for (key, value) in parsed where key != "__metadata__" {
            guard let info = value as? [String: Any],
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }
            let byteSize = offsets[1] - offsets[0]
            result[key] = TensorInfo(shape: shape, byteSize: byteSize)
        }
        return result
    }

    private func buildLayerProfiles(from tensors: [(name: String, shape: [Int], byteSize: Int)]) -> [LayerProfile] {
        // Group tensors by layer index
        var layerMap: [Int: UInt64] = [:]
        var layerTypes: [Int: LayerType] = [:]
        var maxLayer = -1
        var hasEmbedding = false
        var hasOutput = false

        for tensor in tensors {
            if tensor.name.contains("embed_tokens") || tensor.name.contains("wte") || tensor.name.contains("embed") {
                hasEmbedding = true
                // Count embedding as layer 0's memory
                layerMap[-1, default: 0] += UInt64(tensor.byteSize)
                layerTypes[-1] = .embedding
            } else if tensor.name.contains("lm_head") || tensor.name.contains("output") && !tensor.name.contains("attention") {
                hasOutput = true
                layerMap[-2, default: 0] += UInt64(tensor.byteSize)
                layerTypes[-2] = .output
            } else if let match = tensor.name.range(of: "layers\\.(\\d+)", options: .regularExpression),
                      let layerIdx = Int(tensor.name[match].replacingOccurrences(of: "layers.", with: "")) {
                layerMap[layerIdx, default: 0] += UInt64(tensor.byteSize)
                let isMoE = tensor.name.contains("gate_proj") || tensor.name.contains("experts")
                layerTypes[layerIdx] = isMoE ? .moe : .transformer
                maxLayer = max(maxLayer, layerIdx)
            }
        }

        var profiles: [LayerProfile] = []

        // Embedding as layer 0
        if hasEmbedding, let mem = layerMap[-1] {
            profiles.append(LayerProfile(layerIndex: 0, parameterCount: mem / 2, estimatedMemoryBytes: mem, layerType: .embedding))
        }

        // Transformer layers
        for i in 0...maxLayer {
            let mem = layerMap[i, default: 0]
            let type = layerTypes[i, default: .transformer]
            profiles.append(LayerProfile(layerIndex: profiles.count, parameterCount: mem / 2, estimatedMemoryBytes: mem, layerType: type))
        }

        // Output layer
        if hasOutput, let mem = layerMap[-2] {
            profiles.append(LayerProfile(layerIndex: profiles.count, parameterCount: mem / 2, estimatedMemoryBytes: mem, layerType: .output))
        }

        return profiles
    }
}

public enum ModelAnalyzerError: Error, LocalizedError {
    case invalidHeader
    case invalidJSON
    case fileNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidHeader: "Invalid safetensors header"
        case .invalidJSON: "Invalid JSON in safetensors header"
        case .fileNotFound(let path): "Model file not found: \(path)"
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `swift test --filter ModelAnalyzerTests 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/ModelAnalyzer.swift Tests/NovaMLXDistributedTests/ModelAnalyzerTests.swift
git commit -m "feat(distributed): add ModelAnalyzer with safetensors header parsing and shard plan computation"
```

---

## Task 5: ShardEngine — ComputePolicy and Pipeline

**Files:**
- Create: `Sources/NovaMLXDistributed/ShardEngine.swift`
- Create: `Tests/NovaMLXDistributedTests/ShardEngineTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// Tests/NovaMLXDistributedTests/ShardEngineTests.swift
import Testing
@testable import NovaMLXDistributed

@Suite("Shard Engine")
struct ShardEngineTests {

    @Test("FitInMemoryPolicy initializes with valid assignment")
    func fitInMemoryPolicyInit() {
        let assignment = ShardAssignment(nodeId: "test", startLayer: 0, endLayer: 20, memoryEstimate: 80_000_000)
        let policy = FitInMemoryPolicy(assignment: assignment)
        #expect(policy.isReady == false) // Not ready until weights bound
    }

    @Test("ShardEngine builds with uninitialized group")
    func shardEngineUninitializedGroup() {
        let assignment = ShardAssignment(nodeId: "test", startLayer: 0, endLayer: 10, memoryEstimate: 40_000_000)
        let group = DistributedGroup.uninitialized
        let engine = ShardEngine(group: group, assignment: assignment)
        #expect(engine.isLastShard == false)  // size is 0, so rank 0 != size-1
    }

    @Test("ShardEngine detects last shard correctly")
    func lastShardDetection() {
        // Can't create a real group, but test the logic with uninitialized
        let assignment = ShardAssignment(nodeId: "test", startLayer: 20, endLayer: 40, memoryEstimate: 80_000_000)
        let group = DistributedGroup.uninitialized
        let engine = ShardEngine(group: group, assignment: assignment)
        // With uninitialized group (size=0, rank=-1), isLastShard should be false
        #expect(engine.isLastShard == false)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ShardEngineTests 2>&1 | tail -5`
Expected: FAIL

- [ ] **Step 3: Implement ShardEngine**

```swift
// Sources/NovaMLXDistributed/ShardEngine.swift
import Foundation
import MLX

// MARK: - Compute Policy Protocol

public protocol ComputePolicy: Sendable {
    var isReady: Bool { get }
    func bindWeights() async throws
    func compute(input: MLXArray, cache: inout [Any]) throws -> MLXArray
    func releaseWeights()
}

// MARK: - Fit In Memory Policy

public final class FitInMemoryPolicy: ComputePolicy, @unchecked Sendable {
    private let assignment: ShardAssignment
    public private(set) var isReady: Bool = false

    public init(assignment: ShardAssignment) {
        self.assignment = assignment
    }

    public func bindWeights() async throws {
        // Weight binding will be implemented when integrating with MLXEngine's model loading
        isReady = true
    }

    public func compute(input: MLXArray, cache: inout [Any]) throws -> MLXArray {
        guard isReady else { throw ShardEngineError.notReady }
        // Actual forward pass will delegate to the model's layer-by-layer eval
        return input
    }

    public func releaseWeights() {
        isReady = false
    }
}

// MARK: - Shard Engine

public enum ShardEngineError: Error, LocalizedError {
    case notReady
    case sendFailed(String)
    case recvFailed(String)
    case samplingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notReady: "ShardEngine not ready — weights not bound"
        case .sendFailed(let msg): "Distributed send failed: \(msg)"
        case .recvFailed(let msg): "Distributed recv failed: \(msg)"
        case .samplingFailed(let msg): "Sampling failed: \(msg)"
        }
    }
}

public final class ShardEngine: @unchecked Sendable {
    public let group: DistributedGroup
    public let assignment: ShardAssignment
    public let policy: ComputePolicy

    public init(group: DistributedGroup, assignment: ShardAssignment, policy: ComputePolicy? = nil) {
        self.group = group
        self.assignment = assignment
        self.policy = policy ?? FitInMemoryPolicy(assignment: assignment)
    }

    /// Whether this shard is the last in the pipeline (responsible for sampling)
    public var isLastShard: Bool {
        guard group.size > 0 else { return false }
        return group.rank == group.size - 1
    }

    /// Whether this shard is the first in the pipeline (receives raw tokens)
    public var isFirstShard: Bool {
        guard group.size > 0 else { return false }
        return group.rank == 0
    }

    /// Run prefill: process tokens through assigned layers, send/recv activations
    public func prefill(tokens: MLXArray) async throws -> MLXArray {
        var activation = tokens

        // Non-first shards: receive activation from previous shard
        if !isFirstShard {
            activation = MLXDistributedWrapper.recvLike(activation, from: group.rank - 1, group: group)
        }

        // Forward through assigned layers
        var cache: [Any] = []
        activation = try policy.compute(input: activation, cache: &cache)

        // Non-last shards: send activation to next shard
        if !isLastShard {
            _ = MLXDistributedWrapper.send(activation, to: group.rank + 1, group: group)
            return activation
        }

        // Last shard: sample and return token ID
        return activation
    }

    /// Run decode: same pipeline but with single token input
    public func decode(token: MLXArray) async throws -> MLXArray {
        return try await prefill(tokens: token)
    }
}
```

- [ ] **Step 4: Run tests**

Run: `swift test --filter ShardEngineTests 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/ShardEngine.swift Tests/NovaMLXDistributedTests/ShardEngineTests.swift
git commit -m "feat(distributed): add ShardEngine with ComputePolicy protocol and pipeline logic"
```

---

## Task 6: ClusterManager — Bonjour Discovery and Worker Registration

**Files:**
- Create: `Sources/NovaMLXDistributed/ClusterManager.swift`

- [ ] **Step 1: Implement ClusterManager**

```swift
// Sources/NovaMLXDistributed/ClusterManager.swift
import Foundation
import Network

public enum ClusterError: Error, LocalizedError {
    case noWorkersRegistered
    case workerNotFound(String)
    case alreadyInitialized
    case notCoordinator

    public var errorDescription: String? {
        switch self {
        case .noWorkersRegistered: "No workers registered in cluster"
        case .workerNotFound(let id): "Worker not found: \(id)"
        case .alreadyInitialized: "Cluster already initialized"
        case .notCoordinator: "This node is not the coordinator"
        }
    }
}

public struct WorkerInfo: Codable, Sendable {
    public let nodeId: String
    public let spec: NodeSpec
    public var status: WorkerStatus
    public let registeredAt: Date
    public var lastHeartbeat: Date

    public enum WorkerStatus: String, Codable, Sendable {
        case registering
        case ready
        case loading
        case active
        case syncing
        case disconnected
        case failed
    }
}

public final class ClusterManager: @unchecked Sendable {
    public static let shared = ClusterManager()

    public private(set) var config: ClusterConfig?
    public private(set) var workers: [String: WorkerInfo] = [:]
    public private(set) var isRunning: Bool = false

    private var bonjourService: NetService?
    private var bonjourBrowser: NWBrowser?
    private var heartbeatTimer: Timer?

    private let queue = DispatchQueue(label: "com.novamlx.cluster-manager", qos: .userInitiated)

    private init() {}

    // MARK: - Coordinator Lifecycle

    public func startAsCoordinator(config: ClusterConfig) throws {
        guard !isRunning else { throw ClusterError.alreadyInitialized }
        self.config = config
        self.isRunning = true
        advertiseBonjour()
        startHeartbeatMonitoring()
    }

    public func stop() {
        bonjourService?.stop()
        bonjourBrowser?.cancel()
        heartbeatTimer?.invalidate()
        bonjourService = nil
        bonjourBrowser = nil
        heartbeatTimer = nil
        workers.removeAll()
        isRunning = false
    }

    // MARK: - Worker Registration

    public func registerWorker(spec: NodeSpec) {
        queue.sync {
            let worker = WorkerInfo(
                nodeId: spec.nodeId,
                spec: spec,
                status: .registering,
                registeredAt: Date(),
                lastHeartbeat: Date()
            )
            workers[spec.nodeId] = worker
        }
    }

    public func updateHeartbeat(nodeId: String) {
        queue.sync {
            if var worker = workers[nodeId] {
                worker.lastHeartbeat = Date()
                if worker.status == .disconnected {
                    worker.status = .ready
                }
                workers[nodeId] = worker
            }
        }
    }

    public func removeWorker(nodeId: String) {
        queue.sync {
            workers.removeValue(forKey: nodeId)
        }
    }

    // MARK: - Cluster Status

    public var activeWorkers: [WorkerInfo] {
        queue.sync {
            workers.values.filter { $0.status != .disconnected && $0.status != .failed }
        }
    }

    public var spareWorkers: [WorkerInfo] {
        queue.sync {
            workers.values.filter { $0.status == .ready }
        }
    }

    // MARK: - Bonjour

    private func advertiseBonjour() {
        let service = NetService(domain: "local.", type: "_novamlx._tcp.", name: "NovaMLX-Coordinator", port: Int32(config?.coordinatorPort ?? 6591))
        service.publish(options: [.listenForConnections])
        self.bonjourService = service
    }

    private func startHeartbeatMonitoring() {
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.checkWorkerHealth()
        }
    }

    private func checkWorkerHealth() {
        let now = Date()
        let timeout: TimeInterval = 30.0

        for (id, worker) in workers {
            if now.timeIntervalSince(worker.lastHeartbeat) > timeout && worker.status != .disconnected {
                queue.sync {
                    workers[id]?.status = .disconnected
                }
                // L1: trigger transient disconnect handling
                handleWorkerDisconnected(nodeId: id)
            }
        }
    }

    // MARK: - Fault Recovery (L1 hook)

    public var onWorkerDisconnected: ((String) -> Void)?

    private func handleWorkerDisconnected(nodeId: String) {
        onWorkerDisconnected?(nodeId)
    }

    // MARK: - Discovery Debug

    public func discoveryDebugInfo() -> [String: Any] {
        var info: [String: Any] = [:]
        info["isRunning"] = isRunning
        info["role"] = config?.role.rawValue ?? "none"
        info["registeredWorkers"] = workers.count
        info["workers"] = workers.mapValues { [
            "status": $0.status.rawValue,
            "lastHeartbeat": $0.lastHeartbeat,
            "memory": $0.spec.totalMemoryBytes,
        ] }
        return info
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `./build.sh -c debug 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXDistributed/ClusterManager.swift
git commit -m "feat(distributed): add ClusterManager with Bonjour discovery and worker registration"
```

---

## Task 7: WorkerService — Discovery and Heartbeat

**Files:**
- Create: `Sources/NovaMLXDistributed/WorkerService.swift`

- [ ] **Step 1: Implement WorkerService**

```swift
// Sources/NovaMLXDistributed/WorkerService.swift
import Foundation
import Network

public final class WorkerService: @unchecked Sendable {
    public static let shared = WorkerService()

    public private(set) var coordinatorHost: String?
    public private(set) var coordinatorPort: Int?
    public private(set) var isRegistered: Bool = false
    public private(set) var isRunning: Bool = false

    private var heartbeatTimer: Timer?
    private var bonjourBrowser: NWBrowser?
    private let queue = DispatchQueue(label: "com.novamlx.worker-service", qos: .userInitiated)

    private init() {}

    // MARK: - Lifecycle

    public func start(config: ClusterConfig) {
        coordinatorHost = config.coordinatorHost
        coordinatorPort = config.coordinatorPort
        isRunning = true

        // If coordinator host is explicitly configured, connect directly
        if let host = config.coordinatorHost {
            registerWithCoordinator(host: host, port: config.coordinatorPort)
        } else {
            // Otherwise, discover via Bonjour
            discoverCoordinator()
        }

        startHeartbeat()
    }

    public func stop() {
        heartbeatTimer?.invalidate()
        bonjourBrowser?.cancel()
        heartbeatTimer = nil
        bonjourBrowser = nil
        isRunning = false
        isRegistered = false
    }

    // MARK: - Registration

    private func registerWithCoordinator(host: String, port: Int) {
        // Send registration to coordinator via HTTP
        let spec = collectLocalSpec()
        guard let url = URL(string: "http://\(host):\(port)/admin/api/cluster/workers/register") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(spec)

        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                self?.isRegistered = true
            }
        }.resume()
    }

    // MARK: - Bonjour Discovery

    private func discoverCoordinator() {
        let parameters = NWParameters()
        parameters.includePeerToPeer = true

        let browser = NWBrowser(for: .bonjour(type: "_novamlx._tcp.", domain: "local."), using: parameters)
        browser.stateUpdateHandler = { state in
            if case .ready = state {
                // Browser ready
            }
        }
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            for result in results {
                if case .service(let service) = result.endpoint {
                    self?.handleDiscoveredService(service)
                }
            }
        }
        browser.start(queue: queue)
        self.bonjourBrowser = browser
    }

    private func handleDiscoveredService(_ service: NWEndpoint.Service) {
        // Resolve the service and register
        let netService = NetService(name: service.name, type: service.type, domain: service.domain)
        netService.resolve(withTimeout: 5.0)
        if let host = netService.hostName, let port = netService.port {
            registerWithCoordinator(host: host, port: Int(port))
        }
    }

    // MARK: - Heartbeat

    private func startHeartbeat() {
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.sendHeartbeat()
        }
    }

    private func sendHeartbeat() {
        guard let host = coordinatorHost, let port = coordinatorPort else { return }
        guard let url = URL(string: "http://\(host):\(port)/admin/api/cluster/workers/heartbeat") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let body = ["nodeId": collectLocalSpec().nodeId]
        request.httpBody = try? JSONEncoder().encode(body)

        URLSession.shared.dataTask(with: request).resume()
    }

    // MARK: - Local Spec

    private func collectLocalSpec() -> NodeSpec {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let hostname = ProcessInfo.processInfo.hostName
        let nodeId = "\(hostname)-\(totalMemory)"

        return NodeSpec(
            nodeId: nodeId,
            totalMemoryBytes: totalMemory,
            computeCapability: 1.0,
            hostname: hostname,
            port: coordinatorPort ?? 6591
        )
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `./build.sh -c debug 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXDistributed/WorkerService.swift
git commit -m "feat(distributed): add WorkerService with Bonjour discovery and heartbeat"
```

---

## Task 8: WeightDistributor — Model File Sync

**Files:**
- Create: `Sources/NovaMLXDistributed/WeightDistributor.swift`

- [ ] **Step 1: Implement WeightDistributor**

```swift
// Sources/NovaMLXDistributed/WeightDistributor.swift
import Foundation

public enum WeightDistributorError: Error, LocalizedError {
    case modelNotFound(String)
    case downloadFailed(String)
    case coordinatorUnavailable

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path): "Model not found at: \(path)"
        case .downloadFailed(let msg): "Model download failed: \(msg)"
        case .coordinatorUnavailable: "Coordinator unavailable for model download"
        }
    }
}

public final class WeightDistributor: @unchecked Sendable {
    public static let shared = WeightDistributor()

    public private(set) var activeDownloads: [String: DownloadProgress] = [:]

    private let queue = DispatchQueue(label: "com.novamlx.weight-distributor", qos: .utility)

    public struct DownloadProgress: Codable, Sendable {
        public let modelId: String
        public var bytesDownloaded: UInt64
        public var totalBytes: UInt64
        public var isComplete: Bool

        public var fraction: Double {
            guard totalBytes > 0 else { return 0 }
            return Double(bytesDownloaded) / Double(totalBytes)
        }
    }

    private init() {}

    /// Ensure model files are available locally. Returns local path.
    /// - Path A: Files exist locally -> return immediately
    /// - Path B: Files missing -> download from coordinator
    public func ensureModelAvailable(modelId: String, expectedPath: String, coordinatorHost: String, coordinatorPort: Int) async throws -> String {
        let localURL = URL(fileURLWithPath: expectedPath)

        // Path A: local files exist
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL.path
        }

        // Path B: download from coordinator
        return try await downloadFromCoordinator(modelId: modelId, to: localURL, host: coordinatorHost, port: coordinatorPort)
    }

    private func downloadFromCoordinator(modelId: String, to destination: URL, host: String, port: Int) async throws -> String {
        let urlString = "http://\(host):\(port)/admin/api/cluster/models/\(modelId)/download"
        guard let url = URL(string: urlString) else { throw WeightDistributorError.coordinatorUnavailable }

        var progress = DownloadProgress(modelId: modelId, bytesDownloaded: 0, totalBytes: 0, isComplete: false)
        activeDownloads[modelId] = progress

        defer {
            queue.sync {
                activeDownloads.removeValue(forKey: modelId)
            }
        }

        // Create parent directory
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        // Download with progress tracking
        let (asyncBytes, response) = try await URLSession.shared.bytes(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
              let totalBytes = UInt64(contentLength) else {
            throw WeightDistributorError.downloadFailed("Invalid response from coordinator")
        }

        progress.totalBytes = totalBytes

        let tmpURL = destination.appendingPathExtension("download")
        let fileManager = FileManager.default
        fileManager.createFile(atPath: tmpURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tmpURL)

        var buffer = Data()
        buffer.reserveCapacity(1024 * 1024)  // 1MB chunks

        for try await byte in asyncBytes {
            buffer.append(byte)
            if buffer.count >= 1024 * 1024 {
                try handle.write(contentsOf: buffer)
                progress.bytesDownloaded += UInt64(buffer.count)
                queue.sync { activeDownloads[modelId] = progress }
                buffer.removeAll(keepingCapacity: true)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            progress.bytesDownloaded += UInt64(buffer.count)
        }
        try handle.close()

        // Atomic move
        try fileManager.moveItem(at: tmpURL, to: destination)
        progress.isComplete = true

        return destination.path
    }

    /// Get download progress for a model
    public func syncStatus(modelId: String) -> DownloadProgress? {
        queue.sync {
            activeDownloads[modelId]
        }
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `./build.sh -c debug 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXDistributed/WeightDistributor.swift
git commit -m "feat(distributed): add WeightDistributor with local check and auto-download"
```

---

## Task 9: Fault Recovery — L1, L2, L3a, L3b

**Files:**
- Create: `Sources/NovaMLXDistributed/FaultRecovery.swift`
- Create: `Tests/NovaMLXDistributedTests/FaultRecoveryTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// Tests/NovaMLXDistributedTests/FaultRecoveryTests.swift
import Testing
@testable import NovaMLXDistributed

@Suite("Fault Recovery")
struct FaultRecoveryTests {

    @Test("L1 grace period tracks disconnect time")
    func l1GracePeriod() {
        let recovery = FaultRecoveryManager()
        let now = Date()
        recovery.trackDisconnect(nodeId: "worker-1", at: now)
        #expect(recovery.isInGracePeriod(nodeId: "worker-1", at: now.addingTimeInterval(15)))
        #expect(!recovery.isInGracePeriod(nodeId: "worker-1", at: now.addingTimeInterval(31)))
    }

    @Test("L2 selects spare worker with matching memory")
    func l2SpareSelection() {
        let recovery = FaultRecoveryManager()
        let failedAssignment = ShardAssignment(nodeId: "failed", startLayer: 20, endLayer: 40, memoryEstimate: 40_000_000)
        let spares = [
            NodeSpec(nodeId: "spare-1", totalMemoryBytes: 64 * 1024 * 1024 * 1024, computeCapability: 0.8, hostname: "spare1.local", port: 6591),
            NodeSpec(nodeId: "spare-2", totalMemoryBytes: 32 * 1024 * 1024 * 1024, computeCapability: 0.5, hostname: "spare2.local", port: 6591),
        ]
        let selected = recovery.selectSpareFor(failedAssignment: failedAssignment, spares: spares)
        #expect(selected?.nodeId == "spare-1")  // Best capacity match
    }

    @Test("L3a computes new shard plan for remaining nodes")
    func l3aReshardPlan() {
        let remainingNodes = [
            NodeSpec(nodeId: "node-a", totalMemoryBytes: 128 * 1024 * 1024 * 1024, computeCapability: 1.0, hostname: "a.local", port: 6591),
        ]
        let profiles = (0..<40).map { i in
            LayerProfile(layerIndex: i, parameterCount: 1_000_000, estimatedMemoryBytes: 4_000_000, layerType: .transformer)
        }
        let canReshard = FaultRecoveryManager.canReshard(
            remainingNodes: remainingNodes,
            totalModelMemory: profiles.reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes },
            overheadFactor: 1.3
        )
        // 128GB should be enough for 160MB model * 1.3 = 208MB
        #expect(canReshard)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter FaultRecoveryTests 2>&1 | tail -5`
Expected: FAIL

- [ ] **Step 3: Implement FaultRecovery**

```swift
// Sources/NovaMLXDistributed/FaultRecovery.swift
import Foundation

public final class FaultRecoveryManager: @unchecked Sendable {
    public static let shared = FaultRecoveryManager()

    private var disconnectTimes: [String: Date] = [:]
    private let queue = DispatchQueue(label: "com.novamlx.fault-recovery")
    private let gracePeriodSeconds: TimeInterval = 30.0

    public init() {}

    // MARK: - L1: Transient Disconnect

    public func trackDisconnect(nodeId: String, at time: Date = Date()) {
        queue.sync {
            disconnectTimes[nodeId] = time
        }
    }

    public func isInGracePeriod(nodeId: String, at time: Date = Date()) -> Bool {
        queue.sync {
            guard let disconnectTime = disconnectTimes[nodeId] else { return false }
            return time.timeIntervalSince(disconnectTime) <= gracePeriodSeconds
        }
    }

    public func clearDisconnect(nodeId: String) {
        queue.sync {
            disconnectTimes.removeValue(forKey: nodeId)
        }
    }

    // MARK: - L2: Spare Node Swap

    public func selectSpareFor(failedAssignment: ShardAssignment, spares: [NodeSpec]) -> NodeSpec? {
        // Select the spare with the most capacity that can handle the failed assignment's layers
        let candidates = spares.filter { $0.totalMemoryBytes >= failedAssignment.memoryEstimate }
        return candidates.max(by: { $0.totalMemoryBytes < $1.totalMemoryBytes })
    }

    // MARK: - L3a: Auto-Reshard

    public static func canReshard(remainingNodes: [NodeSpec], totalModelMemory: UInt64, overheadFactor: Double = 1.3) -> Bool {
        let totalCapacity = remainingNodes.reduce(UInt64(0)) { $0 + $1.totalMemoryBytes }
        let requiredMemory = UInt64(Double(totalModelMemory) * overheadFactor)
        return totalCapacity >= requiredMemory
    }

    public func computeReshardPlan(remainingNodes: [NodeSpec], profiles: [LayerProfile]) -> ShardPlan? {
        let totalMemory = profiles.reduce(UInt64(0)) { $0 + $1.estimatedMemoryBytes }
        guard Self.canReshard(remainingNodes: remainingNodes, totalModelMemory: totalMemory) else {
            return nil
        }
        return ShardPlan(profiles: profiles, nodes: remainingNodes, strategy: .minNodes)
    }

    // MARK: - L3b: Hard Fail / Manual Reshard

    public func handleHardFail(modelId: String, reason: String) {
        // Notify through ClusterManager's admin status
        // This will be wired up when admin API routes are added
    }
}
```

- [ ] **Step 4: Run tests**

Run: `swift test --filter FaultRecoveryTests 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/FaultRecovery.swift Tests/NovaMLXDistributedTests/FaultRecoveryTests.swift
git commit -m "feat(distributed): add fault recovery L1/L2/L3a/L3b"
```

---

## Task 10: Extend ServerConfig with Cluster Settings

**Files:**
- Modify: `Sources/NovaMLXCore/Types.swift`
- Modify: `Sources/NovaMLXCore/Configuration.swift`

- [ ] **Step 1: Add cluster field to ServerConfig**

In `Sources/NovaMLXCore/Types.swift`, add to the `ServerConfig` struct (after the `autoLoad` property):

```swift
public let cluster: ClusterSettings?

public struct ClusterSettings: Codable, Sendable, Equatable {
    public let role: String
    public let coordinatorHost: String?
    public let coordinatorPort: Int?
    public let strategy: String?

    public init(role: String, coordinatorHost: String? = nil, coordinatorPort: Int? = nil, strategy: String? = nil) {
        self.role = role
        self.coordinatorHost = coordinatorHost
        self.coordinatorPort = coordinatorPort
        self.strategy = strategy
    }
}
```

Add `cluster` to `CodingKeys`, the `init` (default `nil`), and `init(from decoder:)` with `decodeIfPresent`.

- [ ] **Step 2: Update Configuration.swift to load/save cluster settings**

In `Sources/NovaMLXCore/Configuration.swift`, the `ServerConfig` already contains the cluster field via Step 1. No separate `PersistedConfig` changes needed since `ClusterSettings` is nested inside `ServerConfig`.

- [ ] **Step 3: Build and verify existing tests still pass**

Run: `./build.sh -c debug 2>&1 | tail -5`
Run: `swift test --filter NovaMLXCoreTests 2>&1 | tail -5`
Expected: All builds and tests pass. Existing configs without `cluster` field decode correctly (it's optional).

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXCore/Types.swift Sources/NovaMLXCore/Configuration.swift
git commit -m "feat(distributed): add cluster settings to ServerConfig"
```

---

## Task 11: Admin API Cluster Routes

**Files:**
- Create: `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift`
- Modify: `Sources/NovaMLXAPI/APIServer.swift`

- [ ] **Step 1: Implement ClusterAdminRoutes**

```swift
// Sources/NovaMLXDistributed/ClusterAdminRoutes.swift
import Foundation
import Hummingbird

public final class ClusterAdminRoutes: @unchecked Sendable {
    public static let shared = ClusterAdminRoutes()

    private init() {}

    // MARK: - Cluster Status

    public func clusterStatus() -> [String: Any] {
        let manager = ClusterManager.shared
        var status: [String: Any] = [:]
        status["isRunning"] = manager.isRunning
        status["config"] = manager.config.map { [
            "role": $0.role.rawValue,
            "coordinatorHost": $0.coordinatorHost,
            "coordinatorPort": $0.coordinatorPort,
            "strategy": $0.strategy.rawValue,
        ] }
        status["workers"] = manager.activeWorkers.map { [
            "nodeId": $0.nodeId,
            "status": $0.status.rawValue,
            "memory": $0.spec.totalMemoryBytes,
            "lastHeartbeat": ISO8601DateFormatter().string(from: $0.lastHeartbeat),
        ] }
        return status
    }

    // MARK: - Discovery Debug

    public func discoveryDebug() -> [String: Any] {
        return ClusterManager.shared.discoveryDebugInfo()
    }

    // MARK: - Sync Status

    public func modelSyncStatus(modelId: String) -> [String: Any] {
        guard let progress = WeightDistributor.shared.syncStatus(modelId: modelId) else {
            return ["status": "not_syncing"]
        }
        return [
            "status": progress.isComplete ? "complete" : "syncing",
            "bytesDownloaded": progress.bytesDownloaded,
            "totalBytes": progress.totalBytes,
            "progress": String(format: "%.1f%%", progress.fraction * 100),
        ]
    }

    // MARK: - Shard Plan

    public func currentShardPlan(modelId: String) -> [String: Any]? {
        // Will be populated when a model is loaded with cluster mode
        return nil
    }
}
```

- [ ] **Step 2: Add route registrations to APIServer.swift**

In `Sources/NovaMLXAPI/APIServer.swift`, in the admin router section (after existing admin routes), add:

```swift
// Cluster management routes
Get("/admin/api/cluster/status") { request, context in
    let status = ClusterAdminRoutes.shared.clusterStatus()
    return try await context.response(json: status)
}

Get("/admin/api/cluster/discovery-debug") { request, context in
    let debug = ClusterAdminRoutes.shared.discoveryDebug()
    return try await context.response(json: debug)
}

Get("/admin/api/models/{id}/cluster/sync-status") { request, context in
    guard let modelId = context.parameters.get("id") else {
        throw HTTPError(.badRequest, message: "Missing model ID")
    }
    let status = ClusterAdminRoutes.shared.modelSyncStatus(modelId: modelId)
    return try await context.response(json: status)
}
```

- [ ] **Step 3: Build and verify**

Run: `./build.sh -c debug 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXDistributed/ClusterAdminRoutes.swift Sources/NovaMLXAPI/APIServer.swift
git commit -m "feat(distributed): add cluster admin API routes"
```

---

## Task 12: InferenceService Cluster Dispatch

**Files:**
- Modify: `Sources/NovaMLXInference/InferenceService.swift`

- [ ] **Step 1: Add cluster dispatch check**

In `InferenceService.swift`, before the existing worker mode check (around line 121), add a cluster mode check:

```swift
// Cluster mode: forward to distributed inference pipeline
if clusterMode, let clusterEngine = self.clusterEngine {
    NovaMLXLog.info("[Route:\(reqTag)] -> DistributedShardEngine (model=\(resolvedId))")
    return try await clusterEngine.generate(finalRequest)
}
```

Add a property to `InferenceService`:

```swift
private let clusterMode: Bool
private weak var clusterEngine: AnyObject?  // Will be typed to DistributedInferenceCoordinator when wired

// In init:
self.clusterMode = workerMode && (config.cluster?.role == "coordinator")
```

This is a minimal integration point — the actual `DistributedInferenceCoordinator` will be wired in Task 13.

- [ ] **Step 2: Build and verify existing tests still pass**

Run: `./build.sh -c debug 2>&1 | tail -5`
Run: `swift test --filter NovaMLXInferenceTests 2>&1 | tail -5`
Expected: Build succeeds, tests pass

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXInference/InferenceService.swift
git commit -m "feat(distributed): add cluster dispatch hook in InferenceService"
```

---

## Task 13: Integration — Startup Wiring and Single-Machine Verification

**Files:**
- Modify: `Sources/NovaMLXApp/main.swift`

- [ ] **Step 1: Add cluster startup in AppDelegate**

In `main.swift`, after the existing worker mode setup (around line 104), add cluster initialization:

```swift
// Distributed inference initialization
if let clusterSettings = config.cluster {
    switch clusterSettings.role {
    case "coordinator":
        let clusterConfig = ClusterConfig(
            role: .coordinator,
            coordinatorHost: clusterSettings.coordinatorHost ?? "0.0.0.0",
            coordinatorPort: clusterSettings.coordinatorPort ?? 6591,
            strategy: ClusterStrategy(rawValue: clusterSettings.strategy ?? "minNodes") ?? .minNodes
        )
        try? ClusterManager.shared.startAsCoordinator(config: clusterConfig)
        NovaMLXLog.info("[Cluster] Started as coordinator on port \(clusterConfig.coordinatorPort)")
    case "worker":
        if let host = clusterSettings.coordinatorHost {
            let clusterConfig = ClusterConfig(
                role: .worker,
                coordinatorHost: host,
                coordinatorPort: clusterSettings.coordinatorPort ?? 6591,
                strategy: ClusterStrategy(rawValue: clusterSettings.strategy ?? "minNodes") ?? .minNodes
            )
            WorkerService.shared.start(config: clusterConfig)
            NovaMLXLog.info("[Cluster] Started as worker, coordinator at \(host)")
        }
    default:
        break
    }
}
```

- [ ] **Step 2: Verify single-machine operation is unaffected**

Build release and start without cluster config:

```bash
./build.sh -c release
killall NovaMLX; sleep 2; open dist/NovaMLX.app
```

Run a quick inference test to verify zero overhead:
```bash
curl -s http://127.0.0.1:6590/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<any-loaded-model>","messages":[{"role":"user","content":"hello"}],"max_tokens":5}'
```

Expected: Same response time as before cluster code was added.

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXApp/main.swift
git commit -m "feat(distributed): wire cluster startup in AppDelegate"
```

---

## Task 14: Full Test Suite Verification

**Files:**
- All files modified in previous tasks

- [ ] **Step 1: Run full test suite**

```bash
swift test 2>&1 | tail -20
```

Expected: All existing tests pass, plus new distributed tests.

- [ ] **Step 2: Run build.sh release to verify dist sync**

```bash
./build.sh -c release 2>&1 | tail -10
```

Expected: Build succeeds, dist/NovaMLX.app binaries updated.

- [ ] **Step 3: Final commit with any fixes**

If any tests failed or build issues found, fix and commit:

```bash
git add -A
git commit -m "fix(distributed): address test/build issues from integration"
```

---

## Spec Coverage Checklist

| Spec Section | Task(s) |
|---|---|
| 4.1 MLXDistributed Swift wrappers | Task 3 |
| 4.2 ClusterManager | Task 6 |
| 4.3 WorkerService | Task 7 |
| 4.4 ModelAnalyzer | Task 4 |
| 4.5 ShardEngine + ComputePolicy | Task 5 |
| 4.6 WeightDistributor | Task 8 |
| 5.1 Cluster Setup flow | Task 13 |
| 5.2-5.4 Inference data flow | Task 5 (engine), Task 12 (dispatch) |
| 6 Cluster Sizing Strategy | Task 2 (types), Task 4 (analyzer) |
| 7 Fault Recovery L1/L2/L3a/L3b | Task 9 |
| 8 API Surface | Task 11 |
| 9 Single-Machine Guarantee | Task 13 (startup), Task 14 (verification) |
| 2.1 Backend auto-selection (JACCL/RDMA) | Task 3 |
| 2.2 Thunderbolt-aware Bonjour | Task 6, Task 7 |
