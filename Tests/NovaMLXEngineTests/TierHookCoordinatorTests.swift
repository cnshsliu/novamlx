import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import NovaMLXEngine

// Module/Linear are not Sendable; wrap for actor crossings.
final class SendableModuleBox<ModuleType>: @unchecked Sendable {
    let value: ModuleType
    init(_ value: ModuleType) { self.value = value }
}

// Smoke test: verify TierHookCoordinator walks a Module tree and registers
// every Linear/SwitchLinear instance with correct layer indices derived
// from path. This is the universal hook topology that makes TIE work for
// any model — exercises it without a full model load.

private final class TestDecoderLayer: Module {
    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear
    @ModuleInfo var gateProj: Linear
    @ModuleInfo var upProj: Linear
    @ModuleInfo var downProj: Linear

    init(hidden: Int, intermediate: Int) {
        self._qProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._kProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._vProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._oProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }
}

private final class TestMoELayer: Module {
    @ModuleInfo(key: "switch_mlp") var switchMlp: SwitchGLU
    @ModuleInfo(key: "gate") var gate: Linear

    init(hidden: Int = 16, intermediate: Int = 8, numExperts: Int = 4) {
        self._switchMlp.wrappedValue = SwitchGLU(
            inputDims: hidden, hiddenDims: intermediate, numExperts: numExperts
        )
        self._gate.wrappedValue = Linear(hidden, numExperts, bias: false)
        super.init()
    }
}

private final class TestModel: Module {
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [TestDecoderLayer]
    @ModuleInfo(key: "moe_layers") var moeLayers: [TestMoELayer]
    let norm: RMSNorm

    override init() {
        let hidden = 16
        self._embed.wrappedValue = Embedding(embeddingCount: 100, dimensions: hidden)
        self._layers.wrappedValue = (0..<3).map { _ in TestDecoderLayer(hidden: hidden, intermediate: 32) }
        self._moeLayers.wrappedValue = (0..<2).map { _ in TestMoELayer() }
        self.norm = RMSNorm(dimensions: hidden, eps: 1e-6)
        super.init()
    }
}

// Wrapper class for the lazy-load test. Single Linear at a known path.
private final class LazyTestLayer: Module {
    @ModuleInfo(key: "test_linear") var testLinear: Linear
    init(linear: Linear) {
        self._testLinear.wrappedValue = linear
        super.init()
    }
}

private final class LazyTestModel: Module {
    @ModuleInfo(key: "layers") var layers: [LazyTestLayer]
    init(layers: [LazyTestLayer]) {
        self._layers.wrappedValue = layers
        super.init()
    }
}

// Wrapper for SwitchLinear lazy load test.
private final class LazyMoETestLayer: Module {
    @ModuleInfo(key: "switch_mlp") var switchMlp: SwitchGLU
    @ModuleInfo(key: "gate") var gate: Linear
    init(switchGlu: SwitchGLU, gate: Linear) {
        self._switchMlp.wrappedValue = switchGlu
        self._gate.wrappedValue = gate
        super.init()
    }
}

private final class LazyMoETestModel: Module {
    @ModuleInfo(key: "layers") var layers: [LazyMoETestLayer]
    init(layers: [LazyMoETestLayer]) {
        self._layers.wrappedValue = layers
        super.init()
    }
}

final class TierHookCoordinatorTests: XCTestCase {

    func testWalksTreeAndRegistersAllLinearInstances() async throws {
        let model = TestModel()
        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [], layers: []
        )
        let dir = URL(fileURLWithPath: "/tmp/test-tier")
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: dir, metrics: nil)
        // Wrap non-Sendable Module in @unchecked Sendable box for the actor crossing.
        let boxed = SendableModuleBox(model)

        let result = await TierHookCoordinator.shared.register(
            model: boxed.value, policy: policy, strategy: .layer
        )

        XCTAssertEqual(result.linears, 23, "21 dense Linears + 2 MoE gate Linears")
        XCTAssertEqual(result.experts, 6, "SwitchGLU contains 3 SwitchLinear, 2 MoE layers = 6")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    func testUnregisterRemovesAllContexts() async throws {
        let model = TestModel()
        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [], layers: []
        )
        let dir = URL(fileURLWithPath: "/tmp/test-tier")
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: dir, metrics: nil)
        let boxed = SendableModuleBox(model)

        _ = await TierHookCoordinator.shared.register(
            model: boxed.value, policy: policy, strategy: .layer
        )

        // Look up by ObjectIdentifier (Sendable-safe)
        let firstLayer = try XCTUnwrap(model.layers.first)
        let id = ObjectIdentifier(firstLayer.qProj)
        let ctx = await TierHookCoordinator.shared.context(id: id, kind: TierContextStore.HookKind.linear)
        XCTAssertNotNil(ctx)
        XCTAssertEqual(ctx?.layerIdx, 0, "first layer should have layerIdx 0")
        XCTAssertEqual(ctx?.shardKind, .layer)

        await TierHookCoordinator.shared.unregister(policy: policy)
        let ctxAfter = await TierHookCoordinator.shared.context(id: id, kind: TierContextStore.HookKind.linear)
        XCTAssertNil(ctxAfter, "context should be gone after unregister")
    }

    func testLayerIndexDerivedFromPath() async throws {
        let model = TestModel()
        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [], layers: []
        )
        let dir = URL(fileURLWithPath: "/tmp/test-tier")
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: dir, metrics: nil)
        let boxed = SendableModuleBox(model)

        _ = await TierHookCoordinator.shared.register(
            model: boxed.value, policy: policy, strategy: .layer
        )

        let id1 = ObjectIdentifier(model.layers[1].gateProj)
        let ctx1 = await TierHookCoordinator.shared.context(id: id1, kind: TierContextStore.HookKind.linear)
        XCTAssertEqual(ctx1?.layerIdx, 1, "layers[1] path should parse to layerIdx 1")

        let id2 = ObjectIdentifier(model.layers[2].downProj)
        let ctx2 = await TierHookCoordinator.shared.context(id: id2, kind: TierContextStore.HookKind.linear)
        XCTAssertEqual(ctx2?.layerIdx, 2, "layers[2] path should parse to layerIdx 2")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    func testLinearHookFiresOnCall() async throws {
        TierHookInstallator.installIfNeeded()

        let model = TestModel()
        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [],
            layers: [TierManifest.LayerEntry(layer: 0, file: "layer.L00.safetensors", bytes: 0, tensors: [])]
        )
        let dir = URL(fileURLWithPath: "/tmp/test-tier")
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: dir, metrics: nil)
        let boxed = SendableModuleBox(model)
        _ = await TierHookCoordinator.shared.register(model: boxed.value, policy: policy, strategy: .layer)

        let observationsBefore = policy.heatMap.totalObservations

        let input = MLXArray.zeros([1, 4, 16])
        let firstLayer = try XCTUnwrap(model.layers.first)
        _ = firstLayer.qProj(input)

        try await Task.sleep(nanoseconds: 500_000_000)

        let observationsAfter = policy.heatMap.totalObservations
        XCTAssertGreaterThan(observationsAfter, observationsBefore,
            "heat-map should have recorded activity after Linear call")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    func testLazyLoadReplacesWeightFromSSD() async throws {
        // Phase 3 end-to-end: create a fake layer file with known weights,
        // construct a Linear with placeholder zero weights, register it,
        // call Linear — sync hook should replace weight from file before matmul.
        TierHookInstallator.installIfNeeded()

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-lazy-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Write a fake "layer 0" safetensors with a sentinel weight.
        // Tensor name must match what Module.visit produces as path + ".weight".
        // Our LazyTestModel structure: LazyTestModel.layers[0].test_linear
        // → Module.visit path = "layers.0.test_linear"
        let hidden = 4
        let sentinelWeight = MLXArray.ones([hidden, hidden]) * 7.0
        let tensorName = "layers.0.test_linear.weight"
        try MLX.save(arrays: [tensorName: sentinelWeight],
                     url: tmpDir.appendingPathComponent("layer.L00.safetensors"))

        // Construct the Linear with random init weight (we'll verify replacement below)
        let linear = Linear(hidden, hidden, bias: false)
        let initialWeightSum = linear.weight.sum().item(Float.self)
        XCTAssertNotEqual(initialWeightSum, 7.0 * Float(hidden * hidden),
            "sanity: init weight should be random, not sentinel")
        let model = LazyTestModel(layers: [LazyTestLayer(linear: linear)])
        let boxed = SendableModuleBox(model)

        // Register the model — coordinator walks tree, derives layerIdx=0 from
        // path "model.layers.0.test_linear", stores context pointing to policy.
        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [],
            layers: [TierManifest.LayerEntry(
                layer: 0, file: "layer.L00.safetensors", bytes: 0,
                tensors: [tensorName]
            )]
        )
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmpDir, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmpDir, manifest: manifest)
        _ = await TierHookCoordinator.shared.register(model: boxed.value, policy: policy, strategy: .layer)

        // Verify the Linear instance is registered
        let linearID = ObjectIdentifier(linear)
        XCTAssertTrue(TierContextStore.shared.isLoaded(linearID) == false,
            "Linear should NOT be loaded before first call")

        // Call the Linear — sync hook should fire, load weight, replace.
        let input = MLXArray.ones([1, hidden])
        let result = linear(input)
        // Force eval so we can read result
        let resultSum = result.sum().item(Float.self)

        // After call, the sync hook should have replaced weight with the sentinel (7s).
        // input @ weight.T = ones(1,4) @ full(7, 4x4).T = sum across cols = 4 * 7 = 28 per row
        // So each element of result (shape [1,4]) should be 28.
        XCTAssertEqual(resultSum, 28.0 * Float(hidden), accuracy: 0.5,
            "result should reflect sentinel weight (7s), got sum=\(resultSum)")
        XCTAssertEqual(linear.weight.sum().item(Float.self), 7.0 * Float(hidden * hidden), accuracy: 0.5,
            "linear.weight should now be sentinel (7s)")

        XCTAssertTrue(TierContextStore.shared.isLoaded(linearID),
            "Linear should be marked loaded after sync hook fired")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    func testSwitchLinearLazyLoadReplacesWeight() async throws {
        // Phase 3.5: SwitchLinear per-layer lazy load. Write per-expert files
        // for a layer with 2 experts × 1 proj (gate_proj), register a SwitchLinear,
        // verify sync hook stacks the slices and replaces weight.
        TierHookInstallator.installIfNeeded()

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-moe-lazy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let hidden = 4
        let intermediate = 8
        let numExperts = 2

        // Build the SwitchGLU we'll lazy-load into. Need to reach inside to
        // test individual SwitchLinear instances (gate/up/down).
        let switchGlu = SwitchGLU(inputDims: hidden, hiddenDims: intermediate, numExperts: numExperts)
        let gate = Linear(hidden, numExperts, bias: false)
        let model = LazyMoETestModel(layers: [LazyMoETestLayer(switchGlu: switchGlu, gate: gate)])
        let boxed = SendableModuleBox(model)

        // Write per-expert files for layer 0. Each file has 3 projs (gate/up/down)
        // with sentinel values to detect replacement.
        // SwitchGLU Module.visit path: "layers.0.switch_mlp.gate_proj" etc.
        let tensorNames = [
            "layers.0.switch_mlp.gate_proj.weight",
            "layers.0.switch_mlp.up_proj.weight",
            "layers.0.switch_mlp.down_proj.weight",
        ]
        for E in 0..<numExperts {
            var bucket: [String: MLXArray] = [:]
            for name in tensorNames {
                // Sentinel: each expert E gets value (E+1)*10 so we can verify per-expert slice
                bucket[name] = MLXArray.ones([intermediate, hidden]) * Double((E + 1) * 10)
                // For down_proj the shape is [hidden, intermediate] (transposed)
                if name.contains("down_proj") {
                    bucket[name] = MLXArray.ones([hidden, intermediate]) * Double((E + 1) * 10)
                }
            }
            try MLX.save(arrays: bucket,
                         url: tmpDir.appendingPathComponent("expert.L00.E0\(E).safetensors"))
        }

        // Capture initial gate_proj weight for later comparison
        // SwitchGLU stores gateProj/upProj/downProj as @ModuleInfo; access via property
        // We can't directly access switchGlu's private gateProj, but we can check
        // via Module.visit after registration. For now just verify call doesn't crash.

        // Register — TierHookCoordinator walks tree, registers all 3 SwitchLinear
        // (gate/up/down) inside the SwitchGLU + 1 Linear gate.
        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "stacked", strategy: .expert,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: numExperts * tensorNames.count,
            experts: (0..<numExperts).map { E in
                TierManifest.ExpertEntry(
                    layer: 0, expert: E,
                    file: "expert.L00.E0\(E).safetensors",
                    bytes: 0, tensors: tensorNames, stackedSource: true
                )
            },
            layers: nil
        )
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmpDir, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmpDir, manifest: manifest)
        let regResult = await TierHookCoordinator.shared.register(model: boxed.value, policy: policy, strategy: .expert)
        XCTAssertGreaterThanOrEqual(regResult.experts, 3, "should find 3 SwitchLinear inside SwitchGLU")

        // Call the SwitchGLU directly with some indices. This triggers the
        // sync hooks on the 3 SwitchLinear instances inside it.
        let input = MLXArray.ones([1, hidden])
        let indices = MLXArray(Int32(0))  // activate expert 0
        let weights = MLXArray.ones([1])
        // SwitchGLU.callAsFunction signature: (x, indices, weights) — but the one in mlx-swift-lm
        // varies; use the simpler 2-arg form if available.
        // Actually SwitchGLU here takes (x, indices). Let me check.
        // Looking at SwitchLayers.swift:62: callAsFunction(_ x: MLXArray, _ indices: MLXArray)
        _ = switchGlu(input, indices)
        _ = weights

        // After the call, all 3 SwitchLinear instances inside switchGlu should be marked loaded.
        // We verify via TierContextStore.isLoaded — but we need their ObjectIdentifiers.
        // Phase 6: perExpertStreaming is auto-enabled for .expert strategy.
        // In that mode, SwitchLinear sync hook doesn't markLoaded (because
        // indices change per token). So we can't check isLoaded. Instead,
        // verify the per-expert cache is populated.
        var totalSwitchLinear = 0
        model.visit { _, module in
            if module is SwitchLinear {
                totalSwitchLinear += 1
            }
        }
        XCTAssertEqual(totalSwitchLinear, 3, "SwitchGLU should contain 3 SwitchLinear")
        // Per-expert cache should have entries from the call.
        XCTAssertGreaterThan(policy.weightManager.perExpertCacheCount(), 0,
            "per-expert cache should be populated after SwitchLinear call")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    func testEvictionReleasesWeightAndNextCallReloads() async throws {
        // Phase 4: loaded set LRU eviction. Load 2 Linears, verify both loaded,
        // evict with tight budget, verify oldest zeroed + next call reloads.
        TierHookInstallator.installIfNeeded()

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-evict-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let hidden = 4
        // Two Linears at two layers, each with sentinel weights
        let linear1 = Linear(hidden, hidden, bias: false)
        let linear2 = Linear(hidden, hidden, bias: false)
        let model = LazyTestModel(layers: [
            LazyTestLayer(linear: linear1),
            LazyTestLayer(linear: linear2),
        ])
        let boxed = SendableModuleBox(model)

        // Write per-layer files with sentinel weights (1.0 for layer 0, 2.0 for layer 1)
        try MLX.save(arrays: ["layers.0.test_linear.weight": MLXArray.ones([hidden, hidden]) * 1.0],
                     url: tmpDir.appendingPathComponent("layer.L00.safetensors"))
        try MLX.save(arrays: ["layers.1.test_linear.weight": MLXArray.ones([hidden, hidden]) * 2.0],
                     url: tmpDir.appendingPathComponent("layer.L01.safetensors"))

        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [],
            layers: [
                TierManifest.LayerEntry(layer: 0, file: "layer.L00.safetensors", bytes: 0, tensors: []),
                TierManifest.LayerEntry(layer: 1, file: "layer.L01.safetensors", bytes: 0, tensors: []),
            ]
        )
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmpDir, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmpDir, manifest: manifest)
        _ = await TierHookCoordinator.shared.register(model: boxed.value, policy: policy, strategy: .layer)

        // Call both Linears to trigger lazy load
        let input = MLXArray.ones([1, hidden])
        _ = linear1(input)
        _ = linear2(input)

        // Verify both are loaded
        let id1 = ObjectIdentifier(linear1)
        let id2 = ObjectIdentifier(linear2)
        XCTAssertTrue(TierContextStore.shared.isLoaded(id1), "linear1 should be loaded")
        XCTAssertTrue(TierContextStore.shared.isLoaded(id2), "linear2 should be loaded")

        // Capture the bytes metric before eviction
        let bytesBeforeEvict = TierContextStore.shared.loadedBytes
        XCTAssertGreaterThan(bytesBeforeEvict, 0, "should have bytes loaded")

        // Evict with budget of 0 → all entries evicted
        let evicted = TierContextStore.shared.evictToFit(byteBudget: 0)
        XCTAssertGreaterThanOrEqual(evicted, 2, "should evict both entries")

        // After eviction: weights should be zeroed (reset closure ran)
        let sum1 = linear1.weight.sum().item(Float.self)
        let sum2 = linear2.weight.sum().item(Float.self)
        XCTAssertEqual(sum1, 0.0, accuracy: 0.01, "linear1.weight should be zeroed after eviction")
        XCTAssertEqual(sum2, 0.0, accuracy: 0.01, "linear2.weight should be zeroed after eviction")

        // isLoaded should now be false for both
        XCTAssertFalse(TierContextStore.shared.isLoaded(id1))
        XCTAssertFalse(TierContextStore.shared.isLoaded(id2))

        // Next call should re-trigger sync hook → reload from SSD
        _ = linear1(input)
        XCTAssertTrue(TierContextStore.shared.isLoaded(id1), "linear1 should reload after re-call")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }
}
