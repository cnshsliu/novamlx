import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import NovaMLXEngine

/// Speed contracts for TIE. These encode the I/O and hot-path rules that
/// actually change tokens/s — not residency bookkeeping.
final class TierSpeedTests: XCTestCase {

    // MARK: - Layer file cache

    func testReadLayerHitsCacheOnSecondCall() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        let hidden = 4
        try MLX.save(
            arrays: ["layers.0.q.weight": MLXArray.ones([hidden, hidden])],
            url: tmp.appendingPathComponent("layer.L00.safetensors")
        )

        let manifest = denseManifest(layers: [
            .init(layer: 0, file: "layer.L00.safetensors", bytes: 64, tensors: ["layers.0.q.weight"]),
        ])
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmp, metrics: nil)
        let bindExp = expectation(description: "bind")
        Task {
            do {
                try await policy.weightManager.bind(modelDir: tmp, manifest: manifest)
            } catch {
                XCTFail("bind failed: \(error)")
            }
            bindExp.fulfill()
        }
        wait(for: [bindExp], timeout: 5)
        let mgr = policy.weightManager

        _ = try mgr.readLayer(layer: 0)
        let opsAfterFirst = mgr.ssdReadOps
        XCTAssertGreaterThan(opsAfterFirst, 0, "first readLayer must hit SSD")

        _ = try mgr.readLayer(layer: 0)
        XCTAssertEqual(mgr.ssdReadOps, opsAfterFirst, "second readLayer must not touch SSD")
    }

    func testTwoLinearsInSameLayerShareOneSSDRead() async throws {
        TierHookInstallator.installIfNeeded()
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        let hidden = 4
        let q = Linear(hidden, hidden, bias: false)
        let k = Linear(hidden, hidden, bias: false)
        let model = TwoLinearLayerModel(q: q, k: k)
        try MLX.save(
            arrays: [
                "layers.0.q.weight": MLXArray.ones([hidden, hidden]) * 3,
                "layers.0.k.weight": MLXArray.ones([hidden, hidden]) * 5,
            ],
            url: tmp.appendingPathComponent("layer.L00.safetensors")
        )
        let manifest = denseManifest(layers: [
            .init(layer: 0, file: "layer.L00.safetensors", bytes: 128,
                  tensors: ["layers.0.q.weight", "layers.0.k.weight"]),
        ])
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmp, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmp, manifest: manifest)
        let boxed = SendableModuleBox(model)
        _ = await TierHookCoordinator.shared.register(model: boxed.value, policy: policy, strategy: .layer)

        let input = MLXArray.ones([1, hidden])
        _ = q(input)
        _ = k(input)

        XCTAssertEqual(policy.weightManager.ssdReadOps, 1,
                       "both Linears in one layer must share a single SSD read")

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    // MARK: - Prefetch warms the expert cache

    func testPrefetchExpertsTurnsNextLoadIntoCacheHit() async throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        try writeExpertFile(dir: tmp, layer: 1, expert: 0, value: 2)
        try writeExpertFile(dir: tmp, layer: 1, expert: 1, value: 4)

        let manifest = expertManifest(entries: [
            expertEntry(layer: 1, expert: 0, file: "expert.L01.E00.safetensors"),
            expertEntry(layer: 1, expert: 1, file: "expert.L01.E01.safetensors"),
        ])
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmp, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmp, manifest: manifest)
        let mgr = policy.weightManager

        mgr.prefetchExperts(layer: 1, expertIDs: [0, 1])
        let deadline = Date().addingTimeInterval(5)
        while mgr.perExpertCacheCount() < 2 && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertGreaterThanOrEqual(mgr.perExpertCacheCount(), 2, "prefetch must populate the expert cache")

        let ops = mgr.ssdReadOps
        _ = try mgr.loadActivatedExperts(layer: 1, expertIDs: [0, 1])
        XCTAssertEqual(mgr.ssdReadOps, ops, "load after prefetch must not read SSD again")
    }

    /// Prefetch Tasks and the decode-path load share `perExpertCache`.
    /// An unlocked `Dictionary.count` here SIGSEGV'd DeepSeek-V4-Flash TIE.
    func testConcurrentPrefetchAndLoadDoesNotRace() async throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        var entries: [TierManifest.ExpertEntry] = []
        for expert in 0..<8 {
            try writeExpertFile(dir: tmp, layer: 0, expert: expert, value: Float(expert + 1))
            entries.append(expertEntry(
                layer: 0, expert: expert,
                file: String(format: "expert.L%02d.E%02d.safetensors", 0, expert)
            ))
        }
        let manifest = expertManifest(entries: entries)
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmp, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmp, manifest: manifest)
        let mgr = policy.weightManager

        DispatchQueue.concurrentPerform(iterations: 64) { i in
            let ids = (0..<4).map { ($0 + i) % 8 }
            mgr.prefetchExperts(layer: 0, expertIDs: ids)
            _ = try? mgr.loadActivatedExperts(layer: 0, expertIDs: ids)
        }

        let loaded = try mgr.loadActivatedExperts(layer: 0, expertIDs: Array(0..<8))
        XCTAssertEqual(loaded.count, 8)
        XCTAssertGreaterThanOrEqual(mgr.perExpertCacheCount(), 8)
    }

    // MARK: - Hot-path reuse

    func testForwardCacheSkipsRepeatMaterialize() {
        let cache = TierHotPathCache()
        let indices = MLXArray([Int32(1), Int32(3)])
        let id = ObjectIdentifier(indices)
        XCTAssertNil(cache.reuseForward(layer: 4, indicesId: id))

        let local = MLXArray([Int32(0), Int32(1)])
        cache.rememberForward(layer: 4, indicesId: id, expertIDs: [1, 3], localIndices: local)

        let hit = cache.reuseForward(layer: 4, indicesId: id)
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.expertIDs, [1, 3])

        let other = MLXArray([Int32(1), Int32(3)])
        XCTAssertNil(cache.reuseForward(layer: 4, indicesId: ObjectIdentifier(other)),
                     "a new indices tensor is a new forward")
    }

    func testStackedSetMatchDetectsSameExperts() {
        let cache = TierHotPathCache()
        let swId = ObjectIdentifier(NSObject())
        XCTAssertFalse(cache.stackedMatches(id: swId, uniqueSorted: [2, 5]))
        cache.rememberStacked(id: swId, uniqueSorted: [2, 5])
        XCTAssertTrue(cache.stackedMatches(id: swId, uniqueSorted: [2, 5]))
        XCTAssertFalse(cache.stackedMatches(id: swId, uniqueSorted: [2, 6]))
    }

    // MARK: - Eviction must not yank the working set

    func testEvictToFitSkipsRecentlyTouchedWhenIdleWindowSet() async throws {
        TierHookInstallator.installIfNeeded()
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        let hidden = 4
        let linear = Linear(hidden, hidden, bias: false)
        let model = LazySpeedLayerModel(linear: linear)
        try MLX.save(
            arrays: ["layers.0.test_linear.weight": MLXArray.ones([hidden, hidden])],
            url: tmp.appendingPathComponent("layer.L00.safetensors")
        )
        let manifest = denseManifest(layers: [
            .init(layer: 0, file: "layer.L00.safetensors", bytes: 64, tensors: ["layers.0.test_linear.weight"]),
        ])
        let policy = TieredOffloadPolicy(manifest: manifest, modelDir: tmp, metrics: nil)
        try await policy.weightManager.bind(modelDir: tmp, manifest: manifest)
        let boxed = SendableModuleBox(model)
        _ = await TierHookCoordinator.shared.register(model: boxed.value, policy: policy, strategy: .layer)

        _ = linear(MLXArray.ones([1, hidden]))
        XCTAssertTrue(TierContextStore.shared.isLoaded(ObjectIdentifier(linear)))

        let evicted = TierContextStore.shared.evictToFit(byteBudget: 0, minIdleSeconds: 30)
        XCTAssertEqual(evicted, 0, "freshly used weights must not be evicted mid-generation")
        XCTAssertTrue(TierContextStore.shared.isLoaded(ObjectIdentifier(linear)))

        let forced = TierContextStore.shared.evictToFit(byteBudget: 0, minIdleSeconds: 0)
        XCTAssertGreaterThanOrEqual(forced, 1)

        await TierHookCoordinator.shared.unregister(policy: policy)
    }

    func testHeatMapTopExpertsPrefersHottest() {
        let map = ExpertHeatMap(layerCount: 2, expertCount: 8)
        map.record(layer: 1, experts: [3, 3, 3, 1, 1, 5])
        XCTAssertEqual(map.topExperts(layer: 1, k: 2), [3, 1])
    }

    // MARK: - Helpers

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-speed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func denseManifest(layers: [TierManifest.LayerEntry]) -> TierManifest {
        TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "layer", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: 0, experts: [], layers: layers
        )
    }

    private func expertManifest(entries: [TierManifest.ExpertEntry]) -> TierManifest {
        TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "stacked", strategy: .expert,
            tier0File: "tier0.safetensors", tier0TensorCount: 0, tier0Bytes: 0,
            expertCount: entries.count, experts: entries, layers: nil
        )
    }

    private func expertEntry(layer: Int, expert: Int, file: String) -> TierManifest.ExpertEntry {
        TierManifest.ExpertEntry(
            layer: layer, expert: expert, file: file, bytes: 64,
            tensors: [
                "layers.\(layer).switch_mlp.gate_proj.weight",
                "layers.\(layer).switch_mlp.up_proj.weight",
                "layers.\(layer).switch_mlp.down_proj.weight",
            ],
            stackedSource: true
        )
    }

    private func writeExpertFile(dir: URL, layer: Int, expert: Int, value: Float) throws {
        let hidden = 4
        let intermediate = 8
        var bucket: [String: MLXArray] = [:]
        bucket["layers.\(layer).switch_mlp.gate_proj.weight"] = MLXArray.ones([intermediate, hidden]) * value
        bucket["layers.\(layer).switch_mlp.up_proj.weight"] = MLXArray.ones([intermediate, hidden]) * value
        bucket["layers.\(layer).switch_mlp.down_proj.weight"] = MLXArray.ones([hidden, intermediate]) * value
        let name = String(format: "expert.L%02d.E%02d.safetensors", layer, expert)
        try MLX.save(arrays: bucket, url: dir.appendingPathComponent(name))
    }
}

private final class TwoLinearLayerModel: Module {
    @ModuleInfo(key: "layers") var layers: [TwoLinearLayer]
    init(q: Linear, k: Linear) {
        self._layers.wrappedValue = [TwoLinearLayer(q: q, k: k)]
        super.init()
    }
}

private final class TwoLinearLayer: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    init(q: Linear, k: Linear) {
        self._q.wrappedValue = q
        self._k.wrappedValue = k
        super.init()
    }
}

private final class LazySpeedLayerModel: Module {
    @ModuleInfo(key: "layers") var layers: [LazySpeedLayer]
    init(linear: Linear) {
        self._layers.wrappedValue = [LazySpeedLayer(linear: linear)]
        super.init()
    }
}

private final class LazySpeedLayer: Module {
    @ModuleInfo(key: "test_linear") var testLinear: Linear
    init(linear: Linear) {
        self._testLinear.wrappedValue = linear
        super.init()
    }
}
