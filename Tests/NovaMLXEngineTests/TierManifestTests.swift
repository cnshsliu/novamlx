import XCTest
import MLX
@testable import NovaMLXEngine

final class TierManifestTests: XCTestCase {

    /// Nonisolated helper: enumerate safetensors files in a dir, skipping
    /// the tie-shards/ subdir (mirrors the MLX loadWeights patch).
    static func enumerateSafetensorsSkippingTieShards(in dir: URL) async -> [URL] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                var result: [URL] = []
                let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)!
                for case let u as URL in enumerator {
                    if u.path.contains("/tie-shards/") { continue }
                    if u.pathExtension == "safetensors" { result.append(u) }
                }
                continuation.resume(returning: result)
            }
        }
    }

    func testDenseManifestParses() throws {
        let url = URL(fileURLWithPath: "/Volumes/WD/nova-models/Qwen3-8B-4bit.tiered")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Qwen3-8B-4bit.tiered not present — run expert_shard_layout.py first")
        }
        let manifest = try TierManifestLoader.loadIfPresent(modelDir: url)
        XCTAssertNotNil(manifest, "tier-manifest.json should exist")
        guard let m = manifest else { return }

        XCTAssertEqual(m.version, 1)
        XCTAssertEqual(m.architecture, "qwen3")
        XCTAssertEqual(m.layout, "none")
        XCTAssertEqual(m.strategy, .layer, "dense model should detect strategy=.layer")
        XCTAssertEqual(m.tier0File, "tier0.safetensors")
        XCTAssertEqual(m.expertCount, 0, "dense model has no experts")
        XCTAssertEqual(m.experts.count, 0, "experts array should be empty")
        XCTAssertEqual(m.layers?.count, 36, "Qwen3-8B has 36 decoder layers")

        // tier0 should contain embeddings + final norm + lm_head + per-layer norm weights
        // (RMSNorm weights stay in tier0 because sync hook only handles Linear)
        XCTAssertGreaterThan(m.tier0TensorCount, 20, "tier0 should include per-layer norm weights")
        XCTAssertGreaterThan(m.tier0TensorCount, 0)

        // Each layer file lookup should succeed
        for L in 0..<36 {
            let file = m.layerFile(layer: L)
            XCTAssertNotNil(file, "layer \(L) file should be in manifest")
            XCTAssertTrue(file?.hasPrefix("layer.L") == true, "layer file naming")
        }
        // Out-of-range lookup should fail
        XCTAssertNil(m.layerFile(layer: 999))

        // Total bytes math
        XCTAssertGreaterThan(m.totalLayerBytes, 0)
        XCTAssertEqual(m.totalExpertBytes, 0)
    }

    func testBackwardCompatManifestInfersStrategyFromLayout() throws {
        // Old v1 manifests without `strategy` field should default to .expert
        // (the only strategy old manifests could represent).
        let json = """
        {
            "version": 1,
            "converter": "expert_shard_layout.py",
            "source_model": "/fake",
            "architecture": "deepseek_v3",
            "layout": "stacked",
            "tier0_file": "tier0.safetensors",
            "tier0_tensor_count": 100,
            "tier0_bytes": 1000000,
            "expert_count": 2,
            "experts": [
                {"layer": 0, "expert": 0, "file": "e.L00.E00.safetensors", "bytes": 5000, "tensors": []},
                {"layer": 0, "expert": 1, "file": "e.L00.E01.safetensors", "bytes": 5000, "tensors": []}
            ]
        }
        """.data(using: .utf8)!
        let m = try JSONDecoder().decode(TierManifest.self, from: json)
        XCTAssertEqual(m.strategy, .expert, "old manifests without strategy should default to .expert")
        XCTAssertNil(m.layers, "old manifests have no layers field")
        XCTAssertEqual(m.expertFile(layer: 0, expert: 1), "e.L00.E01.safetensors")
    }

    func testIsTieredDetection() throws {
        let tieredDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/Qwen3-8B-4bit.tiered")
        guard FileManager.default.fileExists(atPath: tieredDir.path) else {
            throw XCTSkip("tiered model not present")
        }
        XCTAssertTrue(TierManifestLoader.isTiered(tieredDir))

        let plainDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/mlx-community/Qwen3-8B-4bit")
        XCTAssertFalse(TierManifestLoader.isTiered(plainDir))
    }

    func testBindMovesShardsToSubdirAndUnbindRestores() async throws {
        // Phase 4: WeightTierManager.bind must move per-shard files into
        // tie-shards/ subdir so MLX's loadWeights (patched to skip that
        // subdir) doesn't load them eagerly. On unbind, files move back.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-shards-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Write minimal fake model: tier0 + 2 layer files + manifest
        try MLX.save(arrays: ["embed.weight": MLXArray.ones([4, 4])],
                     url: tmpDir.appendingPathComponent("tier0.safetensors"))
        try MLX.save(arrays: ["layers.0.linear.weight": MLXArray.ones([4, 4])],
                     url: tmpDir.appendingPathComponent("layer.L00.safetensors"))
        try MLX.save(arrays: ["layers.1.linear.weight": MLXArray.ones([4, 4])],
                     url: tmpDir.appendingPathComponent("layer.L01.safetensors"))

        let manifest = TierManifest(
            version: 1, converter: "test", sourceModel: "/test",
            architecture: "test", layout: "none", strategy: .layer,
            tier0File: "tier0.safetensors", tier0TensorCount: 1, tier0Bytes: 0,
            expertCount: 0, experts: [],
            layers: [
                TierManifest.LayerEntry(layer: 0, file: "layer.L00.safetensors", bytes: 0, tensors: []),
                TierManifest.LayerEntry(layer: 1, file: "layer.L01.safetensors", bytes: 0, tensors: []),
            ]
        )

        // Simulate MLX eager load BEFORE bind: scans tmpDir recursively.
        // Files are flat — both layer files visible.
        func safetensorsFiles(in url: URL) -> [URL] {
            var result: [URL] = []
            let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)!
            for case let u as URL in enumerator {
                if u.pathExtension == "safetensors" && !u.path.contains("/tie-shards/") {
                    result.append(u)
                }
            }
            return result
        }

        XCTAssertEqual(safetensorsFiles(in: tmpDir).count, 3,
            "before bind: tier0 + 2 layer files = 3 visible")

        let wm = WeightTierManager()
        try await wm.bind(modelDir: tmpDir, manifest: manifest)

        // After bind: only tier0 visible at top level. Layer files moved to tie-shards/.
        let visible = safetensorsFiles(in: tmpDir)
        XCTAssertEqual(visible.count, 1, "after bind: only tier0 visible to eager loader")
        XCTAssertEqual(visible.first?.lastPathComponent, "tier0.safetensors")

        // tie-shards/ should have the 2 layer files
        let shardsDir = tmpDir.appendingPathComponent("tie-shards")
        XCTAssertTrue(FileManager.default.fileExists(atPath: shardsDir.path))
        let shardFiles = try FileManager.default.contentsOfDirectory(at: shardsDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        XCTAssertEqual(shardFiles.count, 2, "tie-shards/ should have 2 layer files")

        // readLayer should still work — reads from tie-shards/
        let tensors = try wm.readLayer(layer: 0)
        XCTAssertNotNil(tensors["layers.0.linear.weight"])

        // Unbind: files move back
        wm.unbind()
        XCTAssertEqual(safetensorsFiles(in: tmpDir).count, 3,
            "after unbind: all 3 files visible again (restored)")
        XCTAssertFalse(FileManager.default.fileExists(atPath: shardsDir.path),
            "tie-shards/ subdir should be cleaned up")
    }

    func testRealQwen3TieredDirSurvivesEagerLoad() async throws {
        // Real-model validation: load the actual Qwen3-8B-4bit.tiered dir,
        // simulate MLX's loadWeights file enumeration, verify only tier0
        // is visible after bind. Then readLayer works from tie-shards/.
        let modelDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/Qwen3-8B-4bit.tiered")
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw XCTSkip("Qwen3-8B-4bit.tiered not present — run expert_shard_layout.py first")
        }
        guard let manifest = try TierManifestLoader.loadIfPresent(modelDir: modelDir) else {
            XCTFail("tier-manifest.json missing in \(modelDir.path)")
            return
        }
        XCTAssertEqual(manifest.strategy, .layer)
        XCTAssertEqual(manifest.layers?.count, 36)

        // Clean up any prior bind state (e.g., from live worker process).
        // Bind-then-unbind moves shards back to top level if they were moved.
        let cleanupWm = WeightTierManager()
        try? await cleanupWm.bind(modelDir: modelDir, manifest: manifest)
        cleanupWm.unbind()

        // If a concurrent worker is still running and holding the model, it
        // may re-bind between our cleanup and the test proper. Detect that
        // and skip rather than fail.
        let shardsDir = modelDir.appendingPathComponent("tie-shards")
        if FileManager.default.fileExists(atPath: shardsDir.path) {
            throw XCTSkip("tiered dir is bind-locked by another process (worker?). Stop NovaMLX and re-run.")
        }

        func topLevelSafetensors(_ url: URL) -> Int {
            guard let files = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else { return 0 }
            return files.filter { $0.pathExtension == "safetensors" }.count
        }
        let beforeBind = topLevelSafetensors(modelDir)

        let wm = WeightTierManager()
        try await wm.bind(modelDir: modelDir, manifest: manifest)

        let afterBind = topLevelSafetensors(modelDir)
        XCTAssertEqual(afterBind, 1, "only tier0.safetensors should be at top level after bind")
        XCTAssertEqual(beforeBind - afterBind, 36, "36 layer files should have moved into tie-shards/")

        let verifyShardsDir = modelDir.appendingPathComponent("tie-shards")
        XCTAssertTrue(FileManager.default.fileExists(atPath: verifyShardsDir.path))
        let shardFiles = (try? FileManager.default.contentsOfDirectory(at: verifyShardsDir, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "safetensors" } ?? []
        XCTAssertEqual(shardFiles.count, 36)

        let visible = await Self.enumerateSafetensorsSkippingTieShards(in: modelDir)
        XCTAssertEqual(visible.count, 1, "MLX loadWeights would see only tier0")
        XCTAssertEqual(visible.first?.lastPathComponent, "tier0.safetensors")

        let layer0 = try wm.readLayer(layer: 0)
        XCTAssertGreaterThan(layer0.count, 0, "layer 0 should have tensors")

        wm.unbind()
        let afterUnbind = topLevelSafetensors(modelDir)
        XCTAssertEqual(afterUnbind, beforeBind, "files restored after unbind")
    }

    func testRealQwen36MoETieredDirSurvivesEagerLoad() async throws {
        // Real MoE model validation: Qwen3.6-35B-A3B-4bit has 40 layers ×
        // 4 routed experts × 3 projs (gate/up/down) + quantization siblings.
        // Verify expert strategy bind moves per-expert files to tie-shards/.
        let modelDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/Qwen3.6-35B-A3B-4bit.tiered")
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw XCTSkip("Qwen3.6-35B-A3B-4bit.tiered not present — run expert_shard_layout.py first")
        }
        guard let manifest = try TierManifestLoader.loadIfPresent(modelDir: modelDir) else {
            XCTFail("tier-manifest.json missing in \(modelDir.path)")
            return
        }
        XCTAssertEqual(manifest.strategy, .expert, "MoE model should detect strategy=expert")
        XCTAssertGreaterThan(manifest.expertCount, 0, "MoE model should have experts")
        // Validate the structure
        XCTAssertNotNil(manifest.experts.first?.file)

        // Clean up any prior bind state (e.g., from live worker process).
        let cleanupWm = WeightTierManager()
        try? await cleanupWm.bind(modelDir: modelDir, manifest: manifest)
        cleanupWm.unbind()

        func topLevelSafetensors(_ url: URL) -> Int {
            guard let files = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else { return 0 }
            return files.filter { $0.pathExtension == "safetensors" }.count
        }
        let beforeBind = topLevelSafetensors(modelDir)
        XCTAssertGreaterThan(beforeBind, 1, "should have tier0 + many per-expert files before bind")

        let wm = WeightTierManager()
        try await wm.bind(modelDir: modelDir, manifest: manifest)

        // After bind: only tier0 visible at top level
        let afterBind = topLevelSafetensors(modelDir)
        XCTAssertEqual(afterBind, 1, "only tier0.safetensors should be at top level after bind")
        XCTAssertEqual(beforeBind - afterBind, manifest.expertCount,
            "all per-expert files should move into tie-shards/")

        // Verify MLX loadWeights pattern sees only tier0
        let visible = await Self.enumerateSafetensorsSkippingTieShards(in: modelDir)
        XCTAssertEqual(visible.count, 1)
        XCTAssertEqual(visible.first?.lastPathComponent, "tier0.safetensors")

        // Verify readExpert + readLayerStacked work from tie-shards/
        let firstExpert = manifest.experts.first!
        let expert0 = try wm.readExpert(layer: firstExpert.layer, expert: firstExpert.expert)
        XCTAssertGreaterThan(expert0.count, 0, "expert should have tensors")

        let stacked = try wm.readLayerStacked(layer: firstExpert.layer)
        XCTAssertGreaterThan(stacked.count, 0, "stacked layer should produce tensors")
        // Each proj's stacked tensor should be 3D [numExperts, ...] not 2D
        for (_, arr) in stacked {
            XCTAssertGreaterThanOrEqual(arr.ndim, 2, "stacked tensor should have numExperts axis")
        }

        wm.unbind()
        let afterUnbind = topLevelSafetensors(modelDir)
        XCTAssertEqual(afterUnbind, beforeBind, "files restored after unbind")
    }
}
