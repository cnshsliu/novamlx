import Foundation
import MLX
import Testing
@testable import NovaMLXEngine
import NovaMLXCore

@Suite("ExpertShardConverter")
struct ExpertShardConverterTests {
    @Test("inspect is none on empty dir")
    func inspectEmpty() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let status = ExpertShardConverter.inspect(at: dir)
        #expect(status.status == .none)
        #expect(!ExpertShardConverter.canConvert(at: dir))
    }

    @Test("converts classic MoE in place and can delete")
    func convertClassicAndDelete() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-moe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = """
            {"model_type":"deepseek_v41","num_hidden_layers":1,"n_routed_experts":2}
            """
        try cfg.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        var weights: [String: MLXArray] = [
            "language_model.embed_tokens.weight": MLXArray.ones([4, 2]),
            "language_model.head.weight": MLXArray.ones([4, 2]),
            "norm.weight": MLXArray.ones([2]),
            "language_model.layers.0.ffn.gate.weight": MLXArray.ones([2, 2]),
            "language_model.layers.0.attn.attn_sink": MLXArray.ones([2]),
            "language_model.layers.0.attn_hc.fn": MLXArray.ones([2, 2]),
        ]
        for e in 0..<2 {
            for w in ["w1", "w2", "w3"] {
                weights["language_model.layers.0.ffn.experts.\(e).\(w).weight"] = MLXArray.ones([2, 2])
            }
        }
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))

        #expect(ExpertShardConverter.canConvert(at: dir))
        #expect(ExpertShardConverter.inspect(at: dir).status == .convertible)

        try ExpertShardConverter.convert(at: dir)

        let ready = ExpertShardConverter.inspect(at: dir)
        #expect(ready.status == .ready)
        #expect(ExpertShardConverter.validationError(at: dir) == nil)
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("tier-manifest.json").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("tier0.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("source-shards/model.safetensors").path))
        #expect(!FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path))

        let tier0 = try MLX.loadArrays(url: dir.appendingPathComponent("tier0.safetensors"))
        #expect(tier0["model.layers.0.ffn.gate.weight"] != nil)
        #expect(tier0["model.layers.0.attn.attn_sink"] != nil)
        #expect(tier0["model.layers.0.attn_hc.fn"] != nil)

        try ExpertShardConverter.removeLayout(at: dir)
        #expect(ExpertShardConverter.inspect(at: dir).status == .convertible)
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path))
        #expect(!FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("tier-manifest.json").path))
    }

    @Test("incomplete lock is reported")
    func incompleteLock() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-lock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try "{}".write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "x".write(
            to: dir.appendingPathComponent(ExpertShardConverter.lockName),
            atomically: true, encoding: .utf8
        )
        let status = ExpertShardConverter.inspect(at: dir)
        #expect(status.status == .incomplete)
        #expect(ExpertShardConverter.validationError(at: dir) != nil)
    }
}
