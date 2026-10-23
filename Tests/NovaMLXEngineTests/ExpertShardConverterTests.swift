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

    @Test("nested MiniMax text_config stacked switch_mlp converts")
    func convertMiniMaxNestedSwitch() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tie-minimax-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = """
            {"model_type":"minimax_m3_vl","text_config":{"num_hidden_layers":1,"num_local_experts":2}}
            """
        try cfg.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.ones([4, 2]),
            "language_model.lm_head.weight": MLXArray.ones([4, 2]),
            "language_model.model.norm.weight": MLXArray.ones([2]),
            "language_model.model.layers.0.input_layernorm.weight": MLXArray.ones([2]),
            "language_model.model.layers.0.post_attention_layernorm.weight": MLXArray.ones([2]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.ones([4, 2]),
            "language_model.model.layers.0.self_attn.k_norm.weight": MLXArray.ones([2]),
            "language_model.model.layers.0.self_attn.index_k_norm.weight": MLXArray.ones([2]),
            "language_model.model.layers.0.block_sparse_moe.gate.weight": MLXArray.ones([2, 2]),
            "language_model.model.layers.0.block_sparse_moe.e_score_correction_bias": MLXArray.zeros([2]),
            "language_model.model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight":
                MLXArray.ones([2, 3, 4]),
            "language_model.model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight":
                MLXArray.ones([2, 3, 4]),
            "language_model.model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight":
                MLXArray.ones([2, 2, 3]),
        ]
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))

        #expect(ExpertShardConverter.canConvert(at: dir))
        try ExpertShardConverter.convert(at: dir)
        #expect(ExpertShardConverter.inspect(at: dir).status == .ready)

        let tier0 = try MLX.loadArrays(url: dir.appendingPathComponent("tier0.safetensors"))
        #expect(tier0["model.model.layers.0.input_layernorm.weight"] != nil)
        #expect(tier0["model.model.layers.0.post_attention_layernorm.weight"] != nil)
        #expect(tier0["model.model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(tier0["model.model.layers.0.self_attn.k_norm.weight"] != nil)
        #expect(tier0["model.model.layers.0.self_attn.index_k_norm.weight"] != nil)
        #expect(tier0["model.model.layers.0.block_sparse_moe.gate.weight"] != nil)
        #expect(tier0["model.model.layers.0.block_sparse_moe.e_score_correction_bias"] != nil)
        #expect(tier0["model.lm_head.weight"] != nil)
        #expect(tier0["model.model.norm.weight"] != nil)
        #expect(tier0.keys.contains { $0.contains("switch_mlp") } == false)

        let expert = dir.appendingPathComponent("tie-shards/expert.L00.E000.safetensors")
        #expect(FileManager.default.fileExists(atPath: expert.path))
        let sliced = try MLX.loadArrays(url: expert)
        #expect(sliced["model.layers.0.switch_mlp.gate_proj.weight"]?.shape == [3, 4])
        #expect(sliced["model.layers.0.switch_mlp.down_proj.weight"]?.shape == [2, 3])
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
