import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DFlash2")
struct DFlashTests {
    @Test("Parses nested dflash_config including DFlash2 selector fields")
    func parsesDFlash2Config() throws {
        let json = """
            {
              "model_type": "qwen3",
              "architectures": ["DFlash2DraftModel"],
              "hidden_size": 64,
              "vocab_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 128,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 16,
              "rms_norm_eps": 1e-6,
              "max_position_embeddings": 512,
              "num_target_layers": 4,
              "rope_theta": 10000000.0,
              "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 3,
                "selector_rank": 8,
                "selector_top_k": 4,
                "target_layer_ids": [0, 2]
              }
            }
            """.data(using: .utf8)!
        let cfg = try DFlashConfig.fromJSON(json)
        #expect(cfg.blockSize == 8)
        #expect(cfg.selectorRank == 8)
        #expect(cfg.selectorTopK == 4)
        #expect(cfg.convKernelSize == 2)
        #expect(cfg.targetLayerIds == [0, 2])
        #expect(cfg.maskTokenId == 3)
        #expect(cfg.isDFlash2)
    }

    @Test("Refuses DFlash2 architecture without selector fields")
    func refusesDFlash2WithoutSelector() {
        let json = """
            {
              "model_type": "qwen3",
              "architectures": ["DFlash2DraftModel"],
              "hidden_size": 64, "vocab_size": 32, "num_hidden_layers": 1,
              "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
              "head_dim": 16, "rms_norm_eps": 1e-6, "max_position_embeddings": 128,
              "num_target_layers": 1,
              "dflash_config": { "block_size": 8, "target_layer_ids": [0] }
            }
            """.data(using: .utf8)!
        #expect(throws: Error.self) {
            _ = try DFlashConfig.fromJSON(json)
        }
    }

    @Test("Qwen3.8-27B ids resolve to the DFlash2 companion")
    func qwen38ResolvesDFlash2() {
        let ids = [
            "mlx-community/Qwen3.8-27B-8bit",
            "mlx-community/Qwen3.8-27B-4bit",
            "orcarouter/Qwen3.8-27B-Uncensored-MLX",
        ]
        for id in ids {
            #expect(dflashDraftCandidates(forMainId: id) == ["incoai/Qwen3.8-27B-DFlash2"])
        }
        #expect(dflashDraftCandidates(forMainId: "mlx-community/Qwen3-8B-8bit").isEmpty)
        #expect(dflashDraftCandidates(forMainId: "mlx-community/Qwen3.8-27B-MTP-4bit").isEmpty)
    }

    @Test("Grouped conv prepare preserves [B,L,H] shape")
    func groupedConvShape() {
        let conv = DFlashGroupedConv(hiddenSize: 64, taps: 2, groupSize: 16)
        let x = MLXRandom.normal([1, 8, 64])
        let (convolved, outCoeff) = conv.prepare(x)
        eval(convolved, outCoeff)
        #expect(convolved.shape == [1, 8, 64])
        #expect(outCoeff.shape == [1, 8, 2, 4])
    }

    @Test("Greedy selector walk follows the chain, not per-slot argmax")
    func walkGreedyFollowsChain() {
        let sel = DFlashCandidateSelector(hiddenSize: 8, vocabSize: 32, rank: 4, topK: 3)
        let cand = MLXArray(
            [Int32(10), 11, 12, 20, 21, 22, 30, 31, 32], [3, 3])
        let scores = MLXArray(
            [
                Float(0), 1, 5, 0, 1, 5, 0, 1, 5,
                0, 9, 0, 0, 0, 0, 7, 1, 0,
                0, 4, 1, 9, 0, 0, 0, 0, 9,
            ], [3, 3, 3])
        let picks = sel.walkGreedy(scores: scores, candidateIds: cand)
        eval(picks)
        #expect(picks.asArray(Int.self) == [12, 20, 31])
    }

    @Test("Greedy accept prefix stops at first mismatch and keeps the bonus token")
    func greedyAcceptPrefix() {
        let drafted = [10, 11, 12, 13]
        let target = [10, 11, 99, 13, 7]
        let result = DFlashAccept.greedy(drafted: drafted, targetArgmax: target)
        #expect(result.accepted == 2)
        #expect(result.committed == [10, 11, 99])
    }

    @Test("isDFlashDraftConfig detects architecture tag")
    func detectsDFlashConfig() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash-cfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let json = """
            {"model_type":"qwen3","architectures":["DFlash2DraftModel"],
             "dflash_config":{"block_size":8,"selector_rank":8,"selector_top_k":4}}
            """
        try json.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(isDFlashDraftConfig(at: dir))
        #expect(!isMtpDraftConfig(at: dir))
    }
}
