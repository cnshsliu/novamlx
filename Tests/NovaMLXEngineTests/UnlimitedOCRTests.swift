import Foundation
import Testing
@testable import NovaMLXEngine

@Suite("Unlimited OCR layout and DASH plan")
struct UnlimitedOCRTests {
    @Test("Global page is 273 tokens and decode gathers 512 keys")
    func tokenBudget() {
        #expect(UnlimitedOCRSupport.globalTokenCount() == 273)
        #expect(UnlimitedOCRSupport.localTokenCount(widthCrops: 1, heightCrops: 1) == 0)
        #expect(UnlimitedOCRSupport.localTokenCount(widthCrops: 2, heightCrops: 2) == 420)
        #expect(UnlimitedOCRSupport.pageTokenCount(widthCrops: 2, heightCrops: 2) == 693)
        let params = UnlimitedOCRDashParams()
        #expect(params.attendedKeys == UnlimitedOCRSupport.releasedDecodeKeys)
        #expect(params.attendedKeys == 12 * 32 + 128)
    }

    @Test("Official config is DASH, gen-1 is the ring")
    func configKind() throws {
        let ocr2 = try JSONDecoder().decode(
            UnlimitedOCRConfiguration.self, from: Data(ocr2JSON.utf8))
        #expect(ocr2.isDash)
        #expect(ocr2.dash.attendedKeys == 512)
        #expect(ocr2.dash.blockSize == 32)
        #expect(ocr2.dash.localWindow == 128)
        #expect(ocr2.imageTokenID == 128815)
        #expect(ocr2.pageTokenID == 128827)
        #expect(ocr2.hiddenLayers == 12)
        #expect(ocr2.headDim == 128)

        let gen1 = try JSONDecoder().decode(
            UnlimitedOCRConfiguration.self,
            from: Data(#"{"model_type":"unlimited-ocr","language_config":{"sliding_window_size":128}}"#.utf8))
        #expect(!gen1.isDash)
        #expect(gen1.slidingWindow == 128)
    }

    @Test("Weight rename keeps MLX packs and maps the HF tree")
    func rename() {
        #expect(
            UnlimitedOCRSupport.renameWeight("model.layers.0.self_attn.q_proj.weight")
                == "language_model.model.layers.0.self_attn.q_proj.weight")
        #expect(UnlimitedOCRSupport.renameWeight("model.view_seperator") == "view_separator")
        #expect(UnlimitedOCRSupport.renameWeight("lm_head.weight") == "language_model.lm_head.weight")
        #expect(
            UnlimitedOCRSupport.renameWeight("language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight")
                == "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight")
        #expect(UnlimitedOCRSupport.renameWeight("model.layers.0.self_attn.indexer.k_proj.weight") == nil)
        #expect(
            UnlimitedOCRSupport.shouldQuantizeLanguage(path: "language_model.model.layers.1.mlp.switch_mlp.gate_proj"))
        #expect(UnlimitedOCRSupport.shouldQuantizeLanguage(path: "model.layers.1.mlp.switch_mlp.gate_proj"))
        #expect(UnlimitedOCRSupport.shouldQuantizeLanguage(path: "lm_head"))
        #expect(!UnlimitedOCRSupport.shouldQuantizeLanguage(
            path: "language_model.model.layers.1.self_attn.dash_router.q_proj"))
        #expect(!UnlimitedOCRSupport.shouldQuantizeLanguage(path: "sam_model.patch_embed.proj"))
    }

    @Test("Language bits default to 4 and can stay BF16")
    func bits() {
        #expect(UnlimitedOCRSupport.languageBits(from: nil) == 4)
        #expect(UnlimitedOCRSupport.languageBits(from: "0") == nil)
        #expect(UnlimitedOCRSupport.languageBits(from: "bf16") == nil)
        #expect(UnlimitedOCRSupport.languageBits(from: "8") == 8)
    }

    @Test("Page markers isolate image blocks")
    func pages() {
        var ids = [UnlimitedOCRSupport.pageTokenID]
        ids += Array(repeating: UnlimitedOCRSupport.imageTokenID, count: 273)
        ids += [UnlimitedOCRSupport.pageTokenID]
        ids += Array(repeating: UnlimitedOCRSupport.imageTokenID, count: 273)
        let pages = UnlimitedOCRPromptPages(
            ids: ids, blockSize: 32, imageToken: UnlimitedOCRSupport.imageTokenID,
            pageToken: UnlimitedOCRSupport.pageTokenID, perPage: 273)
        #expect(pages.markerCount == 2)
        #expect(pages.promptPages == 2)
        #expect(pages.tokenInPage != nil)
        let first = pages.prefillAllow(positions: [10], blockCount: pages.blockMax.count)
        #expect(first[0].contains(true))
        #expect(first[0].contains(false))
    }

    @Test("Empty text becomes the official prompt")
    func prompts() {
        #expect(UnlimitedOCRProcessor.composePrompt("", images: 1, pageLayout: true) == "<PAGE><image>document parsing.")
        #expect(
            UnlimitedOCRProcessor.composePrompt("document parsing.", images: 3, pageLayout: true)
                == "<PAGE><image><PAGE><image><PAGE><image>Multi page parsing.")
        #expect(
            UnlimitedOCRSupport.remapSAMNeck("sam_model.neck.0.weight")
                == "sam_model.neck.layers.0.weight")
        #expect(
            UnlimitedOCRSupport.remapSAMNeck("sam_model.neck.layers.2.weight")
                == "sam_model.neck.layers.2.weight")
        #expect(UnlimitedOCRSupport.cropGrid(width: 400, height: 400, maxCrops: 16) == (1, 1))
        let wide = UnlimitedOCRSupport.cropGrid(width: 2000, height: 800, maxCrops: 16)
        #expect(wide.0 >= wide.1)
        #expect(wide.0 * wide.1 <= 16)
    }

    @Test("Layout tags are not stop tokens, and a repeated n-gram is banned")
    func stopsAndRepeat() {
        #expect(UnlimitedOCRGeneration.preservesLayoutTags(modelId: "baidu/Unlimited-OCR-2"))
        #expect(UnlimitedOCRGeneration.isTurnSeparator("<|im_end|>"))
        #expect(!UnlimitedOCRGeneration.isTurnSeparator("<table>"))
        #expect(!UnlimitedOCRGeneration.isTurnSeparator("<tr>"))
        #expect(!UnlimitedOCRGeneration.isTurnSeparator("<|det|>"))
        #expect(!UnlimitedOCRGeneration.isTurnSeparator("<PAGE>"))
        let kept = UnlimitedOCRGeneration.stopPatterns(keeping: ["<table>", "<|im_end|>", "<|det|>"])
        #expect(kept == ["<|im_end|>"])

        let banned = UnlimitedOCRGeneration.bannedNextTokens([1, 2, 3, 1, 2], eos: [1], ngram: 3, window: 128)
        #expect(banned == [3])
        let eosKept = UnlimitedOCRGeneration.bannedNextTokens([9, 1, 9], eos: [1], ngram: 2, window: 128)
        #expect(eosKept.isEmpty)
    }

    @Test("Fused DASH plan keeps the 512-key layout and the best remote block")
    func dashPlanReference() {
        let spec = UnlimitedOCRDashPlan.Spec(
            block: 2, localBlocks: 1, local: 2, topM: 3, topK: 2,
            indexDim: 2, indexScale: 1, floor: 0.0001, sigma: 1)
        #expect(spec.attended == 6)
        let plan = UnlimitedOCRDashPlan.reference(
            query: [1, 0],
            summaries: [[3, 0], [1, 0], [0, 1], [10, 0]],
            candidates: [1, 1, 1, 0],
            position: 20,
            spec: spec)
        #expect(plan.indices.count == spec.attended)
        #expect(plan.mask.count == spec.attended)
        #expect(plan.indices.prefix(4) == [0, 1, 2, 3])
        #expect(plan.indices.suffix(2) == [19, 20])
        #expect(plan.mask[4] == 0)
        #expect(plan.mask[5] == 0)
        #expect(plan.mask[0] == plan.mask[1])
        #expect(plan.mask[2] == plan.mask[3])
        #expect(abs(plan.mask[0] - 1.316958) < 0.0005)
        #expect(abs(plan.mask[2] + 1.316958) < 0.0005)
        #expect(abs(plan.mask[0] + plan.mask[2]) < 0.0005)
        #expect(plan.indices.allSatisfy { $0 >= 0 && $0 <= 20 })
        // Host 1.5-entmax. The shaped GPU plan uses the same shift, tau, and square.
        let prior = UnlimitedOCRDashMath.entmax15([1, 0.5, 0])
        #expect(prior.count == 3)
        #expect(abs(prior[0] - 0.624197) < 0.0002)
        #expect(abs(prior[1] - 0.291667) < 0.0002)
        #expect(abs(prior[2] - 0.084136) < 0.0002)
        #expect(prior[0] > prior[1] && prior[1] > prior[2])
        #expect(UnlimitedOCRDashMath.entmax15([]).isEmpty)
    }

    @Test("Seq-1 decode fuse stacks GEMM rows and keeps 512 keys")
    func decodeFuseBudget() {
        let separate = UnlimitedOCRDecodeFuse.gemmBudget(layers: 12, denseLayers: 1, fused: false)
        let fused = UnlimitedOCRDecodeFuse.gemmBudget(layers: 12, denseLayers: 1, fused: true)
        #expect(separate.total == 153)
        #expect(separate.routerScore == 11)
        #expect(fused.total == 83)
        #expect(fused.routerScore == 0)
        #expect(fused.qkv == 12)
        #expect(fused.experts == 22)
        #expect(fused.attentionOut == separate.attentionOut)
        #expect(fused.lmHead == 1)
        #expect(UnlimitedOCRDashParams().attendedKeys == 512)

        let x: [Float] = [1, 2]
        let gate = [[Float(3), 4], [Float(5), 0]]
        let up = [[Float(0), 1]]
        let split = UnlimitedOCRDecodeFuse.referenceConcatDots(x, gate)
            + UnlimitedOCRDecodeFuse.referenceConcatDots(x, up)
        let stacked = UnlimitedOCRDecodeFuse.referenceConcatDots(x, gate + up)
        #expect(split == stacked)
        let score = [[Float(1), 0]]
        let three = UnlimitedOCRDecodeFuse.referenceConcatDots(x, score)
            + UnlimitedOCRDecodeFuse.referenceConcatDots(x, gate)
            + UnlimitedOCRDecodeFuse.referenceConcatDots(x, up)
        #expect(three == UnlimitedOCRDecodeFuse.referenceConcatDots(x, score + gate + up))

        #expect(UnlimitedOCRDecodeFuse.affineBits(packed: 160, groups: 20)?.bits == 4)
        #expect(UnlimitedOCRDecodeFuse.affineBits(packed: 160, groups: 20)?.group == 64)
        #expect(UnlimitedOCRDecodeFuse.affineBits(packed: 320, groups: 20)?.bits == 8)
        #expect(UnlimitedOCRDecodeFuse.affineBits(packed: 320, groups: 20)?.group == 64)
        #expect(UnlimitedOCRDecodeFuse.affineBits(packed: 3, groups: 5) == nil)
        #expect(UnlimitedOCRDecodeFuse.allowsFuse(nil))
        #expect(!UnlimitedOCRDecodeFuse.allowsFuse("0"))
        #expect(UnlimitedOCRDecodeFuse.allowsFuse("on"))

        #expect(
            UnlimitedOCRDecodeFuse.expertRowShape(x: [1, 1, 1280], indices: [1, 1, 6], hidden: 1280)
                == [1, 1, 6, 1280])
    }

    @Test("Seq-1 megakernel matches the affine SwiGLU and 512-key softmax")
    func megaDecodeReference() {
        #expect(!UnlimitedOCRMegaDecode.allows(nil))
        #expect(!UnlimitedOCRMegaDecode.allows("0"))
        #expect(UnlimitedOCRMegaDecode.allows("1"))
        let budget = UnlimitedOCRMegaDecode.budget(layers: 12, denseLayers: 1)
        #expect(budget.mlxGemm == 36)
        #expect(budget.swigluKernels == 23)
        #expect(budget.attendKernels == 12)
        #expect(UnlimitedOCRMegaDecode.nibble4(0x21, 0) == 1)
        #expect(UnlimitedOCRMegaDecode.nibble4(0x21, 1) == 2)
        #expect(UnlimitedOCRMegaDecode.byte8(0x21, 0) == 0x21)

        var x = [Float](repeating: 0, count: 64)
        x[0] = 1
        func plane(_ first: UInt32, rows: Int) -> UnlimitedOCRMegaDecode.Row {
            var words = [UInt32](repeating: 0, count: rows * 8)
            words[0] = first
            return UnlimitedOCRMegaDecode.Row(
                words: words,
                scales: Array(repeating: [1], count: rows),
                biases: Array(repeating: [0], count: rows),
                packed: 8)
        }
        let y = UnlimitedOCRMegaDecode.referenceSwiGLU(
            x: x, gate: plane(2, rows: 64), up: plane(4, rows: 64), down: plane(1, rows: 1), bits: 4)
        let hidden = UnlimitedOCRMegaDecode.silu(2) * 4
        #expect(y.count == 1)
        #expect(abs(y[0] - hidden) < 0.0001)
        #expect(abs(hidden - 7.046377) < 0.001)

        let attended = UnlimitedOCRMegaDecode.referenceAttend(
            q: [1, 0],
            keys: [[1, 0], [0, 1]],
            values: [[3, 0], [0, 4]],
            mask: [0, 0],
            scale: 1)
        #expect(attended.count == 2)
        #expect(attended[0] > attended[1])
        #expect(abs(attended[0] - 2.193) < 0.01)
        #expect(UnlimitedOCRDashParams().attendedKeys == 512)
    }
}

private let ocr2JSON = """
{
  "model_type": "unlimited-ocr",
  "language_config": {
    "hidden_size": 1280,
    "num_hidden_layers": 12,
    "num_attention_heads": 10,
    "v_head_dim": 128,
    "sliding_window_size": 128
  },
  "sparse_attention_config": {
    "training_stage": "dash",
    "use_sparse_attention": true,
    "sparse_block_size": 32,
    "sparse_local_block": 4,
    "dash_topk_blocks": 12,
    "sparse_topk_blocks": 16,
    "image_token_id": 128815,
    "page_token_id": 128827,
    "per_page_image_tokens": 273
  }
}
"""
