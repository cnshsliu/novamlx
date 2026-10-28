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
