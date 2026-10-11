import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
@testable import NovaMLXEngine
import Testing

@Suite("DSpark")
struct DSparkTests {
    private static let speculatorsJSON = """
        {
          "architectures": ["DSparkDraftModel"],
          "speculators_model_type": "dspark",
          "speculators_version": "0.9.0.dev28",
          "block_size": 8,
          "mask_token_id": 248077,
          "aux_hidden_state_layer_ids": [4, 12, 20, 28, 36, 44, 52, 60],
          "draft_vocab_size": 248320,
          "markov_head_type": "vanilla",
          "markov_rank": 256,
          "enable_confidence_head": true,
          "confidence_head_with_markov": true,
          "sample_from_anchor": true,
          "sliding_window_non_causal": false,
          "speculators_config": {
            "algorithm": "dspark",
            "default_proposal_method": "greedy",
            "proposal_methods": [
              {
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 8,
                "verifier_accept_k": 1
              }
            ]
          },
          "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_hidden_layers": 5,
            "num_attention_heads": 20,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "rms_norm_eps": 1e-06,
            "attention_bias": false,
            "rope_parameters": {
              "rope_theta": 10000000,
              "rope_type": "default",
              "partial_rotary_factor": 1.0
            },
            "sliding_window": 2048,
            "use_sliding_window": true,
            "layer_types": [
              "sliding_attention", "sliding_attention", "sliding_attention",
              "sliding_attention", "sliding_attention"
            ]
          }
        }
        """.data(using: .utf8)!

    private static func jsonObject(_ data: Data) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test("Speculators packaging decodes into the DeepSpec schema")
    func parsesSpeculatorsPackaging() throws {
        let data = Self.speculatorsJSON
        let object = try Self.jsonObject(data)
        #expect(DSparkConfig.isDSparkDraftConfig(jsonObject: object))

        let cfg = try DSparkConfig.fromJSON(data)
        #expect(cfg.family == .qwen3)
        #expect(cfg.blockSize == 8)
        #expect(cfg.logitsStart == 0)
        #expect(cfg.kdraft == 8)
        #expect(cfg.targetLayerIds == [4, 12, 20, 28, 36, 44, 52, 60])
        #expect(cfg.maskTokenId == 248077)
        #expect(cfg.markovRank == 256)
        #expect(cfg.markovHeadType == "vanilla")
        #expect(cfg.enableConfidenceHead)
        #expect(cfg.confidenceHeadWithMarkov)
        #expect(cfg.slidingWindow == 2048)
        #expect(cfg.causalBlock)
        #expect(!cfg.attentionSink)

        // draft_vocab_size == vocab_size means no reduction — no d2t table.
        #expect(cfg.draftVocabSize == nil)
        #expect(cfg.vocabSize == 248_320)
        #expect(cfg.outVocabSize == 248_320)

        // Flattened backbone fields come from transformer_layer_config.
        #expect(cfg.hiddenSize == 5120)
        #expect(cfg.numHiddenLayers == 5)
        #expect(cfg.numAttentionHeads == 20)
        #expect(cfg.numKeyValueHeads == 4)
        #expect(cfg.headDim == 256)
        #expect(cfg.intermediateSize == 17_408)
        #expect(cfg.rmsNormEps == 1e-6)
        #expect(cfg.ropeTheta == 10_000_000)
        #expect(cfg.ropeType == "default")
        #expect(cfg.numTargetLayers == 61)

        let backbone = try #require(cfg.transformerLayerConfig)
        #expect(backbone.modelType == "qwen3")
        #expect(backbone.hiddenSize == 5120)
        #expect(backbone.numHiddenLayers == 5)
        #expect(backbone.numAttentionHeads == 20)
        #expect(backbone.numKeyValueHeads == 4)
        #expect(backbone.headDim == 256)
        #expect(backbone.ropeTheta == 10_000_000)
        #expect(backbone.ropeType == "default")
        #expect(backbone.slidingWindow == 2048)
        #expect(backbone.useSlidingWindow)
        #expect(backbone.layerTypes == Array(repeating: "sliding_attention", count: 5))
    }

    @Test("Speculators block width minus proposal width gives the anchor slot")
    func speculatorsAnchorSlotDerivation() throws {
        // block 7 with 6 speculative tokens reserves slot 0 as a pure anchor.
        let json = """
            {
              "architectures": ["DSparkDraftModel"],
              "speculators_model_type": "dspark",
              "block_size": 7,
              "mask_token_id": 4,
              "aux_hidden_state_layer_ids": [1, 2],
              "speculators_config": {
                "algorithm": "dspark",
                "proposal_methods": [{"speculative_tokens": 6}]
              },
              "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 64, "vocab_size": 128, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 2
              }
            }
            """.data(using: .utf8)!
        let cfg = try DSparkConfig.fromJSON(json)
        #expect(cfg.blockSize == 7)
        #expect(cfg.logitsStart == 1)
        #expect(cfg.kdraft == 6)
    }

    @Test("DeepSpec native config honors an explicit logits_start")
    func nativeExplicitLogitsStart() throws {
        let json = """
            {
              "model_type": "qwen3",
              "architectures": ["Qwen3DSparkModel"],
              "hidden_size": 2048,
              "vocab_size": 151936,
              "num_hidden_layers": 4,
              "intermediate_size": 8192,
              "num_attention_heads": 16,
              "num_key_value_heads": 4,
              "head_dim": 128,
              "block_size": 7,
              "mask_token_id": 4,
              "target_layer_ids": [5, 17, 29],
              "logits_start": 1,
              "markov_rank": 128
            }
            """.data(using: .utf8)!
        let object = try Self.jsonObject(json)
        #expect(DSparkConfig.isDSparkDraftConfig(jsonObject: object))
        let cfg = try DSparkConfig.fromJSON(json)
        #expect(cfg.family == .qwen3)
        #expect(cfg.logitsStart == 1)
        #expect(cfg.kdraft == 6)
        #expect(cfg.markovRank == 128)
        #expect(cfg.targetLayerIds == [5, 17, 29])
        #expect(cfg.transformerLayerConfig == nil)
        // No causal/sliding markers on a stock DeepSpec head — bidirectional block.
        #expect(!cfg.causalBlock)
        #expect(cfg.slidingWindow == nil)
    }

    @Test("DeepSpec native config defaults logits_start to 0")
    func nativeDefaultLogitsStart() throws {
        let json = """
            {
              "model_type": "qwen3",
              "architectures": ["DSparkDraftModel"],
              "hidden_size": 1024,
              "vocab_size": 32000,
              "num_hidden_layers": 3,
              "intermediate_size": 4096,
              "num_attention_heads": 8,
              "num_key_value_heads": 2,
              "block_size": 9,
              "mask_token_id": 7,
              "target_layer_ids": [2, 8, 14, 20, 26]
            }
            """.data(using: .utf8)!
        let cfg = try DSparkConfig.fromJSON(json)
        #expect(cfg.logitsStart == 0)
        #expect(cfg.kdraft == 9)
        #expect(cfg.headDim == 128)
        #expect(cfg.ropeTheta == 1_000_000)
    }

    @Test("A speculators config for another algorithm is neither detected nor decoded")
    func rejectsNonDSparkSpeculators() throws {
        let json = """
            {
              "architectures": ["EagleDraftModel"],
              "speculators_model_type": "eagle",
              "speculators_config": {"algorithm": "eagle"}
            }
            """.data(using: .utf8)!
        let object = try Self.jsonObject(json)
        #expect(!DSparkConfig.isDSparkDraftConfig(jsonObject: object))
        #expect(throws: DSparkError.self) {
            _ = try DSparkConfig.fromJSON(json)
        }
    }

    @Test("DFlash2 and plain Qwen3 configs are not DSpark")
    func detectionNegatives() throws {
        let dflash2 = """
            {
              "model_type": "qwen3",
              "architectures": ["DFlash2DraftModel"],
              "dflash_config": {"block_size": 8, "selector_rank": 8, "selector_top_k": 4}
            }
            """.data(using: .utf8)!
        let dflash2Object = try Self.jsonObject(dflash2)
        #expect(!DSparkConfig.isDSparkDraftConfig(jsonObject: dflash2Object))

        let qwen3 = """
            {
              "model_type": "qwen3",
              "architectures": ["Qwen3ForCausalLM"]
            }
            """.data(using: .utf8)!
        let qwen3Object = try Self.jsonObject(qwen3)
        #expect(!DSparkConfig.isDSparkDraftConfig(jsonObject: qwen3Object))
    }

    @Test("Detection accepts each DSpark signal spelling")
    func detectionSignals() throws {
        let byModelType = """
            {"model_type": "dspark"}
            """.data(using: .utf8)!
        let byModelTypeObject = try Self.jsonObject(byModelType)
        #expect(DSparkConfig.isDSparkDraftConfig(jsonObject: byModelTypeObject))

        let byArchitecture = """
            {"model_type": "qwen3", "architectures": ["DSparkDraftModel"]}
            """.data(using: .utf8)!
        let byArchitectureObject = try Self.jsonObject(byArchitecture)
        #expect(DSparkConfig.isDSparkDraftConfig(jsonObject: byArchitectureObject))

        let bySpeculatorsType = """
            {"speculators_model_type": "dspark"}
            """.data(using: .utf8)!
        let bySpeculatorsTypeObject = try Self.jsonObject(bySpeculatorsType)
        #expect(DSparkConfig.isDSparkDraftConfig(jsonObject: bySpeculatorsTypeObject))

        let bySpeculatorsAlgorithm = """
            {"speculators_config": {"algorithm": "dspark"}}
            """.data(using: .utf8)!
        let bySpeculatorsAlgorithmObject = try Self.jsonObject(bySpeculatorsAlgorithm)
        #expect(DSparkConfig.isDSparkDraftConfig(jsonObject: bySpeculatorsAlgorithmObject))
    }

    @Test("isDSparkDraftConfig reads config.json from a model directory")
    func detectionFromModelDir() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dspark-cfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let json = """
            {"model_type":"qwen3","architectures":["DSparkDraftModel"],"block_size":8}
            """
        try json.write(
            to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(DSparkConfig.isDSparkDraftConfig(at: dir))
    }

    // MARK: - Drafter model (synthetic tiny configs, no real weights)

    private static func tinyConfig(
        logitsStart: Int = 0, causalBlock: Bool = true, slidingWindow: Int? = 8,
        markovRank: Int = 8, taps: [Int] = [0, 1, 2]
    ) -> DSparkConfig {
        DSparkConfig(
            family: .qwen3,
            hiddenSize: 64,
            vocabSize: 32,
            numHiddenLayers: 2,
            intermediateSize: 128,
            rmsNormEps: 1e-6,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 16,
            attentionKEqV: false,
            ropeTheta: 10_000,
            partialRotaryFactor: 1.0,
            ropeType: "default",
            causalBlock: causalBlock,
            slidingWindow: slidingWindow,
            blockSize: 4,
            maskTokenId: 31,
            logitsStart: logitsStart,
            targetLayerIds: taps,
            numTargetLayers: 4,
            markovRank: markovRank,
            enableConfidenceHead: true,
            confidenceHeadWithMarkov: true,
            finalLogitSoftcapping: nil,
            mlpActivation: "silu",
            normStyle: "qwen",
            useVNorm: false)
    }

    private static func tinyModel(
        logitsStart: Int = 0, causalBlock: Bool = true, slidingWindow: Int? = 8,
        markovRank: Int = 8, zeroHead: Bool = false
    ) -> DSparkDraftModel {
        let model = DSparkDraftModel(
            tinyConfig(
                logitsStart: logitsStart, causalBlock: causalBlock,
                slidingWindow: slidingWindow, markovRank: markovRank))
        let embed = Embedding(weight: MLXRandom.normal([32, 64]))
        let head = MLXRandom.normal([32, 64])
        model.bind(
            embed: { embed($0) },
            lmHead: zeroHead
                ? { h in MLXArray.zeros(Array(h.shape.dropLast()) + [32]) }
                : { h in matmul(h, head.transposed()) })
        return model
    }

    @Test("selectBlock drafts cap tokens and grows the ctx cache only on committed rows")
    func selectBlockShapes() throws {
        let model = Self.tinyModel()
        let cache = model.makeCtxCache()
        #expect(cache.count == 2)
        #expect(cache[0].offset == 0)

        let fused = MLXRandom.normal([1, 3, 3 * 64])
        let block = MLXArray([Int32(5), 31, 31, 31], [1, 4])
        let draft = model.selectBlock(
            block: block, pendingFused: fused, cache: cache, cap: 3, anchorId: 5)
        #expect(draft.tokens.count == 3)
        #expect(draft.confidences.count == 3)
        #expect(cache[0].offset == 3)
        #expect(try #require(cache[0].keys).dim(2) == 3)

        // A dense step appends context without drafting.
        model.appendContext(MLXRandom.normal([1, 2, 3 * 64]), cache: cache)
        #expect(cache[0].offset == 5)
        #expect(try #require(cache[0].keys).dim(2) == 5)

        // The next block ropes its Q past every cached ctx row.
        let draft2 = model.selectBlock(
            block: block, pendingFused: MLXRandom.normal([1, 1, 3 * 64]), cache: cache,
            cap: 4, anchorId: 5)
        #expect(draft2.tokens.count == 4)
        #expect(cache[0].offset == 6)
    }

    @Test("ctx window bounds the context view of withBlock")
    func ctxWindowViewBounds() throws {
        let cache = DSparkCtxCache()
        cache.append(
            keys: MLXArray.zeros([1, 2, 5, 16]), values: MLXArray.zeros([1, 2, 5, 16]))
        let (kFull, _) = cache.withBlock(
            keys: MLXArray.zeros([1, 2, 4, 16]), values: MLXArray.zeros([1, 2, 4, 16]),
            window: nil)
        #expect(kFull.dim(2) == 9)
        let (kWin, vWin) = cache.withBlock(
            keys: MLXArray.zeros([1, 2, 4, 16]), values: MLXArray.zeros([1, 2, 4, 16]),
            window: 3)
        #expect(kWin.dim(2) == 7)
        #expect(vWin.dim(2) == 7)

        cache.trim(to: 2)
        #expect(cache.offset == 2)
        let trimmedKeys = try #require(cache.keys)
        #expect(trimmedKeys.dim(2) == 2)
    }

    @Test("block mask is causal within the block and windowed over the context")
    func blockMaskValues() throws {
        // ctxLen 5, block 2 at absolute positions 5 and 6, window 3.
        let mask = try #require(
            DSparkBlockMask.make(causal: true, window: 3, blockOffset: 5, ctxLen: 5, qLen: 2))
        #expect(mask.shape == [2, 7])
        eval(mask)
        let m = mask.asArray(Float.self)
        // query 0 (pos 5): causal k<=5, window k>2 -> keys {3,4,5}
        // query 1 (pos 6): causal k<=6, window k>3 -> keys {4,5,6}
        let expected: [[Bool]] = [
            [false, false, false, true, true, true, false],
            [false, false, false, false, true, true, true],
        ]
        for q in 0 ..< 2 {
            for k in 0 ..< 7 {
                #expect((m[q * 7 + k] == 0) == expected[q][k])
            }
        }

        // Nothing to hide on a short context with no causal block.
        #expect(
            DSparkBlockMask.make(causal: false, window: 8, blockOffset: 4, ctxLen: 4, qLen: 2)
                == nil)
        #expect(
            DSparkBlockMask.make(causal: false, window: nil, blockOffset: 4, ctxLen: 4, qLen: 2)
                == nil)
    }

    @Test("Markov walk biases each position with the previous drafted token")
    func markovWalk() throws {
        let model = Self.tinyModel(markovRank: 8)
        // w1: token t -> one-hot e_t (t < 8); w2: e_r -> +10 on token r+1.
        var w1 = [Float](repeating: 0, count: 32 * 8)
        for t in 0 ..< 8 { w1[t * 8 + t] = 1 }
        var w2 = [Float](repeating: 0, count: 32 * 8)
        for r in 0 ..< 8 { w2[(r + 1) * 8 + r] = 10 }
        model.update(
            parameters: ModuleParameters.unflattened([
                "markov_head.markov_w1.weight": MLXArray(w1, [32, 8]),
                "markov_head.markov_w2.weight": MLXArray(w2, [32, 8]),
            ]))

        // Zero base logits: the Markov bias alone decides every position. The anchor
        // (3) biases position 0 onto token 4, then each drafted token biases the next.
        let tokens = model.sampleBlock(MLXArray.zeros([3, 32]), firstPrevToken: 3)
        eval(tokens)
        #expect(tokens.asArray(Int.self) == [4, 5, 6])
    }

    @Test("headSlice starts at logitsStart and selectBlock honors it")
    func headSliceHonorsLogitsStart() throws {
        let model0 = Self.tinyModel(logitsStart: 0)
        let model1 = Self.tinyModel(logitsStart: 1)
        let hidden = MLXArray.arange(0, 4 * 64, dtype: .float32).reshaped([1, 4, 64])
        let s0 = model0.headSlice(hidden, cap: 2)
        #expect(s0.shape == [1, 2, 64])
        eval(s0, hidden)
        #expect(arrayEqual(s0[0, 0], hidden[0, 0]).item(Bool.self))
        let s1 = model1.headSlice(hidden, cap: 2)
        #expect(arrayEqual(s1[0, 0], hidden[0, 1]).item(Bool.self))

        // logitsStart 1: slot 0 is a pure anchor, cap 2 reads slots 1 and 2.
        let cache = model1.makeCtxCache()
        let draft = model1.selectBlock(
            block: MLXArray([Int32(5), 31, 31, 31], [1, 4]),
            pendingFused: MLXRandom.normal([1, 1, 3 * 64]), cache: cache, cap: 2,
            anchorId: 5)
        #expect(draft.tokens.count == 2)
        #expect(model1.kdraft == 3)
        #expect(model1.draftWidth(cap: 2) == 3)
        #expect(model0.draftWidth(cap: 2) == 2)
    }

    @Test("ctx K roped at absolute positions: split appends match, shifted offsets differ")
    func ropeAbsoluteOffsets() throws {
        let model = Self.tinyModel()
        let fused = MLXRandom.normal([1, 4, 3 * 64])

        let cacheA = model.makeCtxCache()
        model.updateContext(fused, offset: 0, cache: cacheA)

        let cacheB = model.makeCtxCache()
        model.updateContext(fused[0..., 0 ..< 2, 0...], offset: 0, cache: cacheB)
        model.updateContext(fused[0..., 2..., 0...], offset: 2, cache: cacheB)

        let keysA = try #require(cacheA[0].keys)
        let keysB = try #require(cacheB[0].keys)
        eval(keysA, keysB)
        #expect(allClose(keysA, keysB, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(cacheA[0].offset == 4)
        #expect(cacheB[0].offset == 4)

        // The same rows roped at a shifted offset are not interchangeable.
        let cacheC = model.makeCtxCache()
        model.updateContext(fused, offset: 3, cache: cacheC)
        let keysC = try #require(cacheC[0].keys)
        eval(keysC)
        #expect(!allClose(keysA, keysC, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    @Test("strict weight load matches the checkpoint tensor names 1:1")
    func weightNameMapping() throws {
        let model = DSparkDraftModel(Self.tinyConfig())
        var weights: [String: MLXArray] = [
            "fc.weight": MLXRandom.normal([64, 3 * 64]),
            "hidden_norm.weight": MLXArray.ones([64]),
            "norm.weight": MLXArray.ones([64]),
            "markov_head.markov_w1.weight": MLXRandom.normal([32, 8]),
            "markov_head.markov_w2.weight": MLXRandom.normal([32, 8]),
            "confidence_head.proj.weight": MLXRandom.normal([1, 64 + 8]),
            "confidence_head.proj.bias": MLXArray.zeros([1]),
        ]
        for layer in 0 ..< 2 {
            let p = "layers.\(layer)"
            weights["\(p).input_layernorm.weight"] = MLXArray.ones([64])
            weights["\(p).post_attention_layernorm.weight"] = MLXArray.ones([64])
            weights["\(p).self_attn.q_proj.weight"] = MLXRandom.normal([4 * 16, 64])
            weights["\(p).self_attn.k_proj.weight"] = MLXRandom.normal([2 * 16, 64])
            weights["\(p).self_attn.v_proj.weight"] = MLXRandom.normal([2 * 16, 64])
            weights["\(p).self_attn.o_proj.weight"] = MLXRandom.normal([64, 4 * 16])
            weights["\(p).self_attn.q_norm.weight"] = MLXArray.ones([16])
            weights["\(p).self_attn.k_norm.weight"] = MLXArray.ones([16])
            weights["\(p).mlp.gate_proj.weight"] = MLXRandom.normal([128, 64])
            weights["\(p).mlp.up_proj.weight"] = MLXRandom.normal([128, 64])
            weights["\(p).mlp.down_proj.weight"] = MLXRandom.normal([64, 128])
        }
        #expect(throws: Never.self) { try model.loadWeights(weights) }

        // The head ships no embed_tokens / lm_head — an extra key must not load.
        var extra = weights
        extra["embed_tokens.weight"] = MLXArray.zeros([32, 64])
        #expect(throws: (any Error).self) { try model.loadWeights(extra) }

        // A missing key must not load either.
        weights.removeValue(forKey: "norm.weight")
        #expect(throws: (any Error).self) { try model.loadWeights(weights) }
    }

    // MARK: - DSparkTokenIterator (scripted mock target)

    /// Scripted verify "target": every logits row has a single +10 peak, so argmax is
    /// fully deterministic. `committed` models the target KV length across
    /// verify/rollback; `verifyInputs` records every verified token window.
    private final class MockDSparkTarget: Module, DFlashTarget, KVCacheDimensionProvider {
        enum Truth {
            /// Row i peaks at tokens[i+1] for i < forcedAccept (draft matches the
            /// target) and at (tokens[i+1] + 1) % V otherwise; the last row peaks at
            /// `bonus`. Exercises the accept path position by position.
            case draftAware(forcedAccept: Int, bonus: Int)
            /// Row i peaks at script[(committed + i + 1) % script.count] — a pure
            /// position function, so acceptance cannot change the emitted stream.
            case positional(script: [Int])
        }

        static let vocab = 32
        static let fusedWidth = 3 * 64

        let truth: Truth
        let embeddingTable = Embedding(weight: MLXRandom.normal([vocab, 64]))
        let head = MLXRandom.normal([vocab, 64])
        private(set) var committed = 0
        private(set) var rollbacks: [(nRejected: Int, specWidth: Int)] = []
        private(set) var verifyInputs: [[Int]] = []

        init(truth: Truth) {
            self.truth = truth
            super.init()
        }

        var kvHeads: [Int] { [] }

        var dflashEmbedding: Embedding { embeddingTable }
        func dflashEmbed(_ tokens: MLXArray) -> MLXArray { embeddingTable(tokens) }
        func dflashLmHead(_ hidden: MLXArray) -> MLXArray {
            matmul(hidden, head.transposed())
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
            prefill: PrefillParameters
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        private func peakRow(_ i: Int, tokens: [Int], width: Int) -> Int {
            switch truth {
            case .draftAware(let forcedAccept, let bonus):
                if i == width - 1 { return bonus }
                let next = tokens[i + 1]
                return i < forcedAccept ? next : (next + 1) % Self.vocab
            case .positional(let script):
                return script[(committed + i + 1) % script.count]
            }
        }

        private func forwardLogits(_ tokens: MLXArray) -> MLXArray {
            let ids = tokens.asArray(Int.self)
            let width = tokens.dim(1)
            var buffer = [Float](repeating: 0, count: width * Self.vocab)
            for i in 0 ..< width {
                buffer[i * Self.vocab + peakRow(i, tokens: ids, width: width)] = 10
            }
            committed += width
            verifyInputs.append(ids)
            return MLXArray(buffer, [1, width, Self.vocab])
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            forwardLogits(inputs)
        }

        func dflashVerify(
            _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
        ) -> (logits: MLXArray, fused: MLXArray) {
            let logits = forwardLogits(tokens)
            let fused = MLXRandom.normal([1, tokens.dim(1), Self.fusedWidth])
            return (logits, fused)
        }

        func dflashRollback(cache: [KVCache], nRejected: Int, specWidth: Int) {
            rollbacks.append((nRejected, specWidth))
            committed -= nRejected
        }
    }

    private static func runIterator(
        target: MockDSparkTarget, maxTokens: Int, prompt: [Int32] = [1, 2, 3]
    ) throws -> (tokens: [Int], iterator: DSparkTokenIterator) {
        let drafter = tinyModel()
        var iterator = try DSparkTokenIterator(
            input: LMInput(text: .init(tokens: MLXArray(prompt))),
            target: target,
            drafter: drafter,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
        )
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return (tokens, iterator)
    }

    @Test("full accept commits anchor plus all drafts to the drafter ctx")
    func iteratorFullAccept() throws {
        let target = MockDSparkTarget(truth: .draftAware(forcedAccept: 4, bonus: 9))
        let (tokens, iterator) = try Self.runIterator(target: target, maxTokens: 6)

        // Prefill last row peaks at the bonus token; the round then commits the 4
        // drafts plus another bonus.
        #expect(tokens.count == 6)
        #expect(tokens[0] == 9)
        let verifyWindow = try #require(target.verifyInputs.last)
        #expect(verifyWindow.count == 5)
        #expect(Array(tokens[1 ..< 5]) == Array(verifyWindow[1 ..< 5]))
        #expect(tokens[5] == 9)

        #expect(iterator.totalProposed == 4)
        #expect(iterator.totalAccepted == 4)
        #expect(target.rollbacks.count == 1)
        #expect(target.rollbacks[0].nRejected == 0)
        #expect(target.rollbacks[0].specWidth == 5)
        // ctx holds the 3 prompt rows + anchor + 4 accepted drafts; the bonus token's
        // row arrives next round. Rejected rows never enter the ctx.
        #expect(iterator.draftCache[0].offset == 8)
        #expect(target.committed == 8)
    }

    @Test("partial accept rolls back the rejected tail and commits only kept rows")
    func iteratorPartialAccept() throws {
        let target = MockDSparkTarget(truth: .draftAware(forcedAccept: 2, bonus: 9))
        let (tokens, iterator) = try Self.runIterator(target: target, maxTokens: 6)

        // Round 1: cap 4, first 2 drafts match -> commit [d0, d1, bonus-peak],
        // rollback (2, 5). Round 2: budget 2 -> cap 1, single draft matches ->
        // commit [e0, 9], rollback (0, 2).
        #expect(tokens.count == 6)
        #expect(target.rollbacks.map(\.nRejected) == [2, 0])
        #expect(target.rollbacks.map(\.specWidth) == [5, 2])
        #expect(iterator.totalProposed == 5)
        #expect(iterator.totalAccepted == 3)

        let round1 = target.verifyInputs[target.verifyInputs.count - 2]
        let d2 = round1[3]
        #expect(tokens[0] == 9)
        #expect(tokens[1] == round1[1])
        #expect(tokens[2] == round1[2])
        #expect(tokens[3] == (d2 + 1) % MockDSparkTarget.vocab)
        let round2 = try #require(target.verifyInputs.last)
        #expect(round2.count == 2)
        #expect(tokens[4] == round2[1])
        #expect(tokens[5] == 9)

        #expect(iterator.draftCache[0].offset == 8)
        #expect(target.committed == 8)
    }

    @Test("zero accept commits only the target argmax; tail budget forces a dense step")
    func iteratorZeroAcceptAndDenseStep() throws {
        let target = MockDSparkTarget(truth: .draftAware(forcedAccept: 0, bonus: 9))
        let (tokens, iterator) = try Self.runIterator(target: target, maxTokens: 3)

        // Round 1: cap 1, draft mismatches -> commit the target argmax only,
        // rollback (1, 2). Round 2: budget 1 -> dense step emitting the bonus peak.
        #expect(tokens.count == 3)
        let round1 = target.verifyInputs[target.verifyInputs.count - 2]
        #expect(round1.count == 2)
        let e0 = round1[1]
        #expect(tokens[0] == 9)
        #expect(tokens[1] == (e0 + 1) % MockDSparkTarget.vocab)
        #expect(tokens[2] == 9)

        #expect(iterator.totalProposed == 1)
        #expect(iterator.totalAccepted == 0)
        #expect(target.rollbacks.map(\.nRejected) == [1])
        #expect(target.rollbacks.map(\.specWidth) == [2])
        // ctx: 3 prompt rows + 1 kept verify row + 1 dense row.
        #expect(iterator.draftCache[0].offset == 5)
        #expect(target.committed == 5)
    }

    @Test("speculative stream matches dense decode token-for-token")
    func iteratorLosslessVsDense() throws {
        let script = [5, 11, 3, 27, 8, 19, 1]
        let maxTokens = 12

        let specTarget = MockDSparkTarget(truth: .positional(script: script))
        let (specTokens, iterator) = try Self.runIterator(
            target: specTarget, maxTokens: maxTokens)

        // Dense reference: same scripted target, one token per forward.
        let denseTarget = MockDSparkTarget(truth: .positional(script: script))
        var denseTokens: [Int] = []
        var logits = denseTarget.callAsFunction(MLXArray([Int32(1), 2, 3], [1, 3]), cache: nil)
        var token = argMax(logits[0, -1]).item(Int.self)
        denseTokens.append(token)
        while denseTokens.count < maxTokens {
            logits = denseTarget.callAsFunction(MLXArray([Int32(token)], [1, 1]), cache: nil)
            token = argMax(logits[0, -1]).item(Int.self)
            denseTokens.append(token)
        }

        let expected = (0 ..< maxTokens).map { script[(3 + $0) % script.count] }
        #expect(specTokens == expected)
        #expect(specTokens == denseTokens)
        // Target KV and drafter ctx agree: prompt + every emitted token but the last
        // (the final pending anchor is never fed back).
        #expect(specTarget.committed == 3 + maxTokens - 1)
        #expect(iterator.draftCache[0].offset == 3 + maxTokens - 1)
    }

    // MARK: - Registry (opt-in Qwen3.8 DSpark candidate)

    @Test("Qwen3.8 DSpark candidate is opt-in only — never auto-injected")
    func qwen38DSparkOptInOnly() {
        let registry = DraftModelRegistry.shared
        #expect(
            registry.dsparkOptInCandidateIds(forMainId: "mlx-community/Qwen3.8-27B-8bit")
                == ["RedHatAI/Qwen3.8-27B-speculator.dspark"])
        #expect(registry.dsparkOptInCandidateIds(forMainId: "Qwen/Qwen3.8-27B").count == 1)
        #expect(
            registry.dsparkOptInCandidateIds(
                forMainId: "RedHatAI/Qwen3.8-27B-speculator.dspark"
            ).isEmpty)
        #expect(
            registry.dsparkOptInCandidateIds(
                forMainId: "mlx-community/Qwen3.8-27B-MTP-4bit"
            ).isEmpty)

        // The auto-inject source (`dsparkCandidate`) stays DeepSeek-only: qwen3.8
        // must not resolve the RedHat head through it.
        #expect(registry.dsparkCandidate(forMainId: "mlx-community/Qwen3.8-27B-8bit") == nil)
    }

    @Test("Qwen3-8B/14B map to the DeepSpec-native heads, opt-in only")
    func qwen3SmallDSparkOptInOnly() {
        let registry = DraftModelRegistry.shared
        #expect(
            registry.dsparkOptInCandidateIds(forMainId: "mlx-community/Qwen3-8B-8bit")
                == ["deepseek-ai/dspark_qwen3_8b_block7"])
        #expect(
            registry.dsparkOptInCandidateIds(forMainId: "mlx-community/Qwen3-14B-8bit")
                == ["deepseek-ai/dspark_qwen3_14b_block7"])
        #expect(registry.dsparkOptInCandidateIds(forMainId: "Qwen/Qwen3-8B").count == 1)
        // No shadowing across the families.
        #expect(
            registry.dsparkOptInCandidateIds(forMainId: "mlx-community/Qwen3.8-27B-8bit")
                == ["RedHatAI/Qwen3.8-27B-speculator.dspark"])
        #expect(
            registry.dsparkOptInCandidateIds(
                forMainId: "deepseek-ai/dspark_qwen3_8b_block7"
            ).isEmpty)
        // Auto-inject stays blind to them.
        #expect(registry.dsparkCandidate(forMainId: "mlx-community/Qwen3-8B-8bit") == nil)
        #expect(registry.dsparkCandidate(forMainId: "mlx-community/Qwen3-14B-8bit") == nil)
    }

    @Test("real RedHat speculators checkpoint loads through DSparkDrafterStore")
    func realCheckpointLoads() throws {
        let dir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Models/RedHatAI/Qwen3.8-27B-speculator.dspark")
        guard FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path)
        else { return }
        #expect(DSparkDrafterStore.isLoadableDSparkDraft(
            modelId: "RedHatAI/Qwen3.8-27B-speculator.dspark"))
        let drafter = try DSparkDrafterStore.shared.drafter(for: dir)
        #expect(drafter.config.blockSize == 8)
        #expect(drafter.config.kdraft == 8)
        #expect(drafter.config.maskTokenId == 248077)
        #expect(drafter.config.targetLayerIds == [4, 12, 20, 28, 36, 44, 52, 60])
        #expect(drafter.config.causalBlock)
        #expect(drafter.config.slidingWindow == 2048)
        // Second fetch hits the store cache.
        #expect(try DSparkDrafterStore.shared.drafter(for: dir) === drafter)
    }

    // MARK: - Qwen3 target taps (DFlashTarget conformance)

    private static func tinyQwen3() throws -> Qwen3Model {
        let json = """
            {
              "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
              "num_attention_heads": 4, "rms_norm_eps": 0.000001, "vocab_size": 99,
              "num_key_value_heads": 2, "head_dim": 16
            }
            """
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
        return Qwen3Model(config)
    }

    @Test("Qwen3 forwardTapped: tap slices agree across runs and with the plain forward")
    func qwen3ForwardTapped() throws {
        let model = try Self.tinyQwen3()
        let ids = MLXArray([Int32(3), 1, 4, 1, 5], [1, 5])

        func verify(_ taps: [Int]) throws -> (logits: MLXArray, fused: MLXArray) {
            let cache = try model.newCache(parameters: GenerateParameters())
            let out = model.dflashVerify(ids, cache: cache, tapLayerIds: taps)
            eval(out.logits, out.fused)
            return out
        }

        let none = try verify([])
        let last = try verify([3])
        let mid = try verify([1])
        let both = try verify([1, 3])

        // No taps: fused is the final pre-norm hidden — identical to tapping the last
        // layer. (Qwen3ModelInner.layers is fileprivate, so taps are cross-checked
        // against each other through the public DFlashTarget interface instead of a
        // manual layer loop.)
        #expect(none.fused.shape == [1, 5, 64])
        #expect(arrayEqual(none.fused, last.fused).item(Bool.self))
        // Two taps concatenate channel-wise in layer order.
        #expect(both.fused.shape == [1, 5, 128])
        #expect(arrayEqual(both.fused[0..., 0..., 0 ..< 64], mid.fused).item(Bool.self))
        #expect(arrayEqual(both.fused[0..., 0..., 64...], last.fused).item(Bool.self))
        // Distinct layers produce distinct hiddens (random tiny weights).
        #expect(!arrayEqual(mid.fused, last.fused).item(Bool.self))

        // Untapped regression: verify logits equal the plain forward's logits.
        let cache = try model.newCache(parameters: GenerateParameters())
        let plain = model(ids, cache: cache)
        eval(plain)
        #expect(arrayEqual(none.logits, plain).item(Bool.self))
    }

    @Test("Qwen3 dflashRollback trims the KV cache by the rejected tail")
    func qwen3RollbackTrimsKV() throws {
        let model = try Self.tinyQwen3()
        let cache = try model.newCache(parameters: GenerateParameters())
        _ = model.dflashVerify(
            MLXArray([Int32(3), 1, 4, 1, 5], [1, 5]), cache: cache, tapLayerIds: [1])
        #expect(cache.compactMap { ($0 as? BaseKVCache)?.offset } == [5, 5, 5, 5])

        model.dflashRollback(cache: cache, nRejected: 2, specWidth: 5)
        #expect(cache.compactMap { ($0 as? BaseKVCache)?.offset } == [3, 3, 3, 3])

        // Zero rejected is a no-op.
        model.dflashRollback(cache: cache, nRejected: 0, specWidth: 3)
        #expect(cache.compactMap { ($0 as? BaseKVCache)?.offset } == [3, 3, 3, 3])
    }

    @Test("iterator init fails loudly on drafter/target tap-width mismatch")
    func iteratorRejectsMismatchedTarget() {
        // 4-tap drafter config expects a 4×64=256-wide fused; the mock emits 3×64.
        let model = DSparkDraftModel(Self.tinyConfig(taps: [0, 1, 2, 3]))
        let target = MockDSparkTarget(truth: .positional(script: [1, 2, 3]))
        #expect(throws: DSparkError.self) {
            _ = try DSparkTokenIterator(
                input: LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3]))),
                target: target,
                drafter: model,
                parameters: GenerateParameters(maxTokens: 4, temperature: 0))
        }
    }

    // MARK: - DeepSpec-native (deepseek-ai) packaging

    private static func safetensorsShape(of key: String, in url: URL) throws -> [Int]? {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else {
            return nil
        }
        let len = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }
        guard let header = try handle.read(upToCount: Int(len)) else { return nil }
        let json = try JSONSerialization.jsonObject(with: header) as? [String: Any]
        return (json?[key] as? [String: Any])?["shape"] as? [Int]
    }

    @Test("DeepSpec-native Qwen3-8B drafter parses; fc is nTaps × hidden wide")
    func realQwen38BDrafterConfig() throws {
        let dir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Models/deepseek-ai/dspark_qwen3_8b_block7")
        let cfgURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfgURL.path) else { return }

        let cfg = try DSparkConfig.fromJSON(Data(contentsOf: cfgURL))
        #expect(cfg.family == .qwen3)
        #expect(cfg.blockSize == 7)
        #expect(cfg.logitsStart == 0)
        #expect(cfg.kdraft == 7)
        #expect(cfg.targetLayerIds == [1, 9, 17, 25, 33])
        #expect(cfg.numTargetLayers == 36)
        #expect(cfg.hiddenSize == 4096)
        #expect(cfg.vocabSize == 151_936)
        // Bidirectional DeepSpec block — no causal/sliding mask.
        #expect(!cfg.causalBlock)
        #expect(cfg.slidingWindow == nil)
        #expect(cfg.draftVocabSize == nil)
        #expect(
            DSparkDrafterStore.isLoadableDSparkDraft(
                modelId: "deepseek-ai/dspark_qwen3_8b_block7"))

        let st = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: st.path) else { return }
        // fc.weight [hidden, nTaps × hidden] — the 36-layer 8B target taps 5 layers
        // at hidden 4096.
        #expect(try Self.safetensorsShape(of: "fc.weight", in: st) == [4096, 5 * 4096])

        // Full store load: strict name + shape verify against the module tree.
        let drafter = try DSparkDrafterStore.shared.drafter(for: dir)
        #expect(drafter.config.blockSize == 7)
        #expect(drafter.kdraft == 7)
    }

    @Test("DeepSpec-native Qwen3-14B drafter parses when downloaded")
    func realQwen314BDrafterConfig() throws {
        let dir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Models/deepseek-ai/dspark_qwen3_14b_block7")
        let cfgURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfgURL.path) else { return }

        let cfg = try DSparkConfig.fromJSON(Data(contentsOf: cfgURL))
        #expect(cfg.family == .qwen3)
        #expect(cfg.blockSize == 7)
        #expect(cfg.kdraft == 7)
        #expect(cfg.targetLayerIds == [1, 10, 19, 28, 37])
        #expect(cfg.numTargetLayers == 40)
        #expect(cfg.hiddenSize == 5120)

        let st = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: st.path) else { return }
        #expect(try Self.safetensorsShape(of: "fc.weight", in: st) == [5120, 5 * 5120])
    }
}
