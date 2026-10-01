import Foundation
import ImageIO
import MLX
import MLXNN
import Testing
@testable import NovaMLXImage

@Suite("Qwen-Image 2.1 schedule")
struct Qwen21ScheduleTests {
    @Test("4-bit MLX keys land on the same modules as the bf16 checkpoint")
    func remapsQuantizedKeys() {
        let text = Qwen21Weights.remapText([
            "language_model.model.layers.0.mlp.down_proj.scales": MLXArray(Float(1)),
            "model.language_model.norm.weight": MLXArray(Float(1)),
        ])
        #expect(text["model.language_model.layers.0.mlp.down_proj.scales"] != nil)
        #expect(text["model.language_model.norm.weight"] != nil)

        let transformer = Qwen21Weights.remapTransformer([
            "modulation.0.weight": MLXArray(Float(1)),
            "modulation.0.scales": MLXArray(Float(1)),
            "time_text_embed.linear_1.biases": MLXArray(Float(1)),
            "img_in.weight": MLXArray(Float(1)),
        ])
        #expect(transformer["modulation.1.weight"] != nil)
        #expect(transformer["modulation.1.scales"] != nil)
        #expect(transformer["time_text_embed.timestep_embedder.linear_1.biases"] != nil)
        #expect(transformer["img_in.weight"] != nil)
        #expect(transformer["modulation.0.weight"] == nil)
    }

    @Test("diffusers convolutions become MLX layout and MLX convolutions stay put")
    func convolutionLayout() {
        let diffusers = Qwen21Weights.convolutionWeight(MLXArray.zeros([96, 4, 3, 3]))
        #expect(diffusers.shape == [96, 3, 3, 4])
        let point = Qwen21Weights.convolutionWeight(MLXArray.zeros([128, 128, 1, 1]))
        #expect(point.shape == [128, 1, 1, 128])
        let mlx = Qwen21Weights.convolutionWeight(MLXArray.zeros([96, 3, 3, 4]))
        #expect(mlx.shape == [96, 3, 3, 4])
        let mlxPoint = Qwen21Weights.convolutionWeight(MLXArray.zeros([128, 1, 1, 128]))
        #expect(mlxPoint.shape == [128, 1, 1, 128])
    }

    @Test("packed modulation weights fit the quantized linear, not the cached float one")
    func quantizedModulationAcceptsPackedWeight() throws {
        let modulation = Qwen21Modulation(dim: 64)
        _ = modulation.leafModules()
        Qwen21Weights.quantizeLayers(modulation, index: 1, bits: 4, group: 64)
        let packed = ModuleParameters.unflattened([
            ("layers.1.weight", MLXArray.zeros([256, 8]).asType(.uint32)),
            ("layers.1.scales", MLXArray.zeros([256, 1])),
            ("layers.1.biases", MLXArray.zeros([256, 1])),
        ])
        try modulation.update(parameters: packed, verify: [.allModelKeysSet, .shapeMismatch])
    }

    @Test("packed attention output weights fit the quantized list slot")
    func quantizedAttentionOutputAcceptsPackedWeight() throws {
        let attention = Qwen21Attention(dim: 64, numHeads: 1, headDim: 64)
        _ = attention.leafModules()
        Qwen21Weights.quantizeOutputs(attention, bits: 4, group: 64)
        let packed = ModuleParameters.unflattened([
            ("to_out.0.weight", MLXArray.zeros([64, 8]).asType(.uint32)),
            ("to_out.0.scales", MLXArray.zeros([64, 1])),
            ("to_out.0.biases", MLXArray.zeros([64, 1])),
        ])
        try attention.update(parameters: packed, verify: [.shapeMismatch])
    }

    @Test("4-bit linear packing matches the MLX community checkpoint")
    func quantizedLinearShape() {
        let linear = Linear(64, 4096, bias: false)
        let quantized = linear.toQuantized(groupSize: 64, bits: 4, mode: .affine) as? QuantizedLinear
        #expect(quantized?.weight.shape == [4096, 8])
        #expect(quantized?.scales.shape == [4096, 1])
        #expect(quantized?.biases?.shape == [4096, 1])
        let embed = Embedding(embeddingCount: 64, dimensions: 4096)
        let quantizedEmbed = embed.toQuantized(groupSize: 64, bits: 4, mode: .affine) as? QuantizedEmbedding
        #expect(quantizedEmbed?.weight.shape == [64, 512])
        #expect(quantizedEmbed?.scales.shape == [64, 64])
        #expect(quantizedEmbed?.biases?.shape == [64, 64])
    }

    @Test("viggle nodes keep the trained low-noise steps")
    func viggleNodes() {
        #expect(Qwen21Schedule.viggleNodes(steps: 6) == [1, 0.9375, 0.875, 0.75, 0.5, 0.25])
        #expect(Qwen21Schedule.viggleNodes(steps: 5) == [1, 0.875, 0.75, 0.5, 0.25])
        #expect(Qwen21Schedule.viggleNodes(steps: 4) == [1, 0.75, 0.5, 0.25])
        #expect(Qwen21Schedule.viggleNodes(steps: 8) == [1, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125])
        let seven = Qwen21Schedule.viggleNodes(steps: 7)
        #expect(seven.count == 7)
        #expect(seven[0] == 1)
        #expect(abs(seven[1] - 0.9583333) < 1e-4)
        #expect(seven[3] == 0.875)
        #expect(seven[6] == 0.25)
    }

    @Test("viggle 1024 schedule shifts and does not pin the last sigma at 0.02")
    func viggleSigmas1024() {
        let sigmas = Qwen21Schedule.sigmas(steps: 6, width: 1024, height: 1024, kind: .viggle)
        #expect(sigmas.count == 7)
        #expect(abs(sigmas[0] - 1) < 1e-5)
        #expect(sigmas[5] > 0.3)
        #expect(sigmas[5] < 0.5)
        #expect(sigmas[6] == 0)
    }

    @Test("turbo LoRA keys land on the Swift modules and the scale is alpha over rank")
    func loraKeysAndScale() {
        #expect(Qwen21LoRA.normalizedPath(for: "transformer.modulation.1.lora_A.weight") == "modulation.layers.1")
        #expect(
            Qwen21LoRA.normalizedPath(for: "transformer.transformer_blocks.3.attn.to_out.0.lora_B.weight")
                == "transformer_blocks.3.attn.to_out.0"
        )
        #expect(
            Qwen21LoRA.normalizedPath(for: "transformer.time_text_embed.timestep_embedder.linear_1.lora_A.weight")
                == "time_text_embed.timestep_embedder.linear_1"
        )
        #expect(
            Qwen21LoRA.normalizedPath(for: "transformer.transformer_blocks.0.img_mlp.gate_layer.lora_A.weight")
                == "transformer_blocks.0.img_mlp.gate_layer"
        )
        #expect(Qwen21LoRA.normalizedPath(for: "transformer.img_in.weight") == nil)
        let meta = #"{"transformer.lora_alpha":256,"transformer.r":256,"transformer.use_rslora":false}"#
        #expect(abs(Qwen21LoRA.scale(metadataJSON: meta, strength: 1) - 1) < 1e-6)
        #expect(abs(Qwen21LoRA.scale(metadataJSON: meta, strength: 0.5) - 0.5) < 1e-6)
        let rs = #"{"transformer.lora_alpha":256,"transformer.r":256,"transformer.use_rslora":true}"#
        #expect(abs(Qwen21LoRA.scale(metadataJSON: rs, strength: 1) - 16) < 1e-4)
        #expect(Qwen21LoRA.scale(metadataJSON: nil, strength: 1) == 1)
    }

    @Test("unmerged LoRA adds B A x beside the base linear")
    func loraResidual() {
        let base = Linear(weight: MLXArray.zeros([2, 3]), bias: nil)
        let a = MLXArray([Float]([1, 0, 0, 0, 1, 0])).reshaped([2, 3])
        let b = MLXArray([Float]([1, 0, 0, 1])).reshaped([2, 2])
        let layer = Qwen21LoRALinear(base: base, a: a, b: b, scale: 1)
        let y = layer(MLXArray([Float]([1, 2, 3])).reshaped([1, 3]))
        eval(y)
        let values = y.asArray(Float.self)
        #expect(values.count == 2)
        #expect(abs(values[0] - 1) < 1e-5)
        #expect(abs(values[1] - 2) < 1e-5)
    }

    @Test("1024 schedule matches the shifted flow-match sigmas")
    func sigmas1024() {
        let sigmas = Qwen21Schedule.sigmas(steps: 40, width: 1024, height: 1024)
        #expect(sigmas.count == 41)
        #expect(abs(sigmas[0] - 1) < 1e-5)
        #expect(abs(sigmas[1] - 0.98696375) < 1e-4)
        #expect(abs(sigmas[20] - 0.65666622) < 1e-4)
        #expect(abs(sigmas[39] - 0.02) < 1e-4)
        #expect(sigmas[40] == 0)
    }

    @Test("image strength picks the same start step as mflux")
    func initStep() {
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: nil) == 0)
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: 0) == 0)
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: 0.4) == 16)
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: 1) == 40)
    }

    @Test("rope axes center the latent grid")
    func ropeAxes() {
        let axes = Qwen21RopeLayout.axes(textLen: 3, height: 2, width: 2)
        #expect(axes.frame == [0, 1, 2, 3, 3, 3, 3])
        #expect(axes.height == [0, 1, 2, -1, -1, 0, 0])
        #expect(axes.width == [0, 1, 2, -1, 0, -1, 0])
        #expect(Qwen21RopeLayout.tableIndex(0) == 0)
        #expect(Qwen21RopeLayout.tableIndex(-1) == 9215)
        #expect(Qwen21RopeLayout.tableIndex(-32) == 9184)
    }

    @Test("four denoising steps produce a 256 image when the checkpoint is present")
    func oneStep() async throws {
        guard ProcessInfo.processInfo.environment["NOVAMLX_QWEN21_SMOKE"] == "1" else { return }
        let folder = URL(fileURLWithPath: "/Users/lucas/Models/Qwen/Qwen-Image-2.1")
        guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("vae/config.json").path) else {
            return
        }
        let pipeline = QwenImage21Pipeline(directoryURL: folder)
        try await pipeline.load()
        let result = try await pipeline.generate(
            prompt: "a red circle on a white background",
            negativePrompt: "",
            steps: 4,
            seed: 1,
            width: 256,
            height: 256
        )
        #expect(result.images.count == 1)
        let image = try #require(result.images.first)
        #expect(image.width == 256)
        #expect(image.height == 256)
        let url = URL(fileURLWithPath: "/tmp/qwen21-swift-4step.png")
        let data = NSMutableData()
        let dest = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)
        #expect(dest != nil)
        if let dest {
            CGImageDestinationAddImage(dest, image, nil)
            #expect(CGImageDestinationFinalize(dest))
            try (data as Data).write(to: url)
        }
    }
}
