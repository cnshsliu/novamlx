import Foundation
import MLX
import MLXNN
import NovaMLXCore

enum Qwen21Weights {
    static func installText(directory: URL, model: Qwen21TextEncoder) throws {
        let raw = try loadIndexed(
            directory: directory.appendingPathComponent("text_encoder"),
            index: "model.safetensors.index.json"
        ) { $0.hasPrefix("model.language_model.") }
        var pairs = [Pair]()
        pairs.append(try linear("embed_tokens.weight", from: "model.language_model.embed_tokens.weight", raw: raw))
        pairs.append(try linear("norm.weight", from: "model.language_model.norm.weight", raw: raw))
        for layer in 0..<model.layers.count {
            let source = "model.language_model.layers.\(layer)"
            let dest = "layers.\(layer)"
            for name in ["input_layernorm", "post_attention_layernorm"] {
                pairs.append(try linear("\(dest).\(name).weight", from: "\(source).\(name).weight", raw: raw))
            }
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                pairs.append(
                    try linear("\(dest).self_attn.\(name).weight", from: "\(source).self_attn.\(name).weight", raw: raw)
                )
            }
            for name in ["gate_proj", "up_proj", "down_proj"] {
                pairs.append(try linear("\(dest).mlp.\(name).weight", from: "\(source).mlp.\(name).weight", raw: raw))
            }
        }
        try apply(pairs, to: model, name: "text encoder")
    }

    static func installTransformer(directory: URL, model: Qwen21Transformer) throws {
        let raw = try loadIndexed(
            directory: directory.appendingPathComponent("transformer"),
            index: "diffusion_pytorch_model.safetensors.index.json",
            keep: nil
        )
        var pairs = [Pair]()
        for key in [
            "img_in.weight",
            "proj_out.weight",
            "norm_out.linear.weight",
            "txt_in.text_norm.weight",
            "txt_in.in_layer.weight",
            "txt_in.out_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
        ] {
            pairs.append(try linear(key, from: key, raw: raw))
        }
        pairs.append(try linear("modulation.layers.1.weight", from: "modulation.1.weight", raw: raw))
        for block in 0..<model.blocks.count {
            let prefix = "transformer_blocks.\(block)"
            for name in ["to_q", "to_k", "to_v", "norm_q", "norm_k"] {
                pairs.append(try linear("\(prefix).attn.\(name).weight", from: "\(prefix).attn.\(name).weight", raw: raw))
            }
            pairs.append(
                try linear("\(prefix).attn.to_out.0.weight", from: "\(prefix).attn.to_out.0.weight", raw: raw)
            )
            for name in ["proj", "out", "gate_layer"] {
                pairs.append(
                    try linear("\(prefix).img_mlp.\(name).weight", from: "\(prefix).img_mlp.\(name).weight", raw: raw)
                )
            }
        }
        try apply(pairs, to: model, name: "transformer")
    }

    static func installVAE(directory: URL, model: Qwen21VAE) throws {
        let file = directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors")
        let raw = try loadArrays(url: file)
        var pairs = [Pair]()
        for name in [
            "encoder.conv_in", "encoder.conv_out", "decoder.conv_in", "decoder.conv_out",
            "quant_conv", "post_quant_conv",
        ] {
            pairs.append(contentsOf: try conv("\(name).conv", from: name, raw: raw))
        }
        for block in 0..<5 {
            let shortcut = block == 1 || block == 2 || block == 3
            pairs.append(contentsOf: try resnets(
                prefix: "encoder.down_blocks.\(block)", count: 2, shortcut: shortcut, raw: raw
            ))
            if block < 4 {
                pairs.append(contentsOf: try conv(
                    "encoder.down_blocks.\(block).downsampler.conv",
                    from: "encoder.down_blocks.\(block).downsampler.resample.1",
                    raw: raw
                ))
            }
        }
        for block in 0..<5 {
            pairs.append(contentsOf: try resnets(
                prefix: "decoder.up_blocks.\(block)", count: 3, shortcut: block >= 2, raw: raw
            ))
            if block < 4 {
                pairs.append(contentsOf: try conv(
                    "decoder.up_blocks.\(block).upsampler.conv",
                    from: "decoder.up_blocks.\(block).upsampler.resample.1",
                    raw: raw
                ))
            }
        }
        for side in ["encoder", "decoder"] {
            pairs.append(contentsOf: try resnets(
                prefix: "\(side).mid_block", count: 2, shortcut: false, raw: raw
            ))
            pairs.append(try gamma("\(side).mid_block.attentions.0.norm.weight", from: "\(side).mid_block.attentions.0.norm.gamma", raw: raw))
            for name in ["to_qkv", "proj"] {
                pairs.append(contentsOf: try conv(
                    "\(side).mid_block.attentions.0.\(name)",
                    from: "\(side).mid_block.attentions.0.\(name)",
                    raw: raw
                ))
            }
            pairs.append(try gamma("\(side).norm_out.weight", from: "\(side).norm_out.gamma", raw: raw))
        }
        try apply(pairs, to: model, name: "VAE")
    }

    static func latentStats(directory: URL) throws -> (mean: [Float], std: [Float]) {
        let url = directory.appendingPathComponent("vae/config.json")
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let mean = (json?["latents_mean"] as? [Double])?.map { Float($0) } ?? []
        let std = (json?["latents_std"] as? [Double])?.map { Float($0) } ?? []
        guard mean.count == 64, std.count == 64 else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 VAE config is missing 64-channel latent stats")
        }
        return (mean, std)
    }

    private typealias Pair = (String, MLXArray)

    private static func apply(_ pairs: [Pair], to model: Module, name: String) throws {
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(pairs),
                verify: [.allModelKeysSet, .shapeMismatch]
            )
        } catch {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 \(name) weights did not match the Swift graph: \(error)")
        }
    }

    private static func linear(_ dest: String, from source: String, raw: [String: MLXArray]) throws -> Pair {
        guard let value = raw[source] else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 checkpoint is missing \(source)")
        }
        return (dest, value)
    }

    private static func gamma(_ dest: String, from source: String, raw: [String: MLXArray]) throws -> Pair {
        guard var value = raw[source] else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 checkpoint is missing \(source)")
        }
        if value.ndim > 1 {
            value = value.reshaped([value.dim(0)])
        }
        return (dest, value)
    }

    private static func conv(_ dest: String, from source: String, raw: [String: MLXArray]) throws -> [Pair] {
        guard var weight = raw["\(source).weight"], let bias = raw["\(source).bias"] else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 checkpoint is missing \(source)")
        }
        if weight.ndim == 4 {
            weight = weight.transposed(0, 2, 3, 1)
        }
        return [("\(dest).weight", weight), ("\(dest).bias", bias)]
    }

    private static func resnets(prefix: String, count: Int, shortcut: Bool, raw: [String: MLXArray]) throws -> [Pair] {
        var pairs = [Pair]()
        for index in 0..<count {
            let base = "\(prefix).resnets.\(index)"
            for norm in ["norm1", "norm2"] {
                pairs.append(try gamma("\(base).\(norm).weight", from: "\(base).\(norm).gamma", raw: raw))
            }
            for name in ["conv1", "conv2"] {
                pairs.append(contentsOf: try conv("\(base).\(name).conv", from: "\(base).\(name)", raw: raw))
            }
            if shortcut && index == 0 {
                pairs.append(contentsOf: try conv("\(base).conv_shortcut.conv", from: "\(base).conv_shortcut", raw: raw))
            }
        }
        return pairs
    }

    private static func loadIndexed(
        directory: URL,
        index: String,
        keep: ((String) -> Bool)?
    ) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent(index)
        let data = try Data(contentsOf: indexURL)
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let map = json["weight_map"] as? [String: String]
        else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 weight index \(index) is unreadable")
        }
        var files: [String: [String]] = [:]
        for (key, file) in map {
            if let keep, !keep(key) { continue }
            files[file, default: []].append(key)
        }
        var tensors: [String: MLXArray] = [:]
        tensors.reserveCapacity(map.count)
        for (file, keys) in files {
            let loaded = try loadArrays(url: directory.appendingPathComponent(file))
            for key in keys {
                guard let value = loaded[key] else {
                    throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 shard \(file) is missing \(key)")
                }
                tensors[key] = value
            }
        }
        return tensors
    }
}
