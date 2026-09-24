import Foundation
import MLX
import MLXNN
import NovaMLXCore

enum Qwen21Weights {
    @discardableResult
    static func installText(directory: URL, model: Qwen21TextEncoder) throws -> Bool {
        let loaded = try loadIndexed(
            directory: directory.appendingPathComponent("text_encoder"),
            index: "model.safetensors.index.json"
        ) {
            $0.hasPrefix("model.language_model.")
                || $0.hasPrefix("language_model.model.")
        }
        let raw = remapText(loaded)
        let packed = raw.keys.contains { $0.hasSuffix(".scales") }
        if packed {
            let spec = quantSpec(in: directory.appendingPathComponent("text_encoder/config.json"))
            quantize(
                model: model, groupSize: spec.group, bits: spec.bits, mode: .affine,
                filter: { _, module in module is Linear || module is Embedding }
            )
        }
        var pairs = [Pair]()
        pairs.append(contentsOf: try parameters("embed_tokens.weight", from: "model.language_model.embed_tokens.weight", raw: raw))
        pairs.append(try linear("norm.weight", from: "model.language_model.norm.weight", raw: raw))
        for layer in 0..<model.layers.count {
            let source = "model.language_model.layers.\(layer)"
            let dest = "layers.\(layer)"
            for name in ["input_layernorm", "post_attention_layernorm"] {
                pairs.append(try linear("\(dest).\(name).weight", from: "\(source).\(name).weight", raw: raw))
            }
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                pairs.append(contentsOf: try parameters(
                    "\(dest).self_attn.\(name).weight",
                    from: "\(source).self_attn.\(name).weight",
                    raw: raw
                ))
            }
            for name in ["q_norm", "k_norm"] {
                pairs.append(try linear(
                    "\(dest).self_attn.\(name).weight",
                    from: "\(source).self_attn.\(name).weight",
                    raw: raw
                ))
            }
            for name in ["gate_proj", "up_proj", "down_proj"] {
                pairs.append(contentsOf: try parameters(
                    "\(dest).mlp.\(name).weight",
                    from: "\(source).mlp.\(name).weight",
                    raw: raw
                ))
            }
        }
        try apply(pairs, to: model, name: "text encoder")
        return packed
    }

    @discardableResult
    static func installTransformer(directory: URL, model: Qwen21Transformer) throws -> Bool {
        let raw = remapTransformer(try loadIndexed(
            directory: directory.appendingPathComponent("transformer"),
            index: "diffusion_pytorch_model.safetensors.index.json",
            keep: nil
        ))
        let packed = raw.keys.contains { $0.hasSuffix(".scales") }
        if packed {
            let spec = quantSpec(in: directory.appendingPathComponent("transformer/config.json"))
            quantize(
                model: model, groupSize: spec.group, bits: spec.bits, mode: .affine,
                filter: { path, module in
                    guard module is Linear else { return false }
                    let last = path.split(separator: ".").last.map(String.init) ?? ""
                    return Int(last) == nil
                }
            )
            quantizeArrayLinears(model, bits: spec.bits, group: spec.group)
        }
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
            pairs.append(contentsOf: try parameters(key, from: key, raw: raw))
        }
        pairs.append(contentsOf: try parameters("modulation.layers.1.weight", from: "modulation.1.weight", raw: raw))
        for block in 0..<model.blocks.count {
            let prefix = "transformer_blocks.\(block)"
            for name in ["to_q", "to_k", "to_v"] {
                pairs.append(contentsOf: try parameters(
                    "\(prefix).attn.\(name).weight", from: "\(prefix).attn.\(name).weight", raw: raw
                ))
            }
            for name in ["norm_q", "norm_k"] {
                pairs.append(try linear(
                    "\(prefix).attn.\(name).weight", from: "\(prefix).attn.\(name).weight", raw: raw
                ))
            }
            pairs.append(contentsOf: try parameters(
                "\(prefix).attn.to_out.0.weight", from: "\(prefix).attn.to_out.0.weight", raw: raw
            ))
            for name in ["proj", "out", "gate_layer"] {
                pairs.append(contentsOf: try parameters(
                    "\(prefix).img_mlp.\(name).weight", from: "\(prefix).img_mlp.\(name).weight", raw: raw
                ))
            }
        }
        try apply(pairs, to: model, name: "transformer")
        return packed
    }

    static func installVAE(directory: URL, model: Qwen21VAE) throws {
        let file = vaeWeightsFile(in: directory)
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

    static func quantizeArrayLinears(_ model: Qwen21Transformer, bits: Int, group: Int = 64) {
        quantizeLayers(model.modulation, index: 1, bits: bits, group: group)
        for block in model.blocks {
            quantizeOutputs(block.attn, bits: bits, group: group)
        }
    }

    /// `layers[i]` and `to_out[0]` are arrays, so `quantize(model:)` skips them.
    /// Replacing the array updates `ModuleInfo`, but `update(parameters:)` still
    /// reads the module cache built before that replacement. Refresh the cache
    /// or the packed 4-bit weight is checked against the original float `Linear`.
    static func quantizeLayers(_ parent: Qwen21Modulation, index: Int, bits: Int, group: Int) {
        var layers = parent.layers
        if let linear = layers[index] as? Linear {
            layers[index] = linear.toQuantized(groupSize: group, bits: bits, mode: .affine)
        }
        parent.layers = layers
        refreshModuleGraph(parent)
    }

    static func quantizeOutputs(_ attention: Qwen21Attention, bits: Int, group: Int) {
        if let quantized = attention.toOut[0].toQuantized(groupSize: group, bits: bits, mode: .affine) as? Linear {
            attention.toOut = [quantized]
        }
        refreshModuleGraph(attention)
    }

    static func refreshModuleGraph(_ module: Module) {
        _ = module.update(modules: NestedDictionary())
    }

    static func remapText(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in raw {
            if key.hasPrefix("language_model.model.") {
                let suffix = key.dropFirst("language_model.model.".count)
                out["model.language_model." + suffix] = value
            } else {
                out[key] = value
            }
        }
        return out
    }

    /// Diffusers stores convolutions as OIHW. An MLX checkpoint already stores
    /// them as OHWI, with the kernel in the middle two axes.
    static func convolutionWeight(_ weight: MLXArray) -> MLXArray {
        guard weight.ndim == 4 else { return weight }
        let inputChannels = weight.dim(1)
        let kernelHeight = weight.dim(2)
        let kernelWidth = weight.dim(3)
        let isDiffusers = kernelHeight == kernelWidth && kernelHeight <= 7 && inputChannels > kernelHeight
        if isDiffusers {
            return weight.transposed(0, 2, 3, 1)
        }
        return weight
    }

    static func remapTransformer(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in raw {
            var mapped = key
            if mapped.hasPrefix("time_text_embed.linear_") {
                mapped = "time_text_embed.timestep_embedder.linear_" + mapped.dropFirst("time_text_embed.linear_".count)
            }
            if mapped.hasPrefix("modulation.0.") {
                mapped = "modulation.1." + mapped.dropFirst("modulation.0.".count)
            }
            out[mapped] = value
        }
        return out
    }

    private typealias Pair = (String, MLXArray)

    private static func quantSpec(in config: URL) -> (group: Int, bits: Int) {
        guard
            let data = try? Data(contentsOf: config),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let quant = json["quantization"] as? [String: Any]
        else {
            return (64, 4)
        }
        let group = quant["group_size"] as? Int ?? 64
        let bits = quant["bits"] as? Int ?? 4
        return (group, bits)
    }

    private static func vaeWeightsFile(in directory: URL) -> URL {
        let names = ["vae/diffusion_pytorch_model.safetensors", "vae/model.safetensors"]
        for name in names {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return directory.appendingPathComponent(names[0])
    }

    private static func parameters(_ dest: String, from source: String, raw: [String: MLXArray]) throws -> [Pair] {
        let weight = try linear(dest, from: source, raw: raw)
        var pairs = [weight]
        guard source.hasSuffix(".weight"), dest.hasSuffix(".weight") else { return pairs }
        let sourceStem = String(source.dropLast(".weight".count))
        let destStem = String(dest.dropLast(".weight".count))
        if let scales = raw[sourceStem + ".scales"] {
            pairs.append((destStem + ".scales", scales))
        }
        if let biases = raw[sourceStem + ".biases"] {
            pairs.append((destStem + ".biases", biases))
        }
        return pairs
    }

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
        weight = convolutionWeight(weight)
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
        let indexURL = try resolveIndex(directory: directory, preferred: index)
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

    private static func resolveIndex(directory: URL, preferred: String) throws -> URL {
        var names = [preferred, "model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"]
        var seen = Set<String>()
        names = names.filter { seen.insert($0).inserted }
        for name in names {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 weight index \(preferred) is missing")
    }
}
