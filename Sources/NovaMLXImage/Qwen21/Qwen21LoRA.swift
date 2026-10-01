import Foundation
import MLX
import MLXNN
import NovaMLXCore

/// Runtime LoRA branch. The residual stays beside the base weight: merging this
/// adapter into bf16 (or into a quantized matrix) drops most of the update.
final class Qwen21LoRALinear: Linear {
    let base: Linear
    let loraA: MLXArray
    let loraB: MLXArray
    let scale: Float

    init(base: Linear, a: MLXArray, b: MLXArray, scale: Float) {
        self.base = base
        self.loraA = a
        self.loraB = b
        self.scale = scale
        super.init(weight: MLXArray.zeros([1, 1]), bias: nil)
        eval(loraA, loraB)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = base(x)
        let hidden = matmul(x, loraA.asType(x.dtype).T)
        let delta = matmul(hidden, loraB.asType(x.dtype).T)
        return y + (delta * scale).asType(y.dtype)
    }
}

enum Qwen21LoRA {
    /// `strength * alpha / rank`, or `alpha / sqrt(rank)` when the file says rsLoRA.
    /// A missing header uses strength, which is 1 for this adapter (alpha equals rank).
    static func scale(metadataJSON: String?, strength: Float) -> Float {
        guard
            let metadataJSON,
            let data = metadataJSON.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return strength }
        let alpha = floatValue(object["transformer.lora_alpha"]) ?? 1
        let rank = floatValue(object["transformer.r"]) ?? 1
        let rsLoRA = object["transformer.use_rslora"] as? Bool ?? false
        let denom = rsLoRA ? sqrt(rank) : rank
        guard denom != 0 else { return strength }
        return strength * alpha / denom
    }

    /// Diffusers key `transformer.<module>.lora_A.weight` → the Swift module path.
    static func normalizedPath(for key: String) -> String? {
        let suffix: String
        if key.hasSuffix(".lora_A.weight") {
            suffix = ".lora_A.weight"
        } else if key.hasSuffix(".lora_B.weight") {
            suffix = ".lora_B.weight"
        } else {
            return nil
        }
        var path = key
        if path.hasPrefix("transformer.") {
            path.removeFirst("transformer.".count)
        }
        path.removeLast(suffix.count)
        if path == "modulation.0" || path == "modulation.1" {
            return "modulation.layers.1"
        }
        return path
    }

    @discardableResult
    static func install(file: URL, on transformer: Qwen21Transformer, strength: Float = 1) throws -> Int {
        let metadata = safetensorsMetadata(file)
        let scale = scale(metadataJSON: metadata["lora_adapter_metadata"], strength: strength)
        let tensors = try loadArrays(url: file)
        var pairs: [String: (MLXArray?, MLXArray?)] = [:]
        for (key, value) in tensors {
            guard let path = normalizedPath(for: key) else { continue }
            var slot = pairs[path] ?? (nil, nil)
            if key.hasSuffix(".lora_A.weight") {
                slot.0 = value
            } else {
                slot.1 = value
            }
            pairs[path] = slot
        }

        var attached = 0
        var missed: [String] = []
        var touchedBlocks = Set<Int>()
        var touchedModulation = false
        var touchedTime = false
        for path in pairs.keys.sorted() {
            guard let slot = pairs[path], let a = slot.0, let b = slot.1 else {
                missed.append(path)
                continue
            }
            let pair = (a, b, scale)
            switch attach(pair, at: path, on: transformer) {
            case .block(let index):
                attached += 1
                touchedBlocks.insert(index)
            case .modulation:
                attached += 1
                touchedModulation = true
            case .time:
                attached += 1
                touchedTime = true
            case .none:
                missed.append(path)
            }
        }
        for index in touchedBlocks {
            let block = transformer.blocks[index]
            Qwen21Weights.refreshModuleGraph(block.attn)
            Qwen21Weights.refreshModuleGraph(block.mlp)
            Qwen21Weights.refreshModuleGraph(block)
        }
        if touchedModulation {
            Qwen21Weights.refreshModuleGraph(transformer.modulation)
        }
        if touchedTime {
            Qwen21Weights.refreshModuleGraph(transformer.timeEmbed.embedder)
            Qwen21Weights.refreshModuleGraph(transformer.timeEmbed)
        }
        if attached == 0 || !missed.isEmpty {
            let sample = missed.prefix(5).joined(separator: ", ")
            throw NovaMLXError.inferenceFailed(
                "Qwen-Image 2.1 turbo LoRA did not match the transformer (\(attached) layers). \(sample)"
            )
        }
        return attached
    }

    private enum Attach {
        case block(Int)
        case modulation
        case time
        case none
    }

    private static func attach(
        _ pair: (MLXArray, MLXArray, Float),
        at path: String,
        on transformer: Qwen21Transformer
    ) -> Attach {
        let parts = path.split(separator: ".").map(String.init)
        if parts.count >= 4, parts[0] == "transformer_blocks", let index = Int(parts[1]),
           transformer.blocks.indices.contains(index)
        {
            let block = transformer.blocks[index]
            if parts[2] == "attn", parts.count == 4 {
                switch parts[3] {
                case "to_q":
                    block.attn.toQ = wrap(block.attn.toQ, pair)
                case "to_k":
                    block.attn.toK = wrap(block.attn.toK, pair)
                case "to_v":
                    block.attn.toV = wrap(block.attn.toV, pair)
                default:
                    return .none
                }
                return .block(index)
            }
            if parts[2] == "attn", parts.count == 5, parts[3] == "to_out", parts[4] == "0" {
                block.attn.toOut = [wrap(block.attn.toOut[0], pair)]
                return .block(index)
            }
            if parts[2] == "img_mlp", parts.count == 4 {
                switch parts[3] {
                case "proj":
                    block.mlp.proj = wrap(block.mlp.proj, pair)
                case "out":
                    block.mlp.outProj = wrap(block.mlp.outProj, pair)
                case "gate_layer":
                    block.mlp.gate = wrap(block.mlp.gate, pair)
                default:
                    return .none
                }
                return .block(index)
            }
        }
        if parts == ["modulation", "layers", "1"] {
            var layers = transformer.modulation.layers
            guard let linear = layers[1] as? Linear else { return .none }
            layers[1] = wrap(linear, pair)
            transformer.modulation.layers = layers
            return .modulation
        }
        if parts == ["time_text_embed", "timestep_embedder", "linear_1"] {
            let embedder = transformer.timeEmbed.embedder
            embedder.linear1 = wrap(embedder.linear1, pair)
            return .time
        }
        if parts == ["time_text_embed", "timestep_embedder", "linear_2"] {
            let embedder = transformer.timeEmbed.embedder
            embedder.linear2 = wrap(embedder.linear2, pair)
            return .time
        }
        return .none
    }

    private static func wrap(_ linear: Linear, _ pair: (MLXArray, MLXArray, Float)) -> Linear {
        Qwen21LoRALinear(base: linear, a: pair.0, b: pair.1, scale: pair.2)
    }

    private static func floatValue(_ value: Any?) -> Float? {
        switch value {
        case let number as NSNumber:
            return number.floatValue
        case let number as Int:
            return Float(number)
        case let number as Double:
            return Float(number)
        default:
            return nil
        }
    }

    private static func safetensorsMetadata(_ file: URL) -> [String: String] {
        guard
            let handle = try? FileHandle(forReadingFrom: file),
            let lengthData = try? handle.read(upToCount: 8),
            lengthData.count == 8
        else { return [:] }
        defer { try? handle.close() }
        let length = lengthData.withUnsafeBytes { raw in
            raw.load(as: UInt64.self).littleEndian
        }
        guard length < 8_000_000, let header = try? handle.read(upToCount: Int(length)),
              header.count == Int(length),
              let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
              let metadata = object["__metadata__"] as? [String: String]
        else { return [:] }
        return metadata
    }
}

enum Qwen21Turbo {
    static func loraFile(in directory: URL) -> URL? {
        let preferred = directory.appendingPathComponent(
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
        )
        if FileManager.default.fileExists(atPath: preferred.path) {
            return preferred
        }
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return nil }
        let matches = files.filter { url in
            let name = url.lastPathComponent.lowercased()
            return name.hasSuffix(".safetensors")
                && name.contains("lora")
                && name.contains("qwen-image-2.1")
        }
        return matches.max { lhs, rhs in
            let left = (try? lhs.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let right = (try? rhs.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if left == right { return lhs.lastPathComponent < rhs.lastPathComponent }
            return left < right
        }
    }

    /// A Viggle folder ships the LoRA and not the text encoder or VAE.
    static func isAdapter(_ directory: URL) -> Bool {
        guard loraFile(in: directory) != nil else { return false }
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.appendingPathComponent("vae/config.json").path) {
            return false
        }
        if fm.fileExists(atPath: directory.appendingPathComponent("text_encoder/config.json").path) {
            return false
        }
        return true
    }

    static func hasTransformer(_ directory: URL) -> Bool {
        let fm = FileManager.default
        for name in [
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/model.safetensors.index.json",
        ] {
            if fm.fileExists(atPath: directory.appendingPathComponent(name).path) {
                return true
            }
        }
        return false
    }

    /// Full bf16 base when it is installed, otherwise the 4-bit checkpoint.
    static func baseDirectory() -> URL? {
        if let env = ProcessInfo.processInfo.environment["NOVAMLX_QWEN21_TURBO_BASE"], !env.isEmpty {
            let url = URL(fileURLWithPath: env, isDirectory: true)
            if hasTransformer(url) { return url }
        }
        for id in ["Qwen/Qwen-Image-2.1", "mlx-community/Qwen-Image-2.1-MLX-4bit"] {
            let url = NovaMLXPaths.directory(forModelId: id)
            if hasTransformer(url) { return url }
        }
        return nil
    }
}
