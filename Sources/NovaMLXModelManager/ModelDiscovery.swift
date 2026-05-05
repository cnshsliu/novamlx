import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct DiscoveredModel: Sendable {
    public let modelId: String
    public let modelPath: URL
    public let modelType: ModelType
    public let family: ModelFamily
    public let estimatedSizeBytes: UInt64
    public let architectures: [String]
    public let configModelType: String
    public let isAdapter: Bool
    public let isComplete: Bool

    public init(
        modelId: String,
        modelPath: URL,
        modelType: ModelType,
        family: ModelFamily,
        estimatedSizeBytes: UInt64,
        architectures: [String],
        configModelType: String,
        isAdapter: Bool = false,
        isComplete: Bool = true
    ) {
        self.modelId = modelId
        self.modelPath = modelPath
        self.modelType = modelType
        self.family = family
        self.estimatedSizeBytes = estimatedSizeBytes
        self.architectures = architectures
        self.configModelType = configModelType
        self.isAdapter = isAdapter
        self.isComplete = isComplete
    }
}

public final class ModelDiscovery: Sendable {
    private static let vlmArchitectures: Set<String> = [
        "LlavaForConditionalGeneration",
        "LlavaNextForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "MllamaForConditionalGeneration",
        "Gemma3ForConditionalGeneration",
        "InternVLChatModel",
        "Idefics3ForConditionalGeneration",
        "PaliGemmaForConditionalGeneration",
        "Phi3VForCausalLM",
        "Pixtral",
        "MolmoForCausalLM",
        "Florence2ForConditionalGeneration",
        "Mistral3ForConditionalGeneration",
        "LlavaQwen2ForCausalLM",
        "Gemma4ForConditionalGeneration",
    ]

    private static let vlmModelTypes: Set<String> = [
        "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
        "gemma3", "llava", "llava_next", "llava-qwen2", "mllama",
        "idefics3", "internvl_chat", "phi3_v", "paligemma", "mistral3",
        "gemma4",
        "pixtral", "molmo", "multi_modality", "florence2", "minicpmv",
        "phi4_siglip", "phi4mm", "deepseek_vl2",
    ]

    private static let embeddingArchitectures: Set<String> = [
        "BertModel", "BertForMaskedLM",
        "XLMRobertaModel", "XLMRobertaForMaskedLM",
        "ModernBertModel", "ModernBertForMaskedLM",
        "Qwen3ForTextEmbedding",
        "SiglipModel", "SiglipVisionModel", "SiglipTextModel",
        "NomicBertModel",
    ]

    private static let embeddingModelTypes: Set<String> = [
        "bert", "xlm-roberta", "xlm_roberta", "modernbert",
        "siglip", "colqwen2_5", "lfm2", "nomic_bert",
        "jina_embeddings_v3",
    ]

    private static let familyByModelType: [String: ModelFamily] = [
        "llama": .llama, "llama2": .llama, "llama3": .llama,
        "mistral": .mistral, "mixtral": .mistral,
        "phi": .phi, "phi3": .phi, "phi3_v": .phi, "phi4mm": .phi,
        "qwen2": .qwen, "qwen2_vl": .qwen, "qwen2_5_vl": .qwen, "qwen3": .qwen, "qwen3_vl": .qwen,
        "gemma": .gemma, "gemma2": .gemma, "gemma3": .gemma, "gemma4": .gemma, "gemma3_text": .gemma, "gemma3n": .gemma, "gemma4_text": .gemma,
        "starcoder2": .starcoder,
        "bailing_moe": .bailing, "bailing_hybrid": .bailing,
        "deepseek_v3": .qwen, "deepseek_v4": .qwen,
        "gpt_oss": .gptOss,
    ]

    private static let familyByArchitecture: [String: ModelFamily] = [
        "LlamaForCausalLM": .llama,
        "MistralForCausalLM": .mistral,
        "MixtralForCausalLM": .mistral,
        "Phi3ForCausalLM": .phi,
        "Phi3VForCausalLM": .phi,
        "Qwen2ForCausalLM": .qwen,
        "Qwen2VLForConditionalGeneration": .qwen,
        "Qwen2_5_VLForConditionalGeneration": .qwen,
        "Gemma2ForCausalLM": .gemma,
        "Gemma3ForConditionalGeneration": .gemma,
        "Gemma4ForConditionalGeneration": .gemma,
        "GptOssForCausalLM": .gptOss,
        "DeepseekV4ForCausalLM": .qwen,
    ]

    public init() {}

    public func discover(in directory: URL) -> [DiscoveredModel] {
        var models: [DiscoveredModel] = []
        let fm = FileManager.default

        guard let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey]) else {
            return models
        }

        for subdir in contents.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard subdir.directoryExists, !subdir.lastPathComponent.hasPrefix(".") else { continue }
            let configPath = subdir.appendingPathComponent("config.json")

            if configPath.fileExists {
                let adapterConfigPath = subdir.appendingPathComponent("adapter_config.json")
                let adapterWeightsPath = subdir.appendingPathComponent("adapters.safetensors")
                if adapterConfigPath.fileExists && adapterWeightsPath.fileExists {
                    if let model = registerModel(at: subdir, id: subdir.lastPathComponent, isAdapter: true) {
                        models.append(model)
                    }
                } else {
                    if let model = registerModel(at: subdir, id: subdir.lastPathComponent) {
                        models.append(model)
                    }
                }
            } else {
                guard let children = try? fm.contentsOfDirectory(at: subdir, includingPropertiesForKeys: [.isDirectoryKey]) else { continue }
                for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                    guard child.directoryExists, !child.lastPathComponent.hasPrefix(".") else { continue }
                    let childConfig = child.appendingPathComponent("config.json")
                    guard childConfig.fileExists else { continue }
                    let adapterConfigPath = child.appendingPathComponent("adapter_config.json")
                    let adapterWeightsPath = child.appendingPathComponent("adapters.safetensors")
                    let isAdapter = adapterConfigPath.fileExists && adapterWeightsPath.fileExists

                    let orgPrefix = subdir.lastPathComponent
                    let modelId = "\(orgPrefix)/\(child.lastPathComponent)"
                    if let model = registerModel(at: child, id: modelId, isAdapter: isAdapter) {
                        models.append(model)
                    }
                }
            }
        }

        if models.isEmpty {
            let rootConfig = directory.appendingPathComponent("config.json")
            if rootConfig.fileExists {
                if let model = registerModel(at: directory, id: directory.lastPathComponent) {
                    models.append(model)
                }
            }
        }

        return models
    }

    private func registerModel(at path: URL, id: String, isAdapter: Bool = false) -> DiscoveredModel? {
        let configPath = path.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configPath) else { return nil }

        let config: HFConfig
        do {
            config = try JSONDecoder().decode(HFConfig.self, from: data)
        } catch {
            NovaMLXLog.warning("Failed to parse config.json for \(id): \(error)")
            return nil
        }

        let modelType = detectModelType(config: config, path: path)
        let family = detectFamily(config: config, modelId: id)
        let size = estimateSize(at: path)
        let complete = Self.checkCompleteness(at: path, isAdapter: isAdapter)

        NovaMLXLog.info("Discovered \(id): type=\(modelType.rawValue), family=\(family.rawValue), archs=\(config.architectures ?? []), size=\(size.bytesFormatted), complete=\(complete)")

        return DiscoveredModel(
            modelId: id,
            modelPath: path,
            modelType: modelType,
            family: family,
            estimatedSizeBytes: size,
            architectures: config.architectures ?? [],
            configModelType: config.modelType ?? "",
            isAdapter: isAdapter,
            isComplete: complete
        )
    }

    /// A model is "complete" if it has non-zero weight files.
    /// - Has at least one .safetensors file with size > 0 (or adapters.safetensors for adapters)
    /// - No zero-byte safetensors files (indicates interrupted download)
    private static func checkCompleteness(at path: URL, isAdapter: Bool) -> Bool {
        let fm = FileManager.default

        if isAdapter {
            let adapterFile = path.appendingPathComponent("adapters.safetensors")
            guard let size = fm.fileSize(at: adapterFile), size > 0 else { return false }
            return true
        }

        // Check for weight files
        guard let contents = try? fm.contentsOfDirectory(at: path, includingPropertiesForKeys: nil) else {
            return false
        }

        // If ANY .download temp files exist, download is still in progress — not complete
        let hasTempFiles = contents.contains { $0.pathExtension == "download" }
        if hasTempFiles {
            #if DEBUG
            NovaMLXLog.info("[Discovery] Incomplete model: .download temp files found in \(path.lastPathComponent)")
            #endif
            return false
        }

        let safetensors = contents.filter { $0.pathExtension == "safetensors" }
        // Must have at least one safetensors file
        guard !safetensors.isEmpty else { return false }

        // All safetensors files must be non-zero (zero-byte = interrupted download)
        for file in safetensors {
            guard let size = fm.fileSize(at: file), size > 0 else {
                #if DEBUG
                NovaMLXLog.info("[Discovery] Incomplete model: zero-byte file \(file.lastPathComponent) in \(path.lastPathComponent)")
                #endif
                return false
            }
        }

        return true
    }

    private func detectModelType(config: HFConfig, path: URL) -> ModelType {
        let architectures = config.architectures ?? []
        let rawType = (config.modelType ?? "").lowercased().replacingOccurrences(of: "-", with: "_")

        // 1. Explicit VLM architectures (highest confidence — these are always multimodal)
        for arch in architectures {
            if Self.vlmArchitectures.contains(arch) { return .vlm }
        }

        // 2. Explicit VLM model_types WITH vision_config present
        if Self.vlmModelTypes.contains(rawType) && config.visionConfig != nil { return .vlm }

        // 3. Ambiguous case: vision_config exists but arch/model_type are NOT in VLM lists.
        //    This happens with text-only fine-tunes of multimodal base models (e.g. Qwen3.5 distilled).
        //    The base model's config.json may retain a vestigial vision_config section, but the
        //    actual model weights lack a vision encoder.
        //
        //    Decision heuristic: if the architecture name contains "CausalLM" or "ForCausalLM",
        //    it's a text-only model by definition (CausalLM = autoregressive language model).
        //    If it contains "ConditionalGeneration" but is NOT in vlmArchitectures, it's a
        //    text-only variant (the VLM variant would have been caught by step 1).
        //
        //    Only classify as VLM if NEITHER condition above holds — truly unknown architectures
        //    with vision_config that don't match any known pattern.
        if config.visionConfig != nil {
            let isDefinitelyLLM = architectures.contains { arch in
                arch.hasSuffix("ForCausalLM") || arch.hasSuffix("CausalLM")
            }
            if isDefinitelyLLM {
                // Text-only model that happens to have vestigial vision_config — ignore it
            } else if !rawType.isEmpty && !Self.vlmModelTypes.contains(rawType) {
                // model_type is set but NOT a known VLM type — likely a text-only variant
                // e.g. model_type="qwen3_5" with arch "Qwen3_5ForConditionalGeneration"
                // The VLM variant would have model_type="qwen3_vl" and be caught by step 2.
            } else {
                // Truly ambiguous: unknown architecture + unknown model_type + vision_config
                return .vlm
            }
        }

        for arch in architectures {
            if Self.embeddingArchitectures.contains(arch) { return .embedding }
        }

        if Self.embeddingModelTypes.contains(rawType) { return .embedding }

        if rawType == "qwen3" || rawType == "gemma3_text" || rawType == "gemma3-text" {
            let llmArchs: Set<String> = ["Qwen3ForCausalLM", "Gemma3ForCausalLM"]
            if architectures.contains(where: { !llmArchs.contains($0) }) {
                return .embedding
            }
        }

        let dirName = path.lastPathComponent.lowercased()
        if dirName.contains("embedding") || dirName.contains("embed") {
            for arch in architectures {
                if Self.embeddingArchitectures.contains(arch) { return .embedding }
            }
        }

        return .llm
    }

    private func detectFamily(config: HFConfig, modelId: String) -> ModelFamily {
        // Resolution order (most-specific first, with data-driven extensions overlaying built-ins):
        //   1. ChatTemplateRegistry user/bundled extensions by model_type
        //   2. Built-in Swift familyByModelType map
        //   3. ChatTemplateRegistry by architecture
        //   4. Built-in familyByArchitecture map
        //   5. Substring match in modelId (last resort)
        //   6. .other
        //
        // Adding support for a new family without recompiling: drop entries
        // into ~/.nova/templates/registry.json under `familyDetection`.

        if let rawType = config.modelType {
            let normalized = rawType.lowercased().replacingOccurrences(of: "-", with: "_")

            // 1. Registry override
            if let f = ChatTemplateRegistry.shared.familyByModelType(normalized) { return f }

            // 2. Built-in
            if let family = Self.familyByModelType[normalized] { return family }

            // Prefix match against built-in (e.g. "qwen3_5" → "qwen3" → .qwen)
            let baseType = normalized.components(separatedBy: CharacterSet(charactersIn: "_0123456789")).first ?? normalized
            for (key, family) in Self.familyByModelType where key.hasPrefix(baseType) { return family }
        }

        if let architectures = config.architectures {
            for arch in architectures {
                // 3. Registry override
                if let f = ChatTemplateRegistry.shared.familyByArchitecture(arch) { return f }
                // 4. Built-in
                if let family = Self.familyByArchitecture[arch] { return family }
            }
        }

        // 5. Substring match — non-conservative, but better than .other for popular families.
        let idLower = modelId.lowercased()
        if idLower.contains("llama") { return .llama }
        if idLower.contains("mistral") || idLower.contains("mixtral") { return .mistral }
        if idLower.contains("phi") { return .phi }
        if idLower.contains("qwen") { return .qwen }
        if idLower.contains("gemma") { return .gemma }
        if idLower.contains("starcoder") { return .starcoder }

        return .other
    }

    private func estimateSize(at url: URL) -> UInt64 {
        FileManager.default.directorySize(at: url)
    }
}

private struct HFConfig: Codable {
    let architectures: [String]?
    let modelType: String?
    let visionConfig: CodableValue?

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case visionConfig = "vision_config"
    }
}

private struct CodableValue: Codable {}
