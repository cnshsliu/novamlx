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

    public init(
        modelId: String,
        modelPath: URL,
        modelType: ModelType,
        family: ModelFamily,
        estimatedSizeBytes: UInt64,
        architectures: [String],
        configModelType: String
    ) {
        self.modelId = modelId
        self.modelPath = modelPath
        self.modelType = modelType
        self.family = family
        self.estimatedSizeBytes = estimatedSizeBytes
        self.architectures = architectures
        self.configModelType = configModelType
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
    ]

    private static let vlmModelTypes: Set<String> = [
        "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
        "gemma3", "llava", "llava_next", "llava-qwen2", "mllama",
        "idefics3", "internvl_chat", "phi3_v", "paligemma", "mistral3",
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
        "gemma": .gemma, "gemma2": .gemma, "gemma3": .gemma,
        "starcoder2": .starcoder,
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
                if !subdir.appendingPathComponent("adapter_config.json").fileExists {
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
                    if child.appendingPathComponent("adapter_config.json").fileExists { continue }

                    let orgPrefix = subdir.lastPathComponent
                    let modelId = "\(orgPrefix)/\(child.lastPathComponent)"
                    if let model = registerModel(at: child, id: modelId) {
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

    private func registerModel(at path: URL, id: String) -> DiscoveredModel? {
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

        NovaMLXLog.info("Discovered \(id): type=\(modelType.rawValue), family=\(family.rawValue), archs=\(config.architectures ?? []), size=\(size.bytesFormatted)")

        return DiscoveredModel(
            modelId: id,
            modelPath: path,
            modelType: modelType,
            family: family,
            estimatedSizeBytes: size,
            architectures: config.architectures ?? [],
            configModelType: config.modelType ?? ""
        )
    }

    private func detectModelType(config: HFConfig, path: URL) -> ModelType {
        let architectures = config.architectures ?? []
        let rawType = (config.modelType ?? "").lowercased().replacingOccurrences(of: "-", with: "_")

        for arch in architectures {
            if Self.vlmArchitectures.contains(arch) { return .vlm }
        }

        if Self.vlmModelTypes.contains(rawType) && config.visionConfig != nil { return .vlm }

        if config.visionConfig != nil {
            let llmArchs: Set<String> = [
                "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
                "Gemma2ForCausalLM", "Phi3ForCausalLM",
            ]
            let hasLLMArch = architectures.contains { llmArchs.contains($0) }
            if !hasLLMArch { return .vlm }
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
        if let rawType = config.modelType {
            let normalized = rawType.lowercased().replacingOccurrences(of: "-", with: "_")
            if let family = Self.familyByModelType[normalized] { return family }
            let baseType = normalized.components(separatedBy: CharacterSet(charactersIn: "_0123456789")).first ?? normalized
            for (key, family) in Self.familyByModelType where key.hasPrefix(baseType) { return family }
        }

        if let architectures = config.architectures {
            for arch in architectures {
                if let family = Self.familyByArchitecture[arch] { return family }
            }
        }

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
