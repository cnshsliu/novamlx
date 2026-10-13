import Foundation

/// GGUF header reader used by model discovery and engine sidecar setup.
/// Tensor payloads stay on disk; only metadata is read.
public enum GGUFError: Error, Sendable {
    case truncated
    case badMagic
    case unsupportedType(UInt32)
    case missingWeights
    case invalidString
}

public struct GGUFFile: Sendable {
    public let architecture: String?
    public let alignment: Int
    public let tokenCount: Int?
    private let integers: [String: Int]

    /// Metadata integer `architecture.<name>`, for example `embedding_length`.
    public func archInt(_ name: String) -> Int? {
        guard let architecture else { return nil }
        return integers["\(architecture).\(name)"]
    }

    public static func parse(url: URL) throws -> GGUFFile {
        let reader = try Reader(url: url)
        defer { reader.close() }
        let magic = try reader.u32()
        guard magic == 0x4655_4747 else { throw GGUFError.badMagic }
        let version = try reader.u32()
        guard version >= 2 else { throw GGUFError.unsupportedType(version) }
        _ = try reader.u64()
        let kvCount = try reader.u64()

        var integers: [String: Int] = [:]
        var architecture: String?
        var alignment = 32
        var tokenCount: Int?
        var index: UInt64 = 0
        while index < kvCount {
            let key = try reader.string()
            let type = try reader.u32()
            let value = try reader.value(type: type)
            switch value {
            case .int(let number):
                integers[key] = number
                if key == "general.alignment" { alignment = number }
            case .string(let text):
                if key == "general.architecture" { architecture = text }
            case .strings(let items):
                if key == "tokenizer.ggml.tokens" { tokenCount = items.count }
            case .none:
                break
            }
            index += 1
        }
        return GGUFFile(
            architecture: architecture,
            alignment: alignment,
            tokenCount: tokenCount,
            integers: integers
        )
    }
}

public enum GGUFHuggingFaceConfig {
    public static func hfModelType(ggufArch: String) -> String {
        switch ggufArch {
        case "qwen2moe": return "qwen2_moe"
        case "qwen3moe": return "qwen3_moe"
        case "deepseek2": return "deepseek_v2"
        default: return ggufArch.replacingOccurrences(of: "-", with: "_")
        }
    }

    public static func architectureName(hfType: String) -> String {
        switch hfType {
        case "llama": return "LlamaForCausalLM"
        case "qwen2": return "Qwen2ForCausalLM"
        case "qwen2_moe": return "Qwen2MoeForCausalLM"
        case "qwen3": return "Qwen3ForCausalLM"
        case "mistral": return "MistralForCausalLM"
        case "gemma": return "GemmaForCausalLM"
        case "gemma2": return "Gemma2ForCausalLM"
        case "phi3": return "Phi3ForCausalLM"
        default:
            guard let first = hfType.first else { return "\(hfType)ForCausalLM" }
            return "\(String(first).uppercased())\(hfType.dropFirst())ForCausalLM"
        }
    }
}

public func hasGGUFWeights(in directory: URL) -> Bool {
    let fm = FileManager.default
    guard directory.directoryExists,
          let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    else { return false }
    return contents.contains { url in
        url.pathExtension.lowercased() == "gguf" && (fm.fileSize(at: url) ?? 0) > 0
    }
}

/// Write `config.json` from GGUF metadata when the directory does not already have one.
public func prepareGGUFModelDirectory(_ directory: URL) throws {
    let fm = FileManager.default
    guard let ggufURL = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .first(where: { $0.pathExtension.lowercased() == "gguf" })
    else { throw GGUFError.missingWeights }
    let gguf = try GGUFFile.parse(url: ggufURL)
    let configURL = directory.appendingPathComponent("config.json")
    guard !configURL.fileExists else { return }

    let arch = gguf.architecture ?? "llama"
    let modelType = GGUFHuggingFaceConfig.hfModelType(ggufArch: arch)
    var config: [String: Any] = [
        "model_type": modelType,
        "architectures": [GGUFHuggingFaceConfig.architectureName(hfType: modelType)],
    ]
    if let hidden = gguf.archInt("embedding_length") { config["hidden_size"] = hidden }
    if let layers = gguf.archInt("block_count") { config["num_hidden_layers"] = layers }
    if let heads = gguf.archInt("attention.head_count") { config["num_attention_heads"] = heads }
    if let kvHeads = gguf.archInt("attention.head_count_kv") { config["num_key_value_heads"] = kvHeads }
    if let ff = gguf.archInt("feed_forward_length") { config["intermediate_size"] = ff }
    if let context = gguf.archInt("context_length") { config["max_position_embeddings"] = context }
    if let vocab = gguf.archInt("vocab_size") ?? gguf.tokenCount { config["vocab_size"] = vocab }

    let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: configURL, options: .atomic)
}

private struct Reader {
    let handle: FileHandle

    init(url: URL) throws {
        handle = try FileHandle(forReadingFrom: url)
    }

    func close() {
        try? handle.close()
    }

    func read(_ count: Int) throws -> Data {
        guard count > 0 else { return Data() }
        let data = try handle.read(upToCount: count) ?? Data()
        guard data.count == count else { throw GGUFError.truncated }
        return data
    }

    func u32() throws -> UInt32 {
        try read(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
    }

    func u64() throws -> UInt64 {
        try read(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    }

    func string() throws -> String {
        let count = try u64()
        guard count <= 16_000_000, let length = Int(exactly: count) else {
            throw GGUFError.invalidString
        }
        let data = try read(length)
        guard let text = String(data: data, encoding: .utf8) else { throw GGUFError.invalidString }
        return text
    }

    enum Value {
        case int(Int)
        case string(String)
        case strings([String])
    }

    func value(type: UInt32) throws -> Value? {
        switch type {
        case 0, 1, 7:
            _ = try read(1)
            return nil
        case 2, 3:
            _ = try read(2)
            return nil
        case 4:
            return .int(Int(try u32()))
        case 5:
            return .int(Int(Int32(bitPattern: try u32())))
        case 6:
            _ = try read(4)
            return nil
        case 8:
            return .string(try string())
        case 9:
            let element = try u32()
            let count = try u64()
            guard count <= 2_000_000 else { throw GGUFError.invalidString }
            if element == 8 {
                var items: [String] = []
                items.reserveCapacity(Int(count))
                for _ in 0 ..< count {
                    items.append(try string())
                }
                return .strings(items)
            }
            for _ in 0 ..< count {
                _ = try value(type: element)
            }
            return nil
        case 10:
            let raw = try u64()
            guard let number = Int(exactly: raw) else { return nil }
            return .int(number)
        case 11:
            let raw = try u64()
            let signed = Int64(bitPattern: raw)
            guard let number = Int(exactly: signed) else { return nil }
            return .int(number)
        case 12:
            _ = try read(8)
            return nil
        default:
            throw GGUFError.unsupportedType(type)
        }
    }
}
