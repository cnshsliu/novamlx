import Foundation

// MARK: - ModelAnalyzerError

/// Errors thrown during model analysis.
public enum ModelAnalyzerError: Error, Sendable, Equatable {
    /// The safetensors header could not be read (too short, bad length, etc.).
    case invalidHeader(String)
    /// The header JSON could not be parsed.
    case invalidJSON(String)
    /// The model directory was not found.
    case fileNotFound(String)
}

// MARK: - TensorDescriptor

/// Parsed tensor info from a safetensors header.
private struct TensorDescriptor: Sendable {
    let name: String
    let dtype: String
    let shape: [Int]
    let byteOffset: Int
    let byteCount: Int

    /// Number of elements (product of shape).
    var elementCount: UInt64 {
        shape.reduce(UInt64(1)) { $0 * UInt64($1) }
    }

    /// Estimated memory in bytes based on dtype.
    var estimatedBytes: UInt64 {
        elementCount * UInt64(bytesPerElement)
    }

    /// Bytes per element for the given dtype string.
    var bytesPerElement: Int {
        switch dtype {
        case "F64": return 8
        case "F32", "I32": return 4
        case "F16", "BF16": return 2
        case "I16", "F8_E4M3", "F8_E5M2": return 1
        case "I8", "U8": return 1
        case "I64": return 8
        case "BOOL": return 1
        default: return 2  // Assume F16 as default
    }
    }
}

// MARK: - ModelAnalyzer

/// Analyzes safetensors model files to produce layer profiles for distributed sharding.
///
/// Scans all `.safetensors` files in a model directory, parses their headers (without
/// reading tensor data), groups tensors by layer index, and produces `LayerProfile` entries
/// suitable for computing a `ShardPlan`.
///
/// Usage:
/// ```swift
/// let profiles = try await ModelAnalyzer.shared.analyze(modelPath: "/path/to/model")
/// let plan = ShardPlan(profiles: profiles, nodes: nodes, strategy: .minNodes)
/// ```
public final class ModelAnalyzer: Sendable {

    /// Shared singleton instance.
    public static let shared = ModelAnalyzer()

    private init() {}

    // MARK: - Public API

    /// Analyze all safetensors files in a model directory and return layer profiles.
    ///
    /// - Parameter modelPath: Absolute path to a directory containing `.safetensors` files.
    /// - Returns: Ordered array of `LayerProfile` entries: embedding first, then transformer
    ///   layers in order, then output layer.
    /// - Throws: `ModelAnalyzerError` on I/O or parsing failures.
    public func analyze(modelPath: String) async throws -> [LayerProfile] {
        let dirURL = URL(fileURLWithPath: modelPath)

        // Verify directory exists
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir),
              isDir.boolValue
        else {
            throw ModelAnalyzerError.fileNotFound(modelPath)
        }

        // Find all .safetensors files
        let enumerator = FileManager.default.enumerator(
            at: dirURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        )
        var safetensorURLs: [URL] = []
        while let item = enumerator?.nextObject() as? URL {
            if item.pathExtension == "safetensors" {
                safetensorURLs.append(item)
            }
        }

        guard !safetensorURLs.isEmpty else {
            throw ModelAnalyzerError.fileNotFound("No .safetensors files in \(modelPath)")
        }

        // Parse all tensors from all files
        var allTensors: [TensorDescriptor] = []
        for url in safetensorURLs {
            let tensors = try readSafetensorsTensorMap(url: url)
            allTensors.append(contentsOf: tensors)
        }

        // Build layer profiles
        return buildLayerProfiles(from: allTensors)
    }

    // MARK: - Private: Safetensors Header Parsing

    /// Read the full tensor map from a safetensors file header.
    ///
    /// Format: first 8 bytes = little-endian u64 header length, next N bytes = JSON.
    /// Each key (except `__metadata__`) describes a tensor with dtype, shape, and data_offsets.
    private func readSafetensorsTensorMap(url: URL) throws -> [TensorDescriptor] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        // Read 8-byte header length
        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else {
            throw ModelAnalyzerError.invalidHeader("Failed to read header length from \(url.lastPathComponent)")
        }
        let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        guard headerLen > 0, headerLen < 512 * 1024 * 1024 else {
            throw ModelAnalyzerError.invalidHeader(
                "Invalid header length \(headerLen) in \(url.lastPathComponent)")
        }

        // Read JSON header
        guard let jsonData = try handle.read(upToCount: Int(headerLen)),
              jsonData.count == Int(headerLen)
        else {
            throw ModelAnalyzerError.invalidHeader(
                "Failed to read \(headerLen) bytes of header from \(url.lastPathComponent)")
        }

        // Parse JSON
        let parsed: [String: Any]
        do {
            guard let obj = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
                throw ModelAnalyzerError.invalidJSON("Header is not a JSON object in \(url.lastPathComponent)")
            }
            parsed = obj
        } catch let error as ModelAnalyzerError {
            throw error
        } catch {
            throw ModelAnalyzerError.invalidJSON("JSON parse error in \(url.lastPathComponent): \(error)")
        }

        // Extract tensor descriptors
        var descriptors: [TensorDescriptor] = []
        for (name, value) in parsed where name != "__metadata__" {
            guard let info = value as? [String: Any],
                  let dtype = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2
            else {
                continue
            }
            descriptors.append(TensorDescriptor(
                name: name,
                dtype: dtype,
                shape: shape,
                byteOffset: offsets[0],
                byteCount: offsets[1] - offsets[0]
            ))
        }

        return descriptors
    }

    // MARK: - Private: Layer Profile Construction

    /// Group tensors into layer profiles.
    ///
    /// Groups tensors by matching `layers\.(\d+)` in their names for transformer layers,
    /// detects embedding tensors via `embed_tokens` / `wte` / `embed`, and output tensors
    /// via `lm_head` / standalone `output`.
    ///
    /// MoE layers are detected via `gate_proj` or `experts` in tensor names within a transformer layer.
    private func buildLayerProfiles(from tensors: [TensorDescriptor]) -> [LayerProfile] {
        // Group tensors by layer
        struct LayerGroup {
            var transformerIndex: Int?
            var isEmbedding: Bool = false
            var isOutput: Bool = false
            var isMoE: Bool = false
            var tensors: [TensorDescriptor] = []
        }

        var layerMap: [Int: LayerGroup] = [:]  // keyed by transformer layer index
        var embeddingGroup = LayerGroup(isEmbedding: true)
        var outputGroup = LayerGroup(isOutput: true)

        // Pattern for matching transformer layer indices
        let layerPattern = /model\.layers\.(\d+)/

        for tensor in tensors {
            if let match = tensor.name.firstMatch(of: layerPattern) {
                let idx = Int(match.1)!
                if layerMap[idx] == nil {
                    layerMap[idx] = LayerGroup(transformerIndex: idx)
                }
                layerMap[idx]!.tensors.append(tensor)

                // Detect MoE: only if tensor path contains "experts" or "moe"
                // (dense MLP also uses "gate_proj" so we don't use that alone)
                if tensor.name.contains("experts") || tensor.name.contains("moe") {
                    layerMap[idx]!.isMoE = true
                }
            } else if isEmbeddingTensor(tensor.name) {
                embeddingGroup.tensors.append(tensor)
            } else if isOutputTensor(tensor.name) {
                outputGroup.tensors.append(tensor)
            }
        }

        // Build profiles in order: embedding, transformer layers (sorted), output
        var profiles: [LayerProfile] = []
        var layerIndex = 0

        // Embedding
        if !embeddingGroup.tensors.isEmpty {
            let (params, mem) = computeProfileStats(for: embeddingGroup.tensors)
            profiles.append(LayerProfile(
                layerIndex: layerIndex,
                parameterCount: params,
                estimatedMemoryBytes: mem,
                layerType: .embedding
            ))
            layerIndex += 1
        }

        // Transformer layers (sorted by original index)
        let sortedIndices = layerMap.keys.sorted()
        for idx in sortedIndices {
            guard let group = layerMap[idx] else { continue }
            let (params, mem) = computeProfileStats(for: group.tensors)
            let type: LayerType = group.isMoE ? .moe : .transformer
            profiles.append(LayerProfile(
                layerIndex: layerIndex,
                parameterCount: params,
                estimatedMemoryBytes: mem,
                layerType: type
            ))
            layerIndex += 1
        }

        // Output
        if !outputGroup.tensors.isEmpty {
            let (params, mem) = computeProfileStats(for: outputGroup.tensors)
            profiles.append(LayerProfile(
                layerIndex: layerIndex,
                parameterCount: params,
                estimatedMemoryBytes: mem,
                layerType: .output
            ))
        }

        return profiles
    }

    /// Whether a tensor name belongs to the embedding layer.
    private func isEmbeddingTensor(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.contains("embed_tokens")
            || lower.contains("wte")
            || (lower.contains("embed") && !lower.contains("layers"))
    }

    /// Whether a tensor name belongs to the output (lm_head) layer.
    private func isOutputTensor(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.contains("lm_head")
            || (lower.hasSuffix("output.weight") && !lower.contains("layers"))
            || lower.contains("model.output")
    }

    /// Compute total parameter count and estimated memory for a set of tensors.
    private func computeProfileStats(for tensors: [TensorDescriptor]) -> (parameterCount: UInt64, memoryBytes: UInt64) {
        var totalParams: UInt64 = 0
        var totalMemory: UInt64 = 0
        for tensor in tensors {
            totalParams += tensor.elementCount
            totalMemory += tensor.estimatedBytes
        }
        return (totalParams, totalMemory)
    }
}
