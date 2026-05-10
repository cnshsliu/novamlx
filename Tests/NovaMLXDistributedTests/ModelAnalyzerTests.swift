import Foundation
import Testing
@testable import NovaMLXDistributed

@Suite("ModelAnalyzer")
struct ModelAnalyzerTests {

    // MARK: - Helpers

    /// Build a synthetic safetensors file in a temporary directory.
    ///
    /// Creates a valid safetensors header (8-byte LE length prefix + JSON) followed by
    /// enough zero padding to satisfy the data offsets described in the header.
    private func createSyntheticSafetensors(
        name: String,
        tensors: [String: (dtype: String, shape: [Int], offsets: (Int, Int))],
        in dir: URL
    ) -> URL {
        let url = dir.appendingPathComponent(name)

        // Build the header JSON
        var header: [String: Any] = [:]
        for (name, info) in tensors {
            header[name] = [
                "dtype": info.dtype,
                "shape": info.shape,
                "data_offsets": [info.offsets.0, info.offsets.1],
            ]
        }
        header["__metadata__"] = ["format": "pt"]

        let json = try! JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(json)

        // Compute the maximum data end offset and pad to that size
        let maxEnd = tensors.values.map(\.offsets.1).max() ?? 0
        let headerTotalSize = 8 + json.count
        if data.count < headerTotalSize + maxEnd {
            data.append(Data(count: headerTotalSize + maxEnd - data.count))
        }

        try! data.write(to: url)
        return url
    }

    /// Create a standard 4-layer transformer model directory for testing.
    /// - Returns: the directory URL containing .safetensors files.
    private func createTestModelDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // 4096 hidden, 4 layers, each proj weight is [4096, 4096] F16 = 33554432 bytes
        let projSize = 4096 * 4096 * 2
        // Embedding: [32000, 4096] F16 = 262144000 bytes
        let embedSize = 32000 * 4096 * 2
        // lm_head: [32000, 4096] F16 = same as embed
        let lmHeadSize = embedSize

        var offset = 0

        // File 1: embeddings + layers 0-1
        let tensors1: [String: (String, [Int], (Int, Int))] = [
            "model.embed_tokens.weight": ("F16", [32000, 4096], (offset, offset + embedSize)),
        ]
        offset += embedSize
        var t1 = tensors1
        for layer in 0..<2 {
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let name = "model.layers.\(layer).self_attn.\(proj).weight"
                t1[name] = ("F16", [4096, 4096], (offset, offset + projSize))
                offset += projSize
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let name = "model.layers.\(layer).mlp.\(proj).weight"
                t1[name] = ("F16", [4096, 4096], (offset, offset + projSize))
                offset += projSize
            }
            // layer_norm
            t1["model.layers.\(layer).input_layernorm.weight"] = ("F32", [4096], (offset, offset + 4096 * 4))
            offset += 4096 * 4
            t1["model.layers.\(layer).post_attention_layernorm.weight"] = ("F32", [4096], (offset, offset + 4096 * 4))
            offset += 4096 * 4
        }
        _ = createSyntheticSafetensors(name: "model-00001-of-00002.safetensors", tensors: t1, in: dir)

        // File 2: layers 2-3 + lm_head
        offset = 0
        var t2: [String: (String, [Int], (Int, Int))] = [:]
        for layer in 2..<4 {
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let name = "model.layers.\(layer).self_attn.\(proj).weight"
                t2[name] = ("F16", [4096, 4096], (offset, offset + projSize))
                offset += projSize
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let name = "model.layers.\(layer).mlp.\(proj).weight"
                t2[name] = ("F16", [4096, 4096], (offset, offset + projSize))
                offset += projSize
            }
            t2["model.layers.\(layer).input_layernorm.weight"] = ("F32", [4096], (offset, offset + 4096 * 4))
            offset += 4096 * 4
            t2["model.layers.\(layer).post_attention_layernorm.weight"] = ("F32", [4096], (offset, offset + 4096 * 4))
            offset += 4096 * 4
        }
        t2["lm_head.weight"] = ("F16", [32000, 4096], (offset, offset + lmHeadSize))
        offset += lmHeadSize

        _ = createSyntheticSafetensors(name: "model-00002-of-00002.safetensors", tensors: t2, in: dir)

        return dir
    }

    /// Create a MoE model directory with experts in the MLP.
    private func createMoETestModelDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // 2 layers, MoE with 8 experts, each expert has gate/up/down
        let hiddenDim = 1024
        let intermediateDim = 2048
        let expertCount = 8
        let f16 = 2
        let expertParamBytes = intermediateDim * hiddenDim * f16  // one expert weight

        var offset = 0
        var tensors: [String: (String, [Int], (Int, Int))] = [:]

        tensors["model.embed_tokens.weight"] = ("F16", [32000, hiddenDim], (offset, offset + 32000 * hiddenDim * f16))
        offset += 32000 * hiddenDim * f16

        for layer in 0..<2 {
            // Self-attention
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let name = "model.layers.\(layer).self_attn.\(proj).weight"
                let sz = hiddenDim * hiddenDim * f16
                tensors[name] = ("F16", [hiddenDim, hiddenDim], (offset, offset + sz))
                offset += sz
            }
            // MoE: gate_proj signals MoE detection
            let gateSize = expertCount * hiddenDim * f16
            tensors["model.layers.\(layer).block_sparse_moe.gate.weight"] = ("F16", [expertCount, hiddenDim], (offset, offset + gateSize))
            offset += gateSize

            // Experts
            for e in 0..<expertCount {
                for proj in ["w1", "w2", "w3"] {
                    let name = "model.layers.\(layer).block_sparse_moe.experts.\(e).\(proj).weight"
                    tensors[name] = ("F16", [intermediateDim, hiddenDim], (offset, offset + expertParamBytes))
                    offset += expertParamBytes
                }
            }

            // Layer norms
            let lnBytes = hiddenDim * 4
            tensors["model.layers.\(layer).input_layernorm.weight"] = ("F32", [hiddenDim], (offset, offset + lnBytes))
            offset += lnBytes
        }

        tensors["lm_head.weight"] = ("F16", [32000, hiddenDim], (offset, offset + 32000 * hiddenDim * f16))
        offset += 32000 * hiddenDim * f16

        _ = createSyntheticSafetensors(name: "model.safetensors", tensors: tensors, in: dir)
        return dir
    }

    // MARK: - Tests

    @Test("Transformer layers are identified from safetensors headers")
    func transformerLayersIdentified() async throws {
        let dir = try createTestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)

        // Should have: embedding + 4 transformer layers + output = 6
        #expect(profiles.count == 6)

        let transformerLayers = profiles.filter { $0.layerType == .transformer }
        #expect(transformerLayers.count == 4)
    }

    @Test("Embedding layer is identified")
    func embeddingLayerIdentified() async throws {
        let dir = try createTestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)

        let embedding = profiles.first { $0.layerType == .embedding }
        #expect(embedding != nil)
        // Embedding should be layerIndex 0
        #expect(embedding?.layerIndex == 0)
        // Should have substantial parameters (32000 * 4096 * 2 bytes)
        #expect(embedding!.estimatedMemoryBytes > 0)
    }

    @Test("Output layer is identified")
    func outputLayerIdentified() async throws {
        let dir = try createTestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)

        let output = profiles.last { $0.layerType == .output }
        #expect(output != nil)
        // Output should be the last layer
        #expect(output?.layerIndex == profiles.count - 1)
    }

    @Test("Layer ordering: embedding first, transformer middle, output last")
    func layerOrdering() async throws {
        let dir = try createTestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)

        // First should be embedding
        #expect(profiles.first?.layerType == .embedding)
        // Last should be output
        #expect(profiles.last?.layerType == .output)
        // Middle should be all transformer
        let middle = profiles.dropFirst().dropLast()
        #expect(middle.allSatisfy { $0.layerType == .transformer })

        // Layer indices should be sequential
        for (i, p) in profiles.enumerated() {
            #expect(p.layerIndex == i)
        }
    }

    @Test("MoE layers detected via gate_proj / experts patterns")
    func moeLayerDetection() async throws {
        let dir = try createMoETestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)

        // Both transformer layers should be MoE
        let moeLayers = profiles.filter { $0.layerType == .moe }
        #expect(moeLayers.count == 2)

        // MoE layers should have more memory than a regular transformer would
        for moe in moeLayers {
            #expect(moe.parameterCount > 0)
            #expect(moe.estimatedMemoryBytes > 0)
        }
    }

    @Test("ShardPlan respects memory ratios from profiles")
    func shardPlanRespectsMemoryRatios() async throws {
        let dir = try createTestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)
        #expect(profiles.count == 6)

        let nodeA = NodeSpec(
            nodeId: "mac-a",
            totalMemoryBytes: 128 * 1024 * 1024 * 1024,
            computeCapability: 1.0,
            hostname: "mac-a.local",
            port: 6591
        )
        let nodeB = NodeSpec(
            nodeId: "mac-b",
            totalMemoryBytes: 64 * 1024 * 1024 * 1024,
            computeCapability: 0.6,
            hostname: "mac-b.local",
            port: 6591
        )

        let plan = ShardPlan(profiles: profiles, nodes: [nodeA, nodeB], strategy: .minNodes)

        // Should produce 2 assignments
        #expect(plan.assignments.count == 2)

        // All layers must be covered
        let totalCovered = plan.assignments.reduce(0) { $0 + ($1.endLayer - $1.startLayer) }
        #expect(totalCovered == profiles.count)

        // Node A (128GB) should get roughly 2x the layers of Node B (64GB)
        let layersA = plan.assignments[0].endLayer - plan.assignments[0].startLayer
        let layersB = plan.assignments[1].endLayer - plan.assignments[1].startLayer
        // Ratio should be approximately 2:1 (128:64)
        let ratio = Double(layersA) / Double(layersB)
        #expect(ratio > 1.5 && ratio < 2.5)
    }

    @Test("Error thrown for missing directory")
    func fileNotFoundThrows() async {
        do {
            _ = try await ModelAnalyzer.shared.analyze(
                modelPath: "/tmp/novamlx_nonexistent_\(UUID().uuidString)"
            )
            #expect(Bool(false), "Should have thrown")
        } catch let error as ModelAnalyzerError {
            if case .fileNotFound = error {
                // Expected
            } else {
                #expect(Bool(false), "Expected fileNotFound, got \(error)")
            }
        } catch {
            #expect(Bool(false), "Unexpected error type: \(error)")
        }
    }

    @Test("Error thrown for invalid safetensors header")
    func invalidHeaderThrows() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Write a file with only 3 bytes (too short for the 8-byte header length)
        let badURL = dir.appendingPathComponent("bad.safetensors")
        try Data([0x01, 0x02, 0x03]).write(to: badURL)

        do {
            _ = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)
            #expect(Bool(false), "Should have thrown")
        } catch let error as ModelAnalyzerError {
            if case .invalidHeader = error {
                // Expected
            } else {
                #expect(Bool(false), "Expected invalidHeader, got \(error)")
            }
        } catch {
            #expect(Bool(false), "Unexpected error type: \(error)")
        }
    }

    @Test("Parameter count matches tensor sizes")
    func parameterCountAccurate() async throws {
        let dir = try createTestModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = try await ModelAnalyzer.shared.analyze(modelPath: dir.path)

        // Each transformer layer has 4 attention projs + 3 MLP projs (each [4096,4096] F16)
        // + 2 layernorms (each [4096] F32)
        // Attention params: 4 * 4096 * 4096 = 67108864
        // MLP params: 3 * 4096 * 4096 = 50331648
        // LayerNorm params: 2 * 4096 = 8192
        // Total params per transformer layer: 117448704
        let transformerLayers = profiles.filter { $0.layerType == .transformer }
        for layer in transformerLayers {
            #expect(layer.parameterCount == 117_448_704)
        }

        // Embedding: 32000 * 4096 = 131072000 params
        let embedding = profiles.first { $0.layerType == .embedding }
        #expect(embedding?.parameterCount == 131_072_000)

        // Output: 32000 * 4096 = 131072000 params
        let output = profiles.first { $0.layerType == .output }
        #expect(output?.parameterCount == 131_072_000)
    }
}
