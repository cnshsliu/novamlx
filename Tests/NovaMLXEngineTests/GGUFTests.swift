import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("GGUF")
struct GGUFTests {

    @Test("Parses GGUF magic, metadata, and F32 tensor")
    func parsesF32() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("toy.gguf")
        let values: [Float] = [1, 2, 3, 4]
        try GGUFTestWriter.write(
            to: url,
            architecture: "llama",
            extra: [
                "llama.embedding_length": .uint32(4),
                "llama.block_count": .uint32(1),
                "llama.attention.head_count": .uint32(1),
                "llama.feed_forward_length": .uint32(8),
                "llama.context_length": .uint32(32),
            ],
            tensors: [
                GGUFTestWriter.Tensor(
                    name: "token_embd.weight",
                    dimensions: [4, 1],
                    type: .f32,
                    payload: values.withUnsafeBufferPointer { Data(buffer: $0) }
                )
            ]
        )

        let gguf = try GGUFFile.parse(url: url)
        #expect(gguf.architecture == "llama")
        #expect(gguf.alignment == 1)
        #expect(gguf.archInt("embedding_length") == 4)
        #expect(gguf.tensors["token_embd.weight"] != nil)

        try prepareGGUFModelDirectory(dir)
        let cfgURL = dir.appendingPathComponent("config.json")
        let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any]
        #expect(cfg?["model_type"] as? String == "llama")
        #expect(cfg?["hidden_size"] as? Int == 4)
        #expect(cfg?["num_hidden_layers"] as? Int == 1)
        #expect(cfg?["vocab_size"] as? Int == 4)

        let weights = try loadGGUFWeights(modelDirectory: dir)
        let arr = try #require(weights["model.embed_tokens.weight"])
        #expect(arr.size == 4)
    }

    @Test("Q4_0 high nibble is the second half of the block")
    func q4_0Layout() throws {
        var block = Data()
        block.append(contentsOf: [0x00, 0x40])
        block.append(contentsOf: Array(repeating: UInt8(0x21), count: 16))
        let info = GGUFTensorInfo(
            name: "w",
            dimensions: [32],
            type: .q4_0,
            dataOffset: 0,
            dataLength: 18,
            totalElements: 32
        )
        let arr = try GGUFDequant.toMLXArray(data: block, info: info)
        let floats = arr.asArray(Float.self)
        #expect(floats.count == 32)
        #expect(abs(floats[0] - Float(Float16(2.0) * Float16(-7))) < 0.02)
        #expect(abs(floats[16] - Float(Float16(2.0) * Float16(-6))) < 0.02)
    }

    @Test("Q4_K dequant produces 256 values per super-block")
    func q4_KBlockSize() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("q4k.gguf")
        var payload = Data(count: 144)
        // d = 1.0 f16 at bytes 0-1, dmin = 0 at 2-3, scales/qs zero
        var one = Float16(1.0)
        withUnsafeBytes(of: &one) { payload.replaceSubrange(0..<2, with: $0) }

        try GGUFTestWriter.write(
            to: url,
            architecture: "llama",
            extra: [
                "llama.embedding_length": .uint32(256),
                "llama.block_count": .uint32(1),
                "llama.attention.head_count": .uint32(1),
                "llama.feed_forward_length": .uint32(256),
                "llama.vocab_size": .uint32(256),
            ],
            tensors: [
                GGUFTestWriter.Tensor(
                    name: "output_norm.weight",
                    dimensions: [256],
                    type: .q4_K,
                    payload: payload
                )
            ]
        )
        let weights = try loadGGUFWeights(modelDirectory: dir)
        let arr = try #require(weights["model.norm.weight"])
        #expect(arr.size == 256)
    }

    @Test("Unsupported IQ quant fails closed")
    func rejectsIQ() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("iq.gguf")
        try GGUFTestWriter.write(
            to: url,
            architecture: "llama",
            extra: [
                "llama.embedding_length": .uint32(32),
                "llama.block_count": .uint32(1),
                "llama.attention.head_count": .uint32(1),
                "llama.feed_forward_length": .uint32(32),
                "llama.vocab_size": .uint32(32),
            ],
            tensors: [
                GGUFTestWriter.Tensor(
                    name: "token_embd.weight",
                    dimensions: [32],
                    type: .iq4_nl,
                    payload: Data(count: 32)
                )
            ]
        )
        #expect(throws: GGUFError.self) {
            _ = try loadGGUFWeights(modelDirectory: dir)
        }
    }

    @Test("Name map covers llama block tensors")
    func nameMap() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("names.gguf")
        let f32 = Data(repeating: 0, count: 16)
        try GGUFTestWriter.write(
            to: url,
            architecture: "llama",
            extra: [
                "llama.embedding_length": .uint32(4),
                "llama.block_count": .uint32(1),
                "llama.attention.head_count": .uint32(1),
                "llama.feed_forward_length": .uint32(4),
                "llama.vocab_size": .uint32(4),
            ],
            tensors: [
                GGUFTestWriter.Tensor(name: "blk.0.attn_q.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.attn_k.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.attn_v.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.attn_output.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.ffn_gate.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.ffn_up.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.ffn_down.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.attn_norm.weight", dimensions: [4], type: .f32, payload: f32),
                GGUFTestWriter.Tensor(name: "blk.0.ffn_norm.weight", dimensions: [4], type: .f32, payload: f32),
            ]
        )
        let weights = try loadGGUFWeights(modelDirectory: dir)
        #expect(weights["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(weights["model.layers.0.mlp.gate_proj.weight"] != nil)
        #expect(weights["model.layers.0.input_layernorm.weight"] != nil)
        #expect(weights["model.layers.0.post_attention_layernorm.weight"] != nil)
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

// MARK: - Minimal GGUF writer for tests

enum GGUFTestWriter {
    enum Meta {
        case uint32(UInt32)
        case uint64(UInt64)
        case string(String)
        case stringArray([String])
    }

    struct Tensor {
        var name: String
        var dimensions: [UInt64]
        var type: GGUFTensorType
        var payload: Data
    }

    static func write(
        to url: URL, architecture: String, extra: [String: Meta] = [:], tensors: [Tensor]
    ) throws {
        var data = Data()
        func u32(_ v: UInt32) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }
        func u64(_ v: UInt64) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }
        func str(_ s: String) {
            let utf = Array(s.utf8)
            u64(UInt64(utf.count))
            data.append(contentsOf: utf)
        }
        func writeMeta(_ key: String, _ value: Meta) {
            str(key)
            switch value {
            case .uint32(let v):
                u32(4)
                u32(v)
            case .uint64(let v):
                u32(10)
                u64(v)
            case .string(let s):
                u32(8)
                str(s)
            case .stringArray(let items):
                u32(9)
                u32(8)
                u64(UInt64(items.count))
                for item in items { str(item) }
            }
        }

        u32(0x4655_4747)
        u32(3)
        u64(UInt64(tensors.count))
        var kv: [(String, Meta)] = [
            ("general.architecture", .string(architecture)),
            ("general.alignment", .uint64(1)),
        ]
        kv.append(("tokenizer.ggml.tokens", .stringArray((0..<4).map { "t\($0)" })))
        kv.append(contentsOf: extra.map { ($0.key, $0.value) })
        u64(UInt64(kv.count))
        for (k, v) in kv { writeMeta(k, v) }

        var tensorPayloads: [Data] = []
        var running: UInt64 = 0
        for t in tensors {
            str(t.name)
            u32(UInt32(t.dimensions.count))
            for d in t.dimensions { u64(d) }
            u32(t.type.rawValue)
            u64(running)
            tensorPayloads.append(t.payload)
            running += UInt64(t.payload.count)
        }
        for payload in tensorPayloads {
            data.append(payload)
        }
        try data.write(to: url)
    }
}
