import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("Qwen-Image 2.1 discovery")
struct QwenImage21DiscoveryTests {
    @Test("Diffusers class name routes a renamed folder to qwenImage21")
    func classNameRoutes21() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("lab/custom-dit")
        try FileManager.default.createDirectory(at: model.appendingPathComponent("vae"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: model.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        let config = """
        {"_class_name":"QwenImage21Transformer2DModel"}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("transformer/config.json"))

        let found = ModelDiscovery().discover(in: root)
        #expect(found.count == 1)
        #expect(found[0].modelId == "lab/custom-dit")
        #expect(found[0].modelType == .image)
        #expect(found[0].family == .qwenImage21)
    }

    @Test("Qwen-Image 1.x id stays qwenImage")
    func originalStays1x() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("Qwen/Qwen-Image")
        try FileManager.default.createDirectory(at: model.appendingPathComponent("vae"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: model.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        let config = """
        {"_class_name":"QwenImageTransformer2DModel"}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("transformer/config.json"))

        let found = ModelDiscovery().discover(in: root)
        #expect(found.count == 1)
        #expect(found[0].family == .qwenImage)
    }

    @Test("Repo id Qwen/Qwen-Image-2.1 routes to qwenImage21")
    func repoIdRoutes21() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("Qwen/Qwen-Image-2.1")
        try FileManager.default.createDirectory(at: model.appendingPathComponent("vae"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: model.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        let config = """
        {"_class_name":"QwenImage21Transformer2DModel"}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("transformer/config.json"))

        let found = ModelDiscovery().discover(in: root)
        #expect(found.count == 1)
        #expect(found[0].modelId == "Qwen/Qwen-Image-2.1")
        #expect(found[0].family == .qwenImage21)
    }

    @Test("aria2 sidecar inside a diffusers subfolder marks the model incomplete")
    func nestedAria2SidecarIsIncomplete() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("mlx-community/Qwen-Image-2.1-MLX-4bit")
        try FileManager.default.createDirectory(at: model.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: model.appendingPathComponent("vae"), withIntermediateDirectories: true)
        let config = """
        {"_class_name":"QwenImage21Transformer2DModel","quantization":{"bits":4,"group_size":64}}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("transformer/config.json"))
        // Sparse holed download: safetensors is non-zero, but the aria2 control
        // sidecar is still there and lives inside the subfolder.
        try Data(repeating: 1, count: 1024).write(to: model.appendingPathComponent("transformer/model.safetensors"))
        try Data(repeating: 1, count: 16).write(
            to: model.appendingPathComponent("transformer/model.safetensors.aria2"))
        try Data(repeating: 1, count: 1024).write(to: model.appendingPathComponent("vae/model.safetensors"))

        let found = ModelDiscovery().discover(in: root)
        #expect(found.count == 1)
        #expect(found[0].modelId == "mlx-community/Qwen-Image-2.1-MLX-4bit")
        #expect(found[0].isComplete == false)
    }

    @Test("4-bit repo id is the same image family, not a different model kind")
    func fourBitRepoIsSameFamily() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("mlx-community/Qwen-Image-2.1-MLX-4bit")
        try FileManager.default.createDirectory(at: model.appendingPathComponent("vae"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: model.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        let config = """
        {"_class_name":"QwenImage21Transformer2DModel","quantization":{"bits":4,"group_size":64}}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("transformer/config.json"))

        let found = ModelDiscovery().discover(in: root)
        #expect(found.count == 1)
        #expect(found[0].modelId == "mlx-community/Qwen-Image-2.1-MLX-4bit")
        #expect(found[0].family == .qwenImage21)
        #expect(found[0].modelType == .image)
    }
}
