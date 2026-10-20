import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("MOSS-Image discovery")
struct MossImageDiscoveryTests {
    @Test("Repo id OpenMOSS-Team/MOSS-Image-SFT routes to mossImage")
    func sftRepoId() throws {
        let found = try discover(id: "OpenMOSS-Team/MOSS-Image-SFT", className: "MOSSImageTransformer")
        #expect(found.modelId == "OpenMOSS-Team/MOSS-Image-SFT")
        #expect(found.family == .mossImage)
        #expect(found.modelType == .image)
    }

    @Test("Base checkpoint uses the same family")
    func baseRepoId() throws {
        let found = try discover(id: "OpenMOSS-Team/MOSS-Image-Base", className: "MOSSImagePipeline")
        #expect(found.family == .mossImage)
        #expect(found.modelType == .image)
    }

    @Test("Renamed folder follows the diffusers class name")
    func renamedFolder() throws {
        let found = try discover(id: "lab/custom-dit", className: "MOSSImageTransformer")
        #expect(found.modelId == "lab/custom-dit")
        #expect(found.family == .mossImage)
        #expect(found.modelType == .image)
    }

    @Test("mlx-community bf16 mirror id is mossImage")
    func mirrorId() throws {
        let found = try discover(id: "mlx-community/MOSS-Image-SFT-bf16", className: "MOSSImageTransformer")
        #expect(found.family == .mossImage)
    }

    private func discover(id: String, className: String) throws -> DiscoveredModel {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(id)
        try FileManager.default.createDirectory(
            at: model.appendingPathComponent("vae"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: model.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        let config = #"{"_class_name":"\#(className)"}"#.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("transformer/config.json"))
        let found = ModelDiscovery().discover(in: root)
        return try #require(found.first)
    }
}
