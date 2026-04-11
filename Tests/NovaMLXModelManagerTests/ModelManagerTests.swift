import Testing
import Foundation
@testable import NovaMLXModelManager

@Suite("ModelManager Tests")
struct ModelManagerTests {
    private func makeTempManager() -> ModelManager {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        return ModelManager(modelsDirectory: tempDir)
    }

    @Test("Register model")
    func registerModel() {
        let manager = makeTempManager()
        let record = manager.register(
            id: "test-model",
            family: .llama,
            modelType: .llm,
            remoteURL: "https://example.com/model"
        )
        #expect(record.id == "test-model")
        #expect(record.family == .llama)
    }

    @Test("Available models")
    func availableModels() {
        let manager = makeTempManager()
        manager.register(id: "model-1", remoteURL: "https://example.com/1")
        manager.register(id: "model-2", remoteURL: "https://example.com/2")
        #expect(manager.availableModels().count == 2)
    }

    @Test("Downloaded models empty initially")
    func downloadedModelsEmpty() {
        let manager = makeTempManager()
        manager.register(id: "test", remoteURL: "https://example.com/test")
        #expect(manager.downloadedModels().isEmpty)
    }

    @Test("Is downloaded false initially")
    func isDownloadedFalse() {
        let manager = makeTempManager()
        manager.register(id: "test", remoteURL: "https://example.com/test")
        #expect(manager.isDownloaded("test") == false)
    }

    @Test("Is downloaded false for unknown")
    func isDownloadedUnknown() {
        let manager = makeTempManager()
        #expect(manager.isDownloaded("unknown") == false)
    }

    @Test("Get record")
    func getRecord() {
        let manager = makeTempManager()
        manager.register(id: "test", family: .mistral, remoteURL: "https://example.com/test")
        let record = manager.getRecord("test")
        #expect(record != nil)
        #expect(record?.family == .mistral)
    }

    @Test("Get record nil for unknown")
    func getRecordNil() {
        let manager = makeTempManager()
        #expect(manager.getRecord("unknown") == nil)
    }

    @Test("Unregister model")
    func unregisterModel() {
        let manager = makeTempManager()
        manager.register(id: "test", remoteURL: "https://example.com/test")
        manager.unregister("test")
        #expect(manager.getRecord("test") == nil)
    }

    @Test("Register popular models")
    func registerPopularModels() {
        let manager = makeTempManager()
        manager.registerPopularModels()
        let models = manager.availableModels()
        #expect(models.count >= 5)
    }

    @Test("Model record properties")
    func modelRecordProperties() {
        let record = ModelRecord(
            id: "test",
            family: .phi,
            modelType: .llm,
            source: .huggingFace,
            localURL: URL(fileURLWithPath: "/tmp/test"),
            remoteURL: "https://example.com/test",
            sizeBytes: 4_000_000_000,
            version: "2.0"
        )
        #expect(record.id == "test")
        #expect(record.source == .huggingFace)
        #expect(record.sizeBytes == 4_000_000_000)
        #expect(record.version == "2.0")
    }

    @Test("Download progress calculation")
    func downloadProgress() {
        let progress = DownloadProgress(bytesDownloaded: 500, totalBytes: 1000, bytesPerSecond: 100)
        #expect(progress.fractionCompleted == 0.5)
        #expect(progress.bytesPerSecond == 100)
    }

    @Test("Download progress zero total")
    func downloadProgressZeroTotal() {
        let progress = DownloadProgress(bytesDownloaded: 500, totalBytes: 0)
        #expect(progress.fractionCompleted == 0)
    }

    private func createFakeModelConfig(at dir: URL, architectures: [String]? = nil, modelType: String? = nil, hasVisionConfig: Bool = false) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var config: [String: Any] = [:]
        if let architectures { config["architectures"] = architectures }
        if let modelType { config["model_type"] = modelType }
        if hasVisionConfig { config["vision_config"] = ["hidden_size": 1024] }
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))
    }

    @Test("Model discovery finds models in flat structure")
    func discoveryFlat() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(at: tempDir.appendingPathComponent("my-llama"), architectures: ["LlamaForCausalLM"], modelType: "llama")

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].modelId == "my-llama")
        #expect(models[0].modelType == .llm)
        #expect(models[0].family == .llama)
    }

    @Test("Model discovery finds models in org/repo structure")
    func discoveryOrgRepo() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(
            at: tempDir.appendingPathComponent("mlx-community/Phi-3.5-mini-instruct-4bit"),
            architectures: ["Phi3ForCausalLM"],
            modelType: "phi3"
        )

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].modelId == "mlx-community/Phi-3.5-mini-instruct-4bit")
        #expect(models[0].modelType == .llm)
        #expect(models[0].family == .phi)
    }

    @Test("Model discovery detects VLM from architecture")
    func discoveryVLM() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(
            at: tempDir.appendingPathComponent("qwen-vl"),
            architectures: ["Qwen2VLForConditionalGeneration"],
            modelType: "qwen2_vl"
        )

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].modelType == .vlm)
        #expect(models[0].family == .qwen)
    }

    @Test("Model discovery detects VLM from vision_config")
    func discoveryVLMFromVisionConfig() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(
            at: tempDir.appendingPathComponent("some-vlm"),
            modelType: "custom_vlm",
            hasVisionConfig: true
        )

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].modelType == .vlm)
    }

    @Test("Model discovery detects embedding from architecture")
    func discoveryEmbedding() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(
            at: tempDir.appendingPathComponent("bert-embed"),
            architectures: ["BertModel"],
            modelType: "bert"
        )

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].modelType == .embedding)
    }

    @Test("Model discovery detects embedding from model_type")
    func discoveryEmbeddingFromModelType() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(
            at: tempDir.appendingPathComponent("xlm-roberta-embed"),
            modelType: "xlm-roberta"
        )

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].modelType == .embedding)
    }

    @Test("Model discovery skips adapters")
    func discoverySkipsAdapters() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let adapterDir = tempDir.appendingPathComponent("my-adapter")
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
        let config: [String: Any] = ["architectures": ["LlamaForCausalLM"], "model_type": "llama"]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: adapterDir.appendingPathComponent("config.json"))
        try data.write(to: adapterDir.appendingPathComponent("adapter_config.json"))

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.isEmpty)
    }

    @Test("Model discovery skips hidden directories")
    func discoverySkipsHidden() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(at: tempDir.appendingPathComponent(".hidden-model"), modelType: "llama")

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.isEmpty)
    }

    @Test("Model discovery family detection from model ID")
    func discoveryFamilyFromId() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try createFakeModelConfig(at: tempDir.appendingPathComponent("some-gemma-variant"), modelType: "unknown_arch")

        let discovery = ModelDiscovery()
        let models = discovery.discover(in: tempDir)
        #expect(models.count == 1)
        #expect(models[0].family == .gemma)
    }

    @Test("discoverModels registers found models in manager")
    func discoverModelsRegisters() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let manager = ModelManager(modelsDirectory: tempDir)

        let modelDir = tempDir.appendingPathComponent("hub/models/mlx-community/test-llama")
        try createFakeModelConfig(at: modelDir, architectures: ["LlamaForCausalLM"], modelType: "llama")

        let discovered = manager.discoverModels()
        #expect(discovered.count >= 1)

        let record = manager.getRecord("mlx-community/test-llama")
        #expect(record != nil)
        #expect(record?.family == .llama)
        #expect(record?.modelType == .llm)
        #expect(record?.downloadedAt != nil)
    }

    @Test("discoverModels does not overwrite existing loaded records")
    func discoverModelsPreservesExisting() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let manager = ModelManager(modelsDirectory: tempDir)
        manager.register(id: "test-model", family: .mistral, modelType: .llm, remoteURL: "https://example.com/test")

        let modelDir = tempDir.appendingPathComponent("test-model")
        try createFakeModelConfig(at: modelDir, architectures: ["LlamaForCausalLM"], modelType: "llama")

        _ = manager.discoverModels()

        let record = manager.getRecord("test-model")
        #expect(record != nil)
        #expect(record?.family == .mistral)
    }
}
