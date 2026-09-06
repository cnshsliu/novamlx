import Foundation
import MLXLLM
import MLXLMCommon
import NovaMLXCore
import Testing
@testable import NovaMLXEngine

@Suite("MTP support")
struct MtpSupportTests {
    @Test("mtp draft id candidates from 8-bit backbone")
    func draftCandidates() {
        let ids = mtpDraftCandidates(forMainId: "mlx-community/Qwen3.8-27B-8bit")
        #expect(ids.contains("mlx-community/Qwen3.8-27B-MTP-4bit"))
        #expect(ids.contains("mlx-community/Qwen3.8-27B-MTP-8bit"))
    }

    @Test("qwen3_5_mtp config is a draft head")
    func mtpConfigDetection() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-cfg-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let json = """
        {"model_type":"qwen3_5_mtp","text_config":{"hidden_size":64,"vocab_size":128}}
        """
        try json.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(isMtpDraftConfig(at: dir))
        #expect(checkpointHasMtpWeights(at: dir))
    }

    @Test("backbone with only mtp_num_hidden_layers is not treated as having MTP weights")
    func backboneFlagOnly() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-flag-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let json = """
        {"model_type":"qwen3_5","text_config":{"mtp_num_hidden_layers":1,"vocab_size":248320}}
        """
        try json.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try """
        {"weight_map":{"language_model.model.layers.0.self_attn.q_proj.weight":"a.safetensors"}}
        """.write(to: dir.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        #expect(!isMtpDraftConfig(at: dir))
        #expect(!checkpointHasMtpWeights(at: dir))
    }

    @Test("hy_v4 native mtp_layers in the weight map count as MTP weights")
    func hyV4MtpLayers() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-hyv4-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try """
        {"model_type":"hy_v4","num_nextn_predict_layers":1}
        """.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try """
        {"weight_map":{"model.mtp_layers.0.eh_proj.weight":"mtp.safetensors"}}
        """.write(to: dir.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        #expect(!isMtpDraftConfig(at: dir))
        #expect(checkpointHasMtpWeights(at: dir))
    }

    @Test("registry pairs backbone with on-disk MTP")
    func registryPairs() throws {
        let models = NovaMLXPaths.modelsDir
        let mainRel = "mtp-test-org/Backbone-8bit"
        let draftRel = "mtp-test-org/Backbone-MTP-4bit"
        let mainDir = models.appendingPathComponent(mainRel)
        let draftDir = models.appendingPathComponent(draftRel)
        try FileManager.default.createDirectory(at: mainDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: draftDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: models.appendingPathComponent("mtp-test-org"))
        }
        try """
        {"model_type":"qwen3_5","text_config":{"vocab_size":248320}}
        """.write(to: mainDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try """
        {"model_type":"qwen3_5_mtp","text_config":{"vocab_size":248320,"mtp_num_hidden_layers":1}}
        """.write(to: draftDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let found = DraftModelRegistry.shared.mtpCandidate(forMainId: mainRel)
        #expect(found?.draftModelId == draftRel)
    }

    @Test("LLMTypeRegistry has qwen3_5_mtp")
    func registryHasQwen35Mtp() async {
        let json = Data("""
            {"model_type":"qwen3_5_mtp","text_config":{"hidden_size":64,"vocab_size":32,"num_hidden_layers":1,"mtp_num_hidden_layers":1}}
            """.utf8)
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: json, modelType: "qwen3_5_mtp")
        } catch let error as ModelFactoryError {
            if case .unsupportedModelType(let t) = error {
                Issue.record("qwen3_5_mtp not registered — got unsupportedModelType(\"\(t)\")")
            }
        } catch {
            // Construction may fail on dummy config; registration is what we assert.
        }
    }

    @Test("user-facing load of an MTP companion is rejected")
    func rejectDirectLoadMessage() {
        let err = NovaMLXError.mtpCompanionNotLoadable("mlx-community/Qwen3.8-27B-MTP-8bit")
        #expect(err.errorDescription?.contains("cannot be loaded directly") == true)
        #expect(err.errorDescription?.contains("backbone") == true)
    }
}
