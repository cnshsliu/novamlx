import Foundation
import NovaMLXCore
import Testing
@testable import NovaMLXEngine

@Suite("Companion model directory")
struct CompanionModelDirectoryTests {
    private func writeConfig(_ json: String, name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-companion-\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try json.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        return dir
    }

    @Test("Draft config.json marks an id the catalog does not know")
    func draftConfigs() throws {
        let dflash = try writeConfig(
            #"{"model_type":"qwen3","architectures":["DFlashDraftModel"]}"#,
            name: "dflash"
        )
        defer { try? FileManager.default.removeItem(at: dflash) }
        #expect(CompanionModel.isAuxiliary(id: "org/plain", directory: dflash))

        let dspark = try writeConfig(
            #"{"model_type":"deepseek_v41_dspark","architectures":["DSparkDraftModel"]}"#,
            name: "dspark"
        )
        defer { try? FileManager.default.removeItem(at: dspark) }
        #expect(CompanionModel.isAuxiliary(id: "org/plain", directory: dspark))

        let mtp = try writeConfig(#"{"model_type":"qwen3_5_mtp"}"#, name: "mtp")
        defer { try? FileManager.default.removeItem(at: mtp) }
        #expect(CompanionModel.isAuxiliary(id: "org/plain", directory: mtp))
    }

    @Test("A chat backbone with MTP metadata is not a companion")
    func backboneNotCompanion() throws {
        let backbone = try writeConfig(
            #"{"model_type":"qwen3_5","mtp_num_hidden_layers":1,"architectures":["Qwen3_5ForConditionalGeneration"]}"#,
            name: "backbone"
        )
        defer { try? FileManager.default.removeItem(at: backbone) }
        #expect(!CompanionModel.isAuxiliary(id: "mlx-community/Qwen3.8-27B-8bit", directory: backbone))
        #expect(!CompanionModel.isAuxiliary(id: "mlx-community/DeepSeek-V4.1-Flash", directory: backbone))
    }
}
