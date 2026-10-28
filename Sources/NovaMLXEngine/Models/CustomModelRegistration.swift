import Foundation
import MLXLMCommon
import MLXLLM
import MLXVLM

private func createCustom<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

enum CustomModelRegistration {
    private nonisolated(unsafe) static var registered = false

    static func ensureRegistered() async {
        guard !registered else { return }
        registered = true
        await LLMTypeRegistry.shared.registerModelType(
            "bailing_hybrid",
            creator: createCustom(BailingHybridConfiguration.self, BailingHybridModel.init))
        await LLMTypeRegistry.shared.registerModelType(
            "deepseek_v4",
            creator: createCustom(DeepseekV4Configuration.self, DeepseekV4Model.init))
        await LLMTypeRegistry.shared.registerModelType(
            "deepseek_v41",
            creator: { data in
                let flat = try DeepseekV41OfficialLoader.flattenConfig(data)
                let configuration = try JSONDecoder.json5().decode(
                    DeepseekV4Configuration.self, from: flat)
                return DeepseekV4Model(configuration)
            })
        // Decodable-only: Bolt's time_step_limit uses {"__float__": "Infinity"}.
        await LLMTypeRegistry.shared.registerModelType(
            "nemotron_h_puzzle",
            creator: { data in
                let configuration = try JSONDecoder.json5().decode(
                    NemotronHPuzzleConfiguration.self, from: data)
                return NemotronHPuzzleModel(configuration)
            })
        await LLMTypeRegistry.shared.registerModelType(
            "flint",
            creator: createCustom(FlintConfiguration.self, FlintModel.init))
        await QwenMtpRegistration.ensure()
        await UnlimitedOCRRegistration.register()
    }
}
