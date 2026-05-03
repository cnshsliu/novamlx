import Foundation
import MLXLMCommon
import MLXLLM

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
    }
}
