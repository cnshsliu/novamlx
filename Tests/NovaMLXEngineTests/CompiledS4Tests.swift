import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import NovaMLXCore
import NovaMLXEngine
import Testing

// Compiled verifyStep forward at S=4 must match the general path output.
// Isolates the compiled-segment correctness from the MTP machinery.
@Suite("Compiled S=4 forward")
struct CompiledS4Tests {
    @Test func s4ForwardMatches() async throws {
        let modelId = "Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-oQ8e"
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/nova-models/\(modelId)") else {
            Issue.record("model not on disk"); return
        }
        let engine = MLXEngine()
        let config = ModelConfig(identifier: ModelIdentifier(id: modelId, family: .qwen))
        _ = try await engine.loadModel(
            from: URL(fileURLWithPath: "/Volumes/WD/nova-models/\(modelId)"),
            config: config, progress: nil)
        guard let mlxContainer = engine.getContainer(for: modelId)?.mlxContainer else {
            Issue.record("no container"); return
        }

        try await mlxContainer.perform { context in
            guard let model = context.model as? Qwen35Model else {
                Issue.record("not Qwen35Model"); return
            }

            // Two identical cache sets
            let cache1 = try model.newCache(parameters: nil)
            let cache2 = try model.newCache(parameters: nil)

            // Warm with a prompt so mamba/attention states exist
            let prompt = MLXArray(Array(0..<63).map { Int32($0) })[.newAxis, 0...]
            _ = model(prompt, cache: cache1)
            _ = model(prompt, cache: cache2)
            eval(cache1.flatMap { $0.state.compactMap { $0 } })
            eval(cache2.flatMap { $0.state.compactMap { $0 } })

            // S=4: general path — emitDrafterState routes to forward with
            // recordSpec=true (forces non-compiled), applyFinalNorm=false
            let s4 = MLXArray([100, 101, 102, 103].map { Int32($0) })[.newAxis, 0...]
            var generalState = LMOutput.State()
            generalState[mtpEmitFlagKey] = true
            generalState[mtpRecordSpecCaptureKey] = true
            let generalOut = model(
                LMInput.Text(tokens: s4), cache: cache1, state: generalState).logits
            eval(generalOut)

            // S=4: compiled path — recordSpec absent → verifyStep triggers
            var compiledState = LMOutput.State()
            compiledState[mtpEmitFlagKey] = true
            compiledState[mtpRecordSpecCaptureKey] = false
            let compiledOut = model(
                LMInput.Text(tokens: s4), cache: cache2, state: compiledState).logits
            eval(compiledOut)

            // Compare
            let diff = MLX.abs(generalOut - compiledOut).max().item(Float.self)
            print("S4-FORWARD-DIFF: \(diff)")
            #expect(diff < 0.01, "compiled S=4 output differs: \(diff)")
        }
    }
}
