import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import NovaMLXCore
@testable import NovaMLXEngine
import Testing

// Stepwise repro: find the minimum interaction that breaks compiled verify.
// The MTP context: (1) prompt via emitDrafterState, (2) draft chain, (3) compiled S=4.
// This test runs step 1 then step 3 — if it crashes, the issue is the
// emitDrafterState path's cache state, not the draft chain.
@Suite("Compiled S=4 MTP-context repro")
struct CompiledS4ReproTests {
    @Test func emitStateThenCompiledVerify() async throws {
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
            let cache = try model.newCache(parameters: nil)

            // Step 1: process prompt via emitDrafterState path (like MTP prepare)
            let prompt = MLXArray(Array(0..<63).map { Int32($0) })[.newAxis, 0...]
            var mtpState = LMOutput.State()
            mtpState[mtpEmitFlagKey] = true
            mtpState[mtpRecordSpecCaptureKey] = true  // MTP prepare uses recordSpec=true
            let mtpResult = model(
                LMInput.Text(tokens: prompt), cache: cache, state: mtpState)
            eval(mtpResult.logits)
            eval(cache.flatMap { $0.state.compactMap { $0 } })
            print("STEP1-OK: prompt processed, cache offset after: \(cache.first?.offset ?? -1)")

            // Step 2: draft chain (3 head forwards, like MTP speculateRound)
            let draftDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/mlx-community/Qwen3.8-27B-MTP-4bit")
            let drafter = try await QwenMtpDrafterStore.shared.drafter(
                directory: draftDir, vision: false)
            guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel else {
                Issue.record("not stateful"); return
            }
            var draftState = statefulDrafter.makeState(parameters: nil)
            let lastHidden = mtpResult.state?[mtpLastHiddenStatesKey]!
            let lastToken = MLXArray([Int32(50)]).reshaped([1, 1])
            let lastHiddenCol = lastHidden![0..., (-1)..., 0...]
            let draftTokens = statefulDrafter.draftBlock(
                target: model, lastToken: lastToken, lastHidden: lastHiddenCol,
                sharedKV: [:], positionDeltas: nil, queryOffset: 63,
                blockSize: 4, state: &draftState,
                sampler: GenerateParameters(temperature: 0).sampler())
            eval(draftTokens)
            print("STEP2-OK: draft tokens shape=\(draftTokens.shape)")

            // Step 3: compiled verify at S=4 (recordSpec=false → verifyStep)
            let s4 = concatenated([lastToken.reshaped(-1), draftTokens.flattened()])[.newAxis, 0...]
            var verifyState = LMOutput.State()
            verifyState[mtpEmitFlagKey] = true
            verifyState[mtpRecordSpecCaptureKey] = false
            let compiledOut = model(
                LMInput.Text(tokens: s4), cache: cache, state: verifyState).logits
            eval(compiledOut)
            print("STEP3-OK: compiled verify passed, shape=\(compiledOut.shape)")
        }
    }
}
