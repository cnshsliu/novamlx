import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import NovaMLXCore
@testable import NovaMLXEngine
import Testing

// Full compiled-verify round INCLUDING a simulated rejection:
// 1. Process prompt (emitDrafterState, recordSpec=true → general path)
// 2. Draft chain
// 3. Compiled verify (recordSpec=false → verifyStep)
// 4. Simulate rejection: restore pre-verify GDN states, trim KV, re-run accepted prefix
// Steps 1-3 pass (proven). Step 4 is the suspect.
@Suite("Compiled S=4 + rejection")
struct CompiledS4RejectionTests {
    @Test func compiledVerifyWithRejection() async throws {
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
        let draftDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/mlx-community/Qwen3.8-27B-MTP-4bit")

        try await mlxContainer.perform { context in
            guard let model = context.model as? Qwen35Model else {
                Issue.record("not Qwen35Model"); return
            }
            let cache = try model.newCache(parameters: nil)

            // Step 1: process prompt
            let prompt = MLXArray(Array(0..<63).map { Int32($0) })[.newAxis, 0...]
            var mtpState = LMOutput.State()
            mtpState[mtpEmitFlagKey] = true
            let mtpResult = model(
                LMInput.Text(tokens: prompt), cache: cache, state: mtpState)
            eval(mtpResult.logits)

            // Step 2: draft chain
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
            print("STEPS-12-OK")

            // Save pre-verify GDN states (COW refs)
            var preVerifyGDN: [(conv: MLXArray, rec: MLXArray)] = []
            var preVerifyOffset = 0
            for c in cache {
                if let mamba = c as? MambaCache {
                    preVerifyGDN.append((conv: mamba[0]!, rec: mamba[1]!))
                    preVerifyOffset = mamba.offset
                } else if let kv = c as? KVCacheSimple {
                    preVerifyOffset = kv.offset
                }
            }
            let preVerifyKVOffset = { () -> Int in
                for c in cache { if let kv = c as? KVCacheSimple { return kv.offset } }
                return -1
            }()

            // Step 3: compiled verify at S=4
            let s4 = concatenated([lastToken.reshaped(-1), draftTokens.flattened()])[.newAxis, 0...]
            var verifyState = LMOutput.State()
            verifyState[mtpEmitFlagKey] = true
            verifyState[mtpRecordSpecCaptureKey] = false
            let verifyResult = model(
                LMInput.Text(tokens: s4), cache: cache, state: verifyState)
            eval(verifyResult.logits)
            print("STEP3-OK: compiled verify done")

            // Check post-verify state
            let postVerifyKVOffset = { () -> Int in
                for c in cache { if let kv = c as? KVCacheSimple { return kv.offset } }
                return -1
            }()
            print("OFFSETS: preKV=\(preVerifyKVOffset) postKV=\(postVerifyKVOffset) preGDN=\(preVerifyOffset)")

            // Step 4: simulate rejection of 1 token (accepted=2, keep=3)
            // Restore pre-verify GDN states
            var gi = 0
            for c in cache {
                if let mamba = c as? MambaCache {
                    mamba[0] = preVerifyGDN[gi].conv
                    mamba[1] = preVerifyGDN[gi].rec
                    mamba.offset = preVerifyOffset
                    gi += 1
                }
            }
            // Trim KV by 4 (undo verify's writes)
            let trimmed = trimPromptCache(cache, numTokens: 4)
            let postTrimKVOffset = { () -> Int in
                for c in cache { if let kv = c as? KVCacheSimple { return kv.offset } }
                return -1
            }()
            print("STEP4a: GDN restored, KV trimmed=\(trimmed), KV offset=\(postTrimKVOffset)")

            // Step 4b: re-run accepted prefix (3 tokens = bonus + d1 + d2)
            let rerunTokens = concatenated([
                lastToken.reshaped(-1),
                draftTokens.flattened()[0..<2],
            ])[.newAxis, 0...]
            print("STEP4b: rerun tokens shape=\(rerunTokens.shape)")
            var rerunState = LMOutput.State()
            rerunState[mtpEmitFlagKey] = true
            rerunState[mtpRecordSpecCaptureKey] = true  // non-compiled path
            let rerunResult = model(
                LMInput.Text(tokens: rerunTokens), cache: cache, state: rerunState)
            eval(rerunResult.logits)
            print("STEP4-OK: rejection rerun passed, logits shape=\(rerunResult.logits.shape)")
        }
    }
}
