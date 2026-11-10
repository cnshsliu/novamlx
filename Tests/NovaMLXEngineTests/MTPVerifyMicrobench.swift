import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import NovaMLXCore
import NovaMLXEngine
import Testing

// In-process microbench: decompose the verify-forward cost by sequence
// length. Times N back-to-back forwards of S tokens through the real 27B
// backbone (cache warm), S in {1, 4}. The S=1 number is the serial-decode
// floor; the S=4 number is the MTP verify cost; the delta over the bandwidth
// floor (~50ms at 546GB/s) quantifies kernel-launch overhead — the target of
// the compiled-verify-segments project.
@Suite("MTP verify microbench")
struct MTPVerifyMicrobenchTests {
    @Test func verifyCostBySequenceLength() async throws {
        let modelId = "Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-oQ8e"
        let dir = NovaMLXPaths.directory(forModelId: modelId)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            Issue.record("model not on disk")
            return
        }
        let engine = MLXEngine()
        let config = ModelConfig(identifier: ModelIdentifier(id: modelId, family: .qwen))
        _ = try await engine.loadModel(from: dir, config: config, progress: nil)
        guard let mlxContainer = engine.getContainer(for: modelId)?.mlxContainer else {
            Issue.record("no container")
            return
        }

        try await mlxContainer.perform { context in
            guard let model = context.model as? Qwen35Model else {
                Issue.record("not Qwen35Model")
                return
            }
            let cache = try model.newCache(parameters: nil)
            // Warm the cache with a real prefill so mamba/attention states exist.
            let prompt = MLXArray((0..<128).map { Int32($0) })[.newAxis, 0...]
            _ = model(prompt, cache: cache)
            eval(cache.flatMap { $0.state.compactMap { $0 } })

            func timeForwards(S: Int, rounds: Int) -> Double {
                let tokens = MLXArray((0..<S).map { Int32(1000 + $0) })[.newAxis, 0...]
                // one warmup (compiles/traces)
                _ = model(tokens, cache: cache)
                eval(cache.flatMap { $0.state.compactMap { $0 } })
                let t0 = DispatchTime.now()
                for _ in 0..<rounds {
                    _ = model(tokens, cache: cache)
                    // force eval of outputs + states each round
                    eval(cache.flatMap { $0.state.compactMap { $0 } })
                }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
                return ms / Double(rounds)
            }

            let s1 = timeForwards(S: 1, rounds: 20)
            let s4 = timeForwards(S: 4, rounds: 20)
            print(String(format: "MICROBENCH: S=1 %.1fms  S=4 %.1fms  (S4-S1=%.1fms)", s1, s4, s4 - s1))
        }
    }
}
