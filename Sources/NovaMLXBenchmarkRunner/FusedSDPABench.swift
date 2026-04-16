import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import Hub

@MainActor
func runFusedSDPABenchmark() async {
    print("═══════════════════════════════════════════════════════════════════")
    print("        Fused Quantized SDPA Kernel Benchmark")
    print("═══════════════════════════════════════════════════════════════════")
    print("")

    let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    let modelsDir = appSupport.appendingPathComponent("NovaMLX/models", isDirectory: true)
    let hubDownloadBase = modelsDir.appendingPathComponent("hub")

    let engine = MLXEngine()
    let settingsManager = ModelSettingsManager(baseDirectory: appSupport.appendingPathComponent("NovaMLX"))
    let inferenceService = InferenceService(engine: engine, settingsManager: settingsManager)
    let hubApi = HubApi(downloadBase: hubDownloadBase)

    let modelId = "mlx-community/Phi-3.5-mini-instruct-4bit"
    let family = ModelFamily.phi

    let repo = Hub.Repo(id: modelId)
    let localDir = hubApi.localRepoLocation(repo)

    let safetensorsFiles = (try? FileManager.default.contentsOfDirectory(at: localDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }) ?? []
    if safetensorsFiles.isEmpty {
        print("ERROR: Model not found at \(localDir.path)")
        print("Please download it first.")
        return
    }

    print("Loading \(modelId)...")
    let modelConfig = ModelConfig(
        identifier: ModelIdentifier(id: modelId, family: family),
        modelType: .llm,
        maxTokens: 512
    )

    do {
        _ = try await engine.loadModel(from: localDir, config: modelConfig)
    } catch {
        print("ERROR: Failed to load model: \(error)")
        return
    }

    let weightsBytes = MLX.Memory.activeMemory
    print("Weights: \(String(format: "%.2f", Double(weightsBytes) / (1024*1024*1024)))GB")

    print("\nWarmup (no KV quant)...")
    let warmupReq = InferenceRequest(
        model: modelId,
        messages: [ChatMessage(role: .user, content: "Hello")],
        temperature: 0.0,
        maxTokens: 10,
        stream: false
    )
    _ = try? await inferenceService.generate(warmupReq)

    let prompt = "Explain the theory of relativity in simple terms. Include examples and analogies to make it easy to understand for a general audience."
    let maxTokens = 128
    let runsPerConfig = 3

    typealias RunResult = (ttftMs: Double, decodeTps: Double, totalTokens: Int, gpuGB: Double)

    func runBenchmark(label: String) async throws -> [RunResult] {
        var results: [RunResult] = []
        for runIdx in 0..<runsPerConfig {
            let timer = StreamTimer()
            let memBefore = MLX.Memory.activeMemory

            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: prompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )

            let requestStart = ContinuousClock.now
            let stream = inferenceService.stream(req)
            for try await token in stream {
                if token.finishReason != nil { break }
                timer.recordToken()
            }

            let memAfter = MLX.Memory.activeMemory
            let estimatedPromptTokens = prompt.count / 4

            guard let timing = timer.compute(promptTokens: estimatedPromptTokens, requestStart: requestStart) else {
                print("    Run \(runIdx+1)/\(runsPerConfig): FAILED (no timing)")
                continue
            }

            let peakMem = Double(max(memBefore, memAfter)) / (1024*1024*1024)
            results.append(RunResult(ttftMs: timing.ttftMs, decodeTps: timing.decodeTps, totalTokens: timing.totalTokens, gpuGB: peakMem))

            print("    \(label) run \(runIdx+1)/\(runsPerConfig): TTFT=\(String(format: "%.1f", timing.ttftMs))ms Decode=\(String(format: "%.1f", timing.decodeTps))t/s Tokens=\(timing.totalTokens)")
        }
        return results
    }

    func average(_ results: [RunResult], field: KeyPath<RunResult, Double>) -> Double {
        guard !results.isEmpty else { return 0 }
        return results.map { $0[keyPath: field] }.reduce(0, +) / Double(results.count)
    }

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Phase 1: Baseline (no KV quantization)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    settingsManager.updateSettings(modelId) { settings in
        settings.kvBits = nil
        settings.kvGroupSize = nil
    }

    let baselineResults = try! await runBenchmark(label: "baseline")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Phase 2: KV Quantization 8-bit, groupSize=32 (FUSED kernel)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    settingsManager.updateSettings(modelId) { settings in
        settings.kvBits = 8
        settings.kvGroupSize = 32
    }

    let fused8Results = try! await runBenchmark(label: "fused-8bit")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Phase 3: KV Quantization 4-bit, groupSize=32 (FUSED kernel)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    settingsManager.updateSettings(modelId) { settings in
        settings.kvBits = 4
        settings.kvGroupSize = 32
    }

    let fused4Results = try! await runBenchmark(label: "fused-4bit")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Phase 4: Unfused baseline (KV Quant 8-bit, fused kernel DISABLED)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    FusedSDPARegistration.disableFused()

    settingsManager.updateSettings(modelId) { settings in
        settings.kvBits = 8
        settings.kvGroupSize = 32
    }

    let unfused8Results = try! await runBenchmark(label: "unfused-8bit")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Phase 5: Unfused baseline (KV Quant 4-bit, fused kernel DISABLED)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    settingsManager.updateSettings(modelId) { settings in
        settings.kvBits = 4
        settings.kvGroupSize = 32
    }

    let unfused4Results = try! await runBenchmark(label: "unfused-4bit")

    FusedSDPARegistration.register()

    print("\n")
    print("═══════════════════════════════════════════════════════════════════")
    print("                    RESULTS SUMMARY")
    print("═══════════════════════════════════════════════════════════════════")
    print("")
    print(String(format: "%-30s %12s %12s %8s", "Configuration", "Decode TPS", "TTFT (ms)", "GPU GB"))
    print(String(repeating: "-", count: 65))

    func printRow(_ label: String, _ results: [RunResult]) {
        let tps = average(results, field: \.decodeTps)
        let ttft = average(results, field: \.ttftMs)
        let gpu = average(results, field: \.gpuGB)
        print(String(format: "%-30s %12.1f %12.1f %8.2f", label, tps, ttft, gpu))
    }

    printRow("No KV quant (baseline)", baselineResults)
    printRow("KV-8bit FUSED", fused8Results)
    printRow("KV-8bit UNFUSED", unfused8Results)
    printRow("KV-4bit FUSED", fused4Results)
    printRow("KV-4bit UNFUSED", unfused4Results)

    print("")
    print("Regression Analysis:")
    print("─────────────────────")

    let baselineTps = average(baselineResults, field: \.decodeTps)
    let fused8Tps = average(fused8Results, field: \.decodeTps)
    let unfused8Tps = average(unfused8Results, field: \.decodeTps)
    let fused4Tps = average(fused4Results, field: \.decodeTps)
    let unfused4Tps = average(unfused4Results, field: \.decodeTps)

    if unfused8Tps > 0 {
        let unfused8Regression = (baselineTps - unfused8Tps) / baselineTps * 100
        let fused8Regression = (baselineTps - fused8Tps) / baselineTps * 100
        print(String(format: "  KV-8bit unfused regression: %.1f%% (baseline=%.1f, unfused=%.1f)", unfused8Regression, baselineTps, unfused8Tps))
        print(String(format: "  KV-8bit fused regression:    %.1f%% (baseline=%.1f, fused=%.1f)", fused8Regression, baselineTps, fused8Tps))
        if fused8Tps > unfused8Tps {
            let improvement = (fused8Tps - unfused8Tps) / unfused8Tps * 100
            print(String(format: "  KV-8bit fused vs unfused:    +%.1f%% improvement", improvement))
        }
    }

    if unfused4Tps > 0 {
        let unfused4Regression = (baselineTps - unfused4Tps) / baselineTps * 100
        let fused4Regression = (baselineTps - fused4Tps) / baselineTps * 100
        print(String(format: "  KV-4bit unfused regression: %.1f%% (baseline=%.1f, unfused=%.1f)", unfused4Regression, baselineTps, unfused4Tps))
        print(String(format: "  KV-4bit fused regression:    %.1f%% (baseline=%.1f, fused=%.1f)", fused4Regression, baselineTps, fused4Tps))
        if fused4Tps > unfused4Tps {
            let improvement = (fused4Tps - unfused4Tps) / unfused4Tps * 100
            print(String(format: "  KV-4bit fused vs unfused:    +%.1f%% improvement", improvement))
        }
    }

    print("")
    print("Done.")
}
