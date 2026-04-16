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
final class BenchmarkHarness: @unchecked Sendable {
    let engine: MLXEngine
    let settingsManager: ModelSettingsManager
    let inferenceService: InferenceService
    let report: BenchmarkReport
    private let hubApi: HubApi

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDir = appSupport.appendingPathComponent("NovaMLX/models", isDirectory: true)
        let hubDownloadBase = modelsDir.appendingPathComponent("hub")

        self.engine = MLXEngine()
        self.settingsManager = ModelSettingsManager(baseDirectory: appSupport.appendingPathComponent("NovaMLX"))
        self.inferenceService = InferenceService(engine: engine, settingsManager: settingsManager)
        self.report = BenchmarkReport()
        self.hubApi = HubApi(downloadBase: hubDownloadBase)
    }

    func ensureModelDownloaded(_ config: BenchModelConfig) async throws -> URL {
        let repo = Hub.Repo(id: config.hubId)
        let localDir = hubApi.localRepoLocation(repo)

        let safetensorsFiles = try FileManager.default.contentsOfDirectory(at: localDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        if !safetensorsFiles.isEmpty {
            return localDir
        }

        print("  ⬇️ Downloading \(config.hubId)...")
        let allFilenames = try await hubApi.getFilenames(from: repo)

        for filename in allFilenames {
            let fileDest = localDir.appendingPathComponent(filename)
            let fileDir = fileDest.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: fileDir, withIntermediateDirectories: true)

            if FileManager.default.fileExists(atPath: fileDest.path) { continue }

            let sourceURL = hubFileURL(repoId: config.hubId, filename: filename)
            print("    ⬇️ \(filename)")
            try await downloadFile(url: sourceURL, to: fileDest)
        }

        print("  ✅ Downloaded \(config.hubId)")
        return localDir
    }

    func loadModel(_ config: BenchModelConfig) async throws -> LoadTimeMetric {
        let localDir = try await ensureModelDownloaded(config)
        let modelId = config.hubId

        let modelConfig = ModelConfig(
            identifier: ModelIdentifier(id: modelId, family: config.family),
            modelType: config.modelType,
            maxTokens: config.maxTokens
        )

        let loadStart = ContinuousClock.now
        _ = try await engine.loadModel(from: localDir, config: modelConfig)
        let loadEnd = ContinuousClock.now
        let loadMs = Double(loadStart.duration(to: loadEnd).components.attoseconds) / 1e18 * 1000

        let weightsBytes = MLX.Memory.activeMemory
        let weightsGB = Double(weightsBytes) / (1024 * 1024 * 1024)

        let firstInfStart = ContinuousClock.now
        let warmupRequest = InferenceRequest(
            model: modelId,
            messages: [ChatMessage(role: .user, content: "Hi")],
            temperature: 0.0,
            maxTokens: 5,
            stream: false
        )
        _ = try await inferenceService.generate(warmupRequest)
        let firstInfEnd = ContinuousClock.now
        let firstInfMs = Double(firstInfStart.duration(to: firstInfEnd).components.attoseconds) / 1e18 * 1000

        let metric = LoadTimeMetric(
            modelId: modelId,
            family: config.family.rawValue,
            loadTimeMs: loadMs,
            weightsMemoryGB: weightsGB,
            firstInferenceMs: firstInfMs
        )
        report.addLoadTime(metric)
        return metric
    }

    func runStreamingBenchmark(
        config: BenchModelConfig,
        prompt: String,
        maxTokens: Int = 128,
        warmupRuns: Int = 1,
        measuredRuns: Int = 3,
        testName: String = "streaming"
    ) async throws -> [BenchmarkMetric] {
        let modelId = config.hubId
        guard engine.getContainer(for: modelId) != nil else {
            fatalError("Model \(modelId) not loaded")
        }

        for _ in 0..<warmupRuns {
            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: prompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )
            let stream = inferenceService.stream(req)
            for try await _ in stream {}
        }

        var results: [BenchmarkMetric] = []
        let estimatedPromptTokens = estimateTokenCount(prompt: prompt)

        for runIndex in 0..<measuredRuns {
            let timer = StreamTimer()
            let memoryBefore = MLX.Memory.activeMemory

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

            let memoryAfter = MLX.Memory.activeMemory
            let peakMemory = max(memoryBefore, memoryAfter)

            guard let timing = timer.compute(promptTokens: estimatedPromptTokens, requestStart: requestStart) else { continue }

            let metric = BenchmarkMetric(
                modelId: modelId,
                family: config.family.rawValue,
                promptTokens: estimatedPromptTokens,
                completionTokens: timing.totalTokens,
                ttftMs: timing.ttftMs,
                decodeTps: timing.decodeTps,
                prefillTps: timing.prefillTps,
                e2eLatencyMs: timing.e2eMs,
                peakGPUMemoryGB: Double(peakMemory) / (1024 * 1024 * 1024),
                memoryPerTokenKB: timing.totalTokens > 0 ? Double(peakMemory - memoryBefore) / Double(timing.totalTokens) / 1024 : 0,
                streamingIntervalStdDevMs: timing.streamingIntervalStdDevMs,
                generationIndex: runIndex,
                testName: testName
            )
            report.addMetric(metric)
            results.append(metric)

            print("    Run \(runIndex + 1)/\(measuredRuns): TTFT=\(String(format: "%.1f", timing.ttftMs))ms Decode=\(String(format: "%.1f", timing.decodeTps))t/s Tokens=\(timing.totalTokens)")
        }

        return results
    }

    func runConcurrentBenchmark(
        config: BenchModelConfig,
        concurrency: Int,
        maxTokens: Int = 64,
        testName: String = "concurrent"
    ) async throws -> ConcurrentMetric {
        let modelId = config.hubId
        guard engine.getContainer(for: modelId) != nil else {
            fatalError("Model \(modelId) not loaded")
        }

        let wallStart = ContinuousClock.now

        let allResults: [(Int, Int, Double)] = await withTaskGroup(of: (Int, Int, Double).self) { group in
            for i in 0..<concurrency {
                group.addTask {
                    let req = InferenceRequest(
                        model: modelId,
                        messages: [ChatMessage(role: .user, content: "Say hello in one sentence.")],
                        temperature: 0.0,
                        maxTokens: maxTokens,
                        stream: false
                    )
                    let start = ContinuousClock.now
                    do {
                        let result = try await self.inferenceService.generate(req)
                        let elapsed = Double(start.duration(to: .now).components.attoseconds) / 1e18 * 1000
                        return (i, result.completionTokens, elapsed)
                    } catch {
                        return (i, 0, 0)
                    }
                }
            }

            var collected: [(Int, Int, Double)] = []
            for await result in group {
                collected.append(result)
            }
            return collected
        }

        let wallEnd = ContinuousClock.now
        let wallMs = Double(wallStart.duration(to: wallEnd).components.attoseconds) / 1e18 * 1000

        let totalTokens = allResults.map(\.1).reduce(0, +)
        let perRequestTps = allResults.map { _, tokens, elapsedMs in
            elapsedMs > 0 ? Double(tokens) / (elapsedMs / 1000.0) : 0
        }
        let aggregateTps = wallMs > 0 ? Double(totalTokens) / (wallMs / 1000.0) : 0

        let metric = ConcurrentMetric(
            modelId: modelId,
            concurrency: concurrency,
            totalTokens: totalTokens,
            totalWallMs: wallMs,
            aggregateTps: aggregateTps,
            perRequestTps: perRequestTps,
            testName: testName
        )
        report.addConcurrent(metric)
        return metric
    }

    func runKVQuantBenchmark(
        config: BenchModelConfig,
        kvBitsValues: [Int?] = [nil, 8, 4, 2],
        prompt: String = "Explain the theory of relativity in simple terms.",
        maxTokens: Int = 64
    ) async throws -> [KVQuantComparison] {
        let modelId = config.hubId
        guard engine.getContainer(for: modelId) != nil else {
            fatalError("Model \(modelId) not loaded")
        }

        var comparisons: [KVQuantComparison] = []

        for kvBitsValue in kvBitsValues {
            settingsManager.updateSettings(modelId) { settings in
                settings.kvBits = kvBitsValue
                settings.kvGroupSize = kvBitsValue != nil ? 64 : nil
            }

            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: prompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )

            let timer = StreamTimer()
            let memoryBefore = MLX.Memory.activeMemory

            let requestStart = ContinuousClock.now
            let stream = inferenceService.stream(req)
            for try await token in stream {
                if token.finishReason != nil { break }
                timer.recordToken()
            }

            let memoryAfter = MLX.Memory.activeMemory
            let peakMemory = max(memoryBefore, memoryAfter)
            let estimatedPromptTokens = estimateTokenCount(prompt: prompt)

            guard let timing = timer.compute(promptTokens: estimatedPromptTokens, requestStart: requestStart) else { continue }

            let comparison = KVQuantComparison(
                modelId: modelId,
                kvBits: kvBitsValue,
                ttftMs: timing.ttftMs,
                decodeTps: timing.decodeTps,
                peakGPUMemoryGB: Double(peakMemory) / (1024 * 1024 * 1024)
            )
            report.addKVComparison(comparison)
            comparisons.append(comparison)

            let bitsStr = kvBitsValue.map { "\($0)" } ?? "none"
            print("    kvBits=\(bitsStr): TTFT=\(String(format: "%.1f", timing.ttftMs))ms Decode=\(String(format: "%.1f", timing.decodeTps))t/s GPU=\(String(format: "%.2f", Double(peakMemory) / (1024 * 1024 * 1024)))GB")
        }

        return comparisons
    }

    func runWarmupBenchmark(
        config: BenchModelConfig,
        prompt: String = "What is 2+2?",
        maxTokens: Int = 32,
        runs: Int = 5
    ) async throws -> [BenchmarkMetric] {
        let modelId = config.hubId
        guard engine.getContainer(for: modelId) != nil else {
            fatalError("Model \(modelId) not loaded")
        }

        var results: [BenchmarkMetric] = []
        let estimatedPromptTokens = estimateTokenCount(prompt: prompt)

        for runIndex in 0..<runs {
            let timer = StreamTimer()
            let memoryBefore = MLX.Memory.activeMemory

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

            let memoryAfter = MLX.Memory.activeMemory
            let peakMemory = max(memoryBefore, memoryAfter)

            guard let timing = timer.compute(promptTokens: estimatedPromptTokens, requestStart: requestStart) else { continue }

            let metric = BenchmarkMetric(
                modelId: modelId,
                family: config.family.rawValue,
                promptTokens: estimatedPromptTokens,
                completionTokens: timing.totalTokens,
                ttftMs: timing.ttftMs,
                decodeTps: timing.decodeTps,
                prefillTps: timing.prefillTps,
                e2eLatencyMs: timing.e2eMs,
                peakGPUMemoryGB: Double(peakMemory) / (1024 * 1024 * 1024),
                memoryPerTokenKB: timing.totalTokens > 0 ? Double(peakMemory - memoryBefore) / Double(timing.totalTokens) / 1024 : 0,
                streamingIntervalStdDevMs: timing.streamingIntervalStdDevMs,
                generationIndex: runIndex,
                testName: "warmup"
            )
            report.addMetric(metric)
            results.append(metric)

            print("    Warmup run \(runIndex + 1)/\(runs): TTFT=\(String(format: "%.1f", timing.ttftMs))ms Decode=\(String(format: "%.1f", timing.decodeTps))t/s")
        }

        return results
    }

    func unloadCurrentModel() {
        for modelId in engine.listLoadedModels() {
            engine.unloadModel(ModelIdentifier(id: modelId, family: .other))
        }
    }

    private func hubFileURL(repoId: String, filename: String) -> URL {
        var url = URL(string: "https://huggingface.co")!
        url = url.appendingPathComponent(repoId)
        url = url.appendingPathComponent("resolve")
        url = url.appendingPathComponent("main")
        url = url.appendingPathComponent(filename)
        return url
    }

    private func downloadFile(url: URL, to destination: URL) async throws {
        let request = URLRequest(url: url)
        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NovaMLXError.downloadFailed("HTTP error for \(url.lastPathComponent)", underlying: NSError(domain: "NovaMLX", code: -1))
        }

        let tmpDest = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).tmp")
        try? FileManager.default.removeItem(at: tmpDest)
        FileManager.default.createFile(atPath: tmpDest.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tmpDest)
        defer { try? handle.close() }

        var buffer = Data()
        let flushSize = 1024 * 1024

        for try await byte in asyncBytes {
            buffer.append(byte)
            if buffer.count >= flushSize {
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
        }
        try? handle.close()
        try FileManager.default.moveItem(at: tmpDest, to: destination)
    }

    nonisolated func estimateTokenCount(prompt: String) -> Int {
        return prompt.count / 4
    }
}
