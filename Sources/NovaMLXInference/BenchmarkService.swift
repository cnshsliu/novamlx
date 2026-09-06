import Foundation
import MLX
import NovaMLXCore
import NovaMLXEngine
import NovaMLXUtils

public struct BenchmarkRequest: Codable, Sendable {
    public let modelId: String
    public let promptLengths: [Int]
    public let generationLength: Int

    private enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case promptLengths = "prompt_lengths"
        case generationLength = "generation_length"
    }

    public init(modelId: String, promptLengths: [Int] = [1024, 4096], generationLength: Int = 128) {
        self.modelId = modelId
        self.promptLengths = promptLengths
        self.generationLength = generationLength
    }
}

public struct BenchmarkResult: Codable, Sendable {
    public let promptLength: Int
    public let ttftMs: Double
    public let generationTps: Double
    public let processingTps: Double
    public let totalTokens: Int
    public let peakMemoryGB: Double
    public let e2eLatencyS: Double

    private enum CodingKeys: String, CodingKey {
        case promptLength = "prompt_length"
        case ttftMs = "ttft_ms"
        case generationTps = "generation_tps"
        case processingTps = "processing_tps"
        case totalTokens = "total_tokens"
        case peakMemoryGB = "peak_memory_gb"
        case e2eLatencyS = "e2e_latency_s"
    }

    public init(
        promptLength: Int,
        ttftMs: Double,
        generationTps: Double,
        processingTps: Double,
        totalTokens: Int,
        peakMemoryGB: Double,
        e2eLatencyS: Double
    ) {
        self.promptLength = promptLength
        self.ttftMs = ttftMs
        self.generationTps = generationTps
        self.processingTps = processingTps
        self.totalTokens = totalTokens
        self.peakMemoryGB = peakMemoryGB
        self.e2eLatencyS = e2eLatencyS
    }
}

public struct BenchmarkRun: Codable, Sendable {
    public let id: String
    public let modelId: String
    public var results: [BenchmarkResult]
    public var status: String
    public var progress: Double
    public var error: String?

    public init(id: String, modelId: String) {
        self.id = id
        self.modelId = modelId
        self.results = []
        self.status = "running"
        self.progress = 0.0
        self.error = nil
    }
}

public final class BenchmarkService: @unchecked Sendable {
    private let inferenceService: InferenceService
    private var activeRun: BenchmarkRun?
    private let lock = NovaMLXLock()

    public init(inferenceService: InferenceService) {
        self.inferenceService = inferenceService
    }

    public func startBenchmark(_ request: BenchmarkRequest) throws -> BenchmarkRun {
        let currentRun = lock.withLock { activeRun }
        if let active = currentRun, active.status == "running" {
            throw NovaMLXError.apiError("Benchmark already running: \(active.id)")
        }

        guard inferenceService.isModelLoaded(request.modelId) else {
            throw NovaMLXError.modelNotFound(request.modelId)
        }

        let run = BenchmarkRun(id: UUID().uuidString.prefix(8).description, modelId: request.modelId)
        lock.withLock { activeRun = run }

        Task {
            await executeBenchmark(run: run, request: request)
        }

        return run
    }

    public func getActiveRun() -> BenchmarkRun? {
        lock.withLock { activeRun }
    }

    public func cancelActiveRun() {
        lock.withLock {
            if var run = activeRun, run.status == "running" {
                run.status = "cancelled"
                activeRun = run
            }
        }
    }

    private func executeBenchmark(run: BenchmarkRun, request: BenchmarkRequest) async {
        let totalSteps = request.promptLengths.count
        var completedSteps = 0

        for promptLength in request.promptLengths {
            let isCancelled = lock.withLock { activeRun?.status == "cancelled" }
            if isCancelled { return }

            let dummyPrompt = String(repeating: "Hello world. ", count: max(1, promptLength / 12))
            let messages = [ChatMessage(role: .user, content: dummyPrompt)]

            let inferenceRequest = InferenceRequest(
                model: request.modelId,
                messages: messages,
                temperature: 0.0,
                maxTokens: request.generationLength,
                stream: true
            )

            let startTime = Date()
            let memoryBefore = UInt64(MLX.Memory.activeMemory)
            let compiled = inferenceService.engine.getContainer(for: request.modelId)?.compiledDecodeSegmentCount ?? 0
            let nativeMtp = inferenceService.hasNativeMtp(request.modelId)

            do {
                var firstTokenAt: Date?
                var completionTokens = 0
                for try await token in inferenceService.stream(inferenceRequest) {
                    if firstTokenAt == nil { firstTokenAt = Date() }
                    if !token.text.isEmpty { completionTokens += 1 }
                }

                let endTime = Date()
                let e2eLatency = endTime.timeIntervalSince(startTime)
                let ttft = firstTokenAt.map { $0.timeIntervalSince(startTime) } ?? e2eLatency
                let decodeElapsed = firstTokenAt.map { endTime.timeIntervalSince($0) } ?? e2eLatency
                let decodeTokens = max(completionTokens - 1, 1)
                let decodeTps = Double(decodeTokens) / max(decodeElapsed, 0.001)
                let promptTokens = max(promptLength, 1)
                let processingTps = Double(promptTokens) / max(ttft, 0.001)
                let peakMemory = Double(memoryBefore) / (1024 * 1024 * 1024)

                NovaMLXLog.info(
                    "[Bench] model=\(request.modelId) prompt~\(promptLength) TTFT=\(String(format: "%.0f", ttft * 1000))ms decode=\(String(format: "%.1f", decodeTps)) tok/s compiledSegments=\(compiled) nativeMtp=\(nativeMtp) genTokens=\(completionTokens)"
                )

                let benchResult = BenchmarkResult(
                    promptLength: promptLength,
                    ttftMs: ttft * 1000,
                    generationTps: decodeTps,
                    processingTps: processingTps,
                    totalTokens: promptTokens + completionTokens,
                    peakMemoryGB: peakMemory,
                    e2eLatencyS: e2eLatency
                )

                completedSteps += 1
                lock.withLock {
                    activeRun?.results.append(benchResult)
                    activeRun?.progress = Double(completedSteps) / Double(totalSteps)
                    if completedSteps == totalSteps {
                        activeRun?.status = "completed"
                        activeRun?.progress = 1.0
                    }
                }
            } catch {
                lock.withLock {
                    activeRun?.status = "error"
                    activeRun?.error = error.localizedDescription
                }
                return
            }
        }
    }
}
