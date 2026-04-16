import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import Hub

public struct BenchmarkMetric: Codable, Sendable {
    public let modelId: String
    public let family: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let ttftMs: Double
    public let decodeTps: Double
    public let prefillTps: Double
    public let e2eLatencyMs: Double
    public let peakGPUMemoryGB: Double
    public let memoryPerTokenKB: Double
    public let streamingIntervalStdDevMs: Double
    public let generationIndex: Int
    public let testName: String

    public init(
        modelId: String,
        family: String,
        promptTokens: Int,
        completionTokens: Int,
        ttftMs: Double,
        decodeTps: Double,
        prefillTps: Double,
        e2eLatencyMs: Double,
        peakGPUMemoryGB: Double,
        memoryPerTokenKB: Double,
        streamingIntervalStdDevMs: Double,
        generationIndex: Int,
        testName: String
    ) {
        self.modelId = modelId
        self.family = family
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.ttftMs = ttftMs
        self.decodeTps = decodeTps
        self.prefillTps = prefillTps
        self.e2eLatencyMs = e2eLatencyMs
        self.peakGPUMemoryGB = peakGPUMemoryGB
        self.memoryPerTokenKB = memoryPerTokenKB
        self.streamingIntervalStdDevMs = streamingIntervalStdDevMs
        self.generationIndex = generationIndex
        self.testName = testName
    }
}

public struct LoadTimeMetric: Codable, Sendable {
    public let modelId: String
    public let family: String
    public let loadTimeMs: Double
    public let weightsMemoryGB: Double
    public let firstInferenceMs: Double

    public init(modelId: String, family: String, loadTimeMs: Double, weightsMemoryGB: Double, firstInferenceMs: Double) {
        self.modelId = modelId
        self.family = family
        self.loadTimeMs = loadTimeMs
        self.weightsMemoryGB = weightsMemoryGB
        self.firstInferenceMs = firstInferenceMs
    }
}

public struct ConcurrentMetric: Codable, Sendable {
    public let modelId: String
    public let concurrency: Int
    public let totalTokens: Int
    public let totalWallMs: Double
    public let aggregateTps: Double
    public let perRequestTps: [Double]
    public let testName: String

    public init(modelId: String, concurrency: Int, totalTokens: Int, totalWallMs: Double, aggregateTps: Double, perRequestTps: [Double], testName: String) {
        self.modelId = modelId
        self.concurrency = concurrency
        self.totalTokens = totalTokens
        self.totalWallMs = totalWallMs
        self.aggregateTps = aggregateTps
        self.perRequestTps = perRequestTps
        self.testName = testName
    }
}

public struct KVQuantComparison: Codable, Sendable {
    public let modelId: String
    public let kvBits: Int?
    public let ttftMs: Double
    public let decodeTps: Double
    public let peakGPUMemoryGB: Double

    public init(modelId: String, kvBits: Int?, ttftMs: Double, decodeTps: Double, peakGPUMemoryGB: Double) {
        self.modelId = modelId
        self.kvBits = kvBits
        self.ttftMs = ttftMs
        self.decodeTps = decodeTps
        self.peakGPUMemoryGB = peakGPUMemoryGB
    }
}

public final class BenchmarkReport: @unchecked Sendable {
    public private(set) var metrics: [BenchmarkMetric] = []
    public private(set) var loadTimes: [LoadTimeMetric] = []
    public private(set) var concurrentMetrics: [ConcurrentMetric] = []
    public private(set) var kvComparisons: [KVQuantComparison] = []
    private let lock = NSLock()

    public func addMetric(_ m: BenchmarkMetric) {
        lock.withLock { metrics.append(m) }
    }

    public func addLoadTime(_ m: LoadTimeMetric) {
        lock.withLock { loadTimes.append(m) }
    }

    public func addConcurrent(_ m: ConcurrentMetric) {
        lock.withLock { concurrentMetrics.append(m) }
    }

    public func addKVComparison(_ m: KVQuantComparison) {
        lock.withLock { kvComparisons.append(m) }
    }

    public func summary() -> String {
        lock.withLock {
            var lines: [String] = []
            lines.append("═══════════════════════════════════════════════════════════════════")
            lines.append("                    NovaMLX Performance Report")
            lines.append("═══════════════════════════════════════════════════════════════════")
            lines.append("")

            if !metrics.isEmpty {
                lines.append("── Inference Metrics ──────────────────────────────────────────────")
                lines.append(String(format: "%-35s %6s %8s %8s %8s %8s %8s %8s",
                    "Model", "Prompt", "TTFT", "Prefill", "Decode", "E2E(ms)", "GPU GB", "StdDev"))
                lines.append(String(repeating: "─", count: 105))
                for m in metrics {
                    lines.append(String(format: "%-35s %6d %7.1fms %6.0ft/s %6.1ft/s %7.0f %6.2f %6.1fms",
                        String(m.modelId.prefix(35)), m.promptTokens, m.ttftMs, m.prefillTps, m.decodeTps, m.e2eLatencyMs, m.peakGPUMemoryGB, m.streamingIntervalStdDevMs))
                }
                lines.append("")
            }

            if !loadTimes.isEmpty {
                lines.append("── Model Load Times ───────────────────────────────────────────────")
                lines.append(String(format: "%-35s %10s %10s %10s",
                    "Model", "Load(ms)", "WeightsGB", "1stInf(ms)"))
                lines.append(String(repeating: "─", count: 75))
                for m in loadTimes {
                    lines.append(String(format: "%-35s %9.0f %9.2f %9.0f",
                        String(m.modelId.prefix(35)), m.loadTimeMs, m.weightsMemoryGB, m.firstInferenceMs))
                }
                lines.append("")
            }

            if !concurrentMetrics.isEmpty {
                lines.append("── Concurrent Throughput ──────────────────────────────────────────")
                lines.append(String(format: "%-35s %5s %7s %10s %10s",
                    "Model", "Conc", "Tokens", "Wall(ms)", "Agg TPS"))
                lines.append(String(repeating: "─", count: 75))
                for m in concurrentMetrics {
                    lines.append(String(format: "%-35s %5d %7d %9.0f %9.1f",
                        String(m.modelId.prefix(35)), m.concurrency, m.totalTokens, m.totalWallMs, m.aggregateTps))
                }
                lines.append("")
            }

            if !kvComparisons.isEmpty {
                lines.append("── KV Quantization Impact ─────────────────────────────────────────")
                lines.append(String(format: "%-35s %7s %8s %8s %8s",
                    "Model", "kvBits", "TTFT", "Decode", "GPU GB"))
                lines.append(String(repeating: "─", count: 75))
                for m in kvComparisons {
                    let bitsStr = m.kvBits.map { "\($0)" } ?? "none"
                    lines.append(String(format: "%-35s %7s %7.1fms %6.1ft/s %6.2f",
                        String(m.modelId.prefix(35)), bitsStr, m.ttftMs, m.decodeTps, m.peakGPUMemoryGB))
                }
                lines.append("")
            }

            lines.append("═══════════════════════════════════════════════════════════════════")
            return lines.joined(separator: "\n")
        }
    }
}

public struct BenchModelConfig: Sendable {
    public let hubId: String
    public let family: ModelFamily
    public let modelType: ModelType
    public let maxTokens: Int

    public init(hubId: String, family: ModelFamily, modelType: ModelType = .llm, maxTokens: Int = 128) {
        self.hubId = hubId
        self.family = family
        self.modelType = modelType
        self.maxTokens = maxTokens
    }
}

public final class BenchModelRegistry: @unchecked Sendable {
    public static let shared = BenchModelRegistry()

    public let llmModels: [BenchModelConfig] = [
        BenchModelConfig(hubId: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", family: .llama),
        BenchModelConfig(hubId: "mlx-community/Mistral-7B-Instruct-v0.3-4bit", family: .mistral),
        BenchModelConfig(hubId: "mlx-community/Phi-3.5-mini-instruct-4bit", family: .phi),
        BenchModelConfig(hubId: "mlx-community/Qwen2.5-7B-Instruct-4bit", family: .qwen),
        BenchModelConfig(hubId: "mlx-community/gemma-2-9b-it-4bit", family: .gemma),
    ]

    public let vlmModels: [BenchModelConfig] = [
        BenchModelConfig(hubId: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", family: .qwen, modelType: .vlm),
    ]

    public func model(for family: ModelFamily) -> BenchModelConfig? {
        llmModels.first { $0.family == family }
    }
}

public final class StreamTimer: @unchecked Sendable {
    private var tokenTimestamps: [ContinuousClock.Instant] = []
    private let lock = NSLock()

    public func recordToken() {
        lock.withLock {
            tokenTimestamps.append(.now)
        }
    }

    public func reset() {
        lock.withLock { tokenTimestamps.removeAll() }
    }

    public struct TimingResult: Sendable {
        public let ttftMs: Double
        public let decodeTps: Double
        public let prefillTps: Double
        public let streamingIntervalStdDevMs: Double
        public let totalTokens: Int
        public let e2eMs: Double
    }

    public func compute(promptTokens: Int, requestStart: ContinuousClock.Instant? = nil) -> TimingResult? {
        lock.withLock {
            guard tokenTimestamps.count >= 2 else { return nil }
            let firstToken = tokenTimestamps[0]
            let lastToken = tokenTimestamps.last!
            let ttft: Double
            if let requestStart {
                ttft = Double(requestStart.duration(to: firstToken).components.attoseconds) / 1e18 * 1000
            } else {
                ttft = Double(firstToken.duration(to: tokenTimestamps[1]).components.attoseconds) / 1e18 * 1000
            }
            let totalDuration = Double(firstToken.duration(to: lastToken).components.attoseconds) / 1e18 * 1000
            let completionTokens = tokenTimestamps.count - 1
            guard completionTokens > 0, totalDuration > 0 else { return nil }

            let decodeTps = Double(completionTokens) / (totalDuration / 1000.0)
            let prefillTps = promptTokens > 0 ? Double(promptTokens) / (ttft / 1000.0) : 0

            var intervals: [Double] = []
            for i in 2..<tokenTimestamps.count {
                let dur = Double(tokenTimestamps[i - 1].duration(to: tokenTimestamps[i]).components.attoseconds) / 1e18 * 1000
                intervals.append(dur)
            }
            let stdDev: Double
            if intervals.count >= 2 {
                let mean = intervals.reduce(0, +) / Double(intervals.count)
                let variance = intervals.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(intervals.count)
                stdDev = sqrt(variance)
            } else {
                stdDev = 0
            }

            return TimingResult(
                ttftMs: ttft,
                decodeTps: decodeTps,
                prefillTps: prefillTps,
                streamingIntervalStdDevMs: stdDev,
                totalTokens: completionTokens,
                e2eMs: totalDuration
            )
        }
    }
}
