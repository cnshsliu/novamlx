import Foundation
import MLX
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import Testing

@Suite("NovaMLX Performance Benchmarks", .serialized, .enabled(if: ProcessInfo.processInfo.environment["NOVA_BENCH"] == "1"))
@MainActor
struct PerformanceBenchTests {

    private func makeHarness() -> BenchmarkHarness {
        BenchmarkHarness()
    }

    @Test("Single model streaming benchmark - LLaMA")
    func testLLaMAStreamingBenchmark() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .llama) else {
            Issue.record("No LLaMA model configured")
            return
        }

        print("\n🚀 Benchmarking LLaMA family: \(config.hubId)")
        let loadTime = try await harness.loadModel(config)
        print("  📦 Load: \(String(format: "%.0f", loadTime.loadTimeMs))ms, Weights: \(String(format: "%.2f", loadTime.weightsMemoryGB))GB, 1st inference: \(String(format: "%.0f", loadTime.firstInferenceMs))ms")

        let metrics = try await harness.runStreamingBenchmark(
            config: config,
            prompt: "Write a short poem about the ocean.",
            maxTokens: 128,
            warmupRuns: 2,
            measuredRuns: 3,
            testName: "llama_streaming"
        )

        guard !metrics.isEmpty else {
            Issue.record("No metrics collected for LLaMA")
            return
        }

        let avgTTFT = metrics.map(\.ttftMs).reduce(0, +) / Double(metrics.count)
        let avgDecodeTPS = metrics.map(\.decodeTps).reduce(0, +) / Double(metrics.count)
        print("  📊 LLaMA avg: TTFT=\(String(format: "%.1f", avgTTFT))ms Decode=\(String(format: "%.1f", avgDecodeTPS))t/s")

        #expect(avgTTFT > 0)
        #expect(avgDecodeTPS > 0)

        harness.unloadCurrentModel()
    }

    @Test("Single model streaming benchmark - Mistral")
    func testMistralStreamingBenchmark() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .mistral) else {
            Issue.record("No Mistral model configured")
            return
        }

        print("\n🚀 Benchmarking Mistral family: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let metrics = try await harness.runStreamingBenchmark(
            config: config,
            prompt: "Explain quantum computing in 3 sentences.",
            maxTokens: 128,
            warmupRuns: 2,
            measuredRuns: 3,
            testName: "mistral_streaming"
        )

        guard !metrics.isEmpty else {
            Issue.record("No metrics collected for Mistral")
            return
        }

        let avgDecodeTPS = metrics.map(\.decodeTps).reduce(0, +) / Double(metrics.count)
        print("  📊 Mistral avg: Decode=\(String(format: "%.1f", avgDecodeTPS))t/s")
        #expect(avgDecodeTPS > 0)

        harness.unloadCurrentModel()
    }

    @Test("Single model streaming benchmark - Phi")
    func testPhiStreamingBenchmark() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .phi) else {
            Issue.record("No Phi model configured")
            return
        }

        print("\n🚀 Benchmarking Phi family: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let metrics = try await harness.runStreamingBenchmark(
            config: config,
            prompt: "What is the capital of France?",
            maxTokens: 64,
            warmupRuns: 2,
            measuredRuns: 3,
            testName: "phi_streaming"
        )

        guard !metrics.isEmpty else {
            Issue.record("No metrics collected for Phi")
            return
        }

        let avgDecodeTPS = metrics.map(\.decodeTps).reduce(0, +) / Double(metrics.count)
        print("  📊 Phi avg: Decode=\(String(format: "%.1f", avgDecodeTPS))t/s")
        #expect(avgDecodeTPS > 0)

        harness.unloadCurrentModel()
    }

    @Test("Single model streaming benchmark - Qwen")
    func testQwenStreamingBenchmark() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .qwen) else {
            Issue.record("No Qwen model configured")
            return
        }

        print("\n🚀 Benchmarking Qwen family: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let metrics = try await harness.runStreamingBenchmark(
            config: config,
            prompt: "Translate 'Hello World' to Chinese, Japanese, and Korean.",
            maxTokens: 128,
            warmupRuns: 2,
            measuredRuns: 3,
            testName: "qwen_streaming"
        )

        guard !metrics.isEmpty else {
            Issue.record("No metrics collected for Qwen")
            return
        }

        let avgDecodeTPS = metrics.map(\.decodeTps).reduce(0, +) / Double(metrics.count)
        print("  📊 Qwen avg: Decode=\(String(format: "%.1f", avgDecodeTPS))t/s")
        #expect(avgDecodeTPS > 0)

        harness.unloadCurrentModel()
    }

    @Test("Single model streaming benchmark - Gemma")
    func testGemmaStreamingBenchmark() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .gemma) else {
            Issue.record("No Gemma model configured")
            return
        }

        print("\n🚀 Benchmarking Gemma family: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let metrics = try await harness.runStreamingBenchmark(
            config: config,
            prompt: "Summarize the water cycle in a few sentences.",
            maxTokens: 128,
            warmupRuns: 2,
            measuredRuns: 3,
            testName: "gemma_streaming"
        )

        guard !metrics.isEmpty else {
            Issue.record("No metrics collected for Gemma")
            return
        }

        let avgDecodeTPS = metrics.map(\.decodeTps).reduce(0, +) / Double(metrics.count)
        print("  📊 Gemma avg: Decode=\(String(format: "%.1f", avgDecodeTPS))t/s")
        #expect(avgDecodeTPS > 0)

        harness.unloadCurrentModel()
    }

    @Test("Warmup/Metal compilation cache effect")
    func testWarmupEffect() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .phi) else {
            Issue.record("No Phi model configured for warmup test")
            return
        }

        print("\n🔥 Warmup effect test: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let metrics = try await harness.runWarmupBenchmark(
            config: config,
            prompt: "What is 2+2?",
            maxTokens: 32,
            runs: 5
        )

        guard metrics.count >= 2 else {
            Issue.record("Not enough warmup runs")
            return
        }

        let firstRunTPS = metrics.first!.decodeTps
        let lastRunTPS = metrics.last!.decodeTps
        let improvement = lastRunTPS > 0 ? (firstRunTPS / lastRunTPS) : 0
        print("  📊 Warmup: Run1=\(String(format: "%.1f", firstRunTPS))t/s → Run5=\(String(format: "%.1f", lastRunTPS))t/s (ratio=\(String(format: "%.2f", improvement)))")

        harness.unloadCurrentModel()
    }

    @Test("Concurrent throughput - 2 parallel requests")
    func testConcurrent2() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .phi) else {
            Issue.record("No Phi model configured for concurrent test")
            return
        }

        print("\n🔄 Concurrent-2 test: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let warmupReq = InferenceRequest(
            model: config.hubId,
            messages: [ChatMessage(role: .user, content: "Hi")],
            temperature: 0.0,
            maxTokens: 10,
            stream: false
        )
        _ = try await harness.inferenceService.generate(warmupReq)

        let metric = try await harness.runConcurrentBenchmark(
            config: config,
            concurrency: 2,
            maxTokens: 64,
            testName: "concurrent_2"
        )

        print("  📊 Concurrent-2: \(metric.totalTokens) tokens in \(String(format: "%.0f", metric.totalWallMs))ms, aggregate=\(String(format: "%.1f", metric.aggregateTps))t/s")
        #expect(metric.totalTokens > 0)

        harness.unloadCurrentModel()
    }

    @Test("KV quantization impact")
    func testKVQuantImpact() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .phi) else {
            Issue.record("No Phi model configured for KV quant test")
            return
        }

        print("\n💾 KV Quantization test: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let warmupReq = InferenceRequest(
            model: config.hubId,
            messages: [ChatMessage(role: .user, content: "Hi")],
            temperature: 0.0,
            maxTokens: 10,
            stream: false
        )
        _ = try await harness.inferenceService.generate(warmupReq)

        let comparisons = try await harness.runKVQuantBenchmark(
            config: config,
            kvBitsValues: [nil, 8, 4, 2],
            prompt: "Explain the theory of relativity in simple terms.",
            maxTokens: 64
        )

        guard comparisons.count >= 2 else {
            Issue.record("Not enough KV quant comparisons")
            return
        }

        print("  📊 KV Quant: \(comparisons.count) configurations tested")

        harness.unloadCurrentModel()
    }

    @Test("Long context prefill benchmark")
    func testLongContextPrefill() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .phi) else {
            Issue.record("No Phi model configured for long context test")
            return
        }

        print("\n📏 Long context prefill test: \(config.hubId)")
        let _ = try await harness.loadModel(config)

        let longPrompt = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 100)
        let metrics = try await harness.runStreamingBenchmark(
            config: config,
            prompt: longPrompt,
            maxTokens: 64,
            warmupRuns: 1,
            measuredRuns: 3,
            testName: "long_context"
        )

        guard !metrics.isEmpty else {
            Issue.record("No metrics for long context test")
            return
        }

        let avgTTFT = metrics.map(\.ttftMs).reduce(0, +) / Double(metrics.count)
        let avgPrefillTPS = metrics.map(\.prefillTps).reduce(0, +) / Double(metrics.count)
        print("  📊 Long context: TTFT=\(String(format: "%.1f", avgTTFT))ms Prefill=\(String(format: "%.0f", avgPrefillTPS))t/s")

        harness.unloadCurrentModel()
    }

    @Test("Print final performance report")
    func testPrintReport() async throws {
        let harness = makeHarness()
        let report = harness.report
        print("\n" + report.summary())
        #expect(true)
    }
}
