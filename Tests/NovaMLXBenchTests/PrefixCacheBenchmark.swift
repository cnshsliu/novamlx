import Foundation
import MLX
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import Testing

@Suite("Prefix Cache Benchmark", .serialized, .enabled(if: ProcessInfo.processInfo.environment["NOVAMLX_BENCH"] == "1"))
@MainActor
struct PrefixCacheBenchmark {

    private func makeHarness() -> BenchmarkHarness {
        BenchmarkHarness()
    }

    /// Build a synthetic prompt of ~512 tokens by repeating a fixed sentence.
    private func buildLongPrompt() -> String {
        let sentence = "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon. "
        // ~15 tokens per sentence, repeat to get ~512 tokens
        return String(repeating: sentence, count: 35)
    }

    @Test("Prefix cache on/off/cold/warm TTFT benchmark")
    func prefixCacheBenchmark() async throws {
        let harness = makeHarness()
        guard let config = BenchModelRegistry.shared.model(for: .llama) else {
            Issue.record("No LLaMA model configured — skipping benchmark")
            return
        }

        print("\nPrefixCacheBenchmark")
        print("====================")

        // Load model
        let loadTime = try await harness.loadModel(config)
        print("  Model loaded: \(config.hubId)")
        print("  Load: \(String(format: "%.0f", loadTime.loadTimeMs))ms, Weights: \(String(format: "%.2f", loadTime.weightsMemoryGB))GB")

        let modelId = config.hubId
        let longPrompt = buildLongPrompt()
        let shortPrompt = "What is 2+2? "
        let maxTokens = 32

        // --- Mode A: prefix cache OFF ---
        harness.engine.setPrefixCacheEnabled(false)
        // Warmup
        for _ in 0..<2 {
            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: longPrompt + shortPrompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )
            let stream = harness.inferenceService.stream(req)
            for try await _ in stream {}
        }
        // Measured runs
        var modeATTFTs: [Double] = []
        let modeAStart = Date()
        for _ in 0..<30 {
            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: longPrompt + shortPrompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )
            let requestStart = ContinuousClock.now
            var firstToken = true
            var ttftMs: Double = 0
            let stream = harness.inferenceService.stream(req)
            for try await token in stream {
                if firstToken {
                    ttftMs = Double(requestStart.duration(to: .now).components.attoseconds) / 1e18 * 1000
                    firstToken = false
                }
            }
            if !firstToken { modeATTFTs.append(ttftMs) }
        }
        let modeAWall = Date().timeIntervalSince(modeAStart)

        // --- Mode B: prefix cache ON, cold cache ---
        harness.engine.setPrefixCacheEnabled(true)
        harness.engine.clearPrefixCache(for: modelId)
        // Warmup (populates cache)
        for _ in 0..<10 {
            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: longPrompt + shortPrompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )
            let stream = harness.inferenceService.stream(req)
            for try await _ in stream {}
        }
        // Measured runs (cold — each run adds to cache)
        harness.engine.clearPrefixCache(for: modelId)
        var modeBTTFTs: [Double] = []
        let modeBStart = Date()
        for _ in 0..<30 {
            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: longPrompt + shortPrompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )
            let requestStart = ContinuousClock.now
            var firstToken = true
            var ttftMs: Double = 0
            let stream = harness.inferenceService.stream(req)
            for try await token in stream {
                if firstToken {
                    ttftMs = Double(requestStart.duration(to: .now).components.attoseconds) / 1e18 * 1000
                    firstToken = false
                }
            }
            if !firstToken { modeBTTFTs.append(ttftMs) }
        }
        let modeBWall = Date().timeIntervalSince(modeBStart)

        // --- Mode C: prefix cache ON, warm cache (reuse from Mode B) ---
        var modeCTTFTs: [Double] = []
        let modeCStart = Date()
        for _ in 0..<30 {
            let req = InferenceRequest(
                model: modelId,
                messages: [ChatMessage(role: .user, content: longPrompt + shortPrompt)],
                temperature: 0.0,
                maxTokens: maxTokens,
                stream: true
            )
            let requestStart = ContinuousClock.now
            var firstToken = true
            var ttftMs: Double = 0
            let stream = harness.inferenceService.stream(req)
            for try await token in stream {
                if firstToken {
                    ttftMs = Double(requestStart.duration(to: .now).components.attoseconds) / 1e18 * 1000
                    firstToken = false
                }
            }
            if !firstToken { modeCTTFTs.append(ttftMs) }
        }
        let modeCWall = Date().timeIntervalSince(modeCStart)

        // Get cache stats
        let cacheStats = harness.engine.getPrefixCacheStats(for: modelId)

        // Compute median and p90
        func median(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            let mid = sorted.count / 2
            return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
        }
        func p90(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            let idx = Int(Double(sorted.count) * 0.9)
            return sorted[min(idx, sorted.count - 1)]
        }

        let aMedian = median(modeATTFTs)
        let aP90 = p90(modeATTFTs)
        let bMedian = median(modeBTTFTs)
        let bP90 = p90(modeBTTFTs)
        let cMedian = median(modeCTTFTs)
        let cP90 = p90(modeCTTFTs)

        let speedupB = aMedian > 0 ? ((aMedian - bMedian) / aMedian) * 100 : 0
        let speedupC = aMedian > 0 ? ((aMedian - cMedian) / aMedian) * 100 : 0

        print("")
        print("Mode A (off)        : TTFT median = \(String(format: "%.1f", aMedian)) ms, p90 = \(String(format: "%.1f", aP90)) ms, total wall = \(String(format: "%.1f", modeAWall)) s")
        print("Mode B (cold cache) : TTFT median = \(String(format: "%.1f", bMedian)) ms, p90 = \(String(format: "%.1f", bP90)) ms, total wall = \(String(format: "%.1f", modeBWall)) s")
        print("Mode C (warm cache) : TTFT median = \(String(format: "%.1f", cMedian)) ms, p90 = \(String(format: "%.1f", cP90)) ms, total wall = \(String(format: "%.1f", modeCWall)) s")
        print("Speedup B vs A      : \(String(format: "%.1f", speedupB))% (TTFT median)")
        print("Speedup C vs A      : \(String(format: "%.1f", speedupC))% (TTFT median)")
        if let stats = cacheStats {
            print("Cache stats         : hits=\(stats.hits) misses=\(stats.misses) tokensSaved=\(stats.tokensSaved)")
        }

        // Cleanup
        harness.engine.setPrefixCacheEnabled(false)

        #expect(true)
    }
}
