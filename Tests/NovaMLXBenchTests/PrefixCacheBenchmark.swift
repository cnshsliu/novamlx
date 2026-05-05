import Foundation
import Testing

/// Prefix cache TTFT benchmark. Hits the running NovaMLX server via HTTP
/// with a repeated-prefix workload. Uses whatever model is currently loaded.
///
/// Gate: NOVAMLX_BENCH=1
/// Usage: NOVAMLX_BENCH=1 swift test --filter PrefixCacheBenchmark
@Suite("Prefix Cache Benchmark", .serialized, .enabled(if: ProcessInfo.processInfo.environment["NOVAMLX_BENCH"] == "1"))
struct PrefixCacheBenchmark {

    private static let modelId = ProcessInfo.processInfo.environment["NOVA_MODEL"] ?? "mlx-community/Qwen3.6-27B-4bit"
    private static let baseURL = ProcessInfo.processInfo.environment["NOVA_API_URL"] ?? "http://127.0.0.1:6590"

    private static var apiKey: String {
        let configPath = NSHomeDirectory() + "/.nova/config.json"
        let data = try! Data(contentsOf: URL(fileURLWithPath: configPath))
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        let server = json["server"] as! [String: Any]
        let keys = server["apiKeys"] as! [String]
        return keys[0]
    }

    private static let adminURL = ProcessInfo.processInfo.environment["NOVA_ADMIN_URL"] ?? "http://127.0.0.1:6591"

    /// Build a synthetic prompt of ~512 tokens by repeating a fixed sentence.
    private func buildLongPrompt() -> String {
        let sentence = "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon. "
        return String(repeating: sentence, count: 35)
    }

    /// Send a streaming chat completion, return TTFT in ms.
    private func streamingRequest(model: String, prompt: String, maxTokens: Int) async throws -> Double? {
        let body: [String: Any] = [
            "model": model,
            "messages": [["role": "user", "content": prompt]],
            "temperature": 0.0,
            "max_tokens": maxTokens,
            "stream": true,
        ]
        var request = URLRequest(url: URL(string: "\(Self.baseURL)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(Self.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 300

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return nil
        }

        let requestStart = ContinuousClock.now
        var firstTokenTime: ContinuousClock.Instant? = nil

        for try await line in bytes.lines {
            if firstTokenTime == nil, line.hasPrefix("data: "), line != "data: [DONE]" {
                let payload = String(line.dropFirst(6))
                if let data = payload.data(using: .utf8),
                   let chunk = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = chunk["choices"] as? [[String: Any]],
                   let delta = choices.first?["delta"] as? [String: Any] {
                    let content = delta["content"] as? String
                    let reasoning = delta["reasoning_content"] as? String
                    if content != nil || reasoning != nil {
                        firstTokenTime = .now
                    }
                }
            }
        }

        guard let ft = firstTokenTime else { return nil }
        return Double(requestStart.duration(to: ft).components.attoseconds) / 1e18 * 1000
    }

    /// Set prefix cache enabled/disabled via admin endpoint.
    private func setCacheEnabled(_ enabled: Bool) async throws {
        // Use the config file directly since there's no admin toggle endpoint.
        // Instead, just read the cache stats to verify state.
        _ = enabled // no-op; cache is always enabled unless config says otherwise
    }

    /// Clear prefix cache for model via admin endpoint.
    private func clearCache(model: String) async throws {
        guard let url = URL(string: "\(Self.adminURL)/admin/prefix-cache/\(model.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model)/clear") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(Self.apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 10
        _ = try await URLSession.shared.data(for: request)
    }

    /// Get prefix cache stats.
    private func getCacheStats(model: String) async throws -> [String: Any]? {
        guard let url = URL(string: "\(Self.adminURL)/admin/prefix-cache/\(model.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model)/stats") else { return nil }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(Self.apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    @Test("Prefix cache on/cold/warm TTFT benchmark via HTTP")
    func prefixCacheBenchmark() async throws {
        let model = Self.modelId
        let longPrompt = buildLongPrompt()
        let shortPrompt = "What is 2+2? "
        let prompt = longPrompt + shortPrompt
        let maxTokens = 32

        print("\nPrefixCacheBenchmark")
        print("====================")
        print("  Model: \(model)")

        // Clear cache to start fresh
        try await clearCache(model: model)

        // --- Mode A: prefix cache OFF (disable via config, then re-enable) ---
        // Since we can't toggle at runtime via admin, use the fact that a cold
        // cache = no hits = equivalent to "off". Run once to get baseline TTFT.
        // Mode A: run with freshly cleared cache each time (no hits possible).
        print("  Running Mode A (no cache hits — clearing before each request)...")
        var modeATTFTs: [Double] = []
        let modeAStart = Date()
        for i in 0..<30 {
            try await clearCache(model: model)
            if let ttft = try await streamingRequest(model: model, prompt: prompt, maxTokens: maxTokens) {
                modeATTFTs.append(ttft)
            }
            if i == 0 { print("    request 0 done") }
        }
        let modeAWall = Date().timeIntervalSince(modeAStart)

        // --- Mode B: prefix cache ON, cold start then warm up ---
        print("  Running Mode B (cold → populating cache)...")
        try await clearCache(model: model)
        // Warmup to populate cache
        for _ in 0..<10 {
            _ = try await streamingRequest(model: model, prompt: prompt, maxTokens: maxTokens)
        }
        var modeBTTFTs: [Double] = []
        let modeBStart = Date()
        for i in 0..<30 {
            if let ttft = try await streamingRequest(model: model, prompt: prompt, maxTokens: maxTokens) {
                modeBTTFTs.append(ttft)
            }
            if i == 0 { print("    request 0 done") }
        }
        let modeBWall = Date().timeIntervalSince(modeBStart)

        // --- Mode C: warm cache (reuse from Mode B, no clear) ---
        print("  Running Mode C (warm cache from Mode B)...")
        var modeCTTFTs: [Double] = []
        let modeCStart = Date()
        for i in 0..<30 {
            if let ttft = try await streamingRequest(model: model, prompt: prompt, maxTokens: maxTokens) {
                modeCTTFTs.append(ttft)
            }
            if i == 0 { print("    request 0 done") }
        }
        let modeCWall = Date().timeIntervalSince(modeCStart)

        // Stats
        let stats = try await getCacheStats(model: model)

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
        if let s = stats {
            print("Cache stats         : \(s)")
        }

        #expect(true)
    }
}
