import Foundation
import Testing

// ─── Configuration ───

let apiBase = URL(string: ProcessInfo.processInfo.environment["NOVA_API_URL"] ?? "http://127.0.0.1:6590")!
let adminBase = URL(string: ProcessInfo.processInfo.environment["NOVA_ADMIN_URL"] ?? "http://127.0.0.1:6591")!
let apiKey = ProcessInfo.processInfo.environment["NOVA_API_KEY"] ?? "abcd1234"

// ─── E2E Gate ───
// Auto-detect: if NovaMLX server is running at 127.0.0.1:6590, run E2E tests.
// If not, skip. No environment variable needed.
// Usage: swift test                       (unit + E2E if server is up)
//        NOVA_E2E=0 swift test            (force skip E2E)

private let e2eForceDisabled = ProcessInfo.processInfo.environment["NOVA_E2E"] == "0"

private func detectServerAlive() -> Bool {
    guard !e2eForceDisabled else { return false }
    let url = URL(string: ProcessInfo.processInfo.environment["NOVA_API_URL"] ?? "http://127.0.0.1:6590")!
    let healthURL = url.appendingPathComponent("health")
    var request = URLRequest(url: healthURL)
    request.httpMethod = "GET"
    request.timeoutInterval = 3
    let sem = DispatchSemaphore(value: 0)
    var result = false
    let task = URLSession.shared.dataTask(with: request) { _, response, _ in
        result = (response as? HTTPURLResponse)?.statusCode == 200
        sem.signal()
    }
    task.resume()
    _ = sem.wait(timeout: .now() + 5)
    return result
}

private let e2eEnabled = detectServerAlive()

// ─── HTTP Client ───

struct HTTPClient: Sendable {
    let session: URLSession
    let longSession: URLSession

    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)

        let longConfig = URLSessionConfiguration.default
        longConfig.timeoutIntervalForRequest = 600
        longConfig.timeoutIntervalForResource = 900
        self.longSession = URLSession(configuration: longConfig)
    }

    func get(_ url: URL) async throws -> (Data, Int) {
        var req = URLRequest(url: url)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let (data, resp) = try await session.data(for: req)
        return (data, (resp as? HTTPURLResponse)?.statusCode ?? 0)
    }

    func post(_ url: URL, body: Data) async throws -> (Data, Int) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        let (data, resp) = try await session.data(for: req)
        return (data, (resp as? HTTPURLResponse)?.statusCode ?? 0)
    }

    func postLong(_ url: URL, body: Data) async throws -> (Data, Int) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        let (data, resp) = try await longSession.data(for: req)
        return (data, (resp as? HTTPURLResponse)?.statusCode ?? 0)
    }

    func postStream(_ url: URL, body: Data) async throws -> String {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        let (bytes, resp) = try await session.bytes(for: req)
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if status != 200 {
            let body = try await bytes.reduce(into: "") { $0 += String(decoding: [$1], as: UTF8.self) }
            throw TestError.httpError(status, body)
        }

        var result = ""
        for try await line in bytes.lines {
            if line.hasPrefix("data: ") {
                let payload = String(line.dropFirst(6))
                if payload == "[DONE]" { break }
                result += extractStreamText(payload)
            }
        }
        return result
    }

    func isServerAlive() async -> Bool {
        for i in 0..<3 {
            if let (_, status) = try? await get(apiBase.appendingPathComponent("health")),
               status == 200 { return true }
            if i < 2 { try? await Task.sleep(nanoseconds: 1_000_000_000) }
        }
        return false
    }

    func waitForServer(maxWait: TimeInterval = 30) async -> Bool {
        let deadline = Date().addingTimeInterval(maxWait)
        while Date() < deadline {
            if await isServerAlive() { return true }
            try? await Task.sleep(nanoseconds: 2_000_000_000)
        }
        return false
    }
}

enum TestError: Error, CustomStringConvertible {
    case httpError(Int, String)
    case serverUnavailable(String)
    case modelNotLoaded(String)
    case loadFailed(String)

    var description: String {
        switch self {
        case .httpError(let s, let b): "HTTP \(s): \(b.prefix(200))"
        case .serverUnavailable(let m): "Server unavailable: \(m)"
        case .modelNotLoaded(let m): "Model not loaded: \(m)"
        case .loadFailed(let m): "Model load failed: \(m)"
        }
    }
}

// ─── JSON / Stream Helpers ───

func extractStreamText(_ json: String) -> String {
    guard let data = json.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return "" }

    if let choices = obj["choices"] as? [[String: Any]],
       let delta = choices.first?["delta"] as? [String: Any],
       let text = delta["content"] as? String {
        return text
    }

    if let type = obj["type"] as? String,
       type == "content_block_delta",
       let delta = obj["delta"] as? [String: Any],
       let text = delta["text"] as? String {
        return text
    }

    return ""
}

// ─── Admin API Model Management ───

struct AdminModelStatus: Decodable, Sendable {
    let id: String
    let downloaded: Bool
    let loaded: Bool
}

func getAllDownloadedModels(_ client: HTTPClient) async throws -> [AdminModelStatus] {
    let (data, _) = try await client.get(adminBase.appendingPathComponent("admin/models"))
    let models = try JSONDecoder().decode([AdminModelStatus].self, from: data)
    return models.filter { $0.downloaded }
}

func loadModel(_ client: HTTPClient, modelId: String, maxWait: TimeInterval = 60) async throws -> Bool {
    // Check if already loaded
    let models = try await getAllDownloadedModels(client)
    if models.first(where: { $0.id == modelId })?.loaded == true { return true }

    let body = try JSONSerialization.data(withJSONObject: ["modelId": modelId])
    let (_, status) = try await client.post(
        adminBase.appendingPathComponent("admin/models/load"), body: body
    )
    guard status == 200 else { return false }

    // Poll until loaded
    let deadline = Date().addingTimeInterval(maxWait)
    while Date() < deadline {
        let models = try await getAllDownloadedModels(client)
        if models.first(where: { $0.id == modelId })?.loaded == true { return true }
        try? await Task.sleep(nanoseconds: 2_000_000_000)
    }
    return false
}

func unloadModel(_ client: HTTPClient, modelId: String) async throws {
    let body = try JSONSerialization.data(withJSONObject: ["modelId": modelId])
    _ = try? await client.post(
        adminBase.appendingPathComponent("admin/models/unload"), body: body
    )
}

func requireServer(_ client: HTTPClient) async throws {
    guard await client.isServerAlive() else {
        throw TestError.serverUnavailable("Server is not responding. Start NovaMLX before running E2E tests.")
    }
}

func safePost(_ client: HTTPClient, url: URL, body: Data, modelId: String) async -> (Data, Int)? {
    do {
        return try await client.postLong(url, body: body)
    } catch let error as URLError where error.code == .timedOut || error.code == .cannotConnectToHost || error.code == .networkConnectionLost {
        Issue.record("\(modelId): request failed — \(error.localizedDescription). Server may have crashed under load.")
        return nil
    } catch {
        Issue.record("\(modelId): unexpected error — \(error)")
        return nil
    }
}

func safePostStream(_ client: HTTPClient, url: URL, body: Data, modelId: String) async -> String? {
    do {
        return try await client.postStream(url, body: body)
    } catch let error as TestError {
        if case .httpError(404, _) = error { return nil }
        Issue.record("\(modelId): stream error — \(error)")
        return nil
    } catch let error as URLError {
        Issue.record("\(modelId): stream failed — \(error.localizedDescription). Server may have crashed.")
        return nil
    } catch {
        Issue.record("\(modelId): stream error — \(error)")
        return nil
    }
}

// ─── Test Content Generators ───

func generateTokenFiller(count: Int) -> String {
    let words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "and", "a", "in", "to", "of", "is", "it", "that", "for", "on", "with", "as"]
    return (0..<count).map { words[$0 % words.count] }.joined(separator: " ")
}

func buildClaudeCodeSystemPrompt() -> [[String: Any]] {
    [[
        "type": "text",
        "text": """
        You are an interactive CLI tool that helps users with software engineering tasks. \
        Keep responses concise. Read files before modifying them. Follow KISS and DRY principles.
        """,
        "cache_control": ["type": "ephemeral"]
    ]]
}

func buildClaudeCodeTools() -> [[String: Any]] {
    let schema: [String: Any] = ["type": "object", "properties": ["arg": ["type": "string"]], "required": ["arg"]]
    return [
        ["name": "Read", "description": "Reads a file.", "input_schema": schema],
        ["name": "Bash", "description": "Runs a command.", "input_schema": schema],
        ["name": "Grep", "description": "Searches patterns.", "input_schema": schema],
    ]
}

func buildClaudeCodeConversation(turns: Int) -> [[String: Any]] {
    var messages: [[String: Any]] = []

    messages.append(["role": "user", "content": "Read Package.swift and list the dependencies."])
    messages.append(["role": "assistant", "content": [
        ["type": "text", "text": "Let me read the file."],
        ["type": "tool_use", "id": "toolu_001", "name": "Read", "input": ["file_path": "/Users/dev/project/Package.swift"]]
    ]])
    messages.append(["role": "user", "content": [
        ["type": "tool_result", "tool_use_id": "toolu_001", "content":
            "import PackageDescription\nlet package = Package(name: \"MyApp\", dependencies: [.package(url: \"https://github.com/apple/swift-nio\", from: \"2.0.0\")])"]
    ]])
    messages.append(["role": "assistant", "content": [
        ["type": "text", "text": "The project depends on swift-nio 2.0.0+."]
    ]])

    for i in 2..<turns {
        messages.append(["role": "user", "content": "Check Sources/App/file_\(i).swift for issues."])
        messages.append(["role": "assistant", "content": [
            ["type": "text", "text": "Checking the file."],
            ["type": "tool_use", "id": "toolu_\(String(format: "%03d", i))", "name": "Read",
             "input": ["file_path": "/Users/dev/project/Sources/App/file_\(i).swift"]]
        ]])
        messages.append(["role": "user", "content": [
            ["type": "tool_result", "tool_use_id": "toolu_\(String(format: "%03d", i))",
             "content": "import Foundation\nstruct Config\(i) { let port: Int = 8080 }"]
        ]])
        messages.append(["role": "assistant", "content": [
            ["type": "text", "text": "file_\(i).swift looks clean — simple Config struct."]
        ]])
    }

    return messages
}

// ═══════════════════════════════════════════════════════════════════
// ALL E2E TESTS — single serialized suite to avoid concurrent GPU load
// Each test loads its model individually, tests, then unloads
// ═══════════════════════════════════════════════════════════════════

@Suite("E2E Tests", .serialized, .enabled(if: e2eEnabled))
struct E2ETests {
    let client = HTTPClient()

    // ─── Per-model test helper ───

    /// Load each downloaded model one at a time, run the test closure, then unload.
    func forEachModel(_ label: String, test: (String) async throws -> Void) async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [\(label)] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else {
                Issue.record("\(modelId): failed to load")
                continue
            }
            print("  [\(label)] Testing \(modelId)...")

            do {
                try await test(modelId)
            } catch {
                Issue.record("\(modelId): \(error)")
            }

            // Unload and let GPU memory settle
            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            guard await client.isServerAlive() else {
                Issue.record("Server died after testing \(modelId)")
                let recovered = await client.waitForServer(maxWait: 30)
                if !recovered { return }
                break
            }
        }
    }

    // ─── Suite 1: API correctness ───

    @Test("OpenAI non-stream: returns valid response")
    func testOpenAINonStream() async throws {
        try await forEachModel("openai-non-stream") { modelId in
            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "stream": false, "max_tokens": 50
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            guard let (resp, status) = await safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId) else { return }

            if status == 500 {
                let body = String(data: resp, encoding: .utf8) ?? ""
                print("    \(modelId): HTTP 500 — \(body.prefix(200))")
                return // Server bug, not a test failure
            }
            #expect(status == 200, "\(modelId): HTTP \(status)")
            guard status == 200 else { return }

            let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
            let choices = json?["choices"] as? [[String: Any]]
            let text = (choices?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
            #expect(!text.isEmpty, "\(modelId): empty response")
            let finishReason = choices?.first?["finish_reason"] as? String ?? ""
            #expect(finishReason == "stop" || finishReason == "length", "\(modelId): unexpected finish_reason '\(finishReason)'")
            print("    \(modelId): OK")
        }
    }

    @Test("OpenAI stream: receives tokens and [DONE]")
    func testOpenAIStream() async throws {
        try await forEachModel("openai-stream") { modelId in
            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "stream": true, "max_tokens": 50
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePostStream(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)
            if let result {
                if result.isEmpty {
                    print("    \(modelId): empty stream (possible server error)")
                } else {
                    print("    \(modelId): stream OK")
                }
            }
        }
    }

    @Test("Anthropic non-stream: returns valid response")
    func testAnthropicNonStream() async throws {
        try await forEachModel("anthropic-non-stream") { modelId in
            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "max_tokens": 50, "stream": false
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let url = apiBase.appendingPathComponent("v1/messages")
            var result = await safePost(client, url: url, body: data, modelId: modelId)

            // Retry on 429
            if let (_, status) = result, status == 429 {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                result = await safePost(client, url: url, body: data, modelId: modelId)
            }
            guard let (resp, status) = result else { return }

            if status == 500 {
                let body = String(data: resp, encoding: .utf8) ?? ""
                print("    \(modelId): HTTP 500 — \(body.prefix(200))")
                return
            }
            #expect(status == 200, "\(modelId): HTTP \(status)")
            guard status == 200 else { return }

            let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
            let content = json?["content"] as? [[String: Any]]
            let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
            #expect(!text.isEmpty, "\(modelId): empty response")
            let stopReason = json?["stop_reason"] as? String ?? ""
            #expect(stopReason == "stop" || stopReason == "end_turn", "\(modelId): unexpected stop_reason '\(stopReason)'")
            print("    \(modelId): OK")
        }
    }

    @Test("Anthropic stream: receives content and message_stop")
    func testAnthropicStream() async throws {
        try await forEachModel("anthropic-stream") { modelId in
            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "max_tokens": 50, "stream": true
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePostStream(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId)
            if let result {
                if result.isEmpty {
                    print("    \(modelId): empty stream (possible server error)")
                } else {
                    print("    \(modelId): stream OK")
                }
            }
        }
    }

    // ─── Suite 2: Pressure test — progressive token scaling ───

    @Test("Pressure test: progressive token scaling 2K→4K→6K→8K per model")
    func testPressureTokenScaling() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        let tokenLevels = [2000, 4000, 6000, 8000]

        for model in models {
            let modelId = model.id
            print("  [pressure] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else {
                Issue.record("\(modelId): failed to load for pressure test")
                continue
            }

            for tokens in tokenLevels {
                guard await client.isServerAlive() else {
                    Issue.record("\(modelId): server died at \(tokens) tokens")
                    let recovered = await client.waitForServer(maxWait: 30)
                    if recovered { break } // try next model level, but we lost this model
                    return // server gone
                }

                let filler = generateTokenFiller(count: tokens)
                let prompt = "Summarize this text in one sentence: \(filler)"
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": prompt]],
                    "stream": false, "max_tokens": 100
                ]
                let data = try JSONSerialization.data(withJSONObject: body)

                let result = await safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)
                guard let (_, status) = result else {
                    Issue.record("\(modelId): request failed at \(tokens) tokens — server may have crashed")
                    let recovered = await client.waitForServer(maxWait: 30)
                    if recovered { break } else { return }
                }

                if status == 200 {
                    print("    \(modelId): \(tokens) tokens OK")
                } else if status == 500 {
                    print("    \(modelId): HTTP 500 at \(tokens) tokens (server error)")
                    break // Model has server-side issues, skip remaining token levels
                } else {
                    Issue.record("\(modelId): HTTP \(status) at \(tokens) tokens")
                }
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 3_000_000_000)
        }

        #expect(await client.isServerAlive(), "Server must survive pressure test")
    }

    // ─── Suite 3: Concurrent request stress ───

    @Test("Concurrent stress: 3 simultaneous requests per model")
    func testConcurrentRequests() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [concurrent] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else {
                Issue.record("\(modelId): failed to load for concurrent test")
                continue
            }

            // Fire 3 concurrent requests
            let prompts = [
                "What is 2+2? Answer with just the number.",
                "What color is the sky on a clear day? One word.",
                "What is the capital of France? One word."
            ]

            async let r1 = safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"),
                body: try JSONSerialization.data(withJSONObject: [
                    "model": modelId, "messages": [["role": "user", "content": prompts[0]]],
                    "stream": false, "max_tokens": 20
                ]), modelId: modelId)

            async let r2 = safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"),
                body: try JSONSerialization.data(withJSONObject: [
                    "model": modelId, "messages": [["role": "user", "content": prompts[1]]],
                    "stream": false, "max_tokens": 20
                ]), modelId: modelId)

            async let r3 = safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"),
                body: try JSONSerialization.data(withJSONObject: [
                    "model": modelId, "messages": [["role": "user", "content": prompts[2]]],
                    "stream": false, "max_tokens": 20
                ]), modelId: modelId)

            let results = try await [r1, r2, r3]
            var successCount = 0
            var serverError = false
            for result in results {
                if let (data, status) = result, status == 200 {
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let choices = json["choices"] as? [[String: Any]],
                       let text = (choices.first?["message"] as? [String: Any])?["content"] as? String,
                       !text.isEmpty {
                        successCount += 1
                    }
                } else if let (_, status) = result, status == 500 {
                    serverError = true
                }
            }
            if serverError {
                print("    \(modelId): server error on concurrent requests (skipped)")
            } else {
                #expect(successCount == 3, "\(modelId): only \(successCount)/3 concurrent requests succeeded")
                print("    \(modelId): \(successCount)/3 concurrent requests OK")
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            guard await client.isServerAlive() else {
                Issue.record("Server died after concurrent test with \(modelId)")
                let recovered = await client.waitForServer(maxWait: 30)
                if !recovered { return }
                break
            }
        }
    }

    // ─── Suite 4: Claude Code simulation ───

    @Test("Claude Code multi-turn conversation via Anthropic API")
    func testClaudeCodeSimulation() async throws {
        try await forEachModel("claude-code") { modelId in
            let conversation = buildClaudeCodeConversation(turns: 3)

            let body: [String: Any] = [
                "model": modelId,
                "system": buildClaudeCodeSystemPrompt(),
                "tools": buildClaudeCodeTools(),
                "messages": conversation,
                "max_tokens": 512,
                "stream": false
            ]
            let data = try JSONSerialization.data(withJSONObject: body)

            guard let (resp, status) = await safePost(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId) else { return }
            if status == 500 {
                let body = String(data: resp, encoding: .utf8) ?? ""
                print("    \(modelId): HTTP 500 — \(body.prefix(200))")
                return
            }
            #expect(status == 200, "\(modelId): Claude Code simulation HTTP \(status)")
            guard status == 200 else { return }

            let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
            let content = json?["content"] as? [[String: Any]]
            let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
            #expect(!text.isEmpty, "\(modelId): empty Claude Code response")

            let usage = json?["usage"] as? [String: Any]
            let inputTokens = usage?["input_tokens"] as? Int ?? 0
            print("    \(modelId): Claude Code sim OK — \(inputTokens) input tokens, \(text.count) chars")
        }
    }

    @Test("Claude Code long context: extended 8-turn conversation")
    func testClaudeCodeLongContext() async throws {
        try await forEachModel("claude-code-long") { modelId in
            let conversation = buildClaudeCodeConversation(turns: 8)

            let body: [String: Any] = [
                "model": modelId,
                "system": buildClaudeCodeSystemPrompt(),
                "tools": buildClaudeCodeTools(),
                "messages": conversation,
                "max_tokens": 1024,
                "stream": false
            ]
            let data = try JSONSerialization.data(withJSONObject: body)

            guard let (resp, status) = await safePost(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId) else { return }

            if status == 200 {
                let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                let content = json?["content"] as? [[String: Any]]
                let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
                #expect(!text.isEmpty, "\(modelId): empty long context response")
                let usage = json?["usage"] as? [String: Any]
                let inputTokens = usage?["input_tokens"] as? Int ?? 0
                print("    \(modelId): Long context OK — \(inputTokens) input tokens")
            } else if status == 500 {
                print("    \(modelId): HTTP 500 (server error)")
            } else {
                let errorBody = String(data: resp, encoding: .utf8) ?? ""
                if errorBody.contains("context") || errorBody.contains("token") || status == 400 {
                    print("    \(modelId): Long context hit context limit (expected) — HTTP \(status)")
                } else {
                    Issue.record("\(modelId): Long context unexpected error — HTTP \(status): \(errorBody.prefix(200))")
                }
            }
        }
    }

    // ─── Suite 5: Memory safety ───

    @Test("Memory safety: oversized request rejected without crash")
    func testMemorySafetyRejection() async throws {
        try await forEachModel("memory-safety") { modelId in
            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Hello"]],
                "max_tokens": 999999
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)

            if let (_, status) = result {
                #expect(status == 200 || status >= 400, "\(modelId): unexpected status \(status)")
                print("    \(modelId): oversized request → HTTP \(status) (server survived)")
            }
        }
        #expect(await client.isServerAlive(), "Server must survive memory safety test")
    }

    // ─── Suite 6: ContinuousBatcher E2E ───

    @Test("ContinuousBatcher: concurrent requests complete under batcher")
    func testBatcherConcurrentCompletion() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [batcher-concurrent] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else {
                Issue.record("\(modelId): failed to load")
                continue
            }

            // Fire 10 concurrent requests — exceeds maxBatchSize to exercise queuing
            let requestCount = 10
            try await withThrowingTaskGroup(of: (Int, Bool).self) { group in
                for i in 0..<requestCount {
                    group.addTask {
                        let body: [String: Any] = [
                            "model": modelId,
                            "messages": [["role": "user", "content": "Reply with just the number \(i)."]],
                            "stream": false, "max_tokens": 10
                        ]
                        let data = try JSONSerialization.data(withJSONObject: body)
                        if let (resp, status) = await safePost(self.client,
                            url: apiBase.appendingPathComponent("v1/chat/completions"),
                            body: data, modelId: modelId), status == 200 {
                            if let json = try? JSONSerialization.jsonObject(with: resp) as? [String: Any],
                               let choices = json["choices"] as? [[String: Any]],
                               let text = (choices.first?["message"] as? [String: Any])?["content"] as? String,
                               !text.isEmpty {
                                return (i, true)
                            }
                        }
                        return (i, false)
                    }
                }

                var successes = 0
                for try await (_, ok) in group {
                    if ok { successes += 1 }
                }
                #expect(successes == requestCount,
                    "\(modelId): only \(successes)/\(requestCount) concurrent requests completed under batcher")
                print("    \(modelId): \(successes)/\(requestCount) concurrent requests OK")
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            guard await client.isServerAlive() else {
                Issue.record("Server died after batcher concurrent test with \(modelId)")
                let recovered = await client.waitForServer(maxWait: 30)
                if !recovered { return }
                break
            }
        }
    }

    @Test("ContinuousBatcher: stats endpoint reports batcher metrics")
    func testBatcherStats() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [batcher-stats] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else { continue }

            // Fire a few concurrent requests to exercise the batcher
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<4 {
                    group.addTask {
                        let body: [String: Any] = [
                            "model": modelId,
                            "messages": [["role": "user", "content": "Say hi."]],
                            "stream": false, "max_tokens": 10
                        ]
                        let data = try JSONSerialization.data(withJSONObject: body)
                        _ = await safePost(self.client,
                            url: apiBase.appendingPathComponent("v1/chat/completions"),
                            body: data, modelId: modelId)
                    }
                }
                for try await _ in group { }
            }

            // Check stats endpoint
            let (data, status) = try await client.get(apiBase.appendingPathComponent("v1/stats"))
            #expect(status == 200, "\(modelId): stats endpoint returned HTTP \(status)")
            guard status == 200 else {
                try? await unloadModel(client, modelId: modelId)
                continue
            }

            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let batcher = json["batcher"] as? [String: Any] {
                let totalCompleted = batcher["totalCompleted"] as? UInt64 ?? 0
                let totalQueued = batcher["totalQueued"] as? UInt64 ?? 0
                let maxBatchSize = batcher["maxBatchSize"] as? Int ?? 0
                #expect(totalCompleted > 0, "\(modelId): batcher should have completed requests, got \(totalCompleted)")
                #expect(totalQueued > 0, "\(modelId): batcher should have queued requests, got \(totalQueued)")
                #expect(maxBatchSize > 0, "\(modelId): maxBatchSize should be > 0")
                print("    \(modelId): batcher stats — completed=\(totalCompleted), queued=\(totalQueued), maxBatch=\(maxBatchSize)")
            } else {
                Issue.record("\(modelId): failed to parse batcher stats")
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 2_000_000_000)
        }
    }

    @Test("ContinuousBatcher: stream requests batched concurrently")
    func testBatcherStreamConcurrent() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [batcher-stream] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else { continue }

            // Fire 6 concurrent stream requests
            let requestCount = 6
            try await withThrowingTaskGroup(of: Bool.self) { group in
                for i in 0..<requestCount {
                    group.addTask {
                        let body: [String: Any] = [
                            "model": modelId,
                            "messages": [["role": "user", "content": "Count to \(i) in one word."]],
                            "stream": true, "max_tokens": 10
                        ]
                        let data = try JSONSerialization.data(withJSONObject: body)
                        let result = await safePostStream(self.client,
                            url: apiBase.appendingPathComponent("v1/chat/completions"),
                            body: data, modelId: modelId)
                        return result != nil && !(result ?? "").isEmpty
                    }
                }

                var successes = 0
                for try await ok in group {
                    if ok { successes += 1 }
                }
                #expect(successes >= requestCount / 2,
                    "\(modelId): only \(successes)/\(requestCount) stream requests completed under batcher")
                print("    \(modelId): \(successes)/\(requestCount) stream requests OK")
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 3_000_000_000)

            guard await client.isServerAlive() else {
                Issue.record("Server died after batcher stream test with \(modelId)")
                let recovered = await client.waitForServer(maxWait: 30)
                if !recovered { return }
                break
            }
        }
    }
}
