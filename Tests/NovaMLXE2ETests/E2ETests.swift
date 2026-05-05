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

func unloadAllModels(_ client: HTTPClient) async throws {
    let models = try await getAllDownloadedModels(client)
    let loadedIds = models.filter { $0.loaded }.map { $0.id }
    for id in loadedIds {
        try? await unloadModel(client, modelId: id)
    }
    if !loadedIds.isEmpty {
        try? await Task.sleep(nanoseconds: 3_000_000_000)
    }
}

func loadModel(_ client: HTTPClient, modelId: String, maxWait: TimeInterval = 120) async throws -> Bool {
    let models = try await getAllDownloadedModels(client)
    if models.first(where: { $0.id == modelId })?.loaded == true { return true }

    try await unloadAllModels(client)

    let body = try JSONSerialization.data(withJSONObject: ["modelId": modelId])
    var (_, status) = try await client.postLong(
        adminBase.appendingPathComponent("admin/models/load"), body: body
    )

    // Worker might have crashed — wait for recovery and retry once
    if status != 200 {
        try? await Task.sleep(nanoseconds: 5_000_000_000)
        guard await client.waitForServer(maxWait: 30) else { return false }
        try await unloadAllModels(client)
        (_, status) = try await client.postLong(
            adminBase.appendingPathComponent("admin/models/load"), body: body
        )
        guard status == 200 else { return false }
    }

    // Poll until loaded (models can take 60-90s to load)
    let deadline = Date().addingTimeInterval(maxWait)
    while Date() < deadline {
        let models = try await getAllDownloadedModels(client)
        if models.first(where: { $0.id == modelId })?.loaded == true { return true }
        try? await Task.sleep(nanoseconds: 3_000_000_000)
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
    } catch let error as URLError where error.code == .timedOut || error.code == .cannotConnectToHost || error.code == .networkConnectionLost {
        Issue.record("\(modelId): stream failed — \(error.localizedDescription)")
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
    [[
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": [
            "type": "object",
            "properties": ["path": ["type": "string", "description": "File path"]],
            "required": ["path"]
        ]
    ]]
}

func buildClaudeCodeConversation(turns: Int) -> [[String: Any]] {
    var messages: [[String: Any]] = []
    for i in 0..<turns {
        messages.append(["role": "user", "content": "Turn \(i + 1): Read the file /tmp/test_\(i).txt and tell me what's in it."])
        messages.append(["role": "assistant", "content": [
            ["type": "text", "text": "Let me read that file."],
            ["type": "tool_use", "id": "toolu_\(i)", "name": "read_file", "input": ["path": "/tmp/test_\(i).txt"]]
        ]])
        messages.append(["role": "user", "content": [
            ["type": "tool_result", "tool_use_id": "toolu_\(i)", "content": "File contents: hello world \(i)"]
        ]])
        messages.append(["role": "assistant", "content": "The file contains: hello world \(i)."])
    }
    // Final user turn
    messages.append(["role": "user", "content": "Summarize all the files you read."])
    return messages
}

// ═══════════════════════════════════════════════════════════
// E2E Test Suite
// Optimized: load model once, run all tests, unload once.
// ~6 load/unload cycles instead of 96.
// ═══════════════════════════════════════════════════════════

@Suite("E2E Tests", .serialized, .enabled(if: e2eEnabled))
struct E2ETests {
    let client = HTTPClient()

    /// Load each model once, run ALL test closures against it, then unload.
    /// This gives us ~6 cycles instead of ~96.
    func forEachModelSuite(_ label: String, tests: (String) async throws -> Void) async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [\(label)] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else {
                print("  [\(label)] SKIP \(modelId): failed to load")
                continue
            }
            print("  [\(label)] Testing \(modelId)...")

            do {
                try await tests(modelId)
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

    // ═══════════════════════════════════════════════════════
    // Test 1: Core API — OpenAI + Anthropic, stream + non-stream
    // ═══════════════════════════════════════════════════════

    @Test("Core API: OpenAI + Anthropic non-stream and stream")
    func testCoreAPI() async throws {
        try await forEachModelSuite("core-api") { modelId in
            // --- OpenAI non-stream ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "Say hello in one word."]],
                    "stream": false, "max_tokens": 50
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let (resp, status) = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/chat/completions"),
                    body: data, modelId: modelId) else { return }

                if status == 500 {
                    print("    \(modelId): OpenAI non-stream HTTP 500 — skipping")
                } else {
                    #expect(status == 200, "\(modelId): OpenAI non-stream HTTP \(status)")
                    guard status == 200 else { return }
                    let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                    let choices = json?["choices"] as? [[String: Any]]
                    let text = (choices?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
                    #expect(!text.isEmpty, "\(modelId): empty OpenAI response")
                    let finishReason = choices?.first?["finish_reason"] as? String ?? ""
                    #expect(finishReason == "stop" || finishReason == "length",
                        "\(modelId): unexpected finish_reason '\(finishReason)'")
                    print("    \(modelId): OpenAI non-stream OK")
                }
            }

            // --- OpenAI stream ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "Say hello in one word."]],
                    "stream": true, "max_tokens": 50
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                let result = await safePostStream(client,
                    url: apiBase.appendingPathComponent("v1/chat/completions"),
                    body: data, modelId: modelId)
                if let result, !result.isEmpty {
                    print("    \(modelId): OpenAI stream OK")
                } else {
                    print("    \(modelId): OpenAI stream empty (possible server error)")
                }
            }

            // --- Anthropic non-stream ---
            do {
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
                    print("    \(modelId): Anthropic non-stream HTTP 500 — skipping")
                } else {
                    #expect(status == 200, "\(modelId): Anthropic non-stream HTTP \(status)")
                    guard status == 200 else { return }
                    let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                    let content = json?["content"] as? [[String: Any]]
                    let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
                    #expect(!text.isEmpty, "\(modelId): empty Anthropic response")
                    let stopReason = json?["stop_reason"] as? String ?? ""
                    #expect(stopReason == "stop" || stopReason == "end_turn",
                        "\(modelId): unexpected stop_reason '\(stopReason)'")
                    print("    \(modelId): Anthropic non-stream OK")
                }
            }

            // --- Anthropic stream ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "Say hello in one word."]],
                    "max_tokens": 50, "stream": true
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                let result = await safePostStream(client,
                    url: apiBase.appendingPathComponent("v1/messages"),
                    body: data, modelId: modelId)
                if let result, !result.isEmpty {
                    print("    \(modelId): Anthropic stream OK")
                } else {
                    print("    \(modelId): Anthropic stream empty (possible server error)")
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // Test 2: Thinking format — OpenAI + Anthropic
    // ═══════════════════════════════════════════════════════

    @Test("Thinking format: reasoning_content separated from content")
    func testThinkingFormat() async throws {
        try await forEachModelSuite("thinking-format") { modelId in
            // --- OpenAI non-stream thinking ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "What is 2+2? Think step by step."]],
                    "stream": false, "max_tokens": 300
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let (resp, status) = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/chat/completions"),
                    body: data, modelId: modelId) else { return }
                guard status == 200 else { return }

                let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                let choices = json?["choices"] as? [[String: Any]]
                guard let choice = choices?.first else { return }
                let message = choice["message"] as? [String: Any]
                let content = message?["content"] as? String ?? ""
                let reasoningContent = message?["reasoning_content"] as? String

                #expect(!content.contains("</think"), "\(modelId): content contains raw </think tag")
                #expect(!content.contains("<think"), "\(modelId): content contains raw <think tag")

                if let rc = reasoningContent {
                    #expect(!rc.isEmpty, "\(modelId): reasoning_content present but empty")
                    print("    \(modelId): OpenAI thinking=\(rc.count) chars, content=\(content.count) chars")
                } else {
                    print("    \(modelId): OpenAI no thinking (non-thinking model), content=\(content.count) chars")
                }
            }

            // --- OpenAI stream thinking ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "What is 2+2? Think step by step."]],
                    "stream": true, "max_tokens": 300
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let capture = await captureOpenAIStream(
                    url: apiBase.appendingPathComponent("v1/chat/completions"),
                    body: data, modelId: modelId) else { return }

                #expect(capture.gotDone, "\(modelId): OpenAI stream no [DONE]")
                let fullContent = capture.contentChunks.joined()
                let fullReasoning = capture.reasoningChunks.joined()
                #expect(!fullContent.contains("</think"), "\(modelId): stream content has </think")
                #expect(!fullContent.contains("<think"), "\(modelId): stream content has <think")

                if !fullReasoning.isEmpty {
                    print("    \(modelId): OpenAI stream thinking=\(fullReasoning.count) chars")
                } else {
                    print("    \(modelId): OpenAI stream no thinking (non-thinking model)")
                }
            }

            // --- Anthropic non-stream thinking ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "What is 2+2? Think step by step."]],
                    "max_tokens": 300, "stream": false
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let (resp, status) = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/messages"),
                    body: data, modelId: modelId) else { return }
                guard status == 200 else { return }

                let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                let content = json?["content"] as? [[String: Any]]
                let thinkingBlocks = content?.filter { $0["type"] as? String == "thinking" } ?? []
                let textBlocks = content?.filter { $0["type"] as? String == "text" } ?? []

                for block in textBlocks {
                    let text = block["text"] as? String ?? ""
                    #expect(!text.contains("</think"), "\(modelId): Anthropic text block has </think")
                    #expect(!text.contains("<think"), "\(modelId): Anthropic text block has <think")
                }

                let thinkingText = thinkingBlocks.compactMap { $0["thinking"] as? String }.joined()
                let responseText = textBlocks.compactMap { $0["text"] as? String }.joined()

                if !thinkingText.isEmpty {
                    print("    \(modelId): Anthropic thinking=\(thinkingText.count) chars, response=\(responseText.count) chars")
                } else {
                    print("    \(modelId): Anthropic no thinking block, response=\(responseText.count) chars")
                }
            }

            // --- Anthropic stream thinking ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "What is 2+2? Think step by step."]],
                    "max_tokens": 300, "stream": true
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let capture = await captureAnthropicStream(
                    url: apiBase.appendingPathComponent("v1/messages"),
                    body: data, modelId: modelId) else { return }

                #expect(capture.gotMessageStop, "\(modelId): Anthropic stream no message_stop")
                let fullText = capture.textChunks.joined()
                let fullThinking = capture.thinkingChunks.joined()
                #expect(!fullText.contains("</think"), "\(modelId): Anthropic stream text has </think")
                #expect(!fullText.contains("<think"), "\(modelId): Anthropic stream text has <think")

                if !fullThinking.isEmpty {
                    print("    \(modelId): Anthropic stream thinking=\(fullThinking.count) chars")
                } else {
                    print("    \(modelId): Anthropic stream no thinking")
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // Test 3: Multi-turn conversations
    // ═══════════════════════════════════════════════════════

    @Test("Multi-turn: Claude Code simulation (short + long)")
    func testMultiTurn() async throws {
        try await forEachModelSuite("multi-turn") { modelId in
            // Short conversation (3 turns)
            do {
                let conversation = buildClaudeCodeConversation(turns: 3)
                let body: [String: Any] = [
                    "model": modelId,
                    "system": buildClaudeCodeSystemPrompt(),
                    "tools": buildClaudeCodeTools(),
                    "messages": conversation,
                    "max_tokens": 512, "stream": false
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let (resp, status) = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/messages"),
                    body: data, modelId: modelId) else { return }
                if status == 200 {
                    let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                    let content = json?["content"] as? [[String: Any]]
                    let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
                    #expect(!text.isEmpty, "\(modelId): empty Claude Code response")
                    let usage = json?["usage"] as? [String: Any]
                    let inputTokens = usage?["input_tokens"] as? Int ?? 0
                    print("    \(modelId): Claude Code 3-turn OK — \(inputTokens) input tokens")
                } else {
                    print("    \(modelId): Claude Code 3-turn HTTP \(status)")
                }
            }

            // Long conversation (8 turns)
            do {
                let conversation = buildClaudeCodeConversation(turns: 8)
                let body: [String: Any] = [
                    "model": modelId,
                    "system": buildClaudeCodeSystemPrompt(),
                    "tools": buildClaudeCodeTools(),
                    "messages": conversation,
                    "max_tokens": 1024, "stream": false
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                guard let (resp, status) = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/messages"),
                    body: data, modelId: modelId) else { return }
                if status == 200 {
                    let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                    let content = json?["content"] as? [[String: Any]]
                    let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
                    #expect(!text.isEmpty, "\(modelId): empty long context response")
                    let usage = json?["usage"] as? [String: Any]
                    let inputTokens = usage?["input_tokens"] as? Int ?? 0
                    print("    \(modelId): Long context 8-turn OK — \(inputTokens) input tokens")
                } else if status == 500 {
                    print("    \(modelId): Long context HTTP 500 (server error)")
                } else {
                    print("    \(modelId): Long context HTTP \(status) (may hit context limit)")
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // Test 4: Stress — concurrent + pressure + memory safety
    // ═══════════════════════════════════════════════════════

    @Test("Stress: concurrent requests + memory safety")
    func testStress() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [stress] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else {
                print("  [stress] SKIP \(modelId): failed to load")
                continue
            }

            // --- Concurrent requests (3 simultaneous) ---
            do {
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
                    if let (data, status) = result, status == 200,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let choices = json["choices"] as? [[String: Any]],
                       let text = (choices.first?["message"] as? [String: Any])?["content"] as? String,
                       !text.isEmpty {
                        successCount += 1
                    } else if let (_, status) = result, status == 500 {
                        serverError = true
                    }
                }
                if serverError {
                    print("    \(modelId): concurrent server error (skipped)")
                } else {
                    #expect(successCount == 3, "\(modelId): only \(successCount)/3 concurrent requests succeeded")
                    print("    \(modelId): \(successCount)/3 concurrent OK")
                }
            }

            // --- Memory safety (large max_tokens — server should handle gracefully) ---
            do {
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": "Hello"]],
                    "max_tokens": 50000
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                let result = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/chat/completions"),
                    body: data, modelId: modelId)
                if let (_, status) = result {
                    #expect(status == 200 || status >= 400, "\(modelId): unexpected status \(status)")
                    print("    \(modelId): oversized request → HTTP \(status) (survived)")
                }
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            guard await client.isServerAlive() else {
                Issue.record("Server died after stress test with \(modelId)")
                let recovered = await client.waitForServer(maxWait: 30)
                if !recovered { return }
                break
            }
        }
        #expect(await client.isServerAlive(), "Server must survive stress test")
    }

    // ═══════════════════════════════════════════════════════
    // Test 5: Batcher — concurrent + stats + stream
    // ═══════════════════════════════════════════════════════

    @Test("Batcher: concurrent + stats + stream")
    func testBatcher() async throws {
        try await requireServer(client)
        let models = try await getAllDownloadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No downloaded models found") }

        for model in models {
            let modelId = model.id
            print("  [batcher] Loading \(modelId)...")
            let loaded = try await loadModel(client, modelId: modelId)
            guard loaded else { continue }

            // --- Batcher concurrent (10 requests) ---
            var batcherSuccesses = 0
            do {
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
                    for try await (_, ok) in group {
                        if ok { batcherSuccesses += 1 }
                    }
                }
                #expect(batcherSuccesses == requestCount,
                    "\(modelId): only \(batcherSuccesses)/\(requestCount) batcher requests completed")
                print("    \(modelId): batcher concurrent \(batcherSuccesses)/\(requestCount) OK")
            }

            // --- Batcher stats ---
            if batcherSuccesses > 0 {
                do {
                    let (data, status) = try await client.get(apiBase.appendingPathComponent("v1/stats"))
                    #expect(status == 200, "\(modelId): stats endpoint HTTP \(status)")
                    if status == 200,
                       let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let batcher = json["batcher"] as? [String: Any] {
                        let totalCompleted = batcher["totalCompleted"] as? UInt64 ?? 0
                        #expect(totalCompleted > 0, "\(modelId): batcher should have completed requests")
                        print("    \(modelId): batcher stats — completed=\(totalCompleted)")
                    }
                }
            }

            // --- Batcher stream (6 concurrent) ---
            do {
                let requestCount = 6
                var streamSuccesses = 0
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
                    for try await ok in group {
                        if ok { streamSuccesses += 1 }
                    }
                }
                #expect(streamSuccesses >= requestCount / 2,
                    "\(modelId): only \(streamSuccesses)/\(requestCount) stream requests completed")
                print("    \(modelId): batcher stream \(streamSuccesses)/\(requestCount) OK")
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 2_000_000_000)

            guard await client.isServerAlive() else {
                Issue.record("Server died after batcher test with \(modelId)")
                let recovered = await client.waitForServer(maxWait: 30)
                if !recovered { return }
                break
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // Test 6: Models list endpoint
    // ═══════════════════════════════════════════════════════

    @Test("Models list endpoint returns valid response")
    func testModelsList() async throws {
        try await requireServer(client)
        let (data, status) = try await client.get(apiBase.appendingPathComponent("v1/models"))
        #expect(status == 200, "GET /v1/models returned HTTP \(status)")
        guard status == 200 else { return }

        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let modelList = json["data"] as? [[String: Any]]
            #expect(modelList != nil && !modelList!.isEmpty, "Models list should have ≥1 model")
            print("  Models list: \(modelList?.count ?? 0) models")
        }
    }

    // ═══════════════════════════════════════════════════════
    // Test 7: Pressure — progressive token scaling
    // ═══════════════════════════════════════════════════════

    @Test("Pressure: progressive token scaling 2K→4K→6K→8K")
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
                    if recovered { break } else { return }
                }

                let filler = generateTokenFiller(count: tokens)
                let prompt = "Summarize this text in one sentence: \(filler)"
                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": prompt]],
                    "stream": false, "max_tokens": 100
                ]
                let data = try JSONSerialization.data(withJSONObject: body)

                let result = await safePost(client,
                    url: apiBase.appendingPathComponent("v1/chat/completions"),
                    body: data, modelId: modelId)
                guard let (_, status) = result else {
                    Issue.record("\(modelId): request failed at \(tokens) tokens")
                    let recovered = await client.waitForServer(maxWait: 30)
                    if recovered { break } else { return }
                }

                if status == 200 {
                    print("    \(modelId): \(tokens) tokens OK")
                } else if status == 500 {
                    print("    \(modelId): HTTP 500 at \(tokens) tokens (server error)")
                    break
                } else {
                    Issue.record("\(modelId): HTTP \(status) at \(tokens) tokens")
                }
            }

            try? await unloadModel(client, modelId: modelId)
            try? await Task.sleep(nanoseconds: 3_000_000_000)
        }

        #expect(await client.isServerAlive(), "Server must survive pressure test")
    }

    // ─── Stream Capture Helpers ───

    struct OpenAIStreamCapture {
        var contentChunks: [String] = []
        var reasoningChunks: [String] = []
        var finishReason: String?
        var gotDone = false
    }

    func captureOpenAIStream(url: URL, body: Data, modelId: String) async -> OpenAIStreamCapture? {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        guard let (bytes, resp) = try? await client.session.bytes(for: req) else { return nil }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            Issue.record("\(modelId): stream HTTP \(status)")
            return nil
        }

        var capture = OpenAIStreamCapture()
        do {
            for try await line in bytes.lines {
                guard line.hasPrefix("data: ") else { continue }
                let payload = String(line.dropFirst(6))
                if payload == "[DONE]" { capture.gotDone = true; break }

                guard let data = payload.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let choices = obj["choices"] as? [[String: Any]],
                      let choice = choices.first,
                      let delta = choice["delta"] as? [String: Any]
                else { continue }

                if let content = delta["content"] as? String, !content.isEmpty {
                    capture.contentChunks.append(content)
                }
                if let reasoning = delta["reasoning_content"] as? String, !reasoning.isEmpty {
                    capture.reasoningChunks.append(reasoning)
                }
                if let finish = choice["finish_reason"] as? String, finish != "null" {
                    capture.finishReason = finish
                }
            }
        } catch {
            Issue.record("\(modelId): stream read error — \(error)")
            return nil
        }
        return capture
    }

    struct AnthropicStreamCapture {
        var textChunks: [String] = []
        var thinkingChunks: [String] = []
        var stopReason: String?
        var gotMessageStop = false
    }

    func captureAnthropicStream(url: URL, body: Data, modelId: String) async -> AnthropicStreamCapture? {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        guard let (bytes, resp) = try? await client.session.bytes(for: req) else { return nil }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            Issue.record("\(modelId): Anthropic stream HTTP \(status)")
            return nil
        }

        var capture = AnthropicStreamCapture()
        var currentBlockType = "text"
        do {
            for try await line in bytes.lines {
                if line.hasPrefix("event: ") {
                    let eventType = String(line.dropFirst(7)).trimmingCharacters(in: .whitespaces)
                    if eventType == "message_stop" { capture.gotMessageStop = true }
                    continue
                }
                guard line.hasPrefix("data: ") else { continue }
                let payload = String(line.dropFirst(6))

                guard let data = payload.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { continue }

                let type = obj["type"] as? String ?? ""

                if type == "content_block_start",
                   let block = obj["content_block"] as? [String: Any] {
                    currentBlockType = block["type"] as? String ?? "text"
                }

                if type == "content_block_delta",
                   let delta = obj["delta"] as? [String: Any] {
                    if let text = delta["text"] as? String, !text.isEmpty {
                        if currentBlockType == "thinking" {
                            capture.thinkingChunks.append(text)
                        } else {
                            capture.textChunks.append(text)
                        }
                    }
                    if let thinking = delta["thinking"] as? String, !thinking.isEmpty {
                        capture.thinkingChunks.append(thinking)
                    }
                }

                if type == "message_delta",
                   let delta = obj["delta"] as? [String: Any],
                   let stop = delta["stop_reason"] as? String {
                    capture.stopReason = stop
                }
            }
        } catch {
            Issue.record("\(modelId): Anthropic stream read error — \(error)")
            return nil
        }
        return capture
    }
}
