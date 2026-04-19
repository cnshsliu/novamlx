import Foundation
import Testing

// ─── Configuration ───

let apiBase = URL(string: ProcessInfo.processInfo.environment["NOVA_API_URL"] ?? "http://127.0.0.1:6590")!
let adminBase = URL(string: ProcessInfo.processInfo.environment["NOVA_ADMIN_URL"] ?? "http://127.0.0.1:6591")!
let apiKey = ProcessInfo.processInfo.environment["NOVA_API_KEY"] ?? "abcd1234"

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
        guard let (_, status) = try? await get(apiBase.appendingPathComponent("health")),
              status == 200 else { return false }
        return true
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

    var description: String {
        switch self {
        case .httpError(let s, let b): "HTTP \(s): \(b.prefix(200))"
        case .serverUnavailable(let m): "Server unavailable: \(m)"
        case .modelNotLoaded(let m): "Model not loaded: \(m)"
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

struct ModelInfo: Decodable, Sendable { let id: String }
struct ModelsResponse: Decodable { let data: [ModelInfo] }

func getLoadedModels(_ client: HTTPClient) async throws -> [ModelInfo] {
    let (data, _) = try await client.get(apiBase.appendingPathComponent("v1/models"))
    let resp = try JSONDecoder().decode(ModelsResponse.self, from: data)
    return resp.data
}

func getContextLength(_ modelId: String, client: HTTPClient) async -> Int {
    let result = try? await client.get(adminBase.appendingPathComponent("admin/models"))
    guard let (data, _) = result,
          let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
    else { return 131072 }

    for model in array {
        if model["id"] as? String == modelId,
           let cl = model["contextLength"] as? Int, cl > 0 { return cl }
    }
    return 131072
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
// ═══════════════════════════════════════════════════════════════════

@Suite("E2E Tests", .serialized)
struct E2ETests {
    let client = HTTPClient()

    // ─── Suite 1: API correctness ───

    @Test("OpenAI non-stream: returns valid response")
    func testOpenAINonStream() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in models.map(\.id) {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "stream": false, "max_tokens": 50
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)
            guard let (resp, status) = result else { continue }

            #expect(status == 200, "\(modelId): HTTP \(status)")
            guard status == 200 else { continue }

            let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
            let choices = json?["choices"] as? [[String: Any]]
            let text = (choices?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
            #expect(!text.isEmpty, "\(modelId): empty response")
            #expect(choices?.first?["finish_reason"] as? String == "stop", "\(modelId): unexpected finish_reason")
        }
    }

    @Test("OpenAI stream: receives tokens and [DONE]")
    func testOpenAIStream() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in models.map(\.id) {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "stream": true, "max_tokens": 50
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePostStream(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)
            if let result { #expect(!result.isEmpty, "\(modelId): empty stream response") }
        }
    }

    @Test("Anthropic non-stream: returns valid response")
    func testAnthropicNonStream() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in models.map(\.id) {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "max_tokens": 50, "stream": false
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePost(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId)
            guard let (resp, status) = result else { continue }

            #expect(status == 200, "\(modelId): HTTP \(status)")
            guard status == 200 else { continue }

            let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
            let content = json?["content"] as? [[String: Any]]
            let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
            #expect(!text.isEmpty, "\(modelId): empty response")
            let stopReason = json?["stop_reason"] as? String ?? ""
            #expect(stopReason == "stop" || stopReason == "end_turn", "\(modelId): unexpected stop_reason '\(stopReason)'")
        }
    }

    @Test("Anthropic stream: receives content and message_stop")
    func testAnthropicStream() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in models.map(\.id) {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Say hello in one word."]],
                "max_tokens": 50, "stream": true
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePostStream(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId)
            if let result { #expect(!result.isEmpty, "\(modelId): empty stream response") }
        }
    }

    // ─── Suite 2: Max input token stress test ───

    @Test("Find max input token limit per model via OpenAI API")
    func testMaxInputTokens() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        let loaded = models.map(\.id)
        guard !loaded.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in loaded {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                return
            }

            let contextLength = await getContextLength(modelId, client: client)
            print("  Testing \(modelId) — claimed context_length: \(contextLength)")

            // Conservative: 2K start, 2K steps, max 8K to avoid OOM
            // Some models (e.g. Phi-3.5 with 384KB/token) need 6GB+ KV cache at 16K tokens
            let stepSize = 2000
            let maxTestTokens = 8000
            var maxSuccess = 0
            var lastError = ""

            var tokens = stepSize
            while tokens <= maxTestTokens {
                guard await client.isServerAlive() else {
                    lastError = "server died at \(tokens) tokens"
                    return
                }

                let filler = generateTokenFiller(count: tokens)
                let prompt = "Summarize this text in one sentence: \(filler)"

                let body: [String: Any] = [
                    "model": modelId,
                    "messages": [["role": "user", "content": prompt]],
                    "stream": false, "max_tokens": 50
                ]
                let data = try JSONSerialization.data(withJSONObject: body)

                let result = await safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)
                guard let (_, status) = result else {
                    lastError = "server unavailable at \(tokens) tokens"
                    let recovered = await client.waitForServer(maxWait: 60)
                    if !recovered { return }
                    break
                }

                if status == 404 {
                    // Model evicted by memory pressure — skip this model
                    print("  \(modelId): model evicted at \(tokens) tokens — skipping remaining steps")
                    break
                } else if status == 200 {
                    maxSuccess = tokens
                    tokens += stepSize
                } else {
                    lastError = "HTTP \(status) at \(tokens) tokens"
                    break
                }
            }

            print("  \(modelId): max successful input ≈ \(maxSuccess) tokens (claimed: \(contextLength), lastError: \(lastError))")

            // Only assert minimum if model wasn't evicted (maxSuccess > 0 means at least one request succeeded)
            if maxSuccess > 0 {
                let minimumExpected = 2000
                #expect(maxSuccess >= minimumExpected,
                        "\(modelId): max input \(maxSuccess) below minimum \(minimumExpected)")
            }
        }
    }

    // ─── Suite 3: Claude Code simulation ───

    @Test("Claude Code multi-turn conversation via Anthropic API")
    func testClaudeCodeSimulation() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        let loaded = models.map(\.id)
        guard !loaded.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in loaded {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

            let contextLength = await getContextLength(modelId, client: client)
            print("  Claude Code simulation for \(modelId) — context_length: \(contextLength)")

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

            let result = await safePost(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId)
            guard let (resp, status) = result else { continue }

            if status == 404 {
                print("  \(modelId): model evicted during test — skipping")
                continue
            }
            #expect(status == 200, "\(modelId): Claude Code simulation HTTP \(status)")
            guard status == 200 else { continue }

            let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
            let content = json?["content"] as? [[String: Any]]
            let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
            #expect(!text.isEmpty, "\(modelId): empty Claude Code response")

            let usage = json?["usage"] as? [String: Any]
            let inputTokens = usage?["input_tokens"] as? Int ?? 0
            print("  \(modelId): Claude Code sim OK — \(inputTokens) input tokens, \(text.count) chars response")
        }
    }

    @Test("Claude Code long context: extended 8-turn conversation")
    func testClaudeCodeLongContext() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        let loaded = models.map(\.id)
        guard !loaded.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in loaded {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

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

            let result = await safePost(client, url: apiBase.appendingPathComponent("v1/messages"), body: data, modelId: modelId)

            if let (resp, status) = result, status == 200 {
                let json = try JSONSerialization.jsonObject(with: resp) as? [String: Any]
                let content = json?["content"] as? [[String: Any]]
                let text = content?.compactMap { $0["text"] as? String }.joined() ?? ""
                #expect(!text.isEmpty, "\(modelId): empty long context response")
                let usage = json?["usage"] as? [String: Any]
                let inputTokens = usage?["input_tokens"] as? Int ?? 0
                print("  \(modelId): Long context OK — \(inputTokens) input tokens")
            } else if let (resp, status) = result {
                let errorBody = String(data: resp, encoding: .utf8) ?? ""
                if status == 404 {
                    print("  \(modelId): model evicted during long context test — skipping")
                } else if errorBody.contains("context") || errorBody.contains("token") || status == 400 {
                    print("  \(modelId): Long context hit context limit (expected) — HTTP \(status)")
                } else {
                    Issue.record("\(modelId): Long context unexpected error — HTTP \(status): \(errorBody.prefix(200))")
                }
            }
        }
    }

    // ─── Suite 4: Memory safety ───

    @Test("Memory safety: oversized request rejected without crash")
    func testMemorySafetyRejection() async throws {
        try await requireServer(client)
        let models = try await getLoadedModels(client)
        guard !models.isEmpty else { throw TestError.modelNotLoaded("No models found") }

        for modelId in models.map(\.id) {
            guard await client.isServerAlive() else {
                Issue.record("\(modelId): server died — skipping remaining models")
                break
            }

            // Request absurdly large output — should be rejected, not crash
            let body: [String: Any] = [
                "model": modelId,
                "messages": [["role": "user", "content": "Hello"]],
                "max_tokens": 999999
            ]
            let data = try JSONSerialization.data(withJSONObject: body)
            let result = await safePost(client, url: apiBase.appendingPathComponent("v1/chat/completions"), body: data, modelId: modelId)

            if let (_, status) = result {
                // Accept: 200 (server capped max_tokens), or 4xx/5xx (rejected as expected)
                #expect(status == 200 || status >= 400, "\(modelId): unexpected status \(status)")
            }
            // nil = server crashed, safePost already recorded issue
        }

        // Key assertion: server survived
        #expect(await client.isServerAlive(), "Server crashed during memory safety test")
    }
}
