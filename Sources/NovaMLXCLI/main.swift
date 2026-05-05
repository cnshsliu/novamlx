import Foundation
import NovaMLXCore

struct NovaMLXCLI {
    static func run() async {
        let args = CommandLine.arguments.dropFirst()
        guard let command = args.first else {
            printUsage()
            return
        }

        let subArgs = Array(args.dropFirst())

        do {
            switch command {
            case "models":
                try await handleModels(subArgs)
            case "download":
                try await handleDownload(subArgs)
            case "load":
                try await handleLoad(subArgs)
            case "unload":
                try await handleUnload(subArgs)
            case "delete":
                try await handleDelete(subArgs)
            case "search":
                try await handleSearch(subArgs)
            case "status":
                try await handleStatus()
            case "chat":
                try await handleChat(subArgs)
            case "config":
                try await handleConfig(subArgs)
            case "sessions":
                try await handleSessions(subArgs)
            case "cache":
                try await handleCache(subArgs)
            case "adapters":
                try await handleAdapters(subArgs)
            case "turboquant", "tq":
                try await handleTurboQuant(subArgs)
            case "bench":
                try await handleBench(subArgs)
            case "launch":
                try await LaunchCommand.handleLaunch(subArgs)
            case "login":
                try await handleLogin()
            case "logout":
                handleLogout()
            case "account":
                try await handleAccount()
            case "chat-template":
                try await handleChatTemplate(subArgs)
            case "--help", "-h":
                printUsage()
            case "--version", "-v":
                print("NovaMLX CLI v1.0.0")
            default:
                print("Unknown command: \(command)")
                print("Run 'nova --help' for usage.")
            }
        } catch {
            print("Error: \(error.localizedDescription)")
            if !(error is AuthError) {
                print("Is NovaMLX server running? Start it first with: NovaMLX")
            }
        }
    }

    static func printUsage() {
        print("""
        NovaMLX CLI — Manage your local LLM inference server

        USAGE
          nova <command> [arguments]

        MODEL MANAGEMENT
          nova models                     List loaded models
          nova search <query>             Search HuggingFace for models
          nova download <model-id>        Download a model
          nova load <model-id>            Load a model into memory
          nova unload <model-id>          Unload a model from memory
          nova delete <model-id>          Delete a downloaded model

        INFERENCE
          nova chat [model-id]            Open interactive chat
          nova status                     Show server status & GPU memory

        CONFIGURATION
          nova config list                Show current configuration
          nova config set <key> <value>   Set a config value
          nova turboquant [model-id]      Show KV quantization status
          nova turboquant <id> <bits>     Enable KV quantization (2/4/8)
          nova turboquant <id> off        Disable KV quantization

        SESSIONS & CACHE
          nova sessions                   List active sessions
          nova sessions delete <id>       Delete a session
          nova cache <model-id>           Show cache stats
          nova cache <model-id> clear     Clear cache for model

        ADAPTERS (LoRA)
          nova adapters                   List loaded adapters
          nova adapters load <path>       Load a LoRA adapter
          nova adapters unload <name>     Unload an adapter

        BENCHMARKING
          nova bench start [model-id]     Run benchmark
          nova bench status               Show benchmark progress

        AGENTS
          nova launch list                List available agents
          nova launch openclaw            Launch OpenClaw with NovaMLX backend
          nova launch hermes              Launch Hermes Agent with NovaMLX backend
          nova launch opencode            Launch OpenCode with NovaMLX backend

        CLOUD AUTH
          nova login                      Login to NovaMLX Cloud
          nova logout                     Logout from NovaMLX Cloud
          nova account                    Show account & subscription status

        DIAGNOSTICS
          nova chat-template diagnose <model-id>   Inspect chat template & rendering health

        OPTIONS
          --help, -h     Show this help
          --version, -v  Show version
        """)
    }

    // MARK: - Models

    static func handleModels(_ args: [String]) async throws {
        let resp = try await CLIClient.get("/v1/models")
        CLIClient.printJSON(resp.body)
    }

    static func handleSearch(_ args: [String]) async throws {
        guard let query = args.first else {
            print("Usage: nova search <query>")
            return
        }
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let resp = try await CLIClient.get("/admin/api/hf/search?q=\(encoded)", admin: true)
        CLIClient.printJSON(resp.body)
    }

    static func handleDownload(_ args: [String]) async throws {
        guard let modelId = args.first else {
            print("Usage: nova download <model-id>")
            return
        }
        print("Downloading \(modelId)...")
        let resp = try await CLIClient.post("/admin/models/download", body: "{\"modelId\":\"\(modelId)\"}", admin: true)
        if resp.statusCode == 200 {
            print("Download started.")
        } else {
            print("Failed: \(resp.statusCode) — \(resp.body)")
        }
    }

    static func handleLoad(_ args: [String]) async throws {
        guard let modelId = args.first else {
            print("Usage: nova load <model-id>")
            return
        }
        print("Loading \(modelId)...")
        let resp = try await CLIClient.post("/admin/models/load", body: "{\"modelId\":\"\(modelId)\"}", admin: true)
        if resp.statusCode == 200 {
            print("Model loaded.")
        } else {
            print("Failed: \(resp.statusCode) — \(resp.body)")
        }
    }

    static func handleUnload(_ args: [String]) async throws {
        guard let modelId = args.first else {
            print("Usage: nova unload <model-id>")
            return
        }
        let resp = try await CLIClient.post("/admin/models/unload", body: "{\"modelId\":\"\(modelId)\"}", admin: true)
        if resp.statusCode == 200 {
            print("Model unloaded.")
        } else {
            print("Failed: \(resp.statusCode) — \(resp.body)")
        }
    }

    static func handleDelete(_ args: [String]) async throws {
        guard let modelId = args.first else {
            print("Usage: nova delete <model-id>")
            return
        }
        let encoded = modelId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? modelId
        let resp = try await CLIClient.delete("/admin/models/\(encoded)", admin: true)
        if resp.statusCode == 200 {
            print("Model deleted.")
        } else {
            print("Failed: \(resp.statusCode) — \(resp.body)")
        }
    }

    // MARK: - Status

    static func handleStatus() async throws {
        let resp = try await CLIClient.get("/health")
        CLIClient.printJSON(resp.body)
    }

    // MARK: - Chat

    static func handleChat(_ args: [String]) async throws {
        let modelId = args.first
        let modelsResp = try await CLIClient.get("/v1/models")
        guard let data = modelsResp.body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let models = json["data"] as? [[String: Any]]
        else {
            print("Could not parse models list.")
            return
        }

        guard !models.isEmpty else {
            print("No models loaded. Download and load one first:")
            print("  nova download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
            print("  nova load mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
            return
        }

        let selectedModel: String
        if let modelId {
            selectedModel = modelId
        } else if let first = models.first?["id"] as? String {
            selectedModel = first
        } else {
            print("No model available.")
            return
        }

        print("Chatting with \(selectedModel). Type 'exit' to quit.\n")

        while true {
            print("You> ", terminator: "")
            guard let input = readLine(), input != "exit", !input.isEmpty else { break }

            let body = """
            {"model":"\(selectedModel)","messages":[{"role":"user","content":"\(input.replacingOccurrences(of: "\"", with: "\\\""))"}],"stream":false}
            """
            let resp = try await CLIClient.post("/v1/chat/completions", body: body)

            if let rData = resp.body.data(using: .utf8),
               let rJson = try? JSONSerialization.jsonObject(with: rData) as? [String: Any],
               let choices = rJson["choices"] as? [[String: Any]],
               let message = choices.first?["message"] as? [String: Any],
               let content = message["content"] as? String {
                print("AI> \(content)\n")
            } else {
                print("AI> (no response)\n")
            }
        }
    }

    // MARK: - Config

    static func handleConfig(_ args: [String]) async throws {
        guard let sub = args.first else {
            print("Usage: nova config list | set <key> <value>")
            return
        }
        switch sub {
        case "list":
            let resp = try await CLIClient.get("/admin/api/device-info", admin: true)
            CLIClient.printJSON(resp.body)
        case "set":
            print("Config set is not yet available via CLI. Use the admin API directly.")
        default:
            print("Unknown config subcommand: \(sub)")
        }
    }

    // MARK: - Sessions

    static func handleSessions(_ args: [String]) async throws {
        guard let sub = args.first else {
            let resp = try await CLIClient.get("/admin/sessions", admin: true)
            CLIClient.printJSON(resp.body)
            return
        }
        switch sub {
        case "delete":
            guard let id = args.dropFirst().first else {
                print("Usage: nova sessions delete <session-id>")
                return
            }
            let resp = try await CLIClient.delete("/admin/sessions/\(id)", admin: true)
            print(resp.statusCode == 200 ? "Session deleted." : "Failed: \(resp.statusCode)")
        default:
            print("Unknown subcommand: \(sub)")
        }
    }

    // MARK: - Cache

    static func handleCache(_ args: [String]) async throws {
        guard let modelId = args.first else {
            print("Usage: nova cache <model-id> [clear]")
            return
        }
        let encoded = modelId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? modelId
        let sub = args.dropFirst().first
        if sub == "clear" {
            let resp = try await CLIClient.delete("/admin/cache/\(encoded)", admin: true)
            print(resp.statusCode == 200 ? "Cache cleared." : "Failed: \(resp.statusCode)")
        } else {
            let resp = try await CLIClient.get("/admin/cache/\(encoded)/stats", admin: true)
            CLIClient.printJSON(resp.body)
        }
    }

    // MARK: - Adapters

    static func handleAdapters(_ args: [String]) async throws {
        guard let sub = args.first else {
            let resp = try await CLIClient.get("/admin/adapters", admin: true)
            CLIClient.printJSON(resp.body)
            return
        }
        switch sub {
        case "load":
            guard let path = args.dropFirst().first else {
                print("Usage: nova adapters load <path>")
                return
            }
            let resp = try await CLIClient.post("/admin/adapters/load", body: "{\"path\":\"\(path)\"}", admin: true)
            print(resp.statusCode == 200 ? "Adapter loaded." : "Failed: \(resp.statusCode) — \(resp.body)")
        case "unload":
            guard let name = args.dropFirst().first else {
                print("Usage: nova adapters unload <name>")
                return
            }
            let resp = try await CLIClient.post("/admin/adapters/unload", body: "{\"name\":\"\(name)\"}", admin: true)
            print(resp.statusCode == 200 ? "Adapter unloaded." : "Failed: \(resp.statusCode)")
        default:
            print("Unknown subcommand: \(sub)")
        }
    }

    // MARK: - TurboQuant

    static func handleTurboQuant(_ args: [String]) async throws {
        guard let modelId = args.first else {
            let resp = try await CLIClient.get("/admin/api/turboquant", admin: true)
            CLIClient.printJSON(resp.body)
            return
        }
        let encoded = modelId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? modelId
        let value = args.dropFirst().first

        if value == "off" {
            let resp = try await CLIClient.delete("/admin/api/turboquant/\(encoded)", admin: true)
            print(resp.statusCode == 200 ? "TurboQuant disabled for \(modelId)." : "Failed: \(resp.statusCode)")
        } else if let bits = value, let _ = Int(bits) {
            let resp = try await CLIClient.put("/admin/api/turboquant/\(encoded)", body: "{\"bits\":\(bits),\"groupSize\":64}", admin: true)
            print(resp.statusCode == 200 ? "TurboQuant enabled (\(bits)-bit) for \(modelId)." : "Failed: \(resp.statusCode) — \(resp.body)")
        } else {
            let resp = try await CLIClient.get("/admin/api/turboquant", admin: true)
            CLIClient.printJSON(resp.body)
        }
    }

    // MARK: - Bench

    static func handleBench(_ args: [String]) async throws {
        guard let sub = args.first else {
            print("Usage: nova bench start [model-id] | status")
            return
        }
        switch sub {
        case "start":
            let modelId = args.dropFirst().first
            let body = modelId.map { "{\"modelId\":\"\($0)\"}" }
            let resp = try await CLIClient.post("/admin/api/bench/start", body: body, admin: true)
            print(resp.statusCode == 200 ? "Benchmark started." : "Failed: \(resp.statusCode) — \(resp.body)")
        case "status":
            let resp = try await CLIClient.get("/admin/api/bench/status", admin: true)
            CLIClient.printJSON(resp.body)
        case "cancel":
            let resp = try await CLIClient.post("/admin/api/bench/cancel", admin: true)
            print(resp.statusCode == 200 ? "Benchmark cancelled." : "Failed: \(resp.statusCode)")
        default:
            print("Unknown subcommand: \(sub)")
        }
    }

    // MARK: - Cloud Auth

    static func handleLogin() async throws {
        // If already logged in, show current status
        if let session = AuthCache.loadSession(), !session.isEmpty {
            print("Already logged in. Checking subscription...")
            do {
                let client = AuthClient()
                let response = try await client.checkSession(session)
                if response.valid {
                    print("Logged in as: \(response.user?.email ?? "unknown")")
                    print("Plan: \(response.plan?.uppercased() ?? "FREE")")
                    if let expires = response.expiresAt {
                        print("Expires: \(expires.prefix(10))")
                    }
                    print("\nAlready logged in. Use 'nova logout' first to switch accounts.")
                    return
                }
            } catch AuthError.sessionExpired {
                print("Previous session expired. Please login again.\n")
                AuthCache.clearAll()
            } catch {
                // Fall through to login prompt
            }
        }

        print("NovaMLX Cloud Login")
        print("Email: ", terminator: "")
        guard let email = readLine(), !email.isEmpty else {
            print("Email is required.")
            return
        }
        print("Password: ", terminator: "")
        guard let password = readLine(), !password.isEmpty else {
            print("Password is required.")
            return
        }

        let client = AuthClient()
        let response = try await client.login(email: email, password: password)

        // Save session
        try AuthCache.saveSession(response.session)

        // Check subscription
        let check = try await client.checkSession(response.session)
        let cache = AuthCache(
            valid: check.valid,
            plan: check.plan ?? "free",
            status: check.status ?? "none",
            cancelAtPeriodEnd: check.cancelAtPeriodEnd ?? false,
            expiresAt: check.expiresAt,
            cachedAt: Date().timeIntervalSince1970,
            userEmail: response.user.email
        )
        try cache.save()

        print("")
        if check.valid {
            print("Login successful! Plan: \(check.plan?.uppercased() ?? "FREE")", terminator: "")
            if let expires = check.expiresAt {
                print(" (expires \(expires.prefix(10)))")
            } else {
                print("")
            }
        } else {
            print("Login successful, but no active subscription.")
            print("Visit https://novamlx.com/cloud to subscribe.")
        }
    }

    static func handleLogout() {
        if AuthCache.loadSession() == nil {
            print("Not logged in.")
            return
        }
        AuthCache.clearAll()
        print("Logged out.")
    }

    static func handleAccount() async throws {
        guard let session = AuthCache.loadSession() else {
            print("Not logged in. Use 'nova login' first.")
            return
        }

        print("Checking subscription status...\n")
        let client = AuthClient()
        let response = try await client.checkSession(session)

        // Update cache
        let cache = AuthCache(
            valid: response.valid,
            plan: response.plan ?? "free",
            status: response.status ?? "none",
            cancelAtPeriodEnd: response.cancelAtPeriodEnd ?? false,
            expiresAt: response.expiresAt,
            cachedAt: Date().timeIntervalSince1970,
            userEmail: response.user?.email ?? ""
        )
        try cache.save()

        if response.valid {
            print("User: \(response.user?.email ?? "unknown")")
            if let name = response.user?.name { print("Name: \(name)") }
            print("Plan: \(response.plan?.uppercased() ?? "FREE")")
            print("Status: \(response.status ?? "unknown")")
            if let expires = response.expiresAt {
                print("Expires: \(expires.prefix(10))")
            }
            if response.cancelAtPeriodEnd == true {
                print("\nWarning: Subscription cancelled. Will expire at end of billing period.")
            }
            print("\nCloud inference: Available")
        } else {
            print("User: \(response.user?.email ?? "unknown")")
            print("Plan: \(response.plan?.uppercased() ?? "FREE")")
            print("Status: \(response.status ?? "none")")
            print("\nCloud inference: Not available")
            print("Subscribe at: https://novamlx.com/cloud")
        }
    }

    // MARK: - Chat-template diagnostics

    static func handleChatTemplate(_ args: [String]) async throws {
        guard let sub = args.first else {
            print("Usage: nova chat-template diagnose <model-id>")
            return
        }
        switch sub {
        case "diagnose":
            guard args.count >= 2 else {
                print("Usage: nova chat-template diagnose <model-id>")
                return
            }
            let modelId = args[1]
            let encoded = modelId.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? modelId
            let resp = try await CLIClient.get("/admin/api/chat-template/diagnose/\(encoded)", admin: true)
            renderChatTemplateReport(Data(resp.body.utf8))
        case "--help", "-h":
            print("Subcommands:")
            print("  diagnose <model-id>  Inspect chat template files, format detection, and family config")
        default:
            print("Unknown chat-template subcommand: \(sub)")
        }
    }

    /// Pretty-printer for the JSON report from /admin/api/chat-template/diagnose.
    /// Falls back to raw JSON if the structure is unexpected.
    private static func renderChatTemplateReport(_ data: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            CLIClient.printJSON(String(data: data, encoding: .utf8) ?? "")
            return
        }
        func s(_ k: String) -> String { (json[k] as? String) ?? "-" }
        func i(_ k: String) -> Int { (json[k] as? Int) ?? 0 }
        func b(_ k: String) -> Bool { (json[k] as? Bool) ?? false }
        func arr(_ k: String) -> [String] { (json[k] as? [String]) ?? [] }

        let modelId = s("modelId")
        let dir = json["modelDir"] as? String ?? "(not on disk)"
        let family = s("family")
        let archs = arr("architectures").joined(separator: ", ")
        let mt = s("modelType")
        let label = s("familyConfigLabel")

        print("Chat-Template Diagnostic Report")
        print(String(repeating: "─", count: 60))
        print("Model:           \(modelId)")
        print("Directory:       \(dir)")
        print("Family:          \(family)  (\(label))")
        print("Architecture:    \(archs.isEmpty ? "-" : archs)")
        print("model_type:      \(mt)")
        print("")

        // Template file state
        let tcLen = i("tokenizerConfigChatTemplateLength")
        let tcExc = s("tokenizerConfigChatTemplateExcerpt")
        let jinjaPresent = b("chatTemplateJinjaPresent")
        let jinjaLen = i("chatTemplateJinjaLength")
        let quarantined = b("chatTemplateJinjaQuarantined")
        let agree = b("templatesAgree")
        let regOverride = json["registryOverrideName"] as? String

        print("Template files on disk")
        print("  tokenizer_config.json.chat_template:  \(tcLen) chars")
        if !tcExc.isEmpty {
            let firstLine = tcExc.split(separator: "\n", maxSplits: 0, omittingEmptySubsequences: true).first ?? ""
            print("    excerpt: \(firstLine)…")
        }
        print("  chat_template.jinja:                  \(jinjaPresent ? "\(jinjaLen) chars" : "absent")")
        print("  chat_template.jinja.disabled-by-novamlx (quarantine): \(quarantined ? "yes" : "no")")
        print("  chat_template.json:                   \(b("chatTemplateJsonPresent") ? "present" : "absent")")
        print("  templates agree:                      \(agree ? "yes" : "NO — risk of corruption")")
        if let reg = regOverride {
            print("  registry override:                    \(reg) (NovaMLX-maintained template will be applied)")
        } else {
            print("  registry override:                    none")
        }
        print("")

        // Format detection
        if let detected = json["detectedFormats"] as? [[String: Any]], !detected.isEmpty {
            print("Detected formats (highest confidence first):")
            for d in detected {
                let f = d["format"] as? String ?? "?"
                let c = d["confidence"] as? Int ?? 0
                let ev = (d["evidence"] as? [String]) ?? []
                print("  • \(f)  ×\(c)  evidence: \(ev.joined(separator: ", "))")
            }
        } else {
            print("Detected formats: none recognized (template may use a brand-new format)")
        }
        print("")

        // Family interpretation
        print("Family interpretation (from registry)")
        print("  Expected EOS tokens:    \(arr("expectedEosTokens").joined(separator: ", "))")
        print("  Hallucination patterns: \(arr("hallucinationPatterns").joined(separator: ", "))")
        print("  Leakage blacklist:      \(arr("leakageBlacklist").joined(separator: ", "))")
        print("  Thinking markers:       \(arr("thinkingMarkers").joined(separator: ", "))")
        print("  preserve_thinking:      \(b("preserveThinkingSupported") ? "supported" : "not supported")")
        print("")

        // Health
        let healthy = b("healthy")
        let issues = arr("healthIssues")
        if healthy {
            print("Status: HEALTHY ✓")
        } else {
            print("Status: \(issues.count) issue(s) found")
            for issue in issues {
                print("  ! \(issue)")
            }
            print("")
            print("See `architecture.md` → Diagnostic Playbook for fix paths.")
        }
    }
}

await NovaMLXCLI.run()
