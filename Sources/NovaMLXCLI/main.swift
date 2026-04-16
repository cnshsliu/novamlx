import Foundation

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
            print("Is NovaMLX server running? Start it first with: NovaMLX")
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
}

await NovaMLXCLI.run()
