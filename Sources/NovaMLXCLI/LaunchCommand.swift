import Foundation

struct AgentDescriptor {
    let name: String
    let displayName: String
    let binaryNames: [String]
    let description: String
    let installURL: String

    static let allAgents: [AgentDescriptor] = [
        AgentDescriptor(
            name: "openclaw",
            displayName: "OpenClaw",
            binaryNames: ["openclaw"],
            description: "Open-source AI agent framework with plugin system",
            installURL: "https://github.com/openclaw/openclaw"
        ),
        AgentDescriptor(
            name: "hermes",
            displayName: "Hermes Agent",
            binaryNames: ["hermes"],
            description: "Autonomous AI agent with tool use and multi-step reasoning",
            installURL: "https://github.com/hermes-agent/hermes"
        ),
        AgentDescriptor(
            name: "opencode",
            displayName: "OpenCode",
            binaryNames: ["opencode"],
            description: "Terminal-based AI programming assistant",
            installURL: "https://github.com/opencode-ai/opencode"
        ),
    ]

    func findBinary() -> String? {
        for name in binaryNames {
            if let path = which(name) { return path }
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let fallbackDirs = ["/usr/local/bin", "/opt/homebrew/bin", "\(home)/.local/bin"]
        for dir in fallbackDirs {
            for name in binaryNames {
                let path = "\(dir)/\(name)"
                if FileManager.default.isExecutableFile(atPath: path) { return path }
            }
        }
        return nil
    }

    private func which(_ binary: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [binary]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
            if process.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
                if let path, !path.isEmpty { return path }
            }
        } catch {}
        return nil
    }
}

enum LaunchCommand {
    static func handleLaunch(_ args: [String]) async throws {
        guard !args.isEmpty else {
            print("Usage: nova launch <agent>")
            print("Agents: openclaw, hermes, opencode")
            print("Run 'nova launch list' to see available agents.")
            return
        }

        if args[0] == "list" {
            handleList()
            return
        }

        let agentName = args[0]
        guard let agent = AgentDescriptor.allAgents.first(where: { $0.name == agentName }) else {
            print("Unknown agent: \(agentName)")
            print("Available agents: \(AgentDescriptor.allAgents.map(\.name).joined(separator: ", "))")
            return
        }

        guard let binaryPath = agent.findBinary() else {
            print("\(agent.displayName) is not installed.")
            print("Install it from: \(agent.installURL)")
            return
        }

        // Verify NovaMLX server is running
        let ports = CLIClient.portFromFile()
        let baseURL = "http://127.0.0.1:\(ports.inference)"
        do {
            _ = try await CLIClient.get("/health")
        } catch {
            print("NovaMLX server is not running at \(baseURL)")
            print("Start it first with: NovaMLX")
            return
        }

        let apiKey = CLIClient.resolveAPIKey()
        try launchAgent(agent, binaryPath: binaryPath, host: "127.0.0.1", port: ports.inference, apiKey: apiKey)
    }

    static func handleList() {
        print("Available Agent Integrations:\n")
        for agent in AgentDescriptor.allAgents {
            let path = agent.findBinary()
            let status = path != nil ? "✓ installed (\(path!))" : "✗ not installed"
            let padding = String(repeating: " ", count: max(1, 16 - agent.displayName.count))
            print("  \(agent.displayName)\(padding)\(status)")
            print("  \(String(repeating: " ", count: 16))\(agent.description)")
            print()
        }
        print("Launch an agent with: nova launch <agent-name>")
    }

    private static func launchAgent(
        _ agent: AgentDescriptor,
        binaryPath: String,
        host: String,
        port: Int,
        apiKey: String?
    ) throws {
        let baseURL = "http://\(host):\(port)/v1"

        switch agent.name {
        case "openclaw":
            try launchOpenClaw(binaryPath: binaryPath, baseURL: baseURL, apiKey: apiKey)
        case "hermes":
            try launchHermes(binaryPath: binaryPath, baseURL: baseURL, apiKey: apiKey)
        case "opencode":
            try launchOpenCode(binaryPath: binaryPath, baseURL: baseURL, apiKey: apiKey)
        default:
            print("No launch handler for \(agent.displayName)")
        }
    }

    private static func launchOpenClaw(binaryPath: String, baseURL: String, apiKey: String?) throws {
        let configDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".openclaw")
        try FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        var config: [String: Any] = [:]
        let configPath = configDir.appendingPathComponent("openclaw.json")

        // Read existing config to preserve user settings
        if let data = try? Data(contentsOf: configPath),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            config = existing
        }

        // Set/update NovaMLX provider
        var provider: [String: Any] = (config["provider"] as? [String: Any]) ?? [:]
        provider["api"] = "openai-completions"
        provider["baseUrl"] = baseURL
        if let apiKey { provider["apiKey"] = apiKey }
        config["provider"] = provider

        let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: configPath, options: .atomic)

        print("Configured OpenClaw → NovaMLX at \(baseURL)")
        print("Launching \(binaryPath) ...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = []
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
    }

    private static func launchHermes(binaryPath: String, baseURL: String, apiKey: String?) throws {
        let configDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".hermes")
        try FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        let configPath = configDir.appendingPathComponent("config.yaml")
        var yaml = "provider: openai-compatible\n"
        yaml += "base_url: \(baseURL)\n"
        if let apiKey { yaml += "api_key: \(apiKey)\n" }
        yaml += "model: default\n"

        try yaml.write(to: configPath, atomically: true, encoding: .utf8)

        print("Configured Hermes Agent → NovaMLX at \(baseURL)")
        print("Launching \(binaryPath) ...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = []
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
    }

    private static func launchOpenCode(binaryPath: String, baseURL: String, apiKey: String?) throws {
        // OpenCode uses inline JSON config via OPENCODE_CONFIG_CONTENT env var
        var configObj: [String: Any] = [
            "providers": [
                "novamlx": [
                    "type": "openai-compatible",
                    "baseURL": baseURL,
                ] as [String: Any]
            ],
            "defaultProvider": "novamlx",
        ]
        if let apiKey {
            if var providers = configObj["providers"] as? [String: [String: Any]] {
                providers["novamlx"]?["apiKey"] = apiKey
                configObj["providers"] = providers
            }
        }

        let configData = try JSONSerialization.data(withJSONObject: configObj, options: [.sortedKeys])
        let configStr = String(data: configData, encoding: .utf8)!

        print("Configured OpenCode → NovaMLX at \(baseURL)")
        print("Launching \(binaryPath) ...")

        var env = ProcessInfo.processInfo.environment
        env["OPENCODE_CONFIG_CONTENT"] = configStr

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = []
        process.environment = env
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
    }
}
