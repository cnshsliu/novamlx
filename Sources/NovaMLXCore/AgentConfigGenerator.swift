import Foundation
import AppKit
import os.log

public enum AgentConfigGenerator {
    private static let log = Logger(subsystem: "com.novamlx", category: "AgentConfig")

    // MARK: - CLI Command Generation

    public static func generateCLICommand(
        agent: AgentSpec,
        modelName: String,
        port: Int = 6590,
        apiKey: String? = nil
    ) -> String {
        let baseURL = "http://127.0.0.1:\(port)/v1"
        switch agent.configFormat {
        case .codex:
            return "codex --model \(modelName)"
        case .hermes:
            let key = apiKey ?? "$NOVA_API_KEY"
            return "HERMES_API_KEY=\(key) hermes --model \(modelName) --base-url \(baseURL)"
        case .opencode:
            let key = apiKey ?? "your-api-key"
            return "OPENCODE_API_KEY=\(key) opencode --provider custom --base-url \(baseURL) --model \(modelName)"
        case .claudeCode:
            let key = apiKey ?? "$NOVA_API_KEY"
            return "ANTHROPIC_API_KEY=\(key) ANTHROPIC_BASE_URL=\(baseURL) claude"
        case .curl:
            return generateCurlSnippet(baseURL: baseURL, apiKey: apiKey, modelName: modelName)
        case .python:
            return generatePythonSnippet(baseURL: baseURL, apiKey: apiKey, modelName: modelName)
        case .node:
            return generateNodeSnippet(baseURL: baseURL, apiKey: apiKey, modelName: modelName)
        }
    }

    // MARK: - Config Generation (auto-selects by agent type)

    @discardableResult
    public static func generateConfig(
        agent: AgentSpec,
        providers: [TokenhubProvider],
        port: Int = 6590,
        apiKey: String? = nil,
        modelName: String? = nil
    ) -> String? {
        switch agent.configFormat {
        case .codex:
            return generateCodexConfig(providers: providers, port: port, apiKey: apiKey, modelName: modelName ?? "tknet")
        case .hermes:
            let model = modelName ?? "tknet"
            return generateHermesConfig(baseURL: "http://127.0.0.1:\(port)/v1", apiKey: apiKey, model: model)
        case .opencode:
            let provider = modelName ?? "tknet"
            return generateOpenCodeConfig(baseURL: "http://127.0.0.1:\(port)/v1", apiKey: apiKey, model: provider)
        case .claudeCode, .curl, .python, .node:
            return nil
        }
    }

    // MARK: - Codex Config

    @discardableResult
    public static func generateCodexConfig(
        providers: [TokenhubProvider],
        port: Int = 6590,
        apiKey: String? = nil,
        modelName: String = "tknet"
    ) -> String? {
        let enabledProviders = providers.filter { $0.isEnabled }
        guard !enabledProviders.isEmpty else {
            log.warning("[AgentConfig] No enabled providers, skipping Codex config generation")
            return nil
        }

        let codexDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".codex")
        let catalogPath = codexDir.appendingPathComponent("model_catalog.json")
        let configPath = codexDir.appendingPathComponent("config.toml")

        do {
            try FileManager.default.createDirectory(at: codexDir, withIntermediateDirectories: true)
        } catch {
            log.error("[AgentConfig] Failed to create .codex directory: \(error)")
            return "Failed to create .codex directory"
        }

        var warning: String?

        // Generate model catalog
        let lbProviders = enabledProviders.filter { $0.includeInLoadBalance }
        let (lbCtx, mixed) = ModelSpecs.lbContextWindow(from: enabledProviders)
        if mixed {
            let minK = lbCtx / 1024
            let maxK = (enabledProviders.map { $0.effectiveContextWindow }.max() ?? lbCtx) / 1024
            warning = "LB pool has mixed context sizes (\(minK)K~\(maxK)K), using minimum: \(minK)K"
        }

        var models: [[String: Any]] = []

        // LB profile
        if !lbProviders.isEmpty {
            models.append(codexModelEntry(
                slug: "tknet",
                displayName: "TKNet (Load Balanced)",
                description: "NovaMLX TokenHub load-balanced model",
                contextWindow: lbCtx
            ))
        }

        // Per-provider profiles
        for provider in enabledProviders {
            let slug = "tknet:" + provider.id
            let ctx = provider.effectiveContextWindow
            models.append(codexModelEntry(
                slug: slug,
                displayName: provider.name,
                description: "\(provider.name) via NovaMLX TokenHub",
                contextWindow: ctx
            ))
        }

        let catalog: [String: Any] = ["models": models]

        do {
            let catalogData = try JSONSerialization.data(withJSONObject: catalog, options: [.prettyPrinted, .sortedKeys])
            try catalogData.write(to: catalogPath, options: .atomic)
            log.info("[AgentConfig] Wrote Codex model_catalog.json with \(models.count) models")
        } catch {
            log.error("[AgentConfig] Failed to write model_catalog.json: \(error)")
            return "Failed to write model_catalog.json"
        }

        // Update config.toml
        do {
            try updateCodexTOML(configPath: configPath, catalogPath: catalogPath.path, modelName: modelName)
        } catch {
            log.error("[AgentConfig] Failed to update config.toml: \(error)")
            return "Failed to update config.toml"
        }

        return warning
    }

    private static func codexModelEntry(
        slug: String,
        displayName: String,
        description: String,
        contextWindow: Int
    ) -> [String: Any] {
        return [
            "slug": slug,
            "display_name": displayName,
            "description": description,
            "supported_reasoning_levels": [
                ["effort": "none", "description": "No reasoning"],
                ["effort": "low", "description": "Low reasoning effort"],
                ["effort": "medium", "description": "Medium reasoning effort"],
                ["effort": "high", "description": "High reasoning effort"],
            ],
            "default_reasoning_level": "medium",
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 1,
            "availability_nux": NSNull(),
            "upgrade": NSNull(),
            "base_instructions": "You are a helpful coding agent.",
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": NSNull(),
            "apply_patch_tool_type": NSNull(),
            "truncation_policy": [
                "mode": "tokens",
                "limit": Int(Double(contextWindow) * 0.96),
            ],
            "supports_parallel_tool_calls": true,
            "supports_image_detail_original": false,
            "context_window": contextWindow,
            "max_context_window": contextWindow,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        ]
    }

    private static func updateCodexTOML(configPath: URL, catalogPath: String, modelName: String = "tknet") throws {
        var lines: [String] = []
        if FileManager.default.fileExists(atPath: configPath.path) {
            let content = try String(contentsOf: configPath, encoding: .utf8)
            lines = content.components(separatedBy: "\n")
        }

        // NovaMLX-related top-level keys to remove and re-write
        let novamlxTopLevelKeys: Set<String> = [
            "model_catalog_json",
            "model",
            "model_provider",
        ]

        // Remove existing NovaMLX top-level keys
        lines.removeAll { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            for key in novamlxTopLevelKeys {
                if trimmed.hasPrefix(key + " =") || trimmed.hasPrefix(key + "=") {
                    return true
                }
            }
            return false
        }

        // Remove existing [model_providers.novamlx] section (and comment line above it)
        var filtered: [String] = []
        var skipMode = false
        for (_, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("[model_providers") {
                skipMode = true
                // Also remove the "# NovaMLX" comment line immediately before this section
                if !filtered.isEmpty {
                    let prevIdx = filtered.count - 1
                    if filtered[prevIdx].trimmingCharacters(in: .whitespaces).hasPrefix("# NovaMLX") {
                        filtered.removeLast()
                        // Also remove blank line before the comment
                        if !filtered.isEmpty && filtered.last?.trimmingCharacters(in: .whitespaces).isEmpty == true {
                            filtered.removeLast()
                        }
                    }
                }
                continue
            }
            if skipMode && trimmed.hasPrefix("[") && !trimmed.hasPrefix("#") {
                skipMode = false
            }
            if !skipMode {
                filtered.append(line)
            }
        }
        lines = filtered

        // Clean trailing blank lines
        while let last = lines.last, last.trimmingCharacters(in: .whitespaces).isEmpty {
            lines.removeLast()
        }

        // Find insertion point: before first [section]
        var insertIdx = lines.count
        for (i, line) in lines.enumerated() {
            if line.hasPrefix("[") {
                insertIdx = i
                break
            }
        }

        // Build NovaMLX top-level config block
        var novaBlock: [String] = []
        novaBlock.append("model_catalog_json = \"\(catalogPath)\"")
        novaBlock.append("model = \"\(modelName)\"")
        novaBlock.append("model_provider = \"novamlx\"")

        // Insert before first section with a blank line separator
        if insertIdx > 0 && !lines[insertIdx - 1].isEmpty {
            novaBlock.insert("", at: 0)
        }
        for (offset, line) in novaBlock.enumerated() {
            lines.insert(line, at: insertIdx + offset)
        }

        // Ensure blank line before the provider section
        if lines.last?.isEmpty == false { lines.append("") }
        lines.append("")
        lines.append("# NovaMLX TokenHub provider — routes tknet:xxx through local NovaMLX proxy")
        lines.append("[model_providers.novamlx]")
        lines.append("name = \"NovaMLX TokenHub\"")
        lines.append("base_url = \"http://127.0.0.1:6590/v1\"")
        lines.append("env_key = \"NOVA_API_KEY\"")
        lines.append("wire_api = \"responses\"")

        try lines.joined(separator: "\n").write(to: configPath, atomically: true, encoding: .utf8)
        log.info("[AgentConfig] Updated Codex config.toml with model=\(modelName)")
    }

    // MARK: - Hermes Config

    @discardableResult
    public static func generateHermesConfig(
        baseURL: String,
        apiKey: String?,
        model: String
    ) -> String? {
        let hermesDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".hermes")
        let configPath = hermesDir.appendingPathComponent("config.yaml")

        do {
            try FileManager.default.createDirectory(at: hermesDir, withIntermediateDirectories: true)
        } catch {
            return "Failed to create .hermes directory"
        }

        var yaml = "providers:\n"
        yaml += "  novamlx:\n"
        yaml += "    base_url: \(baseURL)\n"
        if let key = apiKey, !key.isEmpty {
            yaml += "    api_key: \(key)\n"
        }
        yaml += "    model: \(model)\n"
        yaml += "default_provider: novamlx\n"

        do {
            try yaml.write(to: configPath, atomically: true, encoding: .utf8)
            log.info("[AgentConfig] Wrote Hermes config.yaml")
        } catch {
            return "Failed to write Hermes config"
        }
        return nil
    }

    // MARK: - OpenCode Config

    @discardableResult
    public static func generateOpenCodeConfig(
        baseURL: String,
        apiKey: String?,
        model: String
    ) -> String? {
        let opencodeDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".config/opencode")
        let configPath = opencodeDir.appendingPathComponent("config.json")

        do {
            try FileManager.default.createDirectory(at: opencodeDir, withIntermediateDirectories: true)
        } catch {
            return "Failed to create opencode config directory"
        }

        var config: [String: Any] = [
            "provider": "custom",
            "base_url": baseURL,
            "model": model,
        ]
        if let key = apiKey, !key.isEmpty {
            config["api_key"] = key
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: config, options: .prettyPrinted)
            try data.write(to: configPath, options: .atomic)
            log.info("[AgentConfig] Wrote OpenCode config.json")
        } catch {
            return "Failed to write OpenCode config"
        }
        return nil
    }

    // MARK: - App Launch

    public static func launchApp(agent: AgentSpec) -> Bool {
        guard let appName = agent.appName else { return false }

        let workspace = NSWorkspace.shared
        let apps = workspace.urlsForApplications(withBundleIdentifier: "com.apple.TextEdit")

        // Try to find the app by name
        let appPath = "/Applications/\(appName).app"
        if FileManager.default.fileExists(atPath: appPath) {
            workspace.open(URL(fileURLWithPath: appPath))
            return true
        }

        // Fallback: try NSWorkspace search
        if let url = workspace.urlForApplication(withBundleIdentifier: "openai.codex") {
            workspace.open(url)
            return true
        }

        log.warning("[AgentConfig] Could not find app: \(appName)")
        return false
    }

    // MARK: - Codex Process Management

    /// Check if Codex app is currently running.
    public static func isCodexRunning() -> Bool {
        let runningApps = NSWorkspace.shared.runningApplications
        return runningApps.contains { app in
            app.bundleIdentifier == "openai.codex"
        }
    }

    /// Terminate the Codex app process (graceful then forced after 3s).
    public static func terminateCodex() {
        let runningApps = NSWorkspace.shared.runningApplications
        for app in runningApps where app.bundleIdentifier == "openai.codex" {
            app.terminate()
            break
        }
        // Give it 3 seconds to quit gracefully, then force
        DispatchQueue.global().asyncAfter(deadline: .now() + 3.0) {
            let stillRunning = NSWorkspace.shared.runningApplications
            for app in stillRunning where app.bundleIdentifier == "openai.codex" {
                app.forceTerminate()
                break
            }
        }
    }

    /// Launch Codex app (or restart if already running).
    /// - Parameters:
    ///   - agent: The agent spec (used to find app name)
    ///   - forceRestart: If true, kill existing process before launching
    ///   - completion: Called on main thread with success boolean
    public static func launchOrRestartCodex(agent: AgentSpec, forceRestart: Bool, completion: @escaping @Sendable (Bool) -> Void) {
        if forceRestart {
            terminateCodex()
            // Wait for the process to fully exit, then launch
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                let success = launchApp(agent: agent)
                completion(success)
            }
        } else {
            let success = launchApp(agent: agent)
            completion(success)
        }
    }

    // MARK: - Code Snippet Generators

    private static func generateCurlSnippet(baseURL: String, apiKey: String?, modelName: String) -> String {
        let key = apiKey ?? "your-api-key"
        return """
        curl \(baseURL)/chat/completions \\
          -H "Content-Type: application/json" \\
          -H "Authorization: Bearer \(key)" \\
          -d '{
            "model": "\(modelName)",
            "messages": [
              {"role": "user", "content": "Hello!"}
            ],
            "stream": false
          }'
        """
    }

    private static func generatePythonSnippet(baseURL: String, apiKey: String?, modelName: String) -> String {
        let key = apiKey ?? "your-api-key"
        return """
        from openai import OpenAI

        client = OpenAI(
            base_url="\(baseURL)",
            api_key="\(key)"
        )

        response = client.chat.completions.create(
            model="\(modelName)",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
        """
    }

    private static func generateNodeSnippet(baseURL: String, apiKey: String?, modelName: String) -> String {
        let key = apiKey ?? "your-api-key"
        return """
        import OpenAI from 'openai';

        const client = new OpenAI({
          baseURL: '\(baseURL)',
          apiKey: '\(key)'
        });

        const response = await client.chat.completions.create({
          model: '\(modelName)',
          messages: [{ role: 'user', content: 'Hello!' }]
        });
        console.log(response.choices[0].message.content);
        """
    }
}
