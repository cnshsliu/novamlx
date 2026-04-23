import SwiftUI
import NovaMLXCore

struct AgentInfo: Identifiable {
    let id: String
    let displayName: String
    let icon: String
    let description: String
    let binaryNames: [String]
    let installURL: String
    var isInstalled: Bool = false
    var resolvedPath: String? = nil

    static let builtIn: [AgentInfo] = [
        AgentInfo(
            id: "openclaw",
            displayName: "OpenClaw",
            icon: "brain.head.profile.fill",
            description: "Open-source AI agent framework with plugin system and tool use",
            binaryNames: ["openclaw"],
            installURL: "https://github.com/openclaw/openclaw"
        ),
        AgentInfo(
            id: "hermes",
            displayName: "Hermes Agent",
            icon: "bolt.horizontal.fill",
            description: "Autonomous AI agent with multi-step reasoning and tool execution",
            binaryNames: ["hermes"],
            installURL: "https://github.com/hermes-agent/hermes"
        ),
        AgentInfo(
            id: "opencode",
            displayName: "OpenCode",
            icon: "terminal.fill",
            description: "Terminal-based AI programming assistant with code editing",
            binaryNames: ["opencode"],
            installURL: "https://github.com/opencode-ai/opencode"
        ),
    ]
}

struct AgentsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    @EnvironmentObject var l10n: L10n

    @State private var agents: [AgentInfo] = AgentInfo.builtIn
    @State private var isRefreshing = false
    @State private var launchMessage: String? = nil
    @State private var launchError: String? = nil

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                overviewSection
                serverStatusSection
                ForEach(agents) { agent in
                    agentCard(agent: agent)
                }
            }
            .padding(24)
        }
        .background(NovaTheme.Colors.background)
        .onAppear { checkInstalledAgents() }
    }

    // MARK: - Overview

    private var overviewSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(l10n.tr("agents.integrations"), icon: "app.badge.checkmark")
            Text(l10n.tr("agents.description"))
                .font(.system(size: 12))
                .foregroundColor(NovaTheme.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .background(NovaTheme.Colors.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.lg)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
        )
    }

    // MARK: - Server Status

    private var serverStatusSection: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                .frame(width: 8, height: 8)

            if appState.isServerRunning {
                Text(l10n.tr("agents.serverRunning", "http://127.0.0.1:\(appState.serverPort)"))
                    .font(.system(size: 12))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
            } else {
                Text(l10n.tr("agents.serverNotRunning"))
                    .font(.system(size: 12))
                    .foregroundColor(NovaTheme.Colors.statusError)
            }

            Spacer()

            Button(action: { checkInstalledAgents() }) {
                Image(systemName: isRefreshing ? "arrow.clockwise" : "arrow.clockwise")
                    .font(.system(size: 11))
                    .rotationEffect(.degrees(isRefreshing ? 360 : 0))
                    .animation(isRefreshing ? .linear(duration: 0.6).repeatForever(autoreverses: false) : .default, value: isRefreshing)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help(l10n.tr("agents.refreshStatus"))
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.lg)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
        )
    }

    // MARK: - Agent Card

    private func agentCard(agent: AgentInfo) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack(spacing: 10) {
                Image(systemName: agent.icon)
                    .font(.system(size: 18))
                    .foregroundColor(NovaTheme.Colors.accent)
                    .frame(width: 32, height: 32)
                    .background(NovaTheme.Colors.accentDim)
                    .clipShape(RoundedRectangle(cornerRadius: 6))

                VStack(alignment: .leading, spacing: 2) {
                    Text(agent.displayName)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(NovaTheme.Colors.textPrimary)
                    Text(agent.description)
                        .font(.system(size: 11))
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                }

                Spacer()

                if agent.isInstalled {
                    Text(l10n.tr("agents.installed"))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(NovaTheme.Colors.statusOK)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(NovaTheme.Colors.statusOK.opacity(0.12))
                        .clipShape(Capsule())
                } else {
                    Text(l10n.tr("agents.notFound"))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(NovaTheme.Colors.statusWarn.opacity(0.12))
                        .clipShape(Capsule())
                }
            }

            // Config preview
            configPreview(for: agent.id)

            // Action buttons
            HStack(spacing: 8) {
                if agent.isInstalled {
                    Button(action: { launchAgent(agent.id) }) {
                        Label(l10n.tr("agents.launch"), systemImage: "play.fill")
                            .font(.system(size: 11))
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(!appState.isServerRunning)
                } else {
                    Button(action: {
                        if let url = URL(string: agent.installURL) {
                            NSWorkspace.shared.open(url)
                        }
                    }) {
                        Label(l10n.tr("agents.install"), systemImage: "arrow.down.circle")
                            .font(.system(size: 11))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Button(action: {
                    copyConfig(for: agent.id)
                }) {
                    Label(l10n.tr("agents.copyConfig"), systemImage: "doc.on.doc")
                        .font(.system(size: 11))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                if let path = agent.resolvedPath {
                    Text(path)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                Spacer()

                if let msg = launchMessage, msg.hasPrefix(agent.id) {
                    Text(msg)
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.statusOK)
                }
            }

            if let err = launchError, err.hasPrefix(agent.id) {
                Text(err)
                    .font(.system(size: 10))
                    .foregroundColor(NovaTheme.Colors.statusError)
            }
        }
        .padding(16)
        .background(NovaTheme.Colors.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.lg)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
        )
    }

    // MARK: - Config Preview

    private func configPreview(for agentId: String) -> some View {
        let configText = generateConfigText(for: agentId)
        return VStack(alignment: .leading, spacing: 4) {
            Text(l10n.tr("agents.configLabel"))
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(NovaTheme.Colors.textTertiary)
            Text(configText)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textSecondary)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(NovaTheme.Colors.rowBackground)
                .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.sm))
        }
    }

    // MARK: - Helpers

    private func generateConfigText(for agentId: String) -> String {
        let port = appState.serverPort
        let key = appState.apiKey ?? "(none)"

        switch agentId {
        case "openclaw":
            return """
            // ~/.openclaw/openclaw.json
            {
              "provider": {
                "api": "openai-completions",
                "baseUrl": "http://127.0.0.1:\(port)/v1",
                "apiKey": "\(key == "(none)" ? "" : key)"
              }
            }
            """
        case "hermes":
            return """
            # ~/.hermes/config.yaml
            provider: openai-compatible
            base_url: http://127.0.0.1:\(port)/v1
            api_key: \(key)
            model: default
            """
        case "opencode":
            let keyField = key == "(none)" ? "" : ",\"apiKey\":\"\(key)\""
            return """
            OPENCODE_CONFIG_CONTENT=
            {"providers":{"novamlx":{"type":"openai-compatible","baseURL":"http://127.0.0.1:\(port)/v1"\(keyField)}},"defaultProvider":"novamlx"}
            """
        default:
            return ""
        }
    }

    private func copyConfig(for agentId: String) {
        let text = generateConfigText(for: agentId)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        launchMessage = l10n.tr("agents.configCopied", agentId)
        launchError = nil
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            if self.launchMessage == self.l10n.tr("agents.configCopied", agentId) {
                self.launchMessage = nil
            }
        }
    }

    private func checkInstalledAgents() {
        isRefreshing = true
        for i in agents.indices {
            let path = findBinary(named: agents[i].binaryNames.first ?? agents[i].id)
            agents[i].isInstalled = path != nil
            agents[i].resolvedPath = path
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            self.isRefreshing = false
        }
    }

    private func findBinary(named name: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]
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
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let fallbacks = ["/usr/local/bin/\(name)", "/opt/homebrew/bin/\(name)", "\(home)/.local/bin/\(name)"]
        return fallbacks.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    private func launchAgent(_ agentId: String) {
        guard let idx = agents.firstIndex(where: { $0.id == agentId }),
              let binaryPath = agents[idx].resolvedPath else { return }

        let port = appState.serverPort
        let apiKey = appState.apiKey
        let baseURL = "http://127.0.0.1:\(port)/v1"

        do {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: binaryPath)

            var env = ProcessInfo.processInfo.environment

            if agentId == "opencode" {
                var keyField = ""
                if let apiKey { keyField = ",\"apiKey\":\"\(apiKey)\"" }
                let configStr = "{\"providers\":{\"novamlx\":{\"type\":\"openai-compatible\",\"baseURL\":\"http://127.0.0.1:\(port)/v1\"\(keyField)}},\"defaultProvider\":\"novamlx\"}"
                env["OPENCODE_CONFIG_CONTENT"] = configStr
            }

            process.environment = env
            process.standardOutput = FileHandle.standardOutput
            process.standardError = FileHandle.standardError

            if agentId == "openclaw" {
                try writeOpenClawConfig(baseURL: baseURL, apiKey: apiKey)
            } else if agentId == "hermes" {
                try writeHermesConfig(baseURL: baseURL, apiKey: apiKey)
            }

            try process.run()
            launchMessage = l10n.tr("agents.launched", agentId)
            launchError = nil
        } catch {
            launchError = l10n.tr("agents.launchError", agentId, error.localizedDescription)
            launchMessage = nil
        }
    }

    private func writeOpenClawConfig(baseURL: String, apiKey: String?) throws {
        let configDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".openclaw")
        try FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        let configPath = configDir.appendingPathComponent("openclaw.json")
        var config: [String: Any] = [:]
        if let data = try? Data(contentsOf: configPath),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            config = existing
        }

        var provider: [String: Any] = (config["provider"] as? [String: Any]) ?? [:]
        provider["api"] = "openai-completions"
        provider["baseUrl"] = baseURL
        if let apiKey { provider["apiKey"] = apiKey }
        config["provider"] = provider

        let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: configPath, options: .atomic)
    }

    private func writeHermesConfig(baseURL: String, apiKey: String?) throws {
        let configDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".hermes")
        try FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        let configPath = configDir.appendingPathComponent("config.yaml")
        var yaml = "provider: openai-compatible\n"
        yaml += "base_url: \(baseURL)\n"
        if let apiKey { yaml += "api_key: \(apiKey)\n" }
        yaml += "model: default\n"
        try yaml.write(to: configPath, atomically: true, encoding: .utf8)
    }
}

// MARK: - Reusable Section Header

private func sectionHeader(_ title: String, icon: String) -> some View {
    HStack(spacing: 8) {
        Image(systemName: icon)
            .font(.system(size: 14))
            .foregroundColor(NovaTheme.Colors.accent)
        Text(title)
            .font(.system(size: 15, weight: .semibold))
            .foregroundColor(NovaTheme.Colors.textPrimary)
    }
}
