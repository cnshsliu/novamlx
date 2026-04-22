import SwiftUI
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

struct SettingsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @State private var turboQuantConfigs: [String: TQConfig] = [:]
    @State private var sessions: [SessionInfo] = []
    @State private var cacheStats: [String: String] = [:]
    @State private var cliInstalled: Bool = false
    @State private var cliInstallMessage: String? = nil

    // Config editor state
    @State private var showConfigEditor = false
    @State private var cfgHost = "127.0.0.1"
    @State private var cfgPort = 6590
    @State private var cfgAdminPort = 6591
    @State private var cfgApiKeys = ""
    @State private var cfgMaxConcurrent = 16
    @State private var cfgTimeout = 300
    @State private var cfgMaxBodyMB = 100.0
    @State private var cfgDefaultModel = ""
    @State private var cfgSaveMessage: String? = nil
    @State private var cfgHasUnsavedChanges = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                serverConfigSection
                cliSection
                turboQuantSection
                sessionsSection
                aboutSection
            }
            .padding(24)
        }
    }

    // MARK: - Server Config

    private var serverConfigSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                sectionHeader("Server", icon: "server.rack")
                Spacer()
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showConfigEditor.toggle() } }) {
                    Image(systemName: showConfigEditor ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                    Text("Edit Config")
                        .font(.system(size: 11))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            settingsRow("Inference API", value: "http://127.0.0.1:\(String(appState.serverPort))")
            settingsRow("Admin API", value: "http://127.0.0.1:\(String(appState.adminPort))")
            settingsRow("Web Chat", value: "http://127.0.0.1:\(String(appState.serverPort))/chat")
            settingsRow("Admin Dashboard", value: "http://127.0.0.1:\(String(appState.adminPort))/admin/dashboard")

            configPathRow

            if showConfigEditor {
                Divider().padding(.vertical, 4)
                configEditorPanel
            }

            HStack {
                Button("Open Chat in Browser") {
                    NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(String(appState.serverPort))/chat")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Open Dashboard") {
                    NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(String(appState.adminPort))/admin/dashboard")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Spacer()

                Button("Open Config File") {
                    let path = FileManager.default.homeDirectoryForCurrentUser
                        .appendingPathComponent(".nova/config.json").path
                    NSWorkspace.shared.open(URL(string: "file://\(path)")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(16)
        .sectionCard()
        .task { loadCurrentConfig() }
    }

    private var configPathRow: some View {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let configPath = homeDir.appendingPathComponent(".nova/config.json").path

        return HStack {
            Text("Config File").font(.system(size: 13)).foregroundColor(.secondary)
            Spacer()
            Text(configPath)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.accent)
                .lineLimit(1)
                .truncationMode(.middle)
                .help("Click to copy path")
                .onTapGesture {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(configPath, forType: .string)
                }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var configEditorPanel: some View {
        VStack(spacing: 10) {
            Text("Server Configuration")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 12) {
                configField("Host", text: $cfgHost, width: 140, placeholder: "127.0.0.1")
                configFieldInt("Port", value: $cfgPort, width: 80, range: 1...65535)
                configFieldInt("Admin Port", value: $cfgAdminPort, width: 80, range: 1...65535)
            }

            HStack(spacing: 12) {
                configFieldInt("Max Concurrent", value: $cfgMaxConcurrent, width: 80, range: 1...128)
                configFieldInt("Timeout (s)", value: $cfgTimeout, width: 80, range: 10...3600)
                configFieldDouble("Max Body (MB)", value: $cfgMaxBodyMB, width: 100, range: 1...1024)
            }

            HStack(spacing: 12) {
                configField("Default Model", text: $cfgDefaultModel, width: nil, placeholder: "e.g. mlx-community/Qwen3-35B-4bit")
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("API Keys (one per line)")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                TextEditor(text: $cfgApiKeys)
                    .font(.system(size: 12, design: .monospaced))
                    .frame(height: 60)
                    .padding(4)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.gray.opacity(0.3)))
            }

            HStack {
                if let msg = cfgSaveMessage {
                    Text(msg)
                        .font(.system(size: 11))
                        .foregroundColor(msg.contains("Error") ? .red : NovaTheme.Colors.statusOK)
                }
                Spacer()
                Button("Reset to Current") { loadCurrentConfig() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                Button("Save to Disk") { saveConfig() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func configField(_ label: String, text: Binding<String>, width: CGFloat?, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.system(size: 11)).foregroundColor(.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))
                .frame(width: width)
        }
    }

    private func configFieldInt(_ label: String, value: Binding<Int>, width: CGFloat, range: ClosedRange<Int>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.system(size: 11)).foregroundColor(.secondary)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))
                .frame(width: width)
        }
    }

    private func configFieldDouble(_ label: String, value: Binding<Double>, width: CGFloat, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.system(size: 11)).foregroundColor(.secondary)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))
                .frame(width: width)
        }
    }

    private func loadCurrentConfig() {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let configPath = homeDir.appendingPathComponent(".nova/config.json")

        guard let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        if let server = json["server"] as? [String: Any] {
            cfgHost = server["host"] as? String ?? "127.0.0.1"
            cfgPort = server["port"] as? Int ?? 6590
            cfgAdminPort = server["adminPort"] as? Int ?? 6591
            cfgMaxConcurrent = server["maxConcurrentRequests"] as? Int ?? 16
            cfgTimeout = server["requestTimeout"] as? Int ?? 300
            cfgMaxBodyMB = server["maxRequestSizeMB"] as? Double ?? 100.0
            if let keys = server["apiKeys"] as? [String] {
                cfgApiKeys = keys.joined(separator: "\n")
            }
        }
        cfgDefaultModel = json["defaultModel"] as? String ?? ""
        if let rootKeys = json["apiKeys"] as? [String], !rootKeys.isEmpty {
            // Prefer server.apiKeys if both exist
        } else if let rootKeys = json["apiKeys"] as? [String] {
            cfgApiKeys = rootKeys.joined(separator: "\n")
        }
        cfgSaveMessage = nil
        cfgHasUnsavedChanges = false
    }

    private func saveConfig() {
        // Validation
        guard cfgPort > 0 && cfgPort <= 65535 else {
            cfgSaveMessage = "Error: Port must be 1-65535"
            return
        }
        guard cfgAdminPort > 0 && cfgAdminPort <= 65535 else {
            cfgSaveMessage = "Error: Admin port must be 1-65535"
            return
        }
        guard cfgPort != cfgAdminPort else {
            cfgSaveMessage = "Error: Port and Admin Port must differ"
            return
        }
        guard cfgMaxConcurrent > 0 else {
            cfgSaveMessage = "Error: Max Concurrent must be > 0"
            return
        }

        let apiKeysArray = cfgApiKeys
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        var serverDict: [String: Any] = [
            "host": cfgHost,
            "port": cfgPort,
            "adminPort": cfgAdminPort,
            "maxConcurrentRequests": cfgMaxConcurrent,
            "requestTimeout": cfgTimeout,
            "maxRequestSizeMB": cfgMaxBodyMB
        ]
        if !apiKeysArray.isEmpty {
            serverDict["apiKeys"] = apiKeysArray
        }

        var configDict: [String: Any] = ["server": serverDict]
        if !cfgDefaultModel.isEmpty {
            configDict["defaultModel"] = cfgDefaultModel
        }

        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let configPath = homeDir.appendingPathComponent(".nova/config.json")

        do {
            let data = try JSONSerialization.data(withJSONObject: configDict, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: configPath, options: .atomic)
            cfgSaveMessage = "Saved! Restart app to apply changes."
            cfgHasUnsavedChanges = false

            // Also update the in-memory config via NovaMLXConfiguration
            Task {
                let config = NovaMLXConfiguration.shared
                let newServerConfig = ServerConfig(
                    host: cfgHost,
                    port: cfgPort,
                    adminPort: cfgAdminPort,
                    apiKeys: apiKeysArray,
                    maxConcurrentRequests: cfgMaxConcurrent,
                    requestTimeout: TimeInterval(cfgTimeout),
                    maxRequestSizeMB: cfgMaxBodyMB
                )
                await config.setServerConfig(newServerConfig)
                await config.setDefaultModel(cfgDefaultModel.isEmpty ? nil : cfgDefaultModel)
            }
        } catch {
            cfgSaveMessage = "Error: \(error.localizedDescription)"
        }
    }

    // MARK: - CLI Tool

    private var cliSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Command Line Tool", icon: "terminal")

            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Install the `nova` CLI to your PATH")
                        .font(.system(size: 13))
                    Text(cliInstallPath)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Spacer()
                if cliInstalled {
                    Label("Installed", systemImage: "checkmark.circle.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(NovaTheme.Colors.statusOK)
                } else {
                    Button("Install") { installCLI() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                }
            }

            if let msg = cliInstallMessage {
                Text(msg)
                    .font(.caption)
                    .foregroundColor(cliInstalled ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
            }
        }
        .padding(16)
        .sectionCard()
    }

    private var cliInstallPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.local/bin/nova"
    }

    private func installCLI() {
        let binaryPath = Bundle.main.bundlePath + "/Contents/MacOS/nova"
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let installDir = "\(home)/.local/bin"
        let linkPath = "\(installDir)/nova"

        if FileManager.default.fileExists(atPath: linkPath) {
            cliInstalled = true
            cliInstallMessage = "Already installed"
            return
        }

        guard FileManager.default.fileExists(atPath: binaryPath) else {
            cliInstallMessage = "CLI binary not found in app bundle"
            return
        }

        do {
            if !FileManager.default.fileExists(atPath: installDir) {
                try FileManager.default.createDirectory(atPath: installDir, withIntermediateDirectories: true)
            }

            try FileManager.default.createSymbolicLink(atPath: linkPath, withDestinationPath: binaryPath)
            cliInstalled = true
            cliInstallMessage = "Installed! Add ~/.local/bin to your PATH if needed."
        } catch {
            cliInstallMessage = "Error: \(error.localizedDescription)"
        }
    }

    // MARK: - TurboQuant

    private var turboQuantSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("KV Quantization (TurboQuant)", icon: "waveform.path.badge.minus")

            Text("Compresses KV cache to reduce GPU memory. Check Status page GPU metric to see the effect.")
                .font(.caption).foregroundColor(.secondary)

            if appState.loadedModels.isEmpty {
                Text("Load a model to configure KV quantization.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                ForEach(appState.loadedModels, id: \.self) { modelId in
                    turboQuantRow(modelId: modelId)
                }
            }
        }
        .padding(16)
        .sectionCard()
        .task(id: appState.loadedModels) {
            await loadPersistedTurboQuantConfigs()
        }
    }

    private func turboQuantRow(modelId: String) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(modelId.components(separatedBy: "/").last ?? modelId)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                    CopyIDButton(id: modelId)
                }
            }

            Spacer()

            let config = turboQuantConfigs[modelId]
            Group {
                tqButton("Off", bits: nil, modelId: modelId, isActive: config?.bits == nil)
                tqButton("8-bit", bits: 8, modelId: modelId, isActive: config?.bits == 8)
                tqButton("4-bit", bits: 4, modelId: modelId, isActive: config?.bits == 4)
                tqButton("2-bit", bits: 2, modelId: modelId, isActive: config?.bits == 2)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .rowCard()
    }

    @ViewBuilder
    private func tqButton(_ label: String, bits: Int?, modelId: String, isActive: Bool) -> some View {
        if isActive {
            Button(label) { tqAction(bits: bits, modelId: modelId) }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        } else {
            Button(label) { tqAction(bits: bits, modelId: modelId) }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    private func tqAction(bits: Int?, modelId: String) {
        // Update TurboQuant runtime config (in-memory)
        if let bits {
            let config = TurboQuantService.Config(bits: bits, groupSize: 64)
            inferenceService.engine.turboQuantService.setConfig(config, forModel: modelId)
        } else {
            inferenceService.engine.turboQuantService.removeConfig(forModel: modelId)
        }

        // Persist to model_settings.json
        inferenceService.settingsManager.updateSettings(modelId) { settings in
            if let bits {
                settings.kvBits = bits
                settings.kvGroupSize = 64
            } else {
                settings.kvBits = nil
                settings.kvGroupSize = nil
            }
        }

        // Update local UI state
        if let bits {
            turboQuantConfigs[modelId] = TQConfig(bits: bits)
        } else {
            turboQuantConfigs[modelId] = nil
        }
    }

    // MARK: - Sessions

    private var sessionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Sessions", icon: "clock.arrow.circlepath")

            HStack {
                Button("Refresh") {
                    Task { await loadSessions() }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Clear All") {
                    Task {
                        let adminPort = appState.adminPort
                        let url = URL(string: "http://127.0.0.1:\(adminPort)/admin/sessions")!
                        var req = URLRequest(url: url)
                        req.httpMethod = "DELETE"
                        _ = try? await URLSession.shared.data(for: req)
                        sessions.removeAll()
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .foregroundColor(NovaTheme.Colors.statusError)
            }

            if sessions.isEmpty {
                Text("No active sessions")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                ForEach(sessions) { session in
                    HStack {
                        Text(session.id).font(.system(size: 12)).lineLimit(1)
                        Spacer()
                        Text(session.model).font(.caption).foregroundColor(.secondary)
                        Button {
                            Task {
                                let adminPort = appState.adminPort
                                let url = URL(string: "http://127.0.0.1:\(adminPort)/admin/sessions/\(session.id)")!
                                var req = URLRequest(url: url)
                                req.httpMethod = "DELETE"
                                _ = try? await URLSession.shared.data(for: req)
                                sessions.removeAll { $0.id == session.id }
                            }
                        } label: {
                            Image(systemName: "xmark.circle").font(.caption)
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .rowCard()
                }
            }
        }
        .padding(16)
        .sectionCard()
    }

    // MARK: - About

    private var aboutSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("About", icon: "info.circle")

            settingsRow("Version", value: NovaMLXCore.version)
            settingsRow("Build", value: NovaMLXCore.buildTimestamp)
            settingsRow("License", value: "Apache 2.0")
            settingsRow("Platform", value: "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")

            HStack(spacing: 12) {
                Button("GitHub") {
                    NSWorkspace.shared.open(URL(string: "https://github.com/cnshsliu/novamlx")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Documentation") {
                    NSWorkspace.shared.open(URL(string: "https://github.com/cnshsliu/novamlx#readme")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(16)
        .sectionCard()
    }

    // MARK: - Helpers

    private func settingsRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label).font(.system(size: 13)).foregroundColor(.secondary)
            Spacer()
            Text(value).font(.system(size: 13, weight: .medium)).foregroundColor(.primary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func loadPersistedTurboQuantConfigs() async {
        for modelId in appState.loadedModels {
            // Check runtime TurboQuant config first, then persisted settings
            if let config = inferenceService.engine.turboQuantService.getConfig(forModel: modelId) {
                turboQuantConfigs[modelId] = TQConfig(bits: config.bits)
            } else {
                let settings = inferenceService.settingsManager.getSettings(modelId)
                if let kvBits = settings.kvBits, kvBits > 0 {
                    turboQuantConfigs[modelId] = TQConfig(bits: kvBits)
                }
            }
        }
    }

    private func loadSessions() async {
        let adminPort = appState.adminPort
        guard let url = URL(string: "http://127.0.0.1:\(adminPort)/admin/sessions") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                sessions = json.compactMap { s in
                    guard let id = s["id"] as? String, let model = s["model"] as? String else { return nil }
                    return SessionInfo(id: id, model: model)
                }
            }
        } catch {}
    }
}

private struct TQConfig {
    let bits: Int
}

private struct SessionInfo: Identifiable {
    let id: String
    let model: String
}
