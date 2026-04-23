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
    @EnvironmentObject var l10n: L10n

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
                languageSection
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
                sectionHeader(l10n.tr("settings.server"), icon: "server.rack")
                Spacer()
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showConfigEditor.toggle() } }) {
                    Image(systemName: showConfigEditor ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                    Text(l10n.tr("settings.editConfig"))
                        .font(.system(size: 11))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            settingsRow(l10n.tr("settings.inferenceApi"), value: "http://127.0.0.1:\(String(appState.serverPort))")
            settingsRow(l10n.tr("settings.adminApi"), value: "http://127.0.0.1:\(String(appState.adminPort))")
            settingsRow(l10n.tr("settings.webChat"), value: "http://127.0.0.1:\(String(appState.serverPort))/chat")
            settingsRow(l10n.tr("settings.adminDashboard"), value: "http://127.0.0.1:\(String(appState.adminPort))/admin/dashboard")

            configPathRow

            if showConfigEditor {
                Divider().padding(.vertical, 4)
                configEditorPanel
            }

            HStack {
                Button(l10n.tr("settings.openChat")) {
                    NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(String(appState.serverPort))/chat")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(l10n.tr("settings.openDashboard")) {
                    NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(String(appState.adminPort))/admin/dashboard")!)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Spacer()

                Button(l10n.tr("settings.openConfig")) {
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
            Text(l10n.tr("settings.configFile")).font(.system(size: 13)).foregroundColor(.secondary)
            Spacer()
            Text(configPath)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.accent)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(l10n.tr("settings.clickCopy"))
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
            Text(l10n.tr("settings.serverConfig"))
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 12) {
                configField(l10n.tr("settings.host"), text: $cfgHost, width: 140, placeholder: l10n.tr("settings.hostPlaceholder"))
                configFieldInt(l10n.tr("settings.port"), value: $cfgPort, width: 80, range: 1...65535)
                configFieldInt(l10n.tr("settings.adminPort"), value: $cfgAdminPort, width: 80, range: 1...65535)
            }

            HStack(spacing: 12) {
                configFieldInt(l10n.tr("settings.maxConcurrent"), value: $cfgMaxConcurrent, width: 80, range: 1...128)
                configFieldInt(l10n.tr("settings.timeout"), value: $cfgTimeout, width: 80, range: 10...3600)
                configFieldDouble(l10n.tr("settings.maxBody"), value: $cfgMaxBodyMB, width: 100, range: 1...1024)
            }

            HStack(spacing: 12) {
                configField(l10n.tr("settings.defaultModel"), text: $cfgDefaultModel, width: nil, placeholder: l10n.tr("settings.modelPlaceholder"))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(l10n.tr("settings.apiKeys"))
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
                Button(l10n.tr("settings.resetCurrent")) { loadCurrentConfig() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                Button(l10n.tr("settings.saveToDisk")) { saveConfig() }
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
        } else if let rootKeys = json["apiKeys"] as? [String] {
            cfgApiKeys = rootKeys.joined(separator: "\n")
        }
        cfgSaveMessage = nil
        cfgHasUnsavedChanges = false
    }

    private func saveConfig() {
        guard cfgPort > 0 && cfgPort <= 65535 else {
            cfgSaveMessage = l10n.tr("settings.portError")
            return
        }
        guard cfgAdminPort > 0 && cfgAdminPort <= 65535 else {
            cfgSaveMessage = l10n.tr("settings.adminPortError")
            return
        }
        guard cfgPort != cfgAdminPort else {
            cfgSaveMessage = l10n.tr("settings.portConflict")
            return
        }
        guard cfgMaxConcurrent > 0 else {
            cfgSaveMessage = l10n.tr("settings.concurrentError")
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
            cfgSaveMessage = l10n.tr("settings.savedRestart")
            cfgHasUnsavedChanges = false

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

    // MARK: - Language

    private var languageSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("settings.language"), icon: "globe")

            HStack(spacing: 12) {
                Text(l10n.tr("settings.systemDefault"))
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)

                Spacer()

                Picker("", selection: Binding(
                    get: { l10n.currentLanguage },
                    set: { lang in l10n.setLanguage(lang) }
                )) {
                    ForEach(AppLanguage.allCases) { lang in
                        Text(lang.displayName).tag(lang)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 220)
            }
        }
        .padding(16)
        .sectionCard()
    }

    // MARK: - CLI Tool

    private var cliSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("settings.cli"), icon: "terminal")

            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(l10n.tr("settings.installCli"))
                        .font(.system(size: 13))
                    Text(cliInstallPath)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Spacer()
                if cliInstalled {
                    Label(l10n.tr("settings.installed"), systemImage: "checkmark.circle.fill")
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
            cliInstallMessage = l10n.tr("settings.alreadyInstalled")
            return
        }

        guard FileManager.default.fileExists(atPath: binaryPath) else {
            cliInstallMessage = l10n.tr("settings.cliNotFound")
            return
        }

        do {
            if !FileManager.default.fileExists(atPath: installDir) {
                try FileManager.default.createDirectory(atPath: installDir, withIntermediateDirectories: true)
            }

            try FileManager.default.createSymbolicLink(atPath: linkPath, withDestinationPath: binaryPath)
            cliInstalled = true
            cliInstallMessage = l10n.tr("settings.cliInstallSuccess")
        } catch {
            cliInstallMessage = "Error: \(error.localizedDescription)"
        }
    }

    // MARK: - TurboQuant

    private var turboQuantSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("settings.turboQuant"), icon: "waveform.path.badge.minus")

            Text(l10n.tr("settings.turboQuantDesc"))
                .font(.caption).foregroundColor(.secondary)

            if appState.loadedModels.isEmpty {
                Text(l10n.tr("settings.loadModel"))
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
                tqButton(l10n.tr("settings.off"), bits: nil, modelId: modelId, isActive: config?.bits == nil)
                tqButton(l10n.tr("settings.bit8"), bits: 8, modelId: modelId, isActive: config?.bits == 8)
                tqButton(l10n.tr("settings.bit4"), bits: 4, modelId: modelId, isActive: config?.bits == 4)
                tqButton(l10n.tr("settings.bit2"), bits: 2, modelId: modelId, isActive: config?.bits == 2)
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
        if let bits {
            let config = TurboQuantService.Config(bits: bits, groupSize: 64)
            inferenceService.engine.turboQuantService.setConfig(config, forModel: modelId)
        } else {
            inferenceService.engine.turboQuantService.removeConfig(forModel: modelId)
        }

        inferenceService.settingsManager.updateSettings(modelId) { settings in
            if let bits {
                settings.kvBits = bits
                settings.kvGroupSize = 64
            } else {
                settings.kvBits = nil
                settings.kvGroupSize = nil
            }
        }

        if let bits {
            turboQuantConfigs[modelId] = TQConfig(bits: bits)
        } else {
            turboQuantConfigs[modelId] = nil
        }
    }

    // MARK: - Sessions

    private var sessionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("settings.sessions"), icon: "clock.arrow.circlepath")

            HStack {
                Button(l10n.tr("settings.refresh")) {
                    Task { await loadSessions() }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(l10n.tr("settings.clearAll")) {
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
                Text(l10n.tr("settings.noSessions"))
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
            sectionHeader(l10n.tr("settings.about"), icon: "info.circle")

            settingsRow(l10n.tr("settings.version"), value: NovaMLXCore.version)
            settingsRow(l10n.tr("settings.build"), value: NovaMLXCore.buildTimestamp)
            settingsRow(l10n.tr("settings.license"), value: l10n.tr("settings.licenseValue"))
            settingsRow(l10n.tr("settings.platform"), value: "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")

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
