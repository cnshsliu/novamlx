import SwiftUI
import NovaMLXCore
import NovaMLXInference

// MARK: - Provider Catalog

struct ProviderCatalogEntry: Identifiable {
    let id: String
    let displayName: String
    let endpoint: String
    let icon: String

    static let entries: [ProviderCatalogEntry] = [
        ProviderCatalogEntry(id: "openai", displayName: "OpenAI", endpoint: "https://api.openai.com/v1", icon: "circle.hexagon"),
        ProviderCatalogEntry(id: "anthropic", displayName: "Anthropic", endpoint: "https://api.anthropic.com/v1", icon: "brain.head.profile"),
        ProviderCatalogEntry(id: "tknet", displayName: "tknet.ai", endpoint: "https://api.tknet.ai/v1", icon: "star.circle"),
        ProviderCatalogEntry(id: "groq", displayName: "Groq", endpoint: "https://api.groq.com/openai/v1", icon: "bolt.horizontal"),
        ProviderCatalogEntry(id: "together", displayName: "Together AI", endpoint: "https://api.together.xyz/v1", icon: "person.2"),
        ProviderCatalogEntry(id: "fireworks", displayName: "Fireworks AI", endpoint: "https://api.fireworks.ai/inference/v1", icon: "fireworks"),
        ProviderCatalogEntry(id: "mistral", displayName: "Mistral", endpoint: "https://api.mistral.ai/v1", icon: "wind"),
        ProviderCatalogEntry(id: "deepseek", displayName: "DeepSeek", endpoint: "https://api.deepseek.com", icon: "magnifyingglass"),
        ProviderCatalogEntry(id: "openrouter", displayName: "OpenRouter", endpoint: "https://openrouter.ai/api/v1", icon: "arrow.triangle.branch"),
        ProviderCatalogEntry(id: "gemini", displayName: "Google Gemini", endpoint: "https://generativelanguage.googleapis.com/v1beta/openai", icon: "star.circle"),
        ProviderCatalogEntry(id: "xai", displayName: "xAI (Grok)", endpoint: "https://api.x.ai/v1", icon: "sparkles"),
        ProviderCatalogEntry(id: "dashscope-cn", displayName: "DashScope China", endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1", icon: "cloud"),
        ProviderCatalogEntry(id: "dashscope-intl", displayName: "DashScope International", endpoint: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", icon: "globe"),
        ProviderCatalogEntry(id: "glm", displayName: "GLM (Zhipu)", endpoint: "https://open.bigmodel.cn/api/paas/v4", icon: "brain"),
        ProviderCatalogEntry(id: "glm-codeplan", displayName: "GLM CodePlan", endpoint: "https://open.bigmodel.cn/api/paas/v4", icon: "hammer"),
        ProviderCatalogEntry(id: "custom", displayName: "Custom", endpoint: "", icon: "wrench.and.screwdriver"),
    ]

    static func matchByEndpoint(_ endpoint: String) -> ProviderCatalogEntry? {
        let trimmed = endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return entries.first { $0.endpoint == trimmed }
    }
}

// MARK: - TokenhubPageView

struct TokenhubPageView: View {
    @ObservedObject var appState: MenuBarAppState
    @EnvironmentObject var l10n: L10n

    @State private var providers: [TokenhubProvider] = []
    @State private var editingProvider: TokenhubProvider?
    @State private var selectedProvider: TokenhubProvider?
    @State private var isCreatingNew = false

    // Right panel mode
    @State private var rightPanelMode: RightPanelMode = .empty

    // Agent launcher
    @State private var selectedAgentPerProvider: [String: AgentSpec] = [:]
    @State private var lbSelectedAgent: AgentSpec = AgentRegistry.all[0]
    @State private var agentToast: String?
    @State private var showAgentToast = false
    @State private var showCodexRestartConfirm = false
    @State private var pendingCodexLaunch: (agent: AgentSpec, modelName: String)?
    @State private var showDeleteConfirm = false
    @State private var pendingDeleteProvider: TokenhubProvider?
    @State private var apiKeyVisibility: [String: Bool] = [:]

    // Form
    @State private var formName = ""
    @State private var formEndpoint = ""
    @State private var formApiKey = ""
    @State private var formRemoteModel = ""
    @State private var formEnabled = false
    @State private var formIsFree = false
    @State private var formSupportsResponses = false
    @State private var formTags = ""
    @State private var isVerified = false

    // Model query
    @State private var availableModels: [String] = []
    @State private var isQueryingModels = false
    @State private var queryError: String?
    @State private var saveError: String?
    @State private var testEndpointResult: String?
    @State private var testEndpointRunning = false
    @State private var testProxyResult: String?
    @State private var testProxyRunning = false

    // Bulk test
    @State private var bulkTestRunning = false
    @State private var bulkTestProgress: [String: String] = [:] // name -> "OK" / "FAIL"

    // Alert
    @State private var alertMessage = ""
    @State private var showAlert = false

    // Provider limit alert
    @State private var showProviderLimitAlert = false

    private let manager = TokenhubManager.shared

    enum RightPanelMode {
        case empty
        case detail
        case editing
    }

    private var isFormActive: Bool {
        rightPanelMode == .editing
    }

    /// True when the form is editing a cloud-managed provider (one tagged
    /// "managed"). Post-Task-6 this replaces the old `isManaged` struct field;
    /// the form shows read-only fields for managed providers.
    private var isEditingManaged: Bool {
        editingProvider?.tags.contains("managed") == true
    }

    var body: some View {
        VStack(spacing: 0) {
            // Top: Load Balance (full width)
            if !lbProviders.isEmpty {
                loadBalanceSection
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                Divider()
                    .padding(.vertical, 0)
            }

            // Bottom: Provider List + Detail
            HStack(spacing: 0) {
                leftPanel
                    .frame(width: 200)
                Divider()
                rightPanel
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task {
            manager.enforceProviderLimits()
            reloadProviders()
        }
        .onChange(of: formName) { resetVerification() }
        .onChange(of: formEndpoint) { resetVerification() }
        .onChange(of: formApiKey) { resetVerification() }
        .onChange(of: formRemoteModel) { resetVerification() }
        .alert("Tokenhub", isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
        .alert("Provider Limit", isPresented: $showProviderLimitAlert) {
            Button("OK", role: .cancel) {}
            Button("Go to Settings") {
                DispatchQueue.main.async {
                    appState.requestedPage = .settings
                }
            }
        } message: {
            Text("Up to 3 custom providers allowed. To add more, enter your tknet.ai API Key in Settings to unlock unlimited providers.")
        }
        .alert("Restart Codex?", isPresented: $showCodexRestartConfirm) {
            Button("Restart", role: .destructive) {
                guard let pending = pendingCodexLaunch else { return }
                AgentConfigGenerator.launchOrRestartCodex(agent: pending.agent, forceRestart: true) { success in
                    DispatchQueue.main.async {
                        agentToast = success ? "\(pending.agent.displayName) restarted with \(pending.modelName)" : "\(pending.agent.displayName) not found"
                        showAgentToast = true
                        pendingCodexLaunch = nil
                    }
                }
            }
            Button("Cancel", role: .cancel) {
                pendingCodexLaunch = nil
            }
        } message: {
            if let pending = pendingCodexLaunch {
                Text("Codex is running. To switch model to \(pending.modelName), Codex must be restarted. Unsaved work may be lost.")
            } else {
                Text("Codex is running and must be restarted.")
            }
        }
        .alert("Delete Provider?", isPresented: $showDeleteConfirm) {
            Button("Delete", role: .destructive) {
                if let provider = pendingDeleteProvider {
                    deleteProvider(provider)
                    pendingDeleteProvider = nil
                }
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteProvider = nil
            }
        } message: {
            if let provider = pendingDeleteProvider {
                Text("Are you sure you want to delete \(provider.name)? This cannot be undone.")
            } else {
                Text("Are you sure you want to delete this provider?")
            }
        }
    }

    // MARK: - Helpers

    /// Mask API Key: show prefix (7 if sk-*, else 4) + asterisks + last 3
    /// Example: "sk-abc123def456" → "sk-abc1*****456"
    /// Example: "ghp_x8z2mN4kR9" → "ghp_*****R9"
    private func maskApiKey(_ key: String) -> String {
        if key.count <= 7 { return String(repeating: "*", count: key.count) }
        let prefixLen = key.hasPrefix("sk-") ? 7 : 4
        let suffixLen = 3
        if key.count <= prefixLen + suffixLen { return String(repeating: "*", count: key.count) }
        let prefix = String(key.prefix(prefixLen))
        let suffix = String(key.suffix(suffixLen))
        let starCount = key.count - prefixLen - suffixLen
        return "\(prefix)\(String(repeating: "*", count: starCount))\(suffix)"
    }

    // MARK: - Left Panel (My Providers only)

    private var leftPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("My Providers")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                if !manager.hasValidTknetKey() {
                    let userCount = providers.filter { !$0.tags.contains("managed") }.count
                    Text("\(userCount)/\(TokenhubManager.freeProviderLimit)")
                        .font(.system(size: 9))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
                Spacer()
                Button(action: bulkTestAll) {
                    if bulkTestRunning {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Image(systemName: "bolt.horizontal.circle.fill")
                            .font(.system(size: 14))
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(bulkTestRunning || providers.isEmpty)
                .help("Test all providers")
                .onHover { isHovering in
                    if isHovering { NSCursor.pointingHand.push() }
                    else { NSCursor.pop() }
                }

                Button(action: startCreating) {
                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 14))
                        .foregroundColor(NovaTheme.Colors.accent)
                }
                .buttonStyle(.plain)
                .onHover { isHovering in
                    if isHovering { NSCursor.pointingHand.push() }
                    else { NSCursor.pop() }
                }

                Button(action: {
                    // Refresh provider list (locals now live in ModelManager)
                    reloadProviders()
                    agentToast = "Refreshed providers"
                    showAgentToast = true
                }) {
                    Image(systemName: "arrow.clockwise.circle.fill")
                        .font(.system(size: 14))
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                }
                .buttonStyle(.plain)
                .help("Refresh providers")
                .onHover { isHovering in
                    if isHovering { NSCursor.pointingHand.push() }
                    else { NSCursor.pop() }
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 12)
            .padding(.bottom, 8)

            if providers.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("None configured yet")
                        .font(.system(size: 11))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                    Text("Click + to add one")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                Spacer()
            } else {
                ScrollView(showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 0) {
                        // Sort: enabled first, then disabled, then by name
                        let sorted = providers.sorted { a, b in
                            if a.isEnabled != b.isEnabled { return a.isEnabled }
                            return a.name < b.name
                        }
                        ForEach(sorted) { provider in
                            myProviderRow(provider)
                        }
                    }
                }
            }
        }
        .background(NovaTheme.Colors.cardBackground)
    }

    private func myProviderRow(_ provider: TokenhubProvider) -> some View {
        Button(action: { selectMyProvider(provider) }) {
            HStack(spacing: 0) {
                Spacer().frame(width: 10)

                // Free indicator: fixed-width slot, visible dot when free
                Group {
                    if provider.isFree {
                        Circle()
                            .fill(NovaTheme.Colors.statusOK.opacity(0.6))
                            .frame(width: 5, height: 5)
                    } else {
                        Color.clear.frame(width: 5, height: 5)
                    }
                }.frame(width: 8)

                Spacer().frame(width: 4)

                // Status dot
                Circle()
                    .fill(bulkIndicatorColor(for: provider))
                    .frame(width: 6, height: 6)

                Spacer().frame(width: 8)

                VStack(alignment: .leading, spacing: 1) {
                    Text(provider.name)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor((editingProvider?.id == provider.id || selectedProvider?.id == provider.id) ? NovaTheme.Colors.accent : NovaTheme.Colors.textPrimary)
                        .lineLimit(1)
                    HStack(spacing: 4) {
                        Text(provider.remoteModel.isEmpty ? "no model" : provider.remoteModel)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                            .lineLimit(1)
                        if provider.requestCount > 0 {
                            Text("\(provider.successCount)/\(provider.requestCount)")
                                .font(.system(size: 8, design: .monospaced))
                                .foregroundColor(NovaTheme.Colors.textTertiary)
                            if provider.avgLatencyMs > 0 {
                                Text(String(format: "%.0fms", provider.avgLatencyMs))
                                    .font(.system(size: 8, design: .monospaced))
                                    .foregroundColor(NovaTheme.Colors.textTertiary)
                            }
                        }
                        if !provider.apiKey.isEmpty {
                            Text(maskApiKey(provider.apiKey))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(NovaTheme.Colors.textTertiary)
                                .lineLimit(1)
                        }
                    }
                }
                Spacer()
                if let result = bulkTestProgress[provider.name] {
                    Text(result)
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(result == "OK" ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background((editingProvider?.id == provider.id || selectedProvider?.id == provider.id) ? NovaTheme.Colors.accent.opacity(0.15) : Color.clear)
            .cornerRadius(6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovering in
            if isHovering { NSCursor.pointingHand.push() }
            else { NSCursor.pop() }
        }
        .contextMenu {
            Button {
                duplicateProvider(provider)
            } label: {
                Label("Duplicate", systemImage: "doc.on.doc")
            }
            Button(role: .destructive) {
                pendingDeleteProvider = provider
                showDeleteConfirm = true
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    // MARK: - Right Panel

    /// Providers shown in the top "Load Balance" section. Post-Task-6 this is
    /// just enabled providers (Task 7+ will replace this with the LB entity's
    /// actual member list). Kept here so the LB header section still renders.
    private var lbProviders: [TokenhubProvider] {
        providers.filter { $0.isEnabled }
    }

    private var rightPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                switch rightPanelMode {
                case .empty:
                    emptyState
                case .detail:
                    if let provider = selectedProvider {
                        providerDetailSection(provider)
                    }
                case .editing:
                    formContent
                }
            }
            .padding(24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Load Balance Section (fixed at top)

    private var loadBalanceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.system(size: 12))
                    .foregroundColor(NovaTheme.Colors.accent)
                Text("Load Balance")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                Spacer()
                let count = lbProviders.count
                Text("\(count) provider\(count == 1 ? "" : "s")")
                    .font(.system(size: 10))
                    .foregroundColor(NovaTheme.Colors.textTertiary)
            }

            // Context window info
            let (ctx, mixed) = ModelSpecs.lbContextWindow(from: providers)
            HStack(spacing: 4) {
                Text("Context:")
                    .font(.system(size: 10))
                    .foregroundColor(NovaTheme.Colors.textTertiary)
                Text(formatContext(ctx))
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                if mixed {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 9))
                        .foregroundColor(.orange)
                    Text("Mixed context sizes in pool — using minimum for compatibility")
                        .font(.system(size: 9))
                        .foregroundColor(.orange)
                }
            }

            // Agent launcher row
            agentLauncherRow(
                agent: lbSelectedAgent,
                onAgentChange: { lbSelectedAgent = $0 },
                modelName: "tknet",
                allProviders: providers
            )
        }
        .padding(14)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(NovaTheme.Colors.accent.opacity(0.2), lineWidth: 1)
        )
        .cornerRadius(8)
    }

    // MARK: - Provider Detail Section

    private func providerDetailSection(_ provider: TokenhubProvider) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with Edit/Copy/Delete
            HStack {
                Circle()
                    .fill(provider.isEnabled ? NovaTheme.Colors.statusOK : NovaTheme.Colors.textTertiary)
                    .frame(width: 8, height: 8)
                Text(provider.name)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                if provider.tags.contains("managed") {
                    Label("Managed", systemImage: "lock.fill")
                        .font(.system(size: 9))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(NovaTheme.Colors.accentDim)
                        .clipShape(Capsule())
                }
                Spacer()

                Button {
                    loadFormFromProvider(provider)
                    rightPanelMode = .editing
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    duplicateProvider(provider)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(role: .destructive) {
                    pendingDeleteProvider = provider
                    showDeleteConfirm = true
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // Info grid
            VStack(alignment: .leading, spacing: 6) {
                detailRow(label: "Upstream Model", value: provider.remoteModel)
                HStack(alignment: .top) {
                    Text("API Model:")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                        .frame(width: 80, alignment: .trailing)
                    Text("tknet:" + provider.id)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                    Button {
                        NSPasteboard.general.clearContents()
                        let copyValue = "tknet:" + provider.id
                        NSPasteboard.general.setString(copyValue, forType: .string)
                        agentToast = "Copied: " + copyValue
                        showAgentToast = true
                    } label: {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 9))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                    }
                    .buttonStyle(.plain)
                }
                detailRow(label: "Endpoint", value: provider.endpoint)
                if !provider.apiKey.isEmpty {
                    HStack(alignment: .top) {
                        Text("API Key:")
                            .font(.system(size: 10))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                            .frame(width: 60, alignment: .trailing)
                        Text(maskApiKey(provider.apiKey))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                    }
                }
                detailRow(label: "Context", value: formatContext(provider.effectiveContextWindow))
                if !provider.tags.isEmpty {
                    detailRow(label: "Tags", value: provider.tags.joined(separator: ", "))
                }
                if provider.requestCount > 0 {
                    HStack(spacing: 12) {
                        Text("Stats:")
                            .font(.system(size: 10))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                        Text("\(provider.successCount)/\(provider.requestCount) OK")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                        if provider.avgLatencyMs > 0 {
                            Text(String(format: "%.0fms avg", provider.avgLatencyMs))
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(NovaTheme.Colors.textTertiary)
                        }
                    }
                }
            }

            Divider()

            // Test buttons (visible in detail view too)
            if !provider.tags.contains("managed") {
                detailTestButtons(provider)
                Divider()
            }

            // Agent launcher
            let agent = selectedAgentPerProvider[provider.id] ?? AgentRegistry.all[0]
            agentLauncherRow(
                agent: agent,
                onAgentChange: { selectedAgentPerProvider[provider.id] = $0 },
                modelName: "tknet:" + provider.id,
                allProviders: providers
            )
        }
        .padding(14)
        .background(NovaTheme.Colors.cardBackground)
        .cornerRadius(8)
    }

    // MARK: - Detail Test Buttons (shared between detail and editing views)

    private func detailTestButtons(_ provider: TokenhubProvider) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button(action: {
                    loadFormFromProvider(provider)
                    testProviderEndpoint()
                    // Restore detail mode after test triggers
                }) {
                    HStack(spacing: 4) {
                        if testEndpointRunning { ProgressView().controlSize(.small) }
                        else { Image(systemName: "antenna.radiowaves.left.and.right") }
                        Text("Test Provider")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(provider.endpoint.isEmpty || provider.apiKey.isEmpty || testEndpointRunning)

                if provider.endpoint.isEmpty {
                    Text("Endpoint required")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                } else if provider.apiKey.isEmpty {
                    Text("API Key required")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.statusError)
                } else if let result = testEndpointResult {
                    Text(result)
                        .font(.system(size: 11))
                        .foregroundColor(result.hasPrefix("OK") ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                }
            }

            HStack(spacing: 8) {
                Button(action: {
                    // Temporarily load form data so testViaNovaMLX can use it
                    loadFormFromProvider(provider)
                    testViaNovaMLX()
                }) {
                    HStack(spacing: 4) {
                        if testProxyRunning { ProgressView().controlSize(.small) }
                        else { Image(systemName: "arrow.triangle.2.circlepath") }
                        Text("Test tknet:\(provider.name)")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(provider.name.isEmpty || provider.apiKey.isEmpty || provider.remoteModel.isEmpty || testProxyRunning)

                if provider.apiKey.isEmpty {
                    Text("API Key required")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.statusError)
                } else if provider.remoteModel.isEmpty {
                    Text("Model required")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                } else if let result = testProxyResult {
                    Text(result)
                        .font(.system(size: 11))
                        .foregroundColor(result.hasPrefix("OK") ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                }
            }
        }
    }

    private func detailRow(label: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(label + ":")
                .font(.system(size: 10))
                .foregroundColor(NovaTheme.Colors.textTertiary)
                .frame(width: 60, alignment: .trailing)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textSecondary)
                .lineLimit(2)
        }
    }

    // MARK: - Agent Launcher Row (shared between LB and provider detail)

    private func agentLauncherRow(
        agent: AgentSpec,
        onAgentChange: @escaping (AgentSpec) -> Void,
        modelName: String,
        allProviders: [TokenhubProvider]
    ) -> some View {
        HStack(spacing: 8) {
            // Agent picker
            Menu {
                ForEach(AgentRegistry.all) { a in
                    Button(action: { onAgentChange(a) }) {
                        Label(a.displayName, systemImage: a.icon)
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: agent.icon)
                        .font(.system(size: 10))
                    Text(agent.displayName)
                        .font(.system(size: 11))
                }
                .foregroundColor(NovaTheme.Colors.textPrimary)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(NovaTheme.Colors.rowBackground)
                .cornerRadius(6)
            }
            .menuStyle(.borderlessButton)

            Spacer()

            // APP button
            if agent.hasApp {
                Button(action: {
                    // Always regenerate full Codex config (catalog + config.toml)
                    let warning = AgentConfigGenerator.generateConfig(
                        agent: agent,
                        providers: allProviders,
                        apiKey: appState.apiKey,
                        modelName: modelName
                    )
                    if let w = warning { agentToast = w }

                    if AgentConfigGenerator.isCodexRunning() {
                        // Codex is running — need restart to pick up new model
                        pendingCodexLaunch = (agent: agent, modelName: modelName)
                        showCodexRestartConfirm = true
                    } else {
                        // Not running — just launch
                        AgentConfigGenerator.launchOrRestartCodex(agent: agent, forceRestart: false) { success in
                            DispatchQueue.main.async {
                                agentToast = success ? "\(agent.displayName) launched with \(modelName)" : "\(agent.displayName) not found"
                                showAgentToast = true
                            }
                        }
                    }
                }) {
                    Label("APP", systemImage: "play.fill")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            // CLI button (copy to clipboard)
            if agent.hasCLI {
                Button(action: {
                    let _ = AgentConfigGenerator.generateConfig(
                        agent: agent,
                        providers: allProviders,
                        apiKey: appState.apiKey,
                        modelName: modelName
                    )
                    let cmd = AgentConfigGenerator.generateCLICommand(
                        agent: agent,
                        modelName: modelName,
                        apiKey: appState.apiKey
                    )
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(cmd, forType: .string)
                    agentToast = "Copied: \(cmd)"
                    showAgentToast = true
                }) {
                    Label("CLI", systemImage: "doc.on.clipboard")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .overlay {
            // Toast overlay
            if showAgentToast, let msg = agentToast {
                Text(msg)
                    .font(.system(size: 9))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(NovaTheme.Colors.cardBackground)
                    .cornerRadius(4)
                    .shadow(radius: 2)
                    .offset(y: -24)
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                            showAgentToast = false
                            agentToast = nil
                        }
                    }
            }
        }
    }

    private func formatContext(_ tokens: Int) -> String {
        if tokens >= 1_000_000 {
            return "\(tokens / 1_000_000)M"
        } else if tokens >= 1024 {
            return "\(tokens / 1024)K"
        }
        return "\(tokens)"
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer().frame(height: 80)
            Image(systemName: "server.rack")
                .font(.system(size: 36))
                .foregroundColor(NovaTheme.Colors.textTertiary)
            Text("Click + to add a cloud provider")
                .font(.system(size: 13))
                .foregroundColor(NovaTheme.Colors.textTertiary)
            Text("Proxy requests to OpenAI, Anthropic, Groq, and more")
                .font(.system(size: 11))
                .foregroundColor(NovaTheme.Colors.textTertiary)
        }
        .frame(maxWidth: .infinity)
    }

    private var formContent: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(isCreatingNew ? "Add Provider" : "Edit: \(editingProvider?.name ?? "")")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                if isEditingManaged {
                    Label("Managed", systemImage: "lock.fill")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(NovaTheme.Colors.accentDim)
                        .clipShape(Capsule())
                }
            }

            // Provider picker (catalog dropdown) — only for new providers
            if isCreatingNew {
                formField(label: "Provider", hint: nil) {
                    providerPicker
                }
            }

            if isEditingManaged {
                // Read-only info for managed providers
                formField(label: "Endpoint", hint: nil) {
                    Text(formEndpoint)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                }
                formField(label: "Model", hint: nil) {
                    Text(formRemoteModel)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                }
                Text("Managed by subscription — API key handled automatically.")
                    .font(.system(size: 10))
                    .foregroundColor(NovaTheme.Colors.textTertiary)
            } else {
                // Name
                formField(label: "Name", hint: "Unique name, e.g. openai-gpt4o") {
                    TextField("my-provider", text: $formName)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                }

                // Endpoint
                formField(label: "Endpoint", hint: nil) {
                    TextField("https://api.example.com/v1", text: $formEndpoint)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                        .disabled(editingProvider?.tags.contains("tknet") == true)
                        .help(editingProvider?.tags.contains("tknet") == true
                            ? "tknet.ai endpoint is fixed"
                            : "API endpoint URL")
                }

                // API Key
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("API Key")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)

                        Spacer()

                        if !formApiKey.isEmpty {
                            Button(action: {
                                apiKeyVisibility[editingProvider?.id ?? ""] = !(apiKeyVisibility[editingProvider?.id ?? ""] ?? false)
                            }) {
                                Image(systemName: (apiKeyVisibility[editingProvider?.id ?? ""] ?? false) ? "eye.slash.fill" : "eye.fill")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    if (apiKeyVisibility[editingProvider?.id ?? ""] ?? false) {
                        TextField("API Key", text: $formApiKey)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12, design: .monospaced))
                    } else {
                        SecureField("API Key", text: $formApiKey)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12, design: .monospaced))
                    }
                }

                // Tags
                formField(label: "Tags", hint: "Comma-separated, e.g. code, translate, free") {
                    TextField("code, translate, free", text: $formTags)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                }

                // Query Models + Model picker
                formField(label: "Model", hint: nil) {
                    HStack(spacing: 8) {
                        if availableModels.isEmpty {
                            TextField("Type model name or query below", text: $formRemoteModel)
                                .textFieldStyle(.roundedBorder)
                                .controlSize(.small)
                        } else {
                            Picker("Model", selection: $formRemoteModel) {
                                Text("Select model...").tag("")
                                ForEach(availableModels, id: \.self) { model in
                                    Text(model).tag(model)
                                }
                            }
                            .controlSize(.small)
                        }

                        Button(action: queryModels) {
                            HStack(spacing: 4) {
                                if isQueryingModels {
                                    ProgressView()
                                        .controlSize(.small)
                                } else {
                                    Image(systemName: "magnifyingglass")
                                }
                                Text("Query")
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(formEndpoint.isEmpty || formApiKey.isEmpty || isQueryingModels)
                    }
                }

                if let err = queryError {
                    Text(err)
                        .font(.system(size: 11))
                        .foregroundColor(NovaTheme.Colors.statusError)
                }
            }

            if let err = saveError {
                Text(err)
                    .font(.system(size: 11))
                    .foregroundColor(NovaTheme.Colors.statusError)
            }

            Divider()

            // Toggles (always available)
            HStack(spacing: 20) {
                Toggle("Enabled", isOn: $formEnabled)
                    .controlSize(.small)
                    .onChange(of: formEnabled) { saveManagedToggles() }
                if !isEditingManaged {
                    Toggle("Free", isOn: $formIsFree)
                        .controlSize(.small)
                        .onChange(of: formIsFree) { saveManagedToggles() }
                    Toggle("/RESPS", isOn: $formSupportsResponses)
                        .controlSize(.small)
                        .onChange(of: formSupportsResponses) { saveManagedToggles() }
                }
            }

            // Test buttons (not for managed)
            if !isEditingManaged {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Button(action: { testProviderEndpoint() }) {
                            HStack(spacing: 4) {
                                if testEndpointRunning { ProgressView().controlSize(.small) }
                                else { Image(systemName: "antenna.radiowaves.left.and.right") }
                                Text("Test Provider")
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(formEndpoint.isEmpty || formApiKey.isEmpty || testEndpointRunning)

                        if formEndpoint.isEmpty {
                            Text("Endpoint required")
                                .font(.system(size: 10))
                                .foregroundColor(NovaTheme.Colors.textTertiary)
                        } else if formApiKey.isEmpty {
                            Text("API Key required")
                                .font(.system(size: 10))
                                .foregroundColor(NovaTheme.Colors.statusError)
                        } else if let result = testEndpointResult {
                            Text(result)
                                .font(.system(size: 11))
                                .foregroundColor(result.hasPrefix("OK") ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                        }
                    }

                    HStack(spacing: 8) {
                        Button(action: { testViaNovaMLX() }) {
                            HStack(spacing: 4) {
                                if testProxyRunning { ProgressView().controlSize(.small) }
                                else { Image(systemName: "arrow.triangle.2.circlepath") }
                                Text("Test tknet:\(formName)")
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(formName.isEmpty || formApiKey.isEmpty || formRemoteModel.isEmpty || testProxyRunning)

                        if formName.isEmpty {
                            Text("Name required")
                                .font(.system(size: 10))
                                .foregroundColor(NovaTheme.Colors.textTertiary)
                        } else if formApiKey.isEmpty {
                            Text("API Key required")
                                .font(.system(size: 10))
                                .foregroundColor(NovaTheme.Colors.statusError)
                        } else if formRemoteModel.isEmpty {
                            Text("Model required")
                                .font(.system(size: 10))
                                .foregroundColor(NovaTheme.Colors.textTertiary)
                        } else if let result = testProxyResult {
                            Text(result)
                                .font(.system(size: 11))
                                .foregroundColor(result.hasPrefix("OK") ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                        }
                    }
                }
            }

            // Buttons
            HStack(spacing: 8) {
                if isEditingManaged {
                    Button("Close") { clearForm() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                } else {
                    Button(action: { saveProvider() }) {
                        Text("Save")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(formName.isEmpty || formEndpoint.isEmpty)

                    if let editing = editingProvider {
                        Button(role: .destructive, action: {
                            pendingDeleteProvider = editing
                            showDeleteConfirm = true
                        }) {
                            Text("Delete")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    Button("Cancel") { clearForm() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
            }
        }
        .padding(16)
        .sectionCard()
    }

    // MARK: - Provider Picker (Menu)

    private var providerPicker: some View {
        Menu {
            ForEach(ProviderCatalogEntry.entries) { entry in
                Button(action: { selectCatalogInForm(entry) }) {
                    Label(entry.displayName, systemImage: entry.icon)
                }
            }
        } label: {
            HStack(spacing: 6) {
                let matched = ProviderCatalogEntry.matchByEndpoint(formEndpoint)
                Image(systemName: matched?.icon ?? "wrench.and.screwdriver")
                    .frame(width: 16)
                Text(matched?.displayName ?? "Select provider...")
                    .foregroundColor(matched == nil ? NovaTheme.Colors.textTertiary : NovaTheme.Colors.textPrimary)
                Spacer()
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8))
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(6)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 0.5)
            )
        }
        .menuStyle(.borderlessButton)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Helper

    private func formField<Content: View>(label: String, hint: String?, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                if let hint {
                    Text(hint)
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
            }
            content()
        }
    }

    private func resetVerification() {
        isVerified = false
    }

    private func markVerified() {
        isVerified = true
    }

    // MARK: - Actions

    private func startCreating() {
        if !manager.hasValidTknetKey() && manager.userProviderCount() >= TokenhubManager.freeProviderLimit {
            showProviderLimitAlert = true
            return
        }
        isCreatingNew = true
        editingProvider = nil
        selectedProvider = nil
        rightPanelMode = .editing
        formName = ""
        formEndpoint = ""
        formApiKey = ""
        formRemoteModel = ""
        formEnabled = false
        formIsFree = false
        formSupportsResponses = false
        formTags = ""
        isVerified = false
        availableModels = []
        queryError = nil
        saveError = nil
        testEndpointResult = nil
        testProxyResult = nil
    }

    private func selectCatalogInForm(_ entry: ProviderCatalogEntry) {
        formEndpoint = entry.endpoint
        formRemoteModel = ""
        availableModels = []
        queryError = nil
    }

    private func selectMyProvider(_ provider: TokenhubProvider) {
        isCreatingNew = false
        editingProvider = nil
        selectedProvider = provider
        rightPanelMode = .detail
    }

    private func loadFormFromProvider(_ provider: TokenhubProvider) {
        editingProvider = provider
        formName = provider.name
        formEndpoint = provider.endpoint
        formApiKey = provider.apiKey
        formRemoteModel = provider.remoteModel
        formEnabled = provider.isEnabled
        formIsFree = provider.isFree
        formSupportsResponses = provider.supportsResponsesAPI
        formTags = provider.tags.joined(separator: ", ")
        isVerified = provider.isEnabled
        availableModels = []
        queryError = nil
        saveError = nil
        testEndpointResult = nil
        testProxyResult = nil
    }

    private func clearForm() {
        isCreatingNew = false
        editingProvider = nil
        formName = ""
        formEndpoint = ""
        formApiKey = ""
        formRemoteModel = ""
        formEnabled = false
        formIsFree = false
        formSupportsResponses = false
        formTags = ""
        isVerified = false
        availableModels = []
        queryError = nil
        saveError = nil
        testEndpointResult = nil
        testProxyResult = nil
        rightPanelMode = selectedProvider != nil ? .detail : .empty
    }

    private func parseTags(_ raw: String) -> [String] {
        raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty }
    }

    /// Save Enabled/Free/Responses toggles for any provider (auto-saved on toggle change).
    private func saveManagedToggles() {
        guard let editing = editingProvider else { return }
        var updated = editing
        updated.isEnabled = formEnabled
        updated.isFree = formIsFree
        updated.supportsResponsesAPI = formSupportsResponses
        try? manager.update(updated)
        editingProvider = updated
        reloadProviders()
    }

    private func saveProvider() {
        saveError = nil

        // Auto-trim all text fields to prevent whitespace issues from copy-paste
        let trimmedName = formName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedEndpoint = formEndpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedApiKey = formApiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedRemoteModel = formRemoteModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTags = formTags.trimmingCharacters(in: .whitespacesAndNewlines)

        let provider = TokenhubProvider(
            name: trimmedName,
            endpoint: trimmedEndpoint,
            apiKey: trimmedApiKey,
            remoteModel: trimmedRemoteModel,
            isEnabled: formEnabled,
            tags: parseTags(trimmedTags),
            isFree: formIsFree,
            supportsResponsesAPI: formSupportsResponses
        )

        if let editing = editingProvider {
            if editing.name != formName {
                if manager.get(formName) != nil {
                    saveError = "Name '\(formName)' already in use"
                    return
                }
                try? manager.delete(editing.name)
                do { try manager.create(provider) }
                catch { saveError = error.localizedDescription; return }
            } else {
                do { try manager.update(provider) }
                catch { saveError = error.localizedDescription; return }
            }
        } else {
            do { try manager.create(provider) }
            catch { saveError = error.localizedDescription; return }
        }
        if isCreatingNew {
            clearForm()
        } else {
            // Return to detail view for the saved provider
            editingProvider = nil
            selectedProvider = provider
            rightPanelMode = .detail
            reloadProviders()
        }
    }

    private func deleteProvider(_ provider: TokenhubProvider) {
        try? manager.delete(provider.name)
        if editingProvider?.id == provider.id { clearForm() }
        if selectedProvider?.id == provider.id {
            selectedProvider = nil
            rightPanelMode = .empty
        }
        reloadProviders()
    }

    private func deleteEditingProvider() {
        guard let p = editingProvider else { return }
        try? manager.delete(p.name)
        clearForm()
        reloadProviders()
    }

    /// Duplicate a provider with a unique name suffix.
    private func duplicateProvider(_ provider: TokenhubProvider) {
        if !manager.hasValidTknetKey() && manager.userProviderCount() >= TokenhubManager.freeProviderLimit {
            showProviderLimitAlert = true
            return
        }
        var baseName = provider.name
        // Strip existing numeric suffix like "-2", "-3" etc.
        if let range = baseName.range(of: #"-\d+$"#, options: .regularExpression) {
            baseName = String(baseName[baseName.startIndex..<range.lowerBound])
        }

        // Find next available unique name
        var candidate: String
        var suffix = 2
        repeat {
            candidate = "\(baseName)-\(suffix)"
            suffix += 1
        } while manager.get(candidate) != nil

        var newProvider = TokenhubProvider(
            name: candidate,
            endpoint: provider.endpoint,
            apiKey: provider.apiKey,
            remoteModel: provider.remoteModel,
            isEnabled: provider.isEnabled,
            tags: provider.tags,
            isFree: provider.isFree,
            supportsResponsesAPI: provider.supportsResponsesAPI,
            visionStrategy: provider.visionStrategy,
            anthropicEndpoint: provider.anthropicEndpoint,
            contextWindowOverride: provider.contextWindowOverride
        )
        // Reset stats for the duplicate
        newProvider.requestCount = 0
        newProvider.successCount = 0
        newProvider.avgLatencyMs = 0

        do {
            try manager.create(newProvider)
            reloadProviders()
            // Enter edit mode with the new provider
            if let created = manager.get(candidate) {
                loadFormFromProvider(created)
                isCreatingNew = false
                editingProvider = created
                rightPanelMode = .editing
            }
            agentToast = "Duplicated as \(candidate)"
            showAgentToast = true
        } catch {
            alertMessage = "Failed to duplicate: \(error.localizedDescription)"
            showAlert = true
        }
    }

    private func reloadProviders() {
        providers = manager.list()
            .sorted { $0.isEnabled && !$1.isEnabled }
    }

    private func bulkIndicatorColor(for provider: TokenhubProvider) -> Color {
        return provider.isEnabled ? NovaTheme.Colors.statusOK : NovaTheme.Colors.textTertiary
    }

    // MARK: - Bulk Test

    private func bulkTestAll() {
        bulkTestRunning = true
        bulkTestProgress = [:]
        let allProviders = manager.list()
        let key = appState.apiKey
        Task {
            await withTaskGroup(of: (String, Bool).self) { group in
                for provider in allProviders {
                    group.addTask {
                        let ok = await Self.testSingleProvider(provider, localApiKey: key)
                        return (provider.name, ok)
                    }
                }
                for await (name, ok) in group {
                    await MainActor.run {
                        bulkTestProgress[name] = ok ? "OK" : "FAIL"
                    }
                    if !ok {
                        if var p = manager.get(name) {
                            p.isEnabled = false
                            try? manager.update(p)
                        }
                    }
                }
            }
            await MainActor.run {
                bulkTestRunning = false
                reloadProviders()
            }
        }
    }

    private static func testSingleProvider(_ provider: TokenhubProvider, localApiKey: String?) async -> Bool {
        guard !provider.remoteModel.isEmpty else { return false }
        let endpoint = provider.endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: endpoint) else { return false }
        var request = URLRequest(url: url.appendingPathComponent("chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        // Cloud-managed providers (tagged "managed") inherit the session token;
        // all others use their own API key.
        let effectiveKey: String
        if provider.tags.contains("managed") {
            effectiveKey = AuthCache.loadSession() ?? ""
        } else {
            effectiveKey = provider.apiKey
        }
        if !effectiveKey.isEmpty {
            request.setValue("Bearer \(effectiveKey)", forHTTPHeaderField: "Authorization")
        }
        let body: [String: Any] = [
            "model": provider.remoteModel,
            "messages": [["role": "user", "content": "Hi"]],
            "max_tokens": 1,
            "stream": false
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            return code == 200
        } catch {
            return false
        }
    }

    private func queryModels() {
        isQueryingModels = true
        queryError = nil
        let endpoint = formEndpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let key = formApiKey
        Task {
            guard let url = URL(string: endpoint) else {
                await MainActor.run {
                    queryError = "Invalid endpoint URL"
                    isQueryingModels = false
                }
                return
            }
            var request = URLRequest(url: url.appendingPathComponent("models"))
            request.timeoutInterval = 15
            if !key.isEmpty {
                request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            }
            do {
                let (data, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                    await MainActor.run {
                        queryError = "HTTP \(code)"
                        isQueryingModels = false
                    }
                    return
                }
                struct ModelsResponse: Decodable { let data: [ModelEntry] }
                struct ModelEntry: Decodable { let id: String }
                let decoded = try JSONDecoder().decode(ModelsResponse.self, from: data)
                let models = decoded.data.map(\.id).sorted()
                await MainActor.run {
                    availableModels = models
                    isQueryingModels = false
                    markVerified()
                    if let first = models.first, formRemoteModel.isEmpty {
                        formRemoteModel = first
                    }
                }
            } catch {
                await MainActor.run {
                    queryError = error.localizedDescription
                    isQueryingModels = false
                }
            }
        }
    }

    // MARK: - Test Provider Endpoint

    private func testProviderEndpoint() {
        testEndpointRunning = true
        testEndpointResult = nil
        let endpoint = formEndpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let key = formApiKey
        Task {
            guard let url = URL(string: endpoint) else {
                await MainActor.run { testEndpointResult = "Invalid URL"; testEndpointRunning = false }
                return
            }
            var request = URLRequest(url: url.appendingPathComponent("models"))
            request.timeoutInterval = 15
            if !key.isEmpty {
                request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            }
            do {
                let (_, response) = try await URLSession.shared.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                await MainActor.run {
                    testEndpointResult = code == 200 ? "OK (\(code))" : "Failed (HTTP \(code))"
                    testEndpointRunning = false
                    if code == 200 { markVerified() }
                }
            } catch {
                await MainActor.run {
                    testEndpointResult = "Error: \(error.localizedDescription)"
                    testEndpointRunning = false
                }
            }
        }
    }

    // MARK: - Test via NovaMLX (provider-name)

    private func testViaNovaMLX() {
        testProxyRunning = true
        testProxyResult = nil
        let providerName = formName
        let port = appState.serverPort
        let key = appState.apiKey
        Task {
            let provider = TokenhubProvider(
                name: formName,
                endpoint: formEndpoint,
                apiKey: formApiKey,
                remoteModel: formRemoteModel,
                isEnabled: true,
                tags: parseTags(formTags),
                isFree: formIsFree,
                supportsResponsesAPI: formSupportsResponses
            )
            if editingProvider == nil {
                if manager.get(providerName) == nil {
                    try? manager.create(provider)
                }
            } else {
                _ = try? manager.update(provider)
            }

            let useResponses = formSupportsResponses
            let path = useResponses ? "/v1/responses" : "/v1/chat/completions"
            guard let url = URL(string: "http://127.0.0.1:\(port)\(path)") else {
                await MainActor.run { testProxyResult = "Invalid local URL"; testProxyRunning = false }
                return
            }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 30
            if let key {
                request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            }

            let modelName = "tknet:\(providerName)"
            let body: [String: Any]
            if useResponses {
                body = [
                    "model": modelName,
                    "input": "Hi, reply with just 'OK'",
                    "max_output_tokens": 10,
                    "stream": false
                ]
            } else {
                body = [
                    "model": modelName,
                    "messages": [["role": "user", "content": "Hi, reply with just 'OK'"]],
                    "max_tokens": 10,
                    "stream": false
                ]
            }
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)

            do {
                let (data, response) = try await URLSession.shared.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                if code == 200 {
                    let _ = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])
                    let providerHeader = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "X-Tokenhub-Provider")
                    await MainActor.run {
                        testProxyResult = "OK via \(providerHeader ?? providerName) (HTTP \(code))"
                        testProxyRunning = false
                        markVerified()
                        reloadProviders()
                    }
                } else {
                    let msg = String(data: data, encoding: .utf8)?.prefix(200).description ?? "unknown"
                    await MainActor.run {
                        testProxyResult = "Failed (HTTP \(code)): \(msg)"
                        testProxyRunning = false
                    }
                }
            } catch {
                await MainActor.run {
                    testProxyResult = "Error: \(error.localizedDescription)"
                    testProxyRunning = false
                }
            }
        }
    }
}
