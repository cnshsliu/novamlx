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
    @State private var isCreatingNew = false

    // Form
    @State private var formName = ""
    @State private var formEndpoint = ""
    @State private var formApiKey = ""
    @State private var formRemoteModel = ""
    @State private var formEnabled = false
    @State private var formIncludeInLB = false
    @State private var formIsFree = false
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

    private let manager = TokenhubManager.shared

    private var isFormActive: Bool {
        isCreatingNew || editingProvider != nil
    }

    private var isEditingManaged: Bool {
        editingProvider?.isManaged == true
    }

    var body: some View {
        HStack(spacing: 0) {
            leftPanel
                .frame(width: 200)
            Divider()
            rightPanel
                .frame(maxWidth: .infinity, maxHeight: .infinity)
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
    }

    // MARK: - Left Panel (My Providers only)

    private var leftPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("My Providers")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                if !CloudAuth.isSubscribed() {
                    let userCount = providers.filter { !$0.isManaged }.count
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
            } else {
                let managed = providers.filter { $0.isManaged }
                let userProviders = providers.filter { !$0.isManaged }

                if !managed.isEmpty {
                    HStack {
                        Image(systemName: "cloud.fill")
                            .font(.system(size: 10))
                            .foregroundColor(NovaTheme.Colors.accent)
                        Text("Cloud Models")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.top, 4)
                    .padding(.bottom, 2)

                    ForEach(managed) { provider in
                        myProviderRow(provider)
                    }

                    if !userProviders.isEmpty {
                        Divider()
                            .padding(.horizontal, 12)
                            .padding(.vertical, 4)
                    }
                }

                ForEach(userProviders) { provider in
                    myProviderRow(provider)
                }
            }

            Spacer()
        }
        .background(NovaTheme.Colors.cardBackground)
    }

    private func myProviderRow(_ provider: TokenhubProvider) -> some View {
        Button(action: { selectMyProvider(provider) }) {
            HStack(spacing: 8) {
                Circle()
                    .fill(bulkIndicatorColor(for: provider))
                    .frame(width: 6, height: 6)
                VStack(alignment: .leading, spacing: 1) {
                    Text(provider.name)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(editingProvider?.id == provider.id ? NovaTheme.Colors.accent : NovaTheme.Colors.textPrimary)
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
            .background(editingProvider?.id == provider.id ? NovaTheme.Colors.accent.opacity(0.15) : Color.clear)
            .overlay(alignment: .leading) {
                if editingProvider?.id == provider.id {
                    RoundedRectangle(cornerRadius: 1)
                        .fill(NovaTheme.Colors.accent)
                        .frame(width: 2)
                }
            }
            .cornerRadius(6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovering in
            if isHovering { NSCursor.pointingHand.push() }
            else { NSCursor.pop() }
        }
        .contextMenu {
            Button(role: .destructive) {
                deleteProvider(provider)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    // MARK: - Right Panel

    private var rightPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if isFormActive {
                    formContent
                } else {
                    emptyState
                }
            }
            .padding(24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                }

                // API Key
                formField(label: "API Key", hint: nil) {
                    SecureField("sk-...", text: $formApiKey)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
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
                Toggle("Load Balance", isOn: $formIncludeInLB)
                    .controlSize(.small)
                    .onChange(of: formIncludeInLB) { saveManagedToggles() }
                if !isEditingManaged {
                    Toggle("Free", isOn: $formIsFree)
                        .controlSize(.small)
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
                        Text(isCreatingNew ? "Save" : "Update")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(formName.isEmpty || formEndpoint.isEmpty)

                    if editingProvider != nil {
                        Button(role: .destructive, action: { deleteEditingProvider() }) {
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
        // Check free-tier limit
        if !CloudAuth.isSubscribed() && manager.userProviderCount() >= TokenhubManager.freeProviderLimit {
            saveError = "Free tier: max \(TokenhubManager.freeProviderLimit) providers. Subscribe for unlimited."
            return
        }
        isCreatingNew = true
        editingProvider = nil
        formName = ""
        formEndpoint = ""
        formApiKey = ""
        formRemoteModel = ""
        formEnabled = false
        formIncludeInLB = false
        formIsFree = false
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
        editingProvider = provider
        formName = provider.name
        formEndpoint = provider.endpoint
        formApiKey = provider.apiKey
        formRemoteModel = provider.remoteModel
        formEnabled = provider.isEnabled
        formIncludeInLB = provider.includeInLoadBalance
        formIsFree = provider.isFree
        formTags = provider.tags.joined(separator: ", ")
        isVerified = provider.isEnabled || provider.includeInLoadBalance
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
        formIncludeInLB = false
        formIsFree = false
        formTags = ""
        isVerified = false
        availableModels = []
        queryError = nil
        saveError = nil
        testEndpointResult = nil
        testProxyResult = nil
    }

    private func parseTags(_ raw: String) -> [String] {
        raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty }
    }

    /// Auto-detect if an endpoint points to a local server.
    private func isLocalEndpoint(_ endpoint: String) -> Bool {
        let lower = endpoint.lowercased()
        return lower.contains("127.0.0.1") || lower.contains("localhost") || lower.contains("::1")
    }

    /// Save Enabled/LB toggles for managed providers (auto-saved on toggle change).
    private func saveManagedToggles() {
        guard let editing = editingProvider, editing.isManaged else { return }
        var updated = editing
        updated.isEnabled = formEnabled
        updated.includeInLoadBalance = formIncludeInLB
        try? manager.update(updated)
        editingProvider = updated
        reloadProviders()
    }

    private func saveProvider() {
        saveError = nil
        let provider = TokenhubProvider(
            name: formName,
            endpoint: formEndpoint,
            apiKey: formApiKey,
            remoteModel: formRemoteModel,
            isEnabled: formEnabled,
            includeInLoadBalance: formIncludeInLB,
            tags: parseTags(formTags),
            isLocal: isLocalEndpoint(formEndpoint),
            isFree: formIsFree
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
            // Stay selected on the updated provider
            editingProvider = provider
            reloadProviders()
        }
    }

    private func deleteProvider(_ provider: TokenhubProvider) {
        try? manager.delete(provider.name)
        if editingProvider?.id == provider.id { clearForm() }
        reloadProviders()
    }

    private func deleteEditingProvider() {
        guard let p = editingProvider else { return }
        try? manager.delete(p.name)
        clearForm()
        reloadProviders()
    }

    private func reloadProviders() {
        providers = manager.list()
            .sorted { $0.isEnabled && !$1.isEnabled }
    }

    private func bulkIndicatorColor(for provider: TokenhubProvider) -> Color {
        if let result = bulkTestProgress[provider.name] {
            return result == "OK" ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError
        }
        return provider.isEnabled ? NovaTheme.Colors.statusOK : NovaTheme.Colors.textTertiary
    }

    // MARK: - Bulk Test

    private func bulkTestAll() {
        bulkTestRunning = true
        bulkTestProgress = [:]
        let allProviders = manager.list()
        Task {
            await withTaskGroup(of: (String, Bool).self) { group in
                for provider in allProviders {
                    group.addTask {
                        let ok = await Self.testSingleProvider(provider)
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

    private static func testSingleProvider(_ provider: TokenhubProvider) async -> Bool {
        guard !provider.remoteModel.isEmpty else { return false }
        let endpoint = provider.endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: endpoint) else { return false }
        var request = URLRequest(url: url.appendingPathComponent("chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        if !provider.apiKey.isEmpty {
            request.setValue("Bearer \(provider.apiKey)", forHTTPHeaderField: "Authorization")
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

    // MARK: - Test via NovaMLX (tknet:provider-name)

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
                includeInLoadBalance: formIncludeInLB,
                tags: parseTags(formTags),
                isLocal: isLocalEndpoint(formEndpoint),
                isFree: formIsFree
            )
            if editingProvider == nil {
                if manager.get(providerName) == nil {
                    try? manager.create(provider)
                }
            } else {
                _ = try? manager.update(provider)
            }

            guard let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else {
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

            let body: [String: Any] = [
                "model": "tknet:\(providerName)",
                "messages": [["role": "user", "content": "Hi, reply with just 'OK'"]],
                "max_tokens": 10,
                "stream": false
            ]
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
