import SwiftUI
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils
import NovaMLXDB

struct APIKeysPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n

    @State private var managedKeys: [APIKey] = []
    @State private var showCreateKeySheet = false
    @State private var newKeyName = ""
    @State private var newKeyRateLimit: Double? = nil
    @State private var newKeyRateBurst: Int? = nil
    @State private var newKeyAllowedModels: [String] = []
    @State private var newKeyAllowedEndpoints: [String] = []
    @State private var newKeyMaxTokens: Int64? = nil
    @State private var newKeyMaxRequests: Int64? = nil
    @State private var newKeyResetPeriod: UsageResetPeriod = .daily
    @State private var createdRawKey: String? = nil
    @State private var expandedKeyId: String? = nil
    @State private var editingKey: APIKey? = nil
    @State private var revealedKeyIds: Set<String> = []
    @State private var revealedRawKeys: [String: String] = [:]
    @State private var showDeleteConfirm: String? = nil

    private let loadedModelNames: [String]

    init(appState: MenuBarAppState, inferenceService: InferenceService, modelManager: ModelManager) {
        self.appState = appState
        self.inferenceService = inferenceService
        self.modelManager = modelManager
        self.loadedModelNames = []
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                keysSection
            }
            .padding(24)
        }
        .sheet(isPresented: $showCreateKeySheet) {
            createKeySheet
        }
        .sheet(item: $editingKey) { key in
            editKeySheet(key)
        }
        .task { loadManagedKeys() }
        .onReceive(NotificationCenter.default.publisher(for: Notification.Name("APIKeysChanged"))) { _ in
            loadManagedKeys()
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionHeader("API Keys", icon: "key.fill", count: managedKeys.count)
                Spacer()
                Button {
                    newKeyName = ""
                    newKeyRateLimit = nil
                    newKeyRateBurst = nil
                    newKeyAllowedModels = []
                    newKeyAllowedEndpoints = []
                    newKeyMaxTokens = nil
                    newKeyMaxRequests = nil
                    newKeyResetPeriod = .daily
                    createdRawKey = nil
                    showCreateKeySheet = true
                } label: {
                    Label("Create Key", systemImage: "plus")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            if managedKeys.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                    Text("No API keys configured. The server is running in open mode (no authentication required).")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                .padding(8)
            }
        }
    }

    // MARK: - Key List

    private var keysSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(managedKeys, id: \.id) { key in
                apiKeyRow(key)
            }
        }
    }

    // MARK: - Key Row

    private func apiKeyRow(_ key: APIKey) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Circle()
                    .fill(key.isActive ? NovaTheme.Colors.statusOK : Color.orange)
                    .frame(width: 6, height: 6)

                Text(key.name)
                    .font(.system(size: 12, weight: .medium))

                // Masked key display with eye icon
                HStack(spacing: 4) {
                    if revealedKeyIds.contains(key.id), let raw = revealedRawKeys[key.id] {
                        Text(raw)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.primary)
                            .textSelection(.enabled)
                    } else {
                        Text(shortMaskedKey(key))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.secondary)
                    }

                    Button {
                        if revealedKeyIds.contains(key.id) {
                            revealedKeyIds.remove(key.id)
                        } else {
                            // Fetch plaintext from SQLite
                            if let raw = try? NovaDB.shared.apiKeyStore.getRawKey(id: key.id) {
                                revealedRawKeys[key.id] = raw
                                revealedKeyIds.insert(key.id)
                            }
                        }
                    } label: {
                        Image(systemName: revealedKeyIds.contains(key.id) ? "eye.slash" : "eye")
                            .font(.system(size: 9))
                            .foregroundColor(key.isLegacyImport ? .secondary.opacity(0.4) : .secondary)
                    }
                    .buttonStyle(.plain)
                    .disabled(key.isLegacyImport)
                    .help(revealedKeyIds.contains(key.id)
                          ? "Hide key"
                          : (key.isLegacyImport
                             ? "Pre-DB key — rotate to enable reveal"
                             : "Reveal plaintext key"))
                }

                Spacer()

                Text("\(key.usage.totalRequests) reqs")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)

                Toggle("", isOn: Binding(
                    get: { key.isEnabled },
                    set: { _ in toggleKey(key.id) }
                ))
                .toggleStyle(.switch)
                .controlSize(.mini)

                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        expandedKeyId = (expandedKeyId == key.id) ? nil : key.id
                        showDeleteConfirm = nil
                    }
                } label: {
                    Image(systemName: expandedKeyId == key.id ? "chevron.up" : "chevron.down")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }

            if expandedKeyId == key.id {
                apiKeyDetails(key)
            }
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.sm))
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.sm)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
        )
    }

    // MARK: - Key Details (expanded)

    private func apiKeyDetails(_ key: APIKey) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()

            // Usage stats
            HStack(spacing: 16) {
                Label("\(formatNumber(key.usage.totalTokensUsed)) tokens", systemImage: "text.word.spacing")
                Label("\(formatNumber(key.usage.totalRequests)) requests", systemImage: "arrow.triangle.2.circlepath")
                if let lastUsed = key.usage.lastUsedAt {
                    Label(timeAgo(lastUsed), systemImage: "clock")
                }
            }
            .font(.system(size: 10))
            .foregroundColor(.secondary)

            // Limits + progress bar
            if key.rateLimitPerSecond != nil || key.maxTokensPerPeriod != nil || key.maxRequestsPerPeriod != nil {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 12) {
                        if let rps = key.rateLimitPerSecond {
                            Text("Rate: \(rps, specifier: "%.0f")/s")
                                .font(.system(size: 10))
                        }
                        if let maxT = key.maxTokensPerPeriod {
                            Text("\(key.usageResetPeriod.rawValue.capitalized) tokens: \(formatNumber(key.usage.periodTokens))/\(formatNumber(maxT))")
                                .font(.system(size: 10))
                        }
                        if let maxR = key.maxRequestsPerPeriod {
                            Text("\(key.usageResetPeriod.rawValue.capitalized) reqs: \(key.usage.periodRequests)/\(maxR)")
                                .font(.system(size: 10))
                        }
                    }
                    .foregroundColor(NovaTheme.Colors.accent)
                    .font(.system(size: 10))

                    // Progress bar
                    if let maxT = key.maxTokensPerPeriod, maxT > 0 {
                        let fraction = min(1.0, Double(key.usage.periodTokens) / Double(maxT))
                        HStack(spacing: 6) {
                            GeometryReader { geo in
                                ZStack(alignment: .leading) {
                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(Color.gray.opacity(0.2))
                                        .frame(height: 6)
                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(fraction > 0.9 ? Color.red : NovaTheme.Colors.accent)
                                        .frame(width: geo.size.width * fraction, height: 6)
                                }
                            }
                            .frame(height: 6)
                            Text("\(Int(fraction * 100))%")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(fraction > 0.9 ? .red : .secondary)
                                .frame(width: 30)
                        }
                    }
                }
            }

            // Per-model usage breakdown
            if !key.usage.perModelTokens.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Usage by model:")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    let sorted = key.usage.perModelTokens.sorted { $0.value > $1.value }
                    ForEach(sorted.prefix(5), id: \.key) { model, tokens in
                        HStack(spacing: 4) {
                            Text(shortModelName(model))
                                .font(.system(size: 9))
                                .lineLimit(1)
                            Spacer()
                            Text(formatNumber(tokens))
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                    }
                    if sorted.count > 5 {
                        Text("+\(sorted.count - 5) more...")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Allowed models
            if let models = key.allowedModels, !models.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Allowed models:")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    FlowLayout(spacing: 4) {
                        ForEach(models, id: \.self) { m in
                            Text(m)
                                .font(.system(size: 9))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(NovaTheme.Colors.accent.opacity(0.1))
                                .clipShape(Capsule())
                        }
                    }
                }
            }

            // Allowed endpoints
            if let endpoints = key.allowedEndpoints, !endpoints.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Allowed endpoints:")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    FlowLayout(spacing: 4) {
                        ForEach(endpoints, id: \.self) { e in
                            Text(e)
                                .font(.system(size: 9))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(NovaTheme.Colors.accent.opacity(0.1))
                                .clipShape(Capsule())
                        }
                    }
                }
            }

            // Actions
            HStack {
                CopyIDButton(id: key.keyPrefix)

                Button {
                    editingKey = key
                } label: {
                    Label("Edit", systemImage: "pencil")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)

                Button("Rotate") {
                    rotateKey(key.id)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .font(.system(size: 10))

                Spacer()

                // Delete with confirmation
                if showDeleteConfirm == key.id {
                    Text("Delete?")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.red)
                    Button("Yes") {
                        deleteKey(key.id)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .font(.system(size: 9))
                    .foregroundColor(.red)

                    Button("No") {
                        showDeleteConfirm = nil
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .font(.system(size: 9))
                } else {
                    Button(role: .destructive) {
                        showDeleteConfirm = key.id
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.red.opacity(0.7))
                }
            }
        }
    }

    // MARK: - Create Key Sheet

    private var createKeySheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let rawKey = createdRawKey {
                Text("API Key Created")
                    .font(.system(size: 13, weight: .semibold))
                Text("Copy this key now — it won't be shown again.")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                HStack {
                    Text(rawKey)
                        .font(.system(size: 11, design: .monospaced))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(rawKey, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.plain)
                }
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))

                Button("Done") {
                    showCreateKeySheet = false
                    loadManagedKeys()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            } else {
                Text("Create API Key")
                    .font(.system(size: 13, weight: .semibold))

                TextField("My API Key", text: $newKeyName)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))

                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Rate limit/s").font(.system(size: 10)).foregroundColor(.secondary)
                        TextField("Unlimited", value: $newKeyRateLimit, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12))
                            .frame(width: 100)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Burst").font(.system(size: 10)).foregroundColor(.secondary)
                        TextField("Default", value: $newKeyRateBurst, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12))
                            .frame(width: 80)
                    }
                }

                // Allowed models
                VStack(alignment: .leading, spacing: 2) {
                    Text("Allowed models (empty = all)").font(.system(size: 10)).foregroundColor(.secondary)
                    ItemInput(
                        items: $newKeyAllowedModels,
                        suggestions: appState.loadedModels,
                        placeholder: "Model name..."
                    )
                }

                // Allowed endpoints
                VStack(alignment: .leading, spacing: 2) {
                    Text("Allowed endpoints (empty = all)").font(.system(size: 10)).foregroundColor(.secondary)
                    ItemInput(
                        items: $newKeyAllowedEndpoints,
                        suggestions: ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/v1/models"],
                        placeholder: "/v1/..."
                    )
                }

                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Max tokens/period").font(.system(size: 10)).foregroundColor(.secondary)
                        TextField("Unlimited", value: $newKeyMaxTokens, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12))
                            .frame(width: 120)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Max requests/period").font(.system(size: 10)).foregroundColor(.secondary)
                        TextField("Unlimited", value: $newKeyMaxRequests, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12))
                            .frame(width: 120)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Reset period").font(.system(size: 10)).foregroundColor(.secondary)
                        Picker("", selection: $newKeyResetPeriod) {
                            ForEach(UsageResetPeriod.allCases, id: \.self) { p in
                                Text(p.rawValue.capitalized).tag(p)
                            }
                        }
                        .pickerStyle(.menu)
                        .frame(width: 90)
                    }
                }

                HStack {
                    Button("Cancel") {
                        showCreateKeySheet = false
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Spacer()

                    Button("Create") {
                        createNewKey()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(newKeyName.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    // MARK: - Edit Key Sheet

    private func editKeySheet(_ key: APIKey) -> some View {
        EditKeySheet(key: key, appState: appState, onSave: { updated in
            Task {
                do {
                    try NovaDB.shared.apiKeyStore.update(id: key.id) { rec in
                        rec.name = updated.name
                        rec.rateLimitPerSecond = updated.rateLimitPerSecond
                        rec.rateLimitBurst = updated.rateLimitBurst
                        rec.allowedModels = Self.encodeJSONField(updated.allowedModels)
                        rec.allowedEndpoints = Self.encodeJSONField(updated.allowedEndpoints)
                        rec.maxTokensPerPeriod = updated.maxTokensPerPeriod
                        rec.maxRequestsPerPeriod = updated.maxRequestsPerPeriod
                        rec.usageResetPeriod = updated.usageResetPeriod.rawValue
                    }
                    editingKey = nil
                    loadManagedKeys()
                } catch {
                    NovaMLXLog.error("[APIKeys] Failed to update key: \(error)")
                }
            }
        }, onCancel: {
            editingKey = nil
        })
    }

    // MARK: - Actions

    private func loadManagedKeys() {
        Task {
            managedKeys = (try? NovaDB.shared.apiKeyStore.listAsAPIKey()) ?? []
        }
    }

    private func createNewKey() {
        let name = newKeyName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return }
        Task {
            do {
                let (_, raw) = try NovaDB.shared.apiKeyStore.create(
                    name: name,
                    rateLimitPerSecond: newKeyRateLimit,
                    rateLimitBurst: newKeyRateBurst,
                    allowedModels: newKeyAllowedModels.isEmpty ? nil : newKeyAllowedModels,
                    allowedEndpoints: newKeyAllowedEndpoints.isEmpty ? nil : newKeyAllowedEndpoints,
                    maxTokensPerPeriod: newKeyMaxTokens,
                    maxRequestsPerPeriod: newKeyMaxRequests,
                    usageResetPeriod: newKeyResetPeriod.rawValue
                )
                createdRawKey = raw
                loadManagedKeys()
            } catch {
                NovaMLXLog.error("[APIKeys] Failed to create key: \(error)")
            }
        }
    }

    private func toggleKey(_ id: String) {
        Task {
            do {
                try NovaDB.shared.apiKeyStore.update(id: id) { rec in
                    rec.isEnabled.toggle()
                }
                loadManagedKeys()
            } catch {
                NovaMLXLog.error("[APIKeys] Failed to toggle key: \(error)")
            }
        }
    }

    private func deleteKey(_ id: String) {
        Task {
            do {
                try NovaDB.shared.apiKeyStore.delete(id: id)
                if expandedKeyId == id { expandedKeyId = nil }
                showDeleteConfirm = nil
                loadManagedKeys()
            } catch {
                NovaMLXLog.error("[APIKeys] Failed to delete key: \(error)")
            }
        }
    }

    private func rotateKey(_ id: String) {
        Task {
            do {
                let (_, raw) = try NovaDB.shared.apiKeyStore.rotate(id: id)
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(raw, forType: .string)
                loadManagedKeys()
            } catch {
                NovaMLXLog.error("[APIKeys] Failed to rotate key: \(error)")
            }
        }
    }

    // MARK: - Helpers

    /// JSON-encode an optional Encodable value into a String for the store's
    /// JSON-string columns (`allowed_models`, `allowed_endpoints`). Returns nil
    /// for nil input or encode failures — matching how the importer writes them.
    private nonisolated static func encodeJSONField<T: Encodable>(_ value: T?) -> String? {
        guard let value else { return nil }
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func shortMaskedKey(_ key: APIKey) -> String {
        if key.keySuffix.isEmpty {
            return key.keyPrefix + "..."
        }
        return String(key.keyPrefix.prefix(15)) + "..." + key.keySuffix
    }

    private func shortModelName(_ id: String) -> String {
        if let slash = id.lastIndex(of: "/") {
            return String(id[id.index(after: slash)...])
        }
        return id
    }

    private func formatNumber(_ n: Int64) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    private func timeAgo(_ date: Date) -> String {
        let interval = Date().timeIntervalSince(date)
        if interval < 60 { return "just now" }
        if interval < 3600 { return "\(Int(interval / 60))m ago" }
        if interval < 86400 { return "\(Int(interval / 3600))h ago" }
        return "\(Int(interval / 86400))d ago"
    }
}

// MARK: - Edit Key Sheet

private struct EditKeySheet: View {
    let key: APIKey
    let appState: MenuBarAppState
    let onSave: (APIKey) -> Void
    let onCancel: () -> Void

    @State private var name: String = ""
    @State private var rateLimitPerSecond: Double? = nil
    @State private var rateLimitBurst: Int? = nil
    @State private var allowedModels: [String] = []
    @State private var allowedEndpoints: [String] = []
    @State private var maxTokensPerPeriod: Int64? = nil
    @State private var maxRequestsPerPeriod: Int64? = nil
    @State private var usageResetPeriod: UsageResetPeriod = .daily

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit API Key")
                .font(.system(size: 13, weight: .semibold))

            Text(key.maskedDisplay)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)

            TextField("Key Name", text: $name)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Rate limit/s").font(.system(size: 10)).foregroundColor(.secondary)
                    TextField("Unlimited", value: $rateLimitPerSecond, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))
                        .frame(width: 100)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Burst").font(.system(size: 10)).foregroundColor(.secondary)
                    TextField("Default", value: $rateLimitBurst, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))
                        .frame(width: 80)
                }
            }

            // Allowed models
            VStack(alignment: .leading, spacing: 2) {
                Text("Allowed models (empty = all)").font(.system(size: 10)).foregroundColor(.secondary)
                ItemInput(
                    items: $allowedModels,
                    suggestions: appState.loadedModels,
                    placeholder: "Model name..."
                )
            }

            // Allowed endpoints
            VStack(alignment: .leading, spacing: 2) {
                Text("Allowed endpoints (empty = all)").font(.system(size: 10)).foregroundColor(.secondary)
                ItemInput(
                    items: $allowedEndpoints,
                    suggestions: ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/v1/models"],
                    placeholder: "/v1/..."
                )
            }

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Max tokens/period").font(.system(size: 10)).foregroundColor(.secondary)
                    TextField("Unlimited", value: $maxTokensPerPeriod, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))
                        .frame(width: 120)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Max requests/period").font(.system(size: 10)).foregroundColor(.secondary)
                    TextField("Unlimited", value: $maxRequestsPerPeriod, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))
                        .frame(width: 120)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Reset period").font(.system(size: 10)).foregroundColor(.secondary)
                    Picker("", selection: $usageResetPeriod) {
                        ForEach(UsageResetPeriod.allCases, id: \.self) { p in
                            Text(p.rawValue.capitalized).tag(p)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 90)
                }
            }

            HStack {
                Button("Cancel") {
                    onCancel()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Spacer()

                Button("Save") {
                    var updated = key
                    updated.name = name.trimmingCharacters(in: .whitespaces)
                    updated.rateLimitPerSecond = rateLimitPerSecond
                    updated.rateLimitBurst = rateLimitBurst
                    updated.allowedModels = allowedModels.isEmpty ? nil : allowedModels
                    updated.allowedEndpoints = allowedEndpoints.isEmpty ? nil : allowedEndpoints
                    updated.maxTokensPerPeriod = maxTokensPerPeriod
                    updated.maxRequestsPerPeriod = maxRequestsPerPeriod
                    updated.usageResetPeriod = usageResetPeriod
                    onSave(updated)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 420)
        .onAppear {
            name = key.name
            rateLimitPerSecond = key.rateLimitPerSecond
            rateLimitBurst = key.rateLimitBurst
            allowedModels = key.allowedModels ?? []
            allowedEndpoints = key.allowedEndpoints ?? []
            maxTokensPerPeriod = key.maxTokensPerPeriod
            maxRequestsPerPeriod = key.maxRequestsPerPeriod
            usageResetPeriod = key.usageResetPeriod
        }
    }
}
