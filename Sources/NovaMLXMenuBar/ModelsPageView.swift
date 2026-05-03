import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

struct ModelsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n

    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var refreshTrigger = false
    @State private var selectedModelCard: ModelCardData?
    @State private var isLoadingCard = false
    @State private var modelToDelete: String?
    @State private var showDeleteConfirmation = false
    @State private var loadingModelId: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                loadedSection
                cloudModelsSection
                downloadedSection
            }
            .padding(24)
        }
        .alert(alertMessage, isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        }
        .sheet(item: $selectedModelCard) { card in
            modelCardSheet(card)
        }
        .alert(l10n.tr("models.deleteConfirm"), isPresented: $showDeleteConfirmation) {
            Button(l10n.tr("models.cancel"), role: .cancel) {
                modelToDelete = nil
            }
            Button(l10n.tr("models.delete"), role: .destructive) {
                if let id = modelToDelete {
                    if let record = modelManager.getRecord(id) {
                        Task {
                            await inferenceService.unloadModel(ModelIdentifier(id: id, family: record.family))
                        }
                    }
                    try? modelManager.deleteModel(id)
                    refreshTrigger.toggle()
                }
                modelToDelete = nil
            }
        } message: {
            Text(l10n.tr("models.deleteMessage", modelToDelete ?? ""))
        }
        .onReceive(NotificationCenter.default.publisher(for: .novaMLXModelsChanged)) { _ in
            refreshTrigger.toggle()
        }
    }

    private var loadedSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("status.activeModels"), icon: "bolt.fill", count: appState.loadedModels.count)

            if appState.loadedModels.isEmpty {
                emptyState(l10n.tr("models.noModelsLoaded"), subtitle: l10n.tr("models.noModelsLoadedSub"))
            } else {
                ForEach(appState.loadedModels, id: \.self) { modelId in
                    modelRow(
                        modelId,
                        subtitle: modelManager.getRecord(modelId)?.family.rawValue ?? l10n.tr("models.unknown"),
                        isLoaded: true,
                        actions: {
                            Button(l10n.tr("models.unload")) {
                                Task {
                                    if let record = modelManager.getRecord(modelId) {
                                        await inferenceService.unloadModel(ModelIdentifier(id: modelId, family: record.family))
                                        refreshTrigger.toggle()
                                    }
                                }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    )
                }
            }
        }
        .sectionCard()
    }

    private var cloudModelsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("models.cloudModels"), icon: "cloud.fill", count: appState.cloudModels.count)

            if !appState.cloudLoggedIn {
                cloudAuthGate
            } else if appState.cloudModels.isEmpty {
                emptyState(l10n.tr("models.noCloudModels"), subtitle: l10n.tr("models.noCloudModelsSub"))
            } else {
                ForEach(appState.cloudModels, id: \.self) { modelId in
                    cloudModelRow(modelId)
                }
            }
        }
        .sectionCard()
    }

    /// Promotional card shown when user is not logged in — drives them to Settings > Cloud Account
    private var cloudAuthGate: some View {
        VStack(spacing: 14) {
            Image(systemName: "cloud.fill")
                .font(.system(size: 28))
                .foregroundColor(NovaTheme.Colors.accent.opacity(0.6))

            VStack(spacing: 4) {
                Text("Unlock Cloud AI Models")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                Text("Run powerful models like Qwen3.5-35B remotely — no local GPU memory needed. Fast inference, zero hardware limits.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
            }

            HStack(spacing: 10) {
                Button {
                    appState.requestedPage = .settings
                } label: {
                    Label("Connect Cloud", systemImage: "arrow.right.circle")
                        .font(.system(size: 12, weight: .medium))
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)

                if let url = URL(string: "\(AuthClient.defaultBaseURL)/cloud") {
                    Button("Learn More") {
                        NSWorkspace.shared.open(url)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(28)
        .background(
            LinearGradient(
                colors: [NovaTheme.Colors.accent.opacity(0.06), NovaTheme.Colors.accent.opacity(0.02)],
                startPoint: .top, endPoint: .bottom
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }

    private func cloudModelRow(_ modelId: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "cloud.fill")
                .foregroundColor(NovaTheme.Colors.accent)
                .font(.system(size: 14))

            VStack(alignment: .leading, spacing: 2) {
                Text(modelId)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)
                    .foregroundColor(NovaTheme.Colors.accent)
            }

            Spacer()

            Text("Cloud")
                .font(.caption2)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(NovaTheme.Colors.accentDim)
                .clipShape(Capsule())
        }
        .rowCard()
    }

    private var downloadedSection: some View {
        let allDownloaded = modelManager.downloadedModels()
        let loaded = Set(inferenceService.listLoadedModels())
        let downloaded = allDownloaded.filter { !loaded.contains($0.id) }

        return VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("models.noInactiveModels"), icon: "arrow.down.circle", count: downloaded.count)

            if downloaded.isEmpty {
                emptyState(l10n.tr("models.noInactiveModels"), subtitle: l10n.tr("models.noInactiveModelsSub"))
            } else {
                ForEach(downloaded, id: \.id) { record in
                    modelRow(
                        record.id,
                        subtitle: "\(record.family.rawValue)  \(record.sizeBytes > 0 ? record.sizeBytes.bytesFormatted : "")",
                        isLoaded: false,
                        actions: {
                            if loadingModelId == record.id {
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .controlSize(.small)
                                    Text(l10n.tr("models.loading"))
                                        .font(.caption).foregroundColor(NovaTheme.Colors.accent)
                                }
                            } else {
                                Button(l10n.tr("models.load")) {
                                    loadingModelId = record.id
                                    Task {
                                        let config = ModelConfig(
                                            identifier: ModelIdentifier(id: record.id, family: record.family),
                                            modelType: record.modelType
                                        )
                                        do {
                                            try await inferenceService.loadModel(at: record.localURL, config: config)
                                            refreshTrigger.toggle()
                                        } catch {
                                            alertMessage = error.localizedDescription
                                            showAlert = true
                                        }
                                        loadingModelId = nil
                                    }
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                            }

                            Button(role: .destructive) {
                                modelToDelete = record.id
                                showDeleteConfirmation = true
                            } label: {
                                Image(systemName: "trash").font(.caption)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    )
                }
            }
        }
        .sectionCard()
    }

    private func modelRow(_ modelId: String, subtitle: String, isLoaded: Bool, @ViewBuilder actions: () -> some View) -> some View {
        HStack(spacing: 12) {
            Circle()
                .fill(isLoaded ? NovaTheme.Colors.statusOK : Color.clear)
                .frame(width: 8, height: 8)
                .overlay(Circle().stroke(Color.secondary.opacity(0.3), lineWidth: 1))

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(modelId).font(.system(size: 13, weight: .medium)).lineLimit(1)
                        .foregroundColor(NovaTheme.Colors.accent)
                        .help(l10n.tr("models.clickDetails"))
                        .onTapGesture { fetchModelCard(repoId: modelId) }
                    CopyIDButton(id: modelId)
                }
                Text(subtitle).font(.caption2).foregroundColor(.secondary)
            }

            Spacer()
            actions()
        }
        .rowCard()
    }

    private func emptyState(_ title: String, subtitle: String) -> some View {
        VStack(spacing: 6) {
            Text(title).font(.headline).foregroundColor(.secondary)
            Text(subtitle).font(.caption).foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(40)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let mb = Double(bytes) / 1024 / 1024
        if mb >= 1024 { return String(format: "%.1f \(l10n.tr("models.gb"))", mb / 1024) }
        return String(format: "%.0f \(l10n.tr("models.mb"))", mb)
    }

    // MARK: - Model Card

    private func fetchModelCard(repoId: String) {
        isLoadingCard = true
        Task {
            let adminPort = appState.adminPort
            let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? repoId
            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/model-info?repo_id=\(encoded)") else {
                isLoadingCard = false
                return
            }
            do {
                var request = URLRequest(url: url)
                if let apiKey = appState.apiKey {
                    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                }
                let (data, response) = try await URLSession.shared.data(for: request)
                guard let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200 else {
                    isLoadingCard = false
                    return
                }
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    var card = ModelCardData(repoId: repoId)
                    card.author = json["author"] as? String
                    card.downloads = json["downloads"] as? Int
                    card.likes = json["likes"] as? Int
                    card.tags = json["tags"] as? [String] ?? []

                    if let cardData = json["cardData"] as? [String: Any] {
                        card.license = cardData["license"] as? String
                        card.language = cardData["language"] as? [String] ?? []
                    }
                    if let config = json["config"] as? [String: Any] {
                        card.architectures = config["architectures"] as? [String] ?? []
                        card.modelType = config["model_type"] as? String
                    }
                    if let siblings = json["siblings"] as? [[String: Any]] {
                        var totalSize: Int64 = 0
                        card.files = siblings.compactMap { f in
                            guard let name = f["rfilename"] as? String else { return nil }
                            let size = f["size"] as? Int64 ?? 0
                            totalSize += size
                            return ModelCardFile(name: name, size: size)
                        }
                        card.totalSize = totalSize
                    }

                    if let record = modelManager.getRecord(repoId) {
                        let dirSize = FileManager.default.directorySize(at: record.localURL)
                        card.localDiskSize = Int64(dirSize)
                    }

                    selectedModelCard = card
                }
            } catch {}
            isLoadingCard = false
        }
    }

    private func modelCardSheet(_ card: ModelCardData) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                HStack(spacing: 12) {
                    Image(systemName: "cube.box")
                        .font(.title2).foregroundColor(NovaTheme.Colors.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(card.repoId).font(.system(size: 15, weight: .semibold)).lineLimit(2)
                        HStack(spacing: 12) {
                            if let author = card.author {
                                Label(author, systemImage: "person.fill").font(.caption).foregroundColor(.secondary)
                            }
                            if let downloads = card.downloads {
                                Label("\(downloads)", systemImage: "arrow.down.circle").font(.caption).foregroundColor(.secondary)
                            }
                            if let likes = card.likes {
                                Label("\(likes)", systemImage: "heart.fill").font(.caption).foregroundColor(.secondary)
                            }
                        }
                    }
                    Spacer()
                }

                Divider()

                // Tags
                if !card.tags.isEmpty {
                    cardSection(l10n.tr("models.tags")) {
                        FlowLayout(spacing: 4) {
                            ForEach(card.tags.filter { !$0.isEmpty }.prefix(12), id: \.self) { tag in
                                Text(tag)
                                    .font(.caption2)
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(NovaTheme.Colors.accentDim)
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }
                }

                // Technical specs
                let hasSpecs = !card.architectures.isEmpty || card.modelType != nil || card.license != nil || !card.language.isEmpty
                if hasSpecs {
                    cardSection(l10n.tr("models.specifications")) {
                        VStack(alignment: .leading, spacing: 6) {
                            if !card.architectures.isEmpty {
                                specRow(l10n.tr("models.architecture"), value: card.architectures.joined(separator: ", "))
                            }
                            if let mt = card.modelType {
                                specRow(l10n.tr("models.modelType"), value: mt)
                            }
                            if let license = card.license {
                                specRow(l10n.tr("models.license"), value: license)
                            }
                            if !card.language.isEmpty {
                                specRow(l10n.tr("models.language"), value: card.language.joined(separator: ", "))
                            }
                            if let record = modelManager.getRecord(card.repoId) {
                                specRow(l10n.tr("models.family"), value: record.family.rawValue)
                                specRow(l10n.tr("models.type"), value: record.modelType.rawValue.uppercased())
                            }
                        }
                    }
                }

                // Size info
                let totalHF = card.totalSize
                let totalLocal = card.localDiskSize
                if totalHF > 0 || (totalLocal ?? 0) > 0 {
                    cardSection(l10n.tr("models.size")) {
                        VStack(alignment: .leading, spacing: 6) {
                            if totalHF > 0 {
                                specRow(l10n.tr("models.downloadSize"), value: formatBytes(totalHF))
                            }
                            if let local = totalLocal, local > 0 {
                                specRow(l10n.tr("models.diskUsage"), value: formatBytes(local))
                            }
                        }
                    }
                }

                // File listing
                if !card.files.isEmpty {
                    cardSection(l10n.tr("models.files", card.files.count)) {
                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(card.files, id: \.name) { file in
                                HStack {
                                    Image(systemName: "doc")
                                        .font(.system(size: 9)).foregroundColor(.secondary)
                                        .frame(width: 14)
                                    Text(file.name)
                                        .font(.system(size: 11, design: .monospaced))
                                        .lineLimit(1)
                                    Spacer()
                                    Text(file.size > 0 ? formatBytes(file.size) : "—")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                }

                // Actions
                HStack {
                    Spacer()
                    Button(l10n.tr("models.close")) { selectedModelCard = nil }
                        .keyboardShortcut(.cancelAction)
                }
            }
            .padding(24)
        }
        .frame(width: 520, height: 560)
    }

    private func cardSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.system(size: 12, weight: .semibold)).foregroundColor(.secondary)
            content()
        }
        .padding(12)
        .background(NovaTheme.Colors.rowBackground)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func specRow(_ label: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .font(.caption).foregroundColor(.secondary)
                .frame(width: 90, alignment: .trailing)
            Text(value)
                .font(.caption)
            Spacer()
        }
    }
}

struct HFSearchResult {
    let id: String
    let tags: [String]
}

// MARK: - Model Card Data

private struct ModelCardData: Identifiable {
    let id = UUID()
    let repoId: String
    var author: String?
    var downloads: Int?
    var likes: Int?
    var tags: [String] = []
    var license: String?
    var language: [String] = []
    var architectures: [String] = []
    var modelType: String?
    var files: [ModelCardFile] = []
    var totalSize: Int64 = 0
    var localDiskSize: Int64? = nil

    init(repoId: String) {
        self.repoId = repoId
    }
}

private struct ModelCardFile: Identifiable {
    let id = UUID()
    let name: String
    let size: Int64
}
