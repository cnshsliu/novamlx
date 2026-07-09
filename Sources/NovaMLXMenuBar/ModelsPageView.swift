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

    @State private var selectedTab: ModelsTab = .local
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var refreshTrigger = false
    @State private var selectedModelCard: ModelCardData?
    @State private var isLoadingCard = false
    @State private var modelToDelete: String?
    @State private var showDeleteConfirmation = false
    @State private var loadingModelId: String?
    @State private var typeFilter: ModelTypeFilter = .all

    enum ModelsTab: String, CaseIterable {
        case local
        case downloads
    }

    enum ModelTypeFilter: String, CaseIterable {
        case all
        case llm
        case vlm
        case embedding
        case audio
        case image

        var label: String {
            switch self {
            case .all: return "All"
            case .llm: return "LLM"
            case .vlm: return "VLM"
            case .embedding: return "Embed"
            case .audio: return "Audio"
            case .image: return "Image"
            }
        }

        var icon: String {
            switch self {
            case .all: return "square.grid.2x2"
            case .llm: return "text.bubble"
            case .vlm: return "eye"
            case .embedding: return "vector"
            case .audio: return "waveform"
            case .image: return "photo"
            }
        }

        func matches(_ modelType: ModelType) -> Bool {
            switch self {
            case .all: return true
            case .llm: return modelType == .llm
            case .vlm: return modelType == .vlm
            case .embedding: return modelType == .embedding
            case .audio: return modelType == .audio
            case .image: return modelType == .image
            }
        }

        var matchType: ModelType? {
            switch self {
            case .all: return nil
            case .llm: return .llm
            case .vlm: return .vlm
            case .embedding: return .embedding
            case .audio: return .audio
            case .image: return .image
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Category filter — shared across tabs, topmost
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(ModelTypeFilter.allCases, id: \.self) { filter in
                        Button {
                            typeFilter = filter
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: filter.icon)
                                    .font(.system(size: 9))
                                Text(filter.label)
                                    .font(.system(size: 11, weight: typeFilter == filter ? .semibold : .regular))
                            }
                            .foregroundColor(typeFilter == filter ? .white : NovaTheme.Colors.accent)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(typeFilter == filter ? NovaTheme.Colors.accent : NovaTheme.Colors.accent.opacity(0.15))
                            .overlay(Capsule().stroke(NovaTheme.Colors.accent.opacity(typeFilter == filter ? 0 : 0.3), lineWidth: 0.5))
                            .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
            }

            Divider().padding(.horizontal, 24)

            // Tab bar
            HStack(spacing: 0) {
                ForEach(ModelsTab.allCases, id: \.self) { tab in
                    Button(action: { selectedTab = tab }) {
                        HStack(spacing: 6) {
                            if tab == .local {
                                Image(systemName: "desktopcomputer")
                                    .font(.system(size: 10))
                            } else {
                                Image(systemName: "arrow.down.circle")
                                    .font(.system(size: 10))
                            }
                            Text(tab == .local ? l10n.tr("app.localModels") : l10n.tr("app.downloads"))
                                .font(.system(size: 12, weight: selectedTab == tab ? .semibold : .regular))
                        }
                        .foregroundColor(selectedTab == tab ? NovaTheme.Colors.accent : NovaTheme.Colors.textTertiary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(selectedTab == tab ? NovaTheme.Colors.accent.opacity(0.1) : Color.clear)
                        .cornerRadius(6)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                Spacer()

                if appState.activeDownloadCount > 0 {
                    Text("\(appState.activeDownloadCount)")
                        .font(.caption2)
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(NovaTheme.Colors.accent)
                        .clipShape(Capsule())
                }
            }
            .padding(.horizontal, 24)
            .padding(.top, 16)
            .padding(.bottom, 4)

            Divider().padding(.horizontal, 24)

            // Tab content
            Group {
                switch selectedTab {
                case .local:
                    localModelsContent
                case .downloads:
                    DownloadsPageView(appState: appState, modelManager: modelManager, typeFilter: $typeFilter)
                        .environmentObject(l10n)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
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

    private var localModelsContent: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 20) {
                    loadedSection
                    downloadedSection
                }
                .padding(24)
            }
        }
    }

    private var loadedSection: some View {
        let loaded = appState.loadedModels.filter { id in
            guard let record = modelManager.getRecord(id) else { return typeFilter == .all }
            return typeFilter.matches(record.modelType)
        }
        let restoring = appState.restoringModels.filter { id in
            guard let record = modelManager.getRecord(id) else { return typeFilter == .all }
            return typeFilter.matches(record.modelType)
        }

        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                sectionHeader(l10n.tr("status.activeModels"), icon: "bolt.fill", count: loaded.count)
                if !restoring.isEmpty {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text("Restoring \(restoring.count) model\(restoring.count > 1 ? "s" : "")...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            if loaded.isEmpty && restoring.isEmpty {
                emptyState(l10n.tr("models.noModelsLoaded"), subtitle: l10n.tr("models.noModelsLoadedSub"))
            } else {
                // Models currently being restored on startup — show spinner rows
                ForEach(restoring, id: \.self) { modelId in
                    restoringModelRow(modelId)
                }
                ForEach(loaded, id: \.self) { modelId in
                    modelRow(
                        modelId,
                        subtitle: modelManager.getRecord(modelId)?.family.rawValue ?? l10n.tr("models.unknown"),
                        isLoaded: true,
                        actions: {
                            specBoostBadge(for: modelId)
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

    /// Spinner row shown for a model that is currently being restored on startup.
    private func restoringModelRow(_ modelId: String) -> some View {
        HStack(spacing: 10) {
            ProgressView()
                .controlSize(.small)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelId)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                Text("Loading…")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding(10)
        .background(NovaTheme.Colors.rowBackground)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(NovaTheme.Colors.accent.opacity(0.2), lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private func specBoostBadge(for modelId: String) -> some View {
        if let boost = appState.specBoostStatus[modelId], boost.draftModelId != modelId {
            switch boost.status {
            case "active":
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.caption2)
                        .foregroundColor(.green)
                    Text(boost.draftDisplayName ?? "Boost")
                        .font(.caption2)
                        .foregroundColor(.green)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.green.opacity(0.15))
                .cornerRadius(4)
            case "eligible":
                if boost.draftDownloaded == true {
                    Button {
                        Task { await appState.boostLoad(modelId: modelId) }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "bolt.horizontal")
                                .font(.caption2)
                            Text(boost.draftDisplayName ?? "Boost")
                                .font(.caption2)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button {
                        Task { await appState.boostDownload(modelId: modelId) }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.down.circle")
                                .font(.caption2)
                            Text(boost.draftDisplayName ?? "Boost")
                                .font(.caption2)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            default:
                EmptyView()
            }
        }
    }

    private var downloadedSection: some View {
        let allDownloaded = modelManager.downloadedModels()
        let loaded = Set(inferenceService.listLoadedModels())
        let downloaded = allDownloaded.filter { !loaded.contains($0.id) && typeFilter.matches($0.modelType) }
        NovaMLXLog.info("[ModelsPage] downloadedSection: allDownloaded=\(allDownloaded.count), loaded=\(loaded.count), filtered=\(downloaded.count), typeFilter=\(typeFilter.rawValue)")

        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                sectionHeader(l10n.tr("models.noInactiveModels"), icon: "arrow.down.circle", count: downloaded.count)
                Spacer()
                Button {
                    refreshTrigger.toggle()
                    // Force rescan of local models directory
                    NotificationCenter.default.post(name: .novaMLXModelsChanged, object: nil)
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

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
                                    Text(l10n.tr("models.loading") + " " + (record.id.components(separatedBy: "/").last ?? record.id))
                                        .font(.caption).foregroundColor(NovaTheme.Colors.accent)
                                        .lineLimit(1)
                                }
                            } else {
                                Button(l10n.tr("models.load")) {
                                    loadingModelId = record.id
                                    NovaMLXLog.info("[ModelsPage] User clicked Load for \(record.id), type=\(record.modelType), family=\(record.family), url=\(record.localURL.path)")
                                    Task {
                                        let config = ModelConfig(
                                            identifier: ModelIdentifier(id: record.id, family: record.family),
                                            modelType: record.modelType
                                        )
                                        NovaMLXLog.info("[ModelsPage] Calling inferenceService.loadModel for \(record.id)")
                                        do {
                                            try await inferenceService.loadModel(at: record.localURL, config: config)
                                            NovaMLXLog.info("[ModelsPage] Load succeeded for \(record.id)")
                                            refreshTrigger.toggle()
                                        } catch {
                                            NovaMLXLog.error("[ModelsPage] Load FAILED for \(record.id): \(error)")
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
                    if isLoaded {
                        Button {
                            appState.pickInPlayground(modelId)
                        } label: {
                            Image(systemName: "play.circle")
                                .font(.system(size: 11))
                                .foregroundColor(NovaTheme.Colors.accent)
                        }
                        .buttonStyle(.plain)
                        .help("Open in Playground")
                    }
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
