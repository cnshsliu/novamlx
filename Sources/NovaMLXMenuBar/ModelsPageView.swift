import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

struct ModelsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @State private var subTab: SubTab = .myModels
    @State private var searchText = ""
    @State private var searchResults: [HFSearchResult] = []
    @State private var isSearching = false
    @State private var manualDownloadInput = ""
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var refreshTrigger = false
    @State private var showApiKeyPrompt = false
    @State private var newApiKey = ""
    @State private var isSavingApiKey = false
    @State private var selectedModelCard: ModelCardData?
    @State private var isLoadingCard = false
    @State private var modelToDelete: String?
    @State private var showDeleteConfirmation = false
    @State private var loadingModelId: String?

    enum SubTab: String, CaseIterable {
        case myModels = "My Models"
        case downloads = "Downloads"
    }

    var body: some View {
        VStack(spacing: 0) {
            subTabBar
            Divider()
            subTabContent
        }
        .alert(alertMessage, isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        }
        .sheet(isPresented: $showApiKeyPrompt) {
            apiKeySetupSheet
        }
        .sheet(item: $selectedModelCard) { card in
            modelCardSheet(card)
        }
        .alert("Delete Model?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {
                modelToDelete = nil
            }
            Button("Delete", role: .destructive) {
                if let id = modelToDelete {
                    // Unload from memory first if loaded
                    if let record = modelManager.getRecord(id) {
                        Task {
                            await inferenceService.unloadModel(ModelIdentifier(id: id, family: record.family))
                        }
                    }
                    // Then delete from disk
                    try? modelManager.deleteModel(id)
                    refreshTrigger.toggle()
                }
                modelToDelete = nil
            }
        } message: {
            Text("This will permanently remove '\(modelToDelete ?? "")' from disk. This cannot be undone.")
        }
        .onReceive(NotificationCenter.default.publisher(for: .novaMLXModelsChanged)) { _ in
            refreshTrigger.toggle()
        }
    }

    // MARK: - Sub-tab Bar

    private var subTabBar: some View {
        HStack(spacing: 0) {
            ForEach(SubTab.allCases, id: \.self) { tab in
                Button {
                    subTab = tab
                } label: {
                    HStack(spacing: 6) {
                        Text(tab.rawValue)
                            .font(.system(size: 13, weight: subTab == tab ? .semibold : .regular))
                        if tab == .downloads && appState.activeDownloadCount > 0 {
                            Text("\(appState.activeDownloadCount)")
                                .font(.caption2)
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(NovaTheme.Colors.accent)
                                .clipShape(Capsule())
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(subTab == tab ? NovaTheme.Colors.accentDim : Color.clear)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .background(NovaTheme.Colors.cardBackground)
    }

    // MARK: - Sub-tab Content

    @ViewBuilder
    private var subTabContent: some View {
        switch subTab {
        case .myModels:
            myModelsTab
        case .downloads:
            downloadsTab
        }
    }

    // MARK: - Downloads Tab (merged browse + downloads)

    private var downloadsTab: some View {
        let activeOrFailed = appState.downloadTasks.values
            .filter { $0.isActive || $0.status == .failed }
            .sorted { $0.startedAt > $1.startedAt }

        return VStack(spacing: 0) {
            // Active/failed downloads pinned at top
            if !activeOrFailed.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    sectionHeader(
                        activeOrFailed.allSatisfy(\.isActive) ? "Downloading" : "Downloads",
                        icon: "arrow.down.circle",
                        count: activeOrFailed.count
                    )
                    ForEach(activeOrFailed, id: \.repoId) { task in
                        topDownloadRow(task)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                Divider()
            }

            searchBar
            Divider()

            ScrollView {
                VStack(spacing: 20) {
                    if !searchResults.isEmpty {
                        searchResultsSection
                    }
                    manualDownloadSection

                    let completed = appState.downloadTasks.values
                        .filter { $0.status == .completed }
                        .sorted { $0.startedAt > $1.startedAt }

                    if !completed.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            sectionHeader("Completed", icon: "checkmark.circle", count: completed.count)
                            ForEach(completed, id: \.repoId) { task in
                                completedDownloadRow(task)
                            }
                        }
                        .sectionCard()
                    }

                    if searchResults.isEmpty && activeOrFailed.isEmpty && appState.downloadTasks.isEmpty {
                        emptyState("No Downloads", subtitle: "Search for models or paste a URL to start downloading.")
                            .padding(.top, 60)
                    }
                }
                .padding(24)
            }
        }
    }

    private var searchBar: some View {
        HStack(spacing: 12) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                TextField("Search HuggingFace models...", text: $searchText)
                    .textFieldStyle(.plain)
                    .onSubmit { performSearch() }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(NovaTheme.Colors.rowBackground)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            Button(action: { performSearch() }) {
                if isSearching {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Search", systemImage: "magnifyingglass")
                }
            }
            .buttonStyle(.bordered)
            .disabled(searchText.isEmpty || isSearching)

            Spacer()
        }
        .padding(16)
    }

    // MARK: - Search Results

    private var searchResultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Search Results", icon: "magnifyingglass", count: searchResults.count)

            ForEach(searchResults, id: \.id) { result in
                searchResultRow(result)
            }
        }
        .sectionCard()
    }

    private func searchResultRow(_ result: HFSearchResult) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(result.id).font(.system(size: 13, weight: .medium)).lineLimit(1)
                    .foregroundColor(NovaTheme.Colors.accent)
                    .help("Click to view model details")
                    .onTapGesture { fetchModelCard(repoId: result.id) }
                if !result.tags.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(result.tags.prefix(3), id: \.self) { tag in
                            Text(tag)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(NovaTheme.Colors.accentDim)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }
            }

            Spacer()

            downloadActionButton(for: result.id)
        }
        .rowCard()
    }

    @ViewBuilder
    private func downloadActionButton(for repoId: String) -> some View {
        if modelManager.isDownloaded(repoId) {
            Label("Downloaded", systemImage: "checkmark.circle.fill")
                .foregroundColor(NovaTheme.Colors.statusOK)
                .font(.caption)
        } else if let task = appState.downloadTasks[repoId], task.isActive {
            HStack(spacing: 6) {
                Text("Downloading")
                    .font(.caption).foregroundColor(NovaTheme.Colors.accent)
                Text("\(Int(task.progress))%")
                    .font(.caption2).foregroundColor(.secondary)
                Button { appState.cancelDownload(repoId: repoId) } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption).foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
        } else if let task = appState.downloadTasks[repoId], task.status == .failed {
            HStack(spacing: 6) {
                Button("Resume") { appState.startDownload(repoId: repoId) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                Button { appState.cancelAndDeleteDownload(
                    repoId: repoId,
                    modelsDirectory: modelManager.modelsDirectory
                ) } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .foregroundColor(.red)
            }
        } else {
            Button("Download") { appState.startDownload(repoId: repoId) }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    // MARK: - Manual Download

    private var manualDownloadSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Manual Download", icon: "link.badge.plus", count: nil)

            Text("Paste a HuggingFace repo ID or URL to download directly.")
                .font(.caption).foregroundColor(.secondary)

            HStack(spacing: 8) {
                TextField("mlx-community/model-name or https://huggingface.co/...", text: $manualDownloadInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))
                    .onSubmit { manualDownload() }

                Button("Download") { manualDownload() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(manualDownloadInput.trimmingCharacters(in: .whitespaces).isEmpty)

                Button { NSWorkspace.shared.open(modelManager.modelsDirectory) } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Open download folder")
            }

            Text("Saved to: \(modelManager.modelsDirectory.path)")
                .font(.caption2).foregroundColor(.secondary)
                .help("Click to copy path")
                .onTapGesture {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(modelManager.modelsDirectory.path, forType: .string)
                }
        }
        .sectionCard()
    }

    /// Row shown in the pinned top section for active and failed downloads
    private func topDownloadRow(_ task: DownloadTaskInfo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Image(systemName: task.isActive ? "arrow.down.circle" : "exclamationmark.triangle.fill")
                    .foregroundColor(task.isActive ? NovaTheme.Colors.accent : NovaTheme.Colors.statusWarn)
                    .font(.title3)

                VStack(alignment: .leading, spacing: 4) {
                    Text(task.repoId).font(.system(size: 13, weight: .medium)).lineLimit(1)
                    if task.isActive {
                        HStack(spacing: 8) {
                            ProgressView(value: task.progress, total: 100)
                                .frame(maxWidth: 200)
                            Text("\(Int(task.progress))%")
                                .font(.caption).foregroundColor(.secondary).frame(width: 36, alignment: .trailing)
                            Text(task.totalBytes > 0
                                 ? "\(formatBytes(task.downloadedBytes)) / \(formatBytes(task.totalBytes))"
                                 : "\(formatBytes(task.downloadedBytes)) downloaded")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                    } else if let error = task.errorMessage {
                        Text(error)
                            .font(.caption).foregroundColor(NovaTheme.Colors.statusError)
                            .lineLimit(2)
                    }
                }

                Spacer()

                if task.isActive {
                    Button { appState.cancelDownload(repoId: task.repoId) } label: {
                        Text("Cancel")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button("Resume") { appState.startDownload(repoId: task.repoId) }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    Button("Delete") {
                        appState.cancelAndDeleteDownload(
                            repoId: task.repoId,
                            modelsDirectory: modelManager.modelsDirectory
                        )
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .foregroundColor(.red)
                }
            }

            if task.isActive {
                let activeFiles = task.fileProgresses.filter { $0.status == "downloading" }
                let completedCount = task.fileProgresses.filter { $0.status == "completed" }.count
                let total = task.fileProgresses.count

                if total > 0 {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(completedCount)/\(total) files completed")
                            .font(.caption2).foregroundColor(.secondary)

                        ForEach(activeFiles) { file in
                            fileProgressBar(file)
                        }

                        ForEach(task.fileProgresses.filter { $0.status == "failed" }) { file in
                            HStack(spacing: 6) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.system(size: 9)).foregroundColor(NovaTheme.Colors.statusWarn)
                                Text(file.filename)
                                    .font(.system(size: 10, design: .monospaced))
                                    .lineLimit(1).foregroundColor(NovaTheme.Colors.statusWarn)
                                Text("retrying...").font(.system(size: 10)).foregroundColor(NovaTheme.Colors.statusWarn)
                            }
                        }
                    }
                    .padding(.leading, 32)
                }
            }
        }
        .rowCard()
    }

    private func fileProgressBar(_ file: FileDownloadInfo) -> some View {
        let filePercent: Double = file.totalBytes > 0
            ? min(Double(file.downloadedBytes) / Double(file.totalBytes) * 100, 100)
            : 0

        return HStack(spacing: 8) {
            Image(systemName: "arrow.down")
                .font(.system(size: 9)).foregroundColor(NovaTheme.Colors.accent)

            Text(file.filename)
                .font(.system(size: 11, design: .monospaced))
                .lineLimit(1)
                .frame(maxWidth: 180, alignment: .leading)

            ProgressView(value: filePercent, total: 100)
                .frame(maxWidth: 120)

            Text(file.totalBytes > 0 ? "\(Int(filePercent))%" : "—")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)
                .frame(width: 32, alignment: .trailing)

            Text(file.totalBytes > 0
                 ? "\(formatBytes(file.downloadedBytes))/\(formatBytes(file.totalBytes))"
                 : "\(formatBytes(file.downloadedBytes)) downloaded")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)
        }
    }

    private func completedDownloadRow(_ task: DownloadTaskInfo) -> some View {
        HStack(spacing: 12) {
            Image(systemName: task.status == .completed ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundColor(task.status == .completed ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                .font(.title3)

            VStack(alignment: .leading, spacing: 2) {
                Text(task.repoId).font(.system(size: 13, weight: .medium)).lineLimit(1)
                HStack(spacing: 6) {
                    Text(task.status == .completed ? "Downloaded successfully" : (task.errorMessage ?? "Failed"))
                        .font(.caption2).foregroundColor(.secondary)
                    if task.downloadedBytes > 0 {
                        Text("(\(formatBytes(task.downloadedBytes)))")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                }
                    .font(.caption2).foregroundColor(.secondary)
            }

            Spacer()

            if task.status == .failed {
                Button("Retry") { appState.startDownload(repoId: task.repoId) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                Button { appState.cancelAndDeleteDownload(
                    repoId: task.repoId,
                    modelsDirectory: modelManager.modelsDirectory
                ) } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .foregroundColor(.red)
            }

            Button { appState.dismissDownload(repoId: task.repoId) } label: {
                Image(systemName: "xmark").font(.caption).foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
        }
        .rowCard()
    }

    // MARK: - My Models Tab

    private var myModelsTab: some View {
        ScrollView {
            VStack(spacing: 20) {
                loadedSection
                downloadedSection
            }
            .padding(24)
        }
    }

    private var loadedSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Active Models", icon: "bolt.fill", count: appState.loadedModels.count)

            if appState.loadedModels.isEmpty {
                emptyState("No models loaded", subtitle: "Load a downloaded model to start using it.")
            } else {
                ForEach(appState.loadedModels, id: \.self) { modelId in
                    modelRow(
                        modelId,
                        subtitle: modelManager.getRecord(modelId)?.family.rawValue ?? "unknown",
                        isLoaded: true,
                        actions: {
                            Button("Unload") {
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

    private var downloadedSection: some View {
        let allDownloaded = modelManager.downloadedModels()
        let loaded = Set(inferenceService.listLoadedModels())
        let downloaded = allDownloaded.filter { !loaded.contains($0.id) }

        return VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Inactive Models", icon: "arrow.down.circle", count: downloaded.count)

            if downloaded.isEmpty {
                emptyState("No inactive models", subtitle: "Go to Browse to search and download models.")
            } else {
                ForEach(downloaded, id: \.id) { record in
                    modelRow(
                        record.id,
                        subtitle: "\(record.family.rawValue)  \(record.sizeBytes > 0 ? record.sizeBytes.bytesFormatted : "")",
                        isLoaded: false,
                        actions: {
                            if loadingModelId == record.id {
                                // Loading animation
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .controlSize(.small)
                                    Text("Loading...")
                                        .font(.caption).foregroundColor(NovaTheme.Colors.accent)
                                }
                            } else {
                                Button("Load") {
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
                        .help("Click to view model details")
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
        if mb >= 1024 { return String(format: "%.1f GB", mb / 1024) }
        return String(format: "%.0f MB", mb)
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

                    // For local models, compute actual disk usage
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
                    cardSection("Tags") {
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
                    cardSection("Specifications") {
                        VStack(alignment: .leading, spacing: 6) {
                            if !card.architectures.isEmpty {
                                specRow("Architecture", value: card.architectures.joined(separator: ", "))
                            }
                            if let mt = card.modelType {
                                specRow("Model Type", value: mt)
                            }
                            if let license = card.license {
                                specRow("License", value: license)
                            }
                            if !card.language.isEmpty {
                                specRow("Language", value: card.language.joined(separator: ", "))
                            }
                            if let record = modelManager.getRecord(card.repoId) {
                                specRow("Family", value: record.family.rawValue)
                                specRow("Type", value: record.modelType.rawValue.uppercased())
                            }
                        }
                    }
                }

                // Size info
                let totalHF = card.totalSize
                let totalLocal = card.localDiskSize
                if totalHF > 0 || (totalLocal ?? 0) > 0 {
                    cardSection("Size") {
                        VStack(alignment: .leading, spacing: 6) {
                            if totalHF > 0 {
                                specRow("Download Size", value: formatBytes(totalHF))
                            }
                            if let local = totalLocal, local > 0 {
                                specRow("Disk Usage", value: formatBytes(local))
                            }
                        }
                    }
                }

                // File listing
                if !card.files.isEmpty {
                    cardSection("Files (\(card.files.count))") {
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
                    Button("Close") { selectedModelCard = nil }
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

    // MARK: - Search

    private func performSearch() {
        guard !searchText.isEmpty else { return }

        if appState.apiKey == nil {
            newApiKey = "sk-novamlx-\(UUID().uuidString.prefix(8))"
            showApiKeyPrompt = true
            return
        }

        isSearching = true
        Task {
            let adminPort = appState.adminPort
            let query = searchText.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? searchText
            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/search?q=\(query)") else {
                alertMessage = "Invalid URL"
                showAlert = true
                isSearching = false
                return
            }
            do {
                var request = URLRequest(url: url)
                if let apiKey = appState.apiKey {
                    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                }
                let (data, response) = try await URLSession.shared.data(for: request)
                if let httpResp = response as? HTTPURLResponse, httpResp.statusCode != 200 {
                    alertMessage = "Search failed (HTTP \(httpResp.statusCode))"
                    showAlert = true
                    isSearching = false
                    return
                }
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let results = json["models"] as? [[String: Any]] {
                    searchResults = results.compactMap { r in
                        guard let id = r["id"] as? String else { return nil }
                        return HFSearchResult(id: id, tags: r["tags"] as? [String] ?? [])
                    }
                    if searchResults.isEmpty {
                        alertMessage = "No results found for '\(searchText)'"
                        showAlert = true
                    }
                } else {
                    alertMessage = "Unexpected response format"
                    showAlert = true
                }
            } catch {
                alertMessage = "Search failed: \(error.localizedDescription)"
                showAlert = true
            }
            isSearching = false
        }
    }

    // MARK: - Manual Download

    private func manualDownload() {
        let input = manualDownloadInput.trimmingCharacters(in: .whitespaces)
        guard !input.isEmpty else { return }

        let repoId: String
        if input.hasPrefix("http") {
            guard let url = URLComponents(string: input),
                  let host = url.host, host.contains("huggingface.co"),
                  let path = url.path.removingPercentEncoding else {
                alertMessage = "Invalid HuggingFace URL"
                showAlert = true
                return
            }
            let segments = path.split(separator: "/").map(String.init)
            guard segments.count >= 2 else {
                alertMessage = "Invalid URL. Expected: https://huggingface.co/owner/repo"
                showAlert = true
                return
            }
            repoId = "\(segments[0])/\(segments[1])"
        } else {
            repoId = input
        }

        appState.startDownload(repoId: repoId)
        subTab = .downloads
        manualDownloadInput = ""
    }

    // MARK: - API Key Setup Sheet

    private var apiKeySetupSheet: some View {
        VStack(spacing: 20) {
            Text("API Key Required").font(.headline)
            Text("The Admin API requires an API key to enable model search. A key has been auto-generated below.")
                .font(.subheadline).foregroundColor(.secondary).multilineTextAlignment(.center)

            HStack {
                Text("API Key:").font(.subheadline)
                TextField("sk-novamlx-...", text: $newApiKey)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
            }

            HStack(spacing: 12) {
                Button("Cancel") { showApiKeyPrompt = false }
                    .keyboardShortcut(.cancelAction)
                Button("Save & Restart") { saveApiKeyAndRestart() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(newApiKey.trimmingCharacters(in: .whitespaces).isEmpty || isSavingApiKey)
            }
        }
        .padding(24)
        .frame(width: 420)
    }

    private func saveApiKeyAndRestart() {
        let key = newApiKey.trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty else { return }
        isSavingApiKey = true
        Task {
            do {
                let configFile = await NovaMLXConfiguration.shared.configFileURL
                try await NovaMLXConfiguration.shared.updateApiKeys([key], file: configFile)
                appState.apiKey = key
                NotificationCenter.default.post(name: .restartNovaMLXServer, object: nil)
                try? await Task.sleep(for: .milliseconds(500))
                showApiKeyPrompt = false
                isSavingApiKey = false
                performSearch()
            } catch {
                alertMessage = "Failed to save API key: \(error.localizedDescription)"
                showAlert = true
                isSavingApiKey = false
            }
        }
    }
}

private struct HFSearchResult {
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

// MARK: - Flow Layout

private struct FlowLayout: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrange(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
        }

        return (CGSize(width: maxWidth, height: y + rowHeight), positions)
    }
}
