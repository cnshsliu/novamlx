import SwiftUI
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

struct DownloadsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n

    @State private var searchText = ""
    @State private var searchResults: [HFSearchResult] = []
    @State private var isSearching = false
    @State private var manualDownloadInput = ""
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var showApiKeyPrompt = false
    @State private var newApiKey = ""
    @State private var isSavingApiKey = false
    @State private var selectedModelCard: ModelCardData?
    @State private var suggestedSearches = ["mlx-community/gemma4", "mlx-community/qwen3.6"]

    private func fetchSuggestedSearches() async {
        guard let url = URL(string: "https://raw.githubusercontent.com/cnshsliu/novamlx/main/suggested-searches.txt") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return }
            let text = String(data: data, encoding: .utf8) ?? ""
            let lines = text.split(separator: "\n").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            if !lines.isEmpty {
                suggestedSearches = lines
            }
        } catch {}
    }

    var body: some View {
        let activeOrFailed = appState.downloadTasks.values
            .filter { $0.isActive || $0.status == .failed }
            .sorted { $0.startedAt > $1.startedAt }

        let completed = appState.downloadTasks.values
            .filter { $0.status == .completed }
            .sorted { $0.startedAt > $1.startedAt }

        ScrollView {
            VStack(spacing: 20) {
                if !activeOrFailed.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        sectionHeader(
                            activeOrFailed.allSatisfy(\.isActive) ? l10n.tr("models.downloading") : l10n.tr("models.downloads"),
                            icon: "arrow.down.circle",
                            count: activeOrFailed.count
                        )
                        ForEach(activeOrFailed, id: \.repoId) { task in
                            topDownloadRow(task)
                        }
                    }
                    .sectionCard()
                }

                searchBar

                if !searchResults.isEmpty {
                    searchResultsSection
                }
                manualDownloadSection

                if !completed.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        sectionHeader(l10n.tr("models.completed"), icon: "checkmark.circle", count: completed.count)
                        ForEach(completed, id: \.repoId) { task in
                            completedDownloadRow(task)
                        }
                    }
                    .sectionCard()
                }

                if searchResults.isEmpty && activeOrFailed.isEmpty && appState.downloadTasks.isEmpty {
                    emptyState(l10n.tr("models.noDownloads"), subtitle: l10n.tr("models.noDownloadsSub"))
                        .padding(.top, 60)
                }
            }
            .padding(24)
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
        .task {
            await fetchSuggestedSearches()
        }
    }

    // MARK: - Search Bar

    private var searchBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                    TextField(l10n.tr("models.searchPlaceholder"), text: $searchText)
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
                        Label(l10n.tr("models.search"), systemImage: "magnifyingglass")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(searchText.isEmpty || isSearching)

                Spacer()
            }

            FlowLayout(spacing: 8) {
                ForEach(suggestedSearches, id: \.self) { suggestion in
                    Button(action: {
                        searchText = suggestion
                        performSearch()
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "sparkle")
                                .font(.caption2)
                            Text(suggestion)
                                .font(.caption)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(NovaTheme.Colors.rowBackground)
                        .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .sectionCard()
    }

    // MARK: - Search Results

    private var searchResultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("models.searchResults"), icon: "magnifyingglass", count: searchResults.count)

            ForEach(searchResults, id: \.id) { result in
                searchResultRow(result)
            }
        }
        .sectionCard()
    }

    private func searchResultRow(_ result: HFSearchResult) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(result.id).font(.system(size: 13, weight: .medium)).lineLimit(1)
                        .foregroundColor(NovaTheme.Colors.accent)
                        .help(l10n.tr("models.clickDetails"))
                        .onTapGesture { fetchModelCard(repoId: result.id) }
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(result.id, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help(l10n.tr("agents.copyConfig"))
                }
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
            Label(l10n.tr("models.downloaded"), systemImage: "checkmark.circle.fill")
                .foregroundColor(NovaTheme.Colors.statusOK)
                .font(.caption)
        } else if let task = appState.downloadTasks[repoId], task.isActive {
            HStack(spacing: 6) {
                Text(l10n.tr("models.downloading"))
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
                Button(l10n.tr("models.resume")) { appState.startDownload(repoId: repoId) }
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
            Button(l10n.tr("models.download")) { appState.startDownload(repoId: repoId) }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    // MARK: - Manual Download

    private var manualDownloadSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("models.manualDownload"), icon: "link.badge.plus", count: nil)

            Text(l10n.tr("models.urlHint"))
                .font(.caption).foregroundColor(.secondary)

            HStack(spacing: 8) {
                TextField(l10n.tr("models.urlPlaceholder"), text: $manualDownloadInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))
                    .onSubmit { manualDownload() }

                Button(l10n.tr("models.download")) { manualDownload() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(manualDownloadInput.trimmingCharacters(in: .whitespaces).isEmpty)

                Button { NSWorkspace.shared.open(modelManager.modelsDirectory) } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help(l10n.tr("models.openFolder"))
            }

            Text(l10n.tr("models.savedTo", modelManager.modelsDirectory.path))
                .font(.caption2).foregroundColor(.secondary)
                .help(l10n.tr("models.clickCopy"))
                .onTapGesture {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(modelManager.modelsDirectory.path, forType: .string)
                }
        }
        .sectionCard()
    }

    // MARK: - Download Rows

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
                                 : "\(formatBytes(task.downloadedBytes)) \(l10n.tr("models.downloadedLabel"))")
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
                        Text(l10n.tr("models.cancel"))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button(l10n.tr("models.resume")) { appState.startDownload(repoId: task.repoId) }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    Button(l10n.tr("models.delete")) {
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
                        Text(l10n.tr("models.filesCompleted", completedCount, total))
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
                                Text(l10n.tr("models.retrying")).font(.system(size: 10)).foregroundColor(NovaTheme.Colors.statusWarn)
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
                 : "\(formatBytes(file.downloadedBytes)) \(l10n.tr("models.downloadedLabel"))")
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
                    Text(task.status == .completed ? l10n.tr("models.downloadedOk") : (task.errorMessage ?? "Failed"))
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
                Button(l10n.tr("models.retry")) { appState.startDownload(repoId: task.repoId) }
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

    // MARK: - Helpers

    private func emptyState(_ title: String, subtitle: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "arrow.down.circle")
                .font(.system(size: 40))
                .foregroundColor(NovaTheme.Colors.accent.opacity(0.4))
            Text(title).font(.headline).foregroundColor(.secondary)
            Text(subtitle).font(.caption).foregroundColor(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        if bytes >= 1_073_741_824 {
            return String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
        } else if bytes >= 1_048_576 {
            return String(format: "%.1f MB", Double(bytes) / 1_048_576)
        } else {
            return "\(bytes) B"
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
                alertMessage = l10n.tr("models.invalidUrl")
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
                    alertMessage = l10n.tr("models.searchFailed", httpResp.statusCode)
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
                        alertMessage = l10n.tr("models.noResults", searchText)
                        showAlert = true
                    }
                } else {
                    alertMessage = l10n.tr("models.unexpectedFormat")
                    showAlert = true
                }
            } catch {
                alertMessage = l10n.tr("models.searchFailedMsg", error.localizedDescription)
                showAlert = true
            }
            isSearching = false
        }
    }

    private func manualDownload() {
        let input = manualDownloadInput.trimmingCharacters(in: .whitespaces)
        guard !input.isEmpty else { return }

        let repoId: String
        if input.hasPrefix("http") {
            guard let url = URLComponents(string: input),
                  let host = url.host, host.contains("huggingface.co"),
                  let path = url.path.removingPercentEncoding else {
                alertMessage = l10n.tr("models.invalidHfUrl")
                showAlert = true
                return
            }
            let segments = path.split(separator: "/").map(String.init)
            guard segments.count >= 2 else {
                alertMessage = l10n.tr("models.invalidUrlMsg")
                showAlert = true
                return
            }
            repoId = "\(segments[0])/\(segments[1])"
        } else {
            repoId = input
        }

        appState.startDownload(repoId: repoId)
        manualDownloadInput = ""
    }

    // MARK: - API Key Setup Sheet

    private var apiKeySetupSheet: some View {
        VStack(spacing: 20) {
            Text(l10n.tr("models.apiKeyRequired")).font(.headline)
            Text(l10n.tr("models.apiKeyMessage"))
                .font(.subheadline).foregroundColor(.secondary).multilineTextAlignment(.center)

            HStack {
                Text(l10n.tr("models.apiKeyLabel")).font(.subheadline)
                TextField(l10n.tr("models.apiKeyPlaceholder"), text: $newApiKey)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
            }

            HStack(spacing: 12) {
                Button(l10n.tr("models.cancel")) { showApiKeyPrompt = false }
                    .keyboardShortcut(.cancelAction)
                Button(l10n.tr("models.saveRestart")) { saveApiKeyAndRestart() }
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
                alertMessage = l10n.tr("models.saveFailed", error.localizedDescription)
                showAlert = true
                isSavingApiKey = false
            }
        }
    }

    // MARK: - Model Card

    private func fetchModelCard(repoId: String) {
        selectedModelCard = ModelCardData(repoId: repoId)
        Task {
            let adminPort = appState.adminPort
            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/model-card?repo_id=\(repoId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? repoId)") else { return }
            do {
                var request = URLRequest(url: url)
                if let apiKey = appState.apiKey {
                    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                }
                let (data, _) = try await URLSession.shared.data(for: request)
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    var card = ModelCardData(repoId: repoId)
                    card.author = json["author"] as? String
                    card.downloads = json["downloads"] as? Int
                    card.likes = json["likes"] as? Int
                    card.tags = json["tags"] as? [String] ?? []
                    card.license = json["license"] as? String
                    card.language = json["language"] as? [String] ?? []
                    card.architectures = json["architectures"] as? [String] ?? []
                    card.modelType = json["model_type"] as? String
                    card.totalSize = json["total_size"] as? Int64 ?? 0
                    if let files = json["files"] as? [[String: Any]] {
                        card.files = files.compactMap { f in
                            guard let name = f["name"] as? String, let size = f["size"] as? Int64 else { return nil }
                            return ModelCardFile(name: name, size: size)
                        }
                    }
                    if let localSize = json["local_disk_size"] as? Int64 {
                        card.localDiskSize = localSize
                    }
                    selectedModelCard = card
                }
            } catch {}
        }
    }

    private func modelCardSheet(_ card: ModelCardData) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(card.repoId).font(.title3.bold())

            if !card.architectures.isEmpty {
                cardSection("Architectures") {
                    ForEach(card.architectures, id: \.self) { arch in
                        specRow("•", value: arch)
                    }
                }
            }

            HStack(spacing: 24) {
                if let downloads = card.downloads {
                    specRow("Downloads", value: "\(downloads)")
                }
                if let likes = card.likes {
                    specRow("Likes", value: "\(likes)")
                }
                if card.totalSize > 0 {
                    specRow("Size", value: formatBytes(Int64(card.totalSize)))
                }
                if let localSize = card.localDiskSize {
                    specRow("Local", value: formatBytes(localSize))
                }
            }

            if !card.files.isEmpty {
                cardSection("Files (\(card.files.count))") {
                    ForEach(card.files) { file in
                        HStack {
                            Text(file.name).font(.system(size: 11, design: .monospaced)).lineLimit(1)
                            Spacer()
                            Text(formatBytes(file.size)).font(.caption).foregroundColor(.secondary)
                        }
                    }
                }
            }

            HStack {
                Spacer()
                Button("OK") { selectedModelCard = nil }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 520, height: 480)
    }

    private func cardSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption.bold()).foregroundColor(.secondary)
            content()
        }
    }

    private func specRow(_ label: String, value: String) -> some View {
        HStack(spacing: 4) {
            Text(label).font(.caption).foregroundColor(.secondary)
                .frame(width: 90, alignment: .trailing)
            Text(value)
                .font(.caption)
            Spacer()
        }
    }
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
