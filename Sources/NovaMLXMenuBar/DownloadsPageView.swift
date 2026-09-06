import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXModelManager
import NovaMLXUtils

struct DownloadsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n
    @State private var typeFilter: ModelsPageView.ModelTypeFilter = .all

    @State private var searchText = ""
    @State private var searchResults: [HFSearchResult] = []
    @State private var searchTotalCount = 0       // total before local regex filter
    @State private var searchRegexError: String?  // invalid-regex hint
    @State private var isSearching = false
    @State private var selectedMirrorOption = "official"
    @State private var customMirrorURL = ""

    // Search options
    @State private var searchRegex = false        // treat input as regex
    @State private var suffixMlxCommunity = false // auto-prepend mlx-community/
    @State private var searchMlxOnly = true       // Hub search: keep mlx-tagged / mlx-community only

    // Catalog (verified) listing
    @State private var catalogEpoch = 0
    @State private var catalogSearchQuery = ""
    @State private var catalogDidLoad = false
    @State private var hubSearchAttempted = false
    @State private var hubSearchError: String?

    // Alert / API key
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var showApiKeyPrompt = false
    @State private var newApiKey = ""
    @State private var isSavingApiKey = false

    // Model card
    @State private var selectedModelCard: ModelCardData?

    // Live Hub query (shown in the combined activity section)
    @State private var currentSearchEndpoint: String?
    @State private var lastSearchSourceName = ""

    // Mirror toast
    @State private var showMirrorChangeNote = false
    @State private var mirrorChangeMessage = ""

    var body: some View {
        let activeOrFailed = appState.downloadTasks.values
            .filter { $0.isActive || $0.status == .failed }
            .sorted { $0.startedAt > $1.startedAt }

        let completed = appState.downloadTasks.values
            .filter { $0.status == .completed }
            .sorted { $0.startedAt > $1.startedAt }

        VStack(spacing: 0) {
            ModelTypeFilterBar(filter: $typeFilter)
            Divider().padding(.horizontal, 24)
            ScrollView {
                VStack(spacing: 20) {
                    searchSection

                    // Hub list after an Advanced / family-glob search. Empty Hub
                    // must not hide behind the verified catalog — that is what
                    // made "Allow unverified downloads" look like a no-op.
                    if hubSearchAttempted || !searchResults.isEmpty {
                        searchResultsSection
                    }
                    if searchResults.isEmpty || !displayedCatalogModels.isEmpty {
                        catalogModelsSection
                    }

                    if !activeOrFailed.isEmpty || isSearching {
                        activitySection(activeOrFailed: activeOrFailed)
                    }

                    if !completed.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            sectionHeader(l10n.tr("models.completed"), icon: "checkmark.circle", count: completed.count)
                            ForEach(completed, id: \.repoId) { task in
                                completedDownloadRow(task)
                            }
                        }
                        .sectionCard()
                    }

                    if searchResults.isEmpty && activeOrFailed.isEmpty && appState.downloadTasks.isEmpty
                        && displayedCatalogModels.isEmpty && !hubSearchAttempted
                    {
                        emptyState(l10n.tr("models.noDownloads"), subtitle: l10n.tr("models.noDownloadsSub"))
                            .padding(.top, 60)
                    }
                }
                .padding(24)
            }
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
        .onAppear {
            if !modelManager.catalogStore.models.isEmpty {
                catalogDidLoad = true
                catalogEpoch += 1
            }
        }
        .task {
            await modelManager.fetchCatalog()
            catalogDidLoad = true
            catalogEpoch += 1
        }
        .onChange(of: searchText) { _, newValue in
            let trimmed = newValue.trimmingCharacters(in: .whitespaces)
            catalogSearchQuery = trimmed
            // Drop stale Hub rows when the query changes; Search / toggle re-runs Hub.
            searchResults = []
            hubSearchAttempted = false
            hubSearchError = nil
        }
        .onChange(of: appState.allowUnlistedDownloads) { _, enabled in
            Task {
                await appState.setAllowUnlistedDownloads(enabled)
                if enabled {
                    let q = searchText.trimmingCharacters(in: .whitespaces)
                    if !q.isEmpty {
                        performSearch()
                    }
                } else {
                    searchResults = []
                    hubSearchAttempted = false
                    hubSearchError = nil
                }
            }
        }
    }

    // MARK: - Search

    private var searchSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: $appState.allowUnlistedDownloads) {
                    Text(l10n.tr("settings.allowUnlisted"))
                        .font(.system(size: 13))
                }
                .toggleStyle(.switch)
                .controlSize(.small)
                .accessibilityIdentifier("downloads-allow-unlisted-toggle")

                Text(l10n.tr("settings.allowUnlistedCaption"))
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if appState.allowUnlistedDownloads {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                        .font(.system(size: 12))
                    Text(l10n.tr("models.unverifiedBanner"))
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            // Search bar
            HStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                    TextField(
                        appState.allowUnlistedDownloads
                            ? l10n.tr("models.searchPlaceholder")
                            : l10n.tr("models.searchCatalogPlaceholder"),
                        text: $searchText
                    )
                        .textFieldStyle(.plain)
                        .accessibilityIdentifier("downloads-search-field")
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
            }

            // Mirror picker + search toggles (one row)
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 16) {
                    HStack(spacing: 6) {
                        Text("Mirror")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Picker("", selection: $selectedMirrorOption) {
                            Text("Official (huggingface.co)").tag("official")
                            Text("ModelScope (China)").tag("modelscope")
                            Text("Custom URL...").tag("custom")
                        }
                        .pickerStyle(.menu)
                        .accessibilityIdentifier("model-source-picker")
                        .onChange(of: selectedMirrorOption) { _, newOption in
                            Task { await appState.setHuggingfaceEndpoint(endpointForMirrorOption(newOption)) }
                            switch newOption {
                            case "official":
                                mirrorChangeMessage = "Switched to official Hugging Face"
                            case "modelscope":
                                mirrorChangeMessage = "Switched to ModelScope. New downloads will use modelscope.cn."
                            default:
                                mirrorChangeMessage = "Mirror changed. New downloads will use the selected source."
                            }
                            showMirrorChangeNote = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                                showMirrorChangeNote = false
                            }
                        }
                    }

                    if appState.allowUnlistedDownloads {
                        Divider().frame(height: 14)

                        Toggle("MLX only", isOn: $searchMlxOnly)
                            .toggleStyle(.switch)
                            .help("Hide GGUF and PyTorch-only repos. GGUF can load (dequantized to FP16) but is not catalog-verified.")
                        Toggle("regex", isOn: $searchRegex)
                            .toggleStyle(.switch)
                            .help("Treat search as a regular expression matched against model ID")
                        Toggle("mlx-community/", isOn: $suffixMlxCommunity)
                            .toggleStyle(.switch)
                            .help("Auto-prepend mlx-community/ so 'Qwen2-VL-7B' matches 'mlx-community/Qwen2-VL-7B-...'")
                    }
                }
                .font(.caption)

                if selectedMirrorOption == "custom" {
                    TextField("Custom endpoint (e.g. https://www.modelscope.cn)", text: $customMirrorURL)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))
                        .accessibilityIdentifier("model-source-custom-field")
                        .onSubmit {
                            Task {
                                let trimmed = customMirrorURL.trimmingCharacters(in: .whitespaces)
                                await appState.setHuggingfaceEndpoint(trimmed.isEmpty ? nil : trimmed)
                            }
                        }
                }

                if showMirrorChangeNote {
                    Text(mirrorChangeMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .sectionCard()
        .task {
            if let endpoint = await appState.huggingfaceEndpoint {
                if endpoint.contains("modelscope") {
                    selectedMirrorOption = "modelscope"
                } else {
                    selectedMirrorOption = "custom"
                    customMirrorURL = endpoint
                }
            } else {
                selectedMirrorOption = "official"
            }
        }
    }

    // MARK: - Verified Catalog

    private var catalogEmptyCopy: String {
        if !catalogSearchQuery.isEmpty {
            return "No matching models"
        }
        if catalogDidLoad {
            return "No verified models"
        }
        return "Loading catalog…"
    }

    private var displayedCatalogModels: [CatalogEntry] {
        _ = catalogEpoch
        if catalogSearchQuery.isEmpty {
            return modelManager.catalogModels(forCategory: typeFilter.matchType)
        }
        return ModelCatalogPolicy.search(
            modelManager.catalogStore.models,
            query: catalogSearchQuery,
            category: typeFilter.matchType
        )
    }

    private var catalogModelsSection: some View {
        let models = displayedCatalogModels
        let catalogTitle = (hubSearchAttempted && appState.allowUnlistedDownloads)
            ? "Also in catalog"
            : l10n.tr("models.verifiedTitle")

        return VStack(alignment: .leading, spacing: 12) {
            sectionHeader(catalogTitle, icon: "checkmark.seal.fill", count: models.count)

            if models.isEmpty {
                Text(catalogEmptyCopy)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            } else {
                LazyVGrid(columns: [
                    GridItem(.flexible(), spacing: 12),
                    GridItem(.flexible(), spacing: 12),
                ], spacing: 12) {
                    ForEach(models) { model in
                        catalogModelCard(model)
                    }
                }
            }
        }
        .sectionCard()
    }

    private func catalogModelCard(_ model: CatalogEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // Name + preview badge + origin link
            HStack {
                Text(model.name)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                if ModelCatalogPolicy.isIdPattern(model.id) {
                    Text(l10n.tr("models.familyBadge"))
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(NovaTheme.Colors.accent.opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
                if model.status == .preview {
                    Text(l10n.tr("models.previewBadge"))
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(.orange)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
                Spacer()
                Button {
                    if let url = URL(string: model.url) {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    Text("HF")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(NovaTheme.Colors.accent.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
                .buttonStyle(.plain)
            }

            // Description
            Text(model.description ?? "")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .lineLimit(2)

            // Tags
            HStack(spacing: 4) {
                ForEach(model.tags.prefix(3), id: \.self) { tag in
                    Text(tag)
                        .font(.system(size: 9))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(tagColor(tag).opacity(0.15))
                        .foregroundColor(tagColor(tag))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
            }

            // Size + Download button
            HStack {
                Text(model.size ?? "")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                Spacer()
                if ModelCatalogPolicy.isIdPattern(model.id) {
                    Button(l10n.tr("models.browseFamily")) {
                        searchText = ModelCatalogPolicy.hubSearchQuery(forPattern: model.id)
                        performSearch()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    downloadActionButton(for: model.id)
                }
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.rowBackground)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
        )
    }

    private func tagColor(_ tag: String) -> Color {
        switch tag {
        case "MLX": return .green
        case "4-bit", "8-bit": return .orange
        case "Vision": return .purple
        case "ASR", "TTS": return .cyan
        case "FLUX": return .pink
        case "Embedding": return .teal
        case "MoE": return .blue
        case "Multilingual": return .indigo
        case "Voice Clone": return .mint
        case "Lightweight": return .yellow
        default: return .gray
        }
    }

    // MARK: - Search Results

    private var searchResultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            let headerTitle = lastSearchSourceName.isEmpty
                ? l10n.tr("models.searchResults")
                : "Results from \(lastSearchSourceName)"
            sectionHeader(headerTitle, icon: "magnifyingglass", count: searchResults.count)

            if let regexError = searchRegexError {
                Text(regexError)
                    .font(.caption)
                    .foregroundColor(.red)
            } else if searchRegex && searchResults.count < searchTotalCount {
                Text("Regex: \(searchText) — \(searchResults.count) of \(searchTotalCount) models")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if let hubSearchError {
                Text(hubSearchError)
                    .font(.caption)
                    .foregroundColor(.red)
            } else if searchResults.isEmpty && !isSearching {
                Text(l10n.tr("models.noResults", searchText))
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                if appState.allowUnlistedDownloads {
                    Text("Hub search lists Hugging Face repos. Matching catalog entries stay below.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

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

    // MARK: - Download Action Button (shared)

    @ViewBuilder
    private func downloadActionButton(for repoId: String) -> some View {
        if modelManager.isDownloaded(repoId) {
            Label(l10n.tr("models.downloaded"), systemImage: "checkmark.circle.fill")
                .foregroundColor(NovaTheme.Colors.statusOK)
                .font(.caption)
        } else if let task = appState.downloadTasks[repoId], task.isActive {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(l10n.tr("models.downloading"))
                        .font(.caption).foregroundColor(NovaTheme.Colors.accent)
                    Text("\(Int(task.progress))%")
                        .font(.caption2).foregroundColor(.secondary)
                    if task.totalBytes > 0 {
                        Text("(\(ByteCountFormatter.string(fromByteCount: task.downloadedBytes, countStyle: .file))/\(ByteCountFormatter.string(fromByteCount: task.totalBytes, countStyle: .file)))")
                            .font(.system(size: 9)).foregroundColor(.secondary)
                    }
                    Spacer()
                    Button { appState.cancelDownload(repoId: repoId) } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption).foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.gray.opacity(0.2))
                            .frame(height: 4)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(NovaTheme.Colors.accent)
                            .frame(width: geo.size.width * min(task.progress / 100, 1.0), height: 4)
                    }
                }
                .frame(height: 4)
                // Phase line: shows Connecting / Downloading X MB/s / Stalled
                // based on per-file speed and secondsSinceLastByte. Without
                // this the user sees a frozen progress bar and assumes the
                // download is dead — the leading cause of Resume click-storms.
                if let phase = downloadPhaseText(for: task) {
                    Text(phase.text)
                        .font(.system(size: 9))
                        .foregroundColor(phase.color)
                }
            }
        } else if let task = appState.downloadTasks[repoId], task.status == .failed {
            VStack(alignment: .trailing, spacing: 2) {
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
                if let errMsg = task.errorMessage {
                    Text(errMsg)
                        .font(.system(size: 9))
                        .foregroundColor(.red.opacity(0.7))
                        .lineLimit(1)
                }
            }
        } else {
            Button(l10n.tr("models.download")) {
                triggerDownload(repoId: repoId)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    private func triggerDownload(repoId: String) {
        appState.startDownload(repoId: repoId)
    }

    // MARK: - Activity (downloads + live search)

    private func activitySection(activeOrFailed: [DownloadTaskInfo]) -> some View {
        let header = activeOrFailed.isEmpty
            ? l10n.tr("models.search")
            : (activeOrFailed.allSatisfy(\.isActive)
                ? l10n.tr("models.downloading")
                : l10n.tr("models.downloads"))

        return VStack(alignment: .leading, spacing: 10) {
            sectionHeader(
                header,
                icon: "arrow.down.circle",
                count: activeOrFailed.isEmpty ? nil : activeOrFailed.count
            )

            HStack(spacing: 6) {
                Text("Mirror")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(currentMirrorDisplayName)
                    .font(.caption2)
            }

            if isSearching, let url = currentSearchEndpoint {
                HStack(alignment: .top, spacing: 6) {
                    ProgressView().controlSize(.mini)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(l10n.tr("models.search"))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Querying \(url)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                    }
                }
            }

            ForEach(activeOrFailed, id: \.repoId) { task in
                topDownloadRow(task)
            }
        }
        .sectionCard()
    }

    private var currentMirrorDisplayName: String {
        switch selectedMirrorOption {
        case "official": return "Official (huggingface.co)"
        case "modelscope": return "ModelScope (China)"
        case "custom": return "Custom"
        default: return "Official"
        }
    }

    private var currentMirrorHostName: String {
        switch selectedMirrorOption {
        case "official": return "huggingface.co"
        case "modelscope": return "modelscope.cn"
        case "custom": return customMirrorURL.isEmpty ? "Custom" : "Custom Mirror"
        default: return "huggingface.co"
        }
    }

    /// `nil` means official huggingface.co (server default).
    private func endpointForMirrorOption(_ option: String) -> String? {
        switch option {
        case "modelscope": return "https://www.modelscope.cn"
        case "custom":
            let trimmed = customMirrorURL.trimmingCharacters(in: .whitespaces)
            return trimmed.isEmpty ? nil : trimmed
        default: return nil
        }
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
                let activeFiles = task.fileProgresses.filter {
                    $0.status == "downloading" || $0.status == "waiting"
                }
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

        return VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Image(systemName: file.status == "waiting" ? "clock" : "arrow.down")
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
                if file.speed > 0 {
                    Text("\(formatBytes(Int64(file.speed)))/s")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
            if let url = file.currentURL, !url.isEmpty {
                Text(url)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .padding(.leading, 18)
                    .help(url)
            }
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

            if appState.isCatalogAdmin,
               task.status == .completed,
               !modelManager.catalogStore.containsExactId(task.repoId) {
                Button(l10n.tr("catalogAdmin.addVerified")) {
                    let record = modelManager.getRecord(task.repoId)
                    appState.promoteToVerifiedCatalog(
                        id: task.repoId,
                        url: record?.remoteURL ?? "",
                        category: record?.modelType ?? .llm,
                        family: record?.family ?? .other,
                        sizeBytes: record?.sizeBytes ?? UInt64(max(0, task.downloadedBytes))
                    )
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

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
        let rawQuery = searchText.trimmingCharacters(in: .whitespaces)
        guard !rawQuery.isEmpty else { return }
        guard !isSearching else { return }

        isSearching = true
        Task {
            await modelManager.fetchCatalog()
            catalogDidLoad = true
            catalogEpoch += 1
            await finishSearch(rawQuery: rawQuery)
        }
    }

    @MainActor
    private func finishSearch(rawQuery: String) async {
        // Advanced off: filter the catalog locally. Hub is only used to expand
        // a family glob when the query is about that family (e.g. "qwen3.8").
        // Searching "ornith" must not miss the catalog row because Qwen3.8-* exists.
        if !appState.allowUnlistedDownloads {
            catalogSearchQuery = rawQuery
            let expandFamilies = ModelCatalogPolicy.shouldExpandFamilyGlobs(
                query: rawQuery,
                catalog: modelManager.catalogStore.models
            )
            if !expandFamilies {
                searchResults = []
                let hits = displayedCatalogModels
                if hits.isEmpty {
                    alertMessage = l10n.tr("models.noResults", rawQuery)
                    showAlert = true
                }
                isSearching = false
                return
            }
        }

        catalogSearchQuery = rawQuery
        hubSearchAttempted = true
        hubSearchError = nil

        if appState.apiKey == nil {
            newApiKey = "sk-novamlx-\(UUID().uuidString.prefix(8))"
            showApiKeyPrompt = true
            isSearching = false
            return
        }

        searchRegexError = nil
        currentSearchEndpoint = endpointForMirrorOption(selectedMirrorOption) ?? "https://huggingface.co"
        lastSearchSourceName = currentMirrorHostName

        Task {
            let adminPort = appState.adminPort

            // Build server query: optionally prepend mlx-community/
            var serverQuery = rawQuery
            if suffixMlxCommunity && !serverQuery.hasPrefix("mlx-community/") {
                serverQuery = "mlx-community/" + serverQuery
            }
            let searchEndpoint = endpointForMirrorOption(selectedMirrorOption)

            // Regex mode: fetch a wider pool so local filtering has more to match
            let limit = searchRegex ? 200 : 50

            var urlComps = URLComponents(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/search")!
            // URLQueryItem percent-encodes; do not pre-encode or "qwen3.8 flash"
            // becomes a search for the literal "qwen3.8%20flash".
            urlComps.queryItems = [
                URLQueryItem(name: "q", value: serverQuery),
                URLQueryItem(name: "limit", value: String(limit)),
                URLQueryItem(name: "mlx_only", value: searchMlxOnly ? "true" : "false"),
            ]
            if let searchEndpoint, !searchEndpoint.isEmpty {
                urlComps.queryItems?.append(URLQueryItem(name: "endpoint", value: searchEndpoint))
            }
            guard let url = urlComps.url else {
                alertMessage = l10n.tr("models.invalidUrl")
                showAlert = true
                isSearching = false
                currentSearchEndpoint = nil
                lastSearchSourceName = ""
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
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    let resultsArray: [[String: Any]] = {
                        if let direct = json["models"] as? [[String: Any]] { return direct }
                        return []
                    }()

                    var results = resultsArray.compactMap { r -> HFSearchResult? in
                        guard let id = r["id"] as? String else { return nil }
                        return HFSearchResult(id: id, tags: r["tags"] as? [String] ?? [])
                    }
                    if searchMlxOnly {
                        results = results.filter {
                            ModelCatalogPolicy.looksLikeMLXRepo(id: $0.id, tags: $0.tags)
                        }
                    }
                    searchTotalCount = results.count

                    // Local regex filter (HF API doesn't support regex)
                    if searchRegex {
                        do {
                            let regex = try NSRegularExpression(pattern: rawQuery, options: .caseInsensitive)
                            results = results.filter { r in
                                let range = NSRange(r.id.startIndex..., in: r.id)
                                return regex.firstMatch(in: r.id, range: range) != nil
                            }
                        } catch {
                            searchRegexError = "Invalid regex: \(error.localizedDescription)"
                        }
                    }

                    searchResults = results

                    if let apiError = json["error"] as? String, !apiError.isEmpty {
                        hubSearchError = apiError
                    } else if searchResults.isEmpty && searchRegexError == nil
                        && !appState.allowUnlistedDownloads
                        && displayedCatalogModels.isEmpty
                    {
                        alertMessage = l10n.tr("models.noResults", rawQuery)
                        showAlert = true
                    }
                } else {
                    hubSearchError = l10n.tr("models.unexpectedFormat")
                    alertMessage = l10n.tr("models.unexpectedFormat")
                    showAlert = true
                }
            } catch {
                alertMessage = l10n.tr("models.searchFailedMsg", error.localizedDescription)
                showAlert = true
            }
            isSearching = false
            currentSearchEndpoint = nil
        }
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
        // Bootstrap flow: when no API key is configured, create a managed key
        // in the SQLite store so internal UI→server calls can authenticate.
        // The user-edited `newApiKey` text field is ignored — the store
        // generates a cryptographically-random raw key. We surface that raw
        // key via `appState.apiKey` for the local Bearer-token calls.
        isSavingApiKey = true
        Task {
            do {
                let (_, raw) = try NovaDB.shared.apiKeyStore.create(
                    name: "Local UI (auto-created)"
                )
                appState.apiKey = raw
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
            let cardEndpoint = endpointForMirrorOption(selectedMirrorOption)
            let cardQuery = cardEndpoint != nil ? "&endpoint=\(cardEndpoint!.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")" : ""
            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/model-card?repo_id=\(repoId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? repoId)\(cardQuery)") else { return }
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

// MARK: - Download Phase Helpers

/// Derives a phase label + color from a task's live per-file stats.
/// Returns nil when no useful phase can be computed (e.g. no files in flight).
/// Decision tree:
///   - Any file with `speed > 0`  → "Downloading · X MB/s"
///   - Else, any file with `secondsSinceLastByte >= 5` → "Stalled (Ns)"
///   - Else, any file with `secondsSinceLastByte != nil` → "Connecting…"
///   - Otherwise fall back to "Waiting…" so the user sees something.
private func downloadPhaseText(for task: DownloadTaskInfo) -> (text: String, color: Color)? {
    let active = task.fileProgresses.filter { $0.status == "downloading" }
    guard !active.isEmpty else { return ("Waiting…", .secondary) }

    // Aggregate speed across all actively-downloading files.
    let totalSpeed = active.reduce(0.0) { $0 + $1.speed }
    if totalSpeed > 0 {
        let mb = totalSpeed / 1_000_000
        if mb >= 0.1 {
            return ("Downloading · \(String(format: "%.1f", mb)) MB/s", .green)
        }
        let kb = totalSpeed / 1000
        return ("Downloading · \(String(format: "%.0f", kb)) KB/s", .green)
    }

    // No speed yet — look at the worst (largest) secondsSinceLastByte.
    let stallSeconds: Double? = active
        .compactMap { $0.secondsSinceLastByte }
        .max()

    if let s = stallSeconds {
        if s >= 5 {
            return ("Stalled (\(Int(s))s since last byte)", .orange)
        }
        return ("Connecting…", .secondary)
    }
    return ("Waiting…", .secondary)
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


