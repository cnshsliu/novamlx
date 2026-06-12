import SwiftUI
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

struct DownloadsPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n
    @Binding var typeFilter: ModelsPageView.ModelTypeFilter

    @State private var searchText = ""
    @State private var searchResults: [HFSearchResult] = []
    @State private var isSearching = false
    @State private var selectedMirrorOption = "official"
    @State private var customMirrorURL = ""

    // Compat warning
    @State private var showCompatWarning = false
    @State private var compatWarningRepoId = ""

    // Alert / API key
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var showApiKeyPrompt = false
    @State private var newApiKey = ""
    @State private var isSavingApiKey = false

    // Model card
    @State private var selectedModelCard: ModelCardData?

    // Backend activity
    @State private var isBackendActivityExpanded = false
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
            ScrollView {
                VStack(spacing: 20) {
                    searchSection

                    // Show search results or suggested models
                    if !searchResults.isEmpty {
                        searchResultsSection
                    } else {
                        suggestedModelsSection
                    }

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

                    backendActivitySection

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
        }
        .alert(alertMessage, isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        }
        .alert("⚠️ Compatibility Notice", isPresented: $showCompatWarning) {
            Button("Cancel", role: .cancel) {
                showCompatWarning = false
            }
            Button("Download Anyway") {
                showCompatWarning = false
                appState.startDownload(repoId: compatWarningRepoId)
            }
        } message: {
            Text("This model doesn't appear to be in MLX format.\nIt may not work with NovaMLX on macOS.")
        }
        .sheet(isPresented: $showApiKeyPrompt) {
            apiKeySetupSheet
        }
        .sheet(item: $selectedModelCard) { card in
            modelCardSheet(card)
        }
    }

    // MARK: - Search

    private var searchSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Search bar
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
            }

            // Mirror picker
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Mirror")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Picker("", selection: $selectedMirrorOption) {
                        Text("Official (huggingface.co)").tag("official")
                        Text("hf-mirror.com (China)").tag("hf-mirror")
                        Text("Custom URL...").tag("custom")
                    }
                    .pickerStyle(.menu)
                    .onChange(of: selectedMirrorOption) { _, newOption in
                        let endpoint: String? = {
                            switch newOption {
                            case "official": return nil
                            case "hf-mirror": return "https://hf-mirror.com"
                            case "custom": return customMirrorURL.isEmpty ? nil : customMirrorURL
                            default: return nil
                            }
                        }()
                        Task { await appState.setHuggingfaceEndpoint(endpoint) }
                        mirrorChangeMessage = (newOption == "official")
                            ? "Switched to official Hugging Face"
                            : "Mirror changed. New downloads will use the selected source."
                        showMirrorChangeNote = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                            showMirrorChangeNote = false
                        }
                    }
                }

                if selectedMirrorOption == "custom" {
                    TextField("Custom endpoint (e.g. https://hf-mirror.com)", text: $customMirrorURL)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))
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
                if endpoint == "https://hf-mirror.com" {
                    selectedMirrorOption = "hf-mirror"
                } else {
                    selectedMirrorOption = "custom"
                    customMirrorURL = endpoint
                }
            } else {
                selectedMirrorOption = "official"
            }
        }
    }

    // MARK: - Suggested Models

    private var suggestedModelsSection: some View {
        let models = modelManager.suggestedModels(forCategory: typeFilter.matchType)
        let downloadedIds = Set(modelManager.downloadedModels().map(\.id))
        let notDownloaded = models.filter { !downloadedIds.contains($0.repo) }

        return VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Suggested", icon: "star.fill", count: notDownloaded.count)

            if notDownloaded.isEmpty {
                Text(l10n.tr("models.allDownloaded"))
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            } else {
                LazyVGrid(columns: [
                    GridItem(.flexible(), spacing: 12),
                    GridItem(.flexible(), spacing: 12),
                ], spacing: 12) {
                    ForEach(notDownloaded) { model in
                        suggestedModelCard(model)
                    }
                }
            }
        }
        .sectionCard()
    }

    private func suggestedModelCard(_ model: ModelManager.SuggestedModel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // Name + HF link
            HStack {
                Text(model.name)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Spacer()
                Button {
                    if let url = URL(string: "https://huggingface.co/\(model.repo)") {
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
            Text(model.description)
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
                Text(model.size)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                Spacer()
                downloadActionButton(for: model.repo)
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
        // Heuristic compat check: if repo doesn't contain known MLX indicators, warn
        let lower = repoId.lowercased()
        let looksLikeMlx = lower.contains("mlx") || lower.contains("4bit") || lower.contains("8bit")
            || lower.contains("gguf") || lower.contains("quant") || lower.contains("safetensors")
        if !looksLikeMlx {
            compatWarningRepoId = repoId
            showCompatWarning = true
            return
        }
        appState.startDownload(repoId: repoId)
    }

    // MARK: - Backend Activity Panel

    private var backendActivitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isBackendActivityExpanded.toggle()
                }
            } label: {
                HStack {
                    Text(backendActivityTitle)
                        .font(.headline)
                    Spacer()
                    if !isBackendActivityExpanded {
                        Text(backendActivitySummary)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    Image(systemName: isBackendActivityExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                }
            }
            .buttonStyle(.plain)

            if isBackendActivityExpanded {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Mirror:")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(currentMirrorDisplayName)
                            .font(.caption2)
                            .foregroundColor(.primary)
                    }

                    if isSearching, let url = currentSearchEndpoint {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Search")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text("Querying: \(url)")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.primary)
                                .lineLimit(3)
                        }
                    }

                    let active = activeDownloadFileDetails
                    if !active.isEmpty {
                        Text("Active Downloads")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        ForEach(active) { detail in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(detail.filename)
                                    .font(.system(size: 11, weight: .semibold))
                                    .lineLimit(1)
                                if let url = detail.currentURL {
                                    Text(url)
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundColor(.secondary)
                                        .lineLimit(2)
                                }
                                HStack(spacing: 12) {
                                    Text("\(detail.progress, specifier: "%.1f")%")
                                    if detail.speed > 0 {
                                        Text("\(formatBytes(Int64(detail.speed)))/s")
                                    }
                                }
                                .font(.caption2)
                            }
                            .padding(8)
                            .background(Color.gray.opacity(0.08))
                            .cornerRadius(6)
                        }
                    } else if !isSearching {
                        Text("No active backend operations")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }

    private var backendActivitySummary: String {
        if isSearching { return "Searching..." }
        let count = appState.downloadTasks.values.filter { $0.isActive }.count
        if count > 0 { return "\(count) active" }
        return "Idle"
    }

    private var backendActivityTitle: String {
        if isSearching, let endpoint = currentSearchEndpoint {
            if endpoint.contains("hf-mirror") { return "hf-mirror.com" }
            return endpoint.contains("huggingface") ? "huggingface.co" : "Custom Mirror"
        }
        let active = appState.downloadTasks.values.filter { $0.isActive }
        if !active.isEmpty { return currentMirrorDisplayName }
        return "Backend Activities"
    }

    private var currentMirrorDisplayName: String {
        switch selectedMirrorOption {
        case "official": return "Official (huggingface.co)"
        case "hf-mirror": return "hf-mirror.com"
        case "custom": return "Custom"
        default: return "Official"
        }
    }

    private var activeDownloadFileDetails: [DownloadFileDetail] {
        appState.downloadTasks.values
            .filter { $0.isActive }
            .flatMap { task in
                task.fileProgresses
                    .filter { $0.status == "downloading" || $0.status == "waiting" }
                    .map { file in
                        DownloadFileDetail(
                            filename: file.filename,
                            currentURL: file.currentURL,
                            speed: file.speed,
                            progress: task.totalBytes > 0 ? Double(file.downloadedBytes) / Double(task.totalBytes) * 100 : 0,
                            retryCount: file.retryCount,
                            isResuming: file.isResuming
                        )
                    }
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
        currentSearchEndpoint = {
            switch selectedMirrorOption {
            case "hf-mirror": return "https://hf-mirror.com"
            case "custom": return customMirrorURL.isEmpty ? nil : customMirrorURL
            default: return "https://huggingface.co"
            }
        }()
        lastSearchSourceName = {
            switch selectedMirrorOption {
            case "official": return "huggingface.co"
            case "hf-mirror": return "hf-mirror.com"
            case "custom": return customMirrorURL.isEmpty ? "Custom" : "Custom Mirror"
            default: return "huggingface.co"
            }
        }()

        Task {
            let adminPort = appState.adminPort
            let query = searchText.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? searchText
            let searchEndpoint: String? = {
                switch selectedMirrorOption {
                case "official": return nil
                case "hf-mirror": return "https://hf-mirror.com"
                case "custom": return customMirrorURL.isEmpty ? nil : customMirrorURL
                default: return nil
                }
            }()
            let endpointQuery = searchEndpoint != nil ? "&endpoint=\(searchEndpoint!.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")" : ""
            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/search?q=\(query)\(endpointQuery)") else {
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

                    searchResults = resultsArray.compactMap { r in
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
            let cardEndpoint: String? = {
                switch selectedMirrorOption {
                case "hf-mirror": return "https://hf-mirror.com"
                case "custom": return customMirrorURL.isEmpty ? nil : customMirrorURL
                default: return nil
                }
            }()
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

struct DownloadFileDetail: Identifiable {
    let id = UUID()
    let filename: String
    let currentURL: String?
    let speed: Double
    let progress: Double
    let retryCount: Int
    let isResuming: Bool
}
