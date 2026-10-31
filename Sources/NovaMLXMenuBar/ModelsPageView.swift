import AppKit
import SwiftUI
import NovaMLXCore
import NovaMLXEngine
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
    @State private var convertingModelId: String?
    @State private var tieToDelete: String?
    @State private var showTieDeleteConfirm = false
    @State private var typeFilter: ModelTypeFilter = .all
    @State private var modelQuery = ""
    @State private var scrollAnchor: String?
    @State private var downloadedSnapshot: [ModelRecord] = []
    @State private var showUnmountedModels = false
    @State private var unmountedSnapshot: [ModelRecord] = []
    /// DFlash / DSpark / MTP companions stay hidden until the user opts in.
    /// `false` is the default. AppStorage keeps the choice across launches.
    @AppStorage("models.showCompanions") private var showCompanions = false
    /// Empty means every configured model root.
    @State private var diskFilter = ModelRootFilter.all
    /// Downloaded and verified are list facets, not buried labels.
    @State private var downloadedFacet: LocalModelFacet
    @State private var verifiedFacet: LocalModelFacet
    /// Catalog snapshot so the not-downloaded facet redraws after a fetch.
    @State private var catalogSnapshot: [CatalogEntry]
    @State private var didFetchCatalog = false
    @State private var companionIds: Set<String> = []
    @State private var redownloadRoots: [NovaMLXPaths.ModelDirStatus] = []
    @State private var redownloadTarget: ModelRecord?
    @State private var showRootPicker = false
    let memory: ModelsPageMemory

    init(appState: MenuBarAppState, inferenceService: InferenceService, modelManager: ModelManager, memory: ModelsPageMemory) {
        self.appState = appState
        self.inferenceService = inferenceService
        self.modelManager = modelManager
        self.memory = memory
        _typeFilter = State(initialValue: memory.typeFilter)
        _modelQuery = State(initialValue: memory.modelQuery)
        _downloadedFacet = State(initialValue: memory.downloadedFacet)
        _verifiedFacet = State(initialValue: memory.verifiedFacet)
        _catalogSnapshot = State(initialValue: modelManager.catalogStore.models)
        _scrollAnchor = State(initialValue: memory.scrollAnchor)
    }

    enum ModelTypeFilter: String, CaseIterable {
        case all
        case llm
        case vlm
        case embedding
        case audio
        case image
        case decision

        var label: String {
            switch self {
            case .all: return "All"
            case .llm: return "LLM"
            case .vlm: return "VLM"
            case .embedding: return "Embed"
            case .audio: return "Audio"
            case .image: return "Image"
            case .decision: return "Decide"
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
            case .decision: return "arrow.triangle.branch"
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
            case .decision: return modelType == .decision
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
            case .decision: return .decision
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            ModelTypeFilterBar(filter: $typeFilter)
            LocalModelFacetBar(downloaded: $downloadedFacet, verified: $verifiedFacet)
            modelFilterField

            Divider().padding(.horizontal, 24)

            localModelsContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .alert(alertMessage, isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        }
        .confirmationDialog(
            l10n.tr("models.tieDeleteConfirm"),
            isPresented: $showTieDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button(l10n.tr("models.tieDelete"), role: .destructive) {
                if let id = tieToDelete {
                    Task {
                        do {
                            if let record = modelManager.getRecord(id),
                               inferenceService.listLoadedModels().contains(id)
                            {
                                await inferenceService.unloadModel(
                                    ModelIdentifier(id: id, family: record.family)
                                )
                            }
                            try await appState.deleteTIE(modelId: id)
                            refreshTrigger.toggle()
                        } catch {
                            alertMessage = error.localizedDescription
                            showAlert = true
                        }
                    }
                }
                tieToDelete = nil
            }
            Button(l10n.tr("models.cancel"), role: .cancel) { tieToDelete = nil }
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
        .confirmationDialog(
            l10n.tr("models.chooseDisk"),
            isPresented: $showRootPicker,
            titleVisibility: .visible
        ) {
            ForEach(redownloadRoots, id: \.path) { root in
                Button(rootButtonLabel(root)) {
                    if let record = redownloadTarget {
                        startRedownload(record, to: URL(fileURLWithPath: root.path, isDirectory: true))
                    }
                    redownloadTarget = nil
                }
            }
            Button(l10n.tr("models.cancel"), role: .cancel) { redownloadTarget = nil }
        } message: {
            if let record = redownloadTarget {
                Text(l10n.tr("models.chooseDiskMessage", record.id))
            }
        }
        .onAppear { Task { await refreshDownloaded() } }
        .onChange(of: modelQuery) { _, value in memory.modelQuery = value }
        .onChange(of: typeFilter) { _, value in memory.typeFilter = value }
        .onChange(of: downloadedFacet) { _, value in memory.downloadedFacet = value }
        .onChange(of: verifiedFacet) { _, value in memory.verifiedFacet = value }
        .onChange(of: scrollAnchor) { _, value in memory.scrollAnchor = value }
        .onChange(of: refreshTrigger) { _, _ in Task { await refreshDownloaded(fullRescan: true) } }
        .onReceive(NotificationCenter.default.publisher(for: .novaMLXModelsChanged)) { _ in
            Task { await refreshDownloaded() }
        }
    }

    private func refreshDownloaded(fullRescan: Bool = false) async {
        // Full rescan re-syncs the registry with what's actually on disk
        // (manually added/removed/moved folders, unmounted disks, incomplete
        // downloads); the light path only refreshes reachability caches.
        if fullRescan {
            _ = await Task.detached(priority: .utility) {
                modelManager.discoverModels()
            }.value
        }
        await modelManager.refreshModelCaches()
        if !didFetchCatalog {
            didFetchCatalog = true
            if modelManager.catalogStore.models.isEmpty {
                await modelManager.fetchCatalog()
            }
        }
        catalogSnapshot = modelManager.catalogStore.models
        downloadedSnapshot = modelManager.downloadedModels()
        unmountedSnapshot = modelManager.unmountedModels()
        let catalog = catalogSnapshot
        var items = downloadedSnapshot.map { (id: $0.id, directory: $0.localURL) }
        let known = Set(items.map(\.id))
        for id in appState.loadedModels + appState.restoringModels where !known.contains(id) {
            let directory = modelManager.getRecord(id)?.localURL
                ?? NovaMLXPaths.directory(forModelId: id)
            items.append((id: id, directory: directory))
        }
        companionIds = await Self.detectCompanionIds(items, catalog: catalog)
        if !diskFilter.isEmpty,
           !ModelRootFilter.options(for: NovaMLXPaths.configuredModelsDirs).contains(where: { $0.path == diskFilter }) {
            diskFilter = ModelRootFilter.all
        }
    }

    /// Companion classification (id, catalog `Companion` tag, draft config).
    /// Config reads stay off the main actor.
    private static func detectCompanionIds(
        _ items: [(id: String, directory: URL)],
        catalog: [CatalogEntry]
    ) async -> Set<String> {
        let snapshot = items
        let tags = catalog
        return await Task.detached(priority: .utility) {
            Set(snapshot.compactMap { item in
                CompanionModel.isAuxiliary(id: item.id, catalog: tags, directory: item.directory)
                    ? item.id : nil
            })
        }.value
    }

    /// Companions stay out of both lists unless the user opts in.
    /// The id / catalog check is synchronous so known companions do not flash
    /// in before the config scan finishes.
    private func passesCompanionFilter(_ id: String, catalog: [CatalogEntry]) -> Bool {
        if showCompanions { return true }
        if companionIds.contains(id) { return false }
        return !CompanionModel.isAuxiliary(id: id, catalog: catalog)
    }

    /// Drops a selection whose root was removed from the models-path list.
    private var resolvedDiskFilter: String {
        if diskFilter.isEmpty { return ModelRootFilter.all }
        let roots = ModelRootFilter.options(for: NovaMLXPaths.configuredModelsDirs)
        return roots.contains(where: { $0.path == diskFilter }) ? diskFilter : ModelRootFilter.all
    }

    private func passesDiskFilter(_ record: ModelRecord) -> Bool {
        ModelRootFilter.matches(modelURL: record.localURL, selectedRoot: resolvedDiskFilter)
    }

    private var statusFacetsNarrowed: Bool {
        downloadedFacet != .any || verifiedFacet != .any
    }

    /// `isDownloaded` is on-disk (or previously downloaded, for an unmounted copy).
    /// `isLoaded` is resident in memory and is independent of `companionReady`.
    private func passesStatusFacets(id: String, isDownloaded: Bool, isLoaded: Bool) -> Bool {
        let row = LocalModelRowState(
            id: id,
            isDownloaded: isDownloaded,
            isVerified: LocalModelListFilter.isVerifiedWeight(id: id, catalog: catalogSnapshot),
            isLoaded: isLoaded
        )
        return LocalModelListFilter.matches(row, downloaded: downloadedFacet, verified: verifiedFacet)
    }

    private var localModelsContent: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 20) {
                    if downloadedFacet != .no {
                        loadedSection
                        downloadedSection
                        if showUnmountedModels && !unmountedSnapshot.isEmpty {
                            unmountedSection
                                .id(Self.unmountedSectionAnchor)
                        }
                    }
                    if showsNotDownloadedSection {
                        notDownloadedSection
                    }
                }
                .padding(24)
            }
            .scrollPosition(id: $scrollAnchor)
        }
    }

    /// Scroll anchor for the unmounted-disks section; model ids are
    /// `org/name`, so this never collides with row ids.
    private static let unmountedSectionAnchor = "section:unmounted"

    /// Models whose files live on a removable disk that is not mounted.
    /// Shown on demand so the user can re-download them to another disk.
    private var unmountedSection: some View {
        let downloading = Set(
            unmountedSnapshot
                .map { $0.id }
                .filter { modelManager.getDownloadStatus($0)?.state == .downloading }
        )
        let catalog = catalogSnapshot
        let unmounted = unmountedSnapshot.filter { record in
            !downloading.contains(record.id)
                && matchesQuery(record.id)
                && typeFilter.matches(record.modelType)
                && passesCompanionFilter(record.id, catalog: catalog)
                && passesStatusFacets(id: record.id, isDownloaded: true, isLoaded: false)
        }

        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                sectionHeader(l10n.tr("models.unmountedSection"), icon: "externaldrive.badge.xmark", count: unmounted.count)
            }

            if unmounted.isEmpty {
                if modelQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !statusFacetsNarrowed {
                    emptyState(l10n.tr("models.noInactiveModels"), subtitle: l10n.tr("models.noInactiveModelsSub"))
                } else {
                    emptyState(l10n.tr("models.noFilterMatch"), subtitle: "")
                }
            } else {
                ForEach(unmounted, id: \.id) { record in
                    modelRow(
                        record.id,
                        subtitle: "\(volumeName(of: record.localURL)) · \(l10n.tr("models.notMounted"))",
                        isLoaded: false,
                        isDownloaded: true,
                        actions: {
                            Button(l10n.tr("models.redownload")) {
                                requestRedownload(record)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .help(l10n.tr("models.redownloadHelp"))

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

    private func volumeName(of url: URL) -> String {
        let parts = url.standardizedFileURL.pathComponents
        guard parts.count > 2, parts[0] == "/", parts[1] == "Volumes" else {
            return url.deletingLastPathComponent().lastPathComponent
        }
        return parts[2]
    }

    private func rootButtonLabel(_ root: NovaMLXPaths.ModelDirStatus) -> String {
        let name = (root.path as NSString).lastPathComponent
        guard let free = root.freeBytes else { return name }
        let freeText = ByteCountFormatter.string(fromByteCount: Int64(free), countStyle: .file)
        return "\(name) — \(freeText) \(l10n.tr("models.free"))"
    }

    /// Re-download flow: one viable disk → download immediately; several →
    /// let the user pick (external disks may be multiple configured roots).
    private func requestRedownload(_ record: ModelRecord) {
        let viable = NovaMLXPaths.modelDirStatuses().filter { $0.mounted && $0.writable }
        guard !viable.isEmpty else {
            alertMessage = l10n.tr("models.noWritableDisks")
            showAlert = true
            return
        }
        if viable.count == 1, let only = viable.first {
            startRedownload(record, to: URL(fileURLWithPath: only.path, isDirectory: true))
            return
        }
        redownloadTarget = record
        redownloadRoots = viable
        showRootPicker = true
    }

    private func startRedownload(_ record: ModelRecord, to root: URL) {
        do {
            try modelManager.startDownload(record.id, preferredRoot: root)
            refreshTrigger.toggle()
        } catch {
            alertMessage = error.localizedDescription
            showAlert = true
        }
    }

    private var modelFilterField: some View {
        HStack(spacing: 8) {
            ModelRegexFilterField(
                text: $modelQuery,
                placeholder: l10n.tr("models.filterPlaceholder")
            )

            if !unmountedSnapshot.isEmpty {
                Button {
                    toggleUnmountedSection()
                } label: {
                    Label(
                        l10n.tr("models.unmountedToggle"),
                        systemImage: showUnmountedModels
                            ? "externaldrive.badge.xmark" : "externaldrive.badge.exclamationmark"
                    )
                    .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help(l10n.tr("models.unmountedToggleHelp"))
            }
        }
        .padding(.horizontal, 24)
        .padding(.bottom, 8)
    }

    /// Toggle the unmounted-disks section; turning it on jumps the scroll
    /// view to the section (it renders at the bottom of the list).
    private func toggleUnmountedSection() {
        showUnmountedModels.toggle()
        if showUnmountedModels {
            scrollAnchor = Self.unmountedSectionAnchor
        }
    }

    private func matchesQuery(_ id: String) -> Bool {
        ModelQueryMatch.matches(modelQuery, in: id)
    }

    private var loadedSection: some View {
        let catalog = catalogSnapshot
        let loaded = appState.loadedModels.filter { id in
            guard matchesQuery(id), passesCompanionFilter(id, catalog: catalog) else { return false }
            guard passesStatusFacets(id: id, isDownloaded: true, isLoaded: true) else { return false }
            guard let record = modelManager.getRecord(id) else { return typeFilter == .all }
            return typeFilter.matches(record.modelType)
        }
        let restoring = appState.restoringModels.filter { id in
            guard matchesQuery(id), passesCompanionFilter(id, catalog: catalog) else { return false }
            guard passesStatusFacets(id: id, isDownloaded: true, isLoaded: false) else { return false }
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
                if modelQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !statusFacetsNarrowed {
                    emptyState(l10n.tr("models.noModelsLoaded"), subtitle: l10n.tr("models.noModelsLoadedSub"))
                } else {
                    emptyState(l10n.tr("models.noFilterMatch"), subtitle: "")
                }
            } else {
                // Models currently being restored on startup — show spinner rows
                ForEach(restoring, id: \.self) { modelId in
                    restoringModelRow(modelId)
                }
                ForEach(loaded, id: \.self) { modelId in
                    let pinned = inferenceService.isPinned(modelId)
                    modelRow(
                        modelId,
                        subtitle: modelManager.getRecord(modelId)?.family.rawValue ?? l10n.tr("models.unknown"),
                        isLoaded: true,
                        isDownloaded: true,
                        actions: {
                            playButton(modelId: modelId)
                            specCompanionToggle(for: modelId)
                            specBoostBadge(for: modelId)
                            tieBadge(for: modelId)
                            addToCatalogButton(modelId: modelId)
                            tieActions(for: modelId, isLoaded: true)
                            pinToggle(for: modelId)
                            if !pinned {
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
            VStack(alignment: .leading, spacing: 2) {
                Text(modelId)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                Text("Loading…")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                LocalModelStatusBadges(
                    modelId: modelId,
                    isVerified: LocalModelListFilter.isVerifiedWeight(id: modelId, catalog: catalogSnapshot),
                    isDownloaded: true,
                    isLoaded: false
                )
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

    private func pinToggle(for modelId: String) -> some View {
        Toggle(isOn: Binding(
            get: { inferenceService.isPinned(modelId) },
            set: { on in
                inferenceService.setPinned(modelId, pinned: on)
                refreshTrigger.toggle()
            }
        )) {
            Text(l10n.tr("models.pin"))
                .font(.caption2.weight(.semibold))
        }
        .toggleStyle(.switch)
        .controlSize(.mini)
        .help(l10n.tr("models.pinHelp"))
        .accessibilityIdentifier("models-pin-\(modelId)")
    }

    @ViewBuilder
    private func specCompanionToggle(for modelId: String) -> some View {
        if inferenceService.mtpSwitchAvailable(modelId) {
            let enabled = inferenceService.isMtpEnabled(modelId)
            let isDFlash = inferenceService.hasCompanionDFlash(modelId)
            let title: String = {
                if isDFlash {
                    return enabled ? l10n.tr("models.dflashOn") : l10n.tr("models.dflashOff")
                }
                return enabled ? l10n.tr("models.mtpOn") : l10n.tr("models.mtpOff")
            }()
            Toggle(isOn: Binding(
                get: { inferenceService.isMtpEnabled(modelId) },
                set: { on in
                    Task {
                        await inferenceService.setMtpEnabled(modelId, enabled: on)
                        refreshTrigger.toggle()
                    }
                }
            )) {
                Text(title)
                    .font(.caption2.weight(.semibold))
            }
            .toggleStyle(.switch)
            .controlSize(.mini)
            .help(isDFlash ? l10n.tr("models.dflashHelp") : l10n.tr("models.mtpHelp"))
        }
    }

    @ViewBuilder
    private func specBoostBadge(for modelId: String) -> some View {
        if let boost = appState.specBoostStatus[modelId], boost.draftModelId != modelId {
            if boost.status == "eligible", boost.draftDownloaded != true {
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
        }
    }

    @ViewBuilder
    private func tieBadge(for modelId: String) -> some View {
        switch appState.tieStatus[modelId]?.status {
        case .ready:
            Text(l10n.tr("models.tie"))
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.orange.opacity(0.18))
                .foregroundColor(.orange)
                .cornerRadius(4)
                .help(l10n.tr("models.tieReady"))
        case .incomplete:
            Text(l10n.tr("models.tieIncomplete"))
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.red.opacity(0.15))
                .foregroundColor(.red)
                .cornerRadius(4)
                .help(appState.tieStatus[modelId]?.message ?? l10n.tr("models.tieIncompleteHint"))
        default:
            EmptyView()
        }
    }

    @ViewBuilder
    private func tieConvertingLabel(for modelId: String) -> some View {
        let tie = appState.tieStatus[modelId]
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 2) {
                Text(l10n.tr("models.tieConverting"))
                    .font(.caption)
                    .foregroundColor(NovaTheme.Colors.accent)
                if let msg = tie?.message {
                    Text(msg)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                if let frac = tie?.fraction {
                    ProgressView(value: frac)
                        .frame(width: 140)
                }
            }
        }
    }

    @ViewBuilder
    private func tieActions(for modelId: String, isLoaded: Bool) -> some View {
        let status = appState.tieStatus[modelId]?.status
        if !isLoaded, status == .convertible || status == .incomplete {
            Button(l10n.tr("models.tieConvert")) {
                convertingModelId = modelId
                Task {
                    do {
                        try await appState.convertTIE(modelId: modelId)
                        refreshTrigger.toggle()
                    } catch {
                        alertMessage = error.localizedDescription
                        showAlert = true
                    }
                    convertingModelId = nil
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help(l10n.tr("models.tieConvertHint"))
        }
        if (status == .ready || status == .incomplete) && !isLoaded {
            Button(l10n.tr("models.tieDelete"), role: .destructive) {
                tieToDelete = modelId
                showTieDeleteConfirm = true
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        } else if (status == .ready || status == .incomplete) && isLoaded {
            Button(l10n.tr("models.tieDelete"), role: .destructive) {
                tieToDelete = modelId
                showTieDeleteConfirm = true
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Unload, then remove TIE layout")
        }
    }

    private var diskFilterBinding: Binding<String> {
        Binding(
            get: { resolvedDiskFilter },
            set: { diskFilter = $0 }
        )
    }

    /// Disk menu and companion checkbox. Both sit on the downloaded-not-loaded
    /// header. The companion choice also filters the loaded list.
    private var downloadedHeaderFilters: some View {
        let roots = ModelRootFilter.options(for: NovaMLXPaths.configuredModelsDirs)
        return HStack(spacing: 10) {
            Picker(l10n.tr("models.diskFilter"), selection: diskFilterBinding) {
                Text(l10n.tr("models.diskAll")).tag(ModelRootFilter.all)
                ForEach(roots) { root in
                    Text(root.label).tag(root.path)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .controlSize(.small)
            .fixedSize()
            .accessibilityIdentifier("models-disk-filter")
            .accessibilityLabel(l10n.tr("models.diskFilter"))
            .help(l10n.tr("models.diskFilterHelp"))

            Toggle(l10n.tr("models.showCompanions"), isOn: $showCompanions)
                .toggleStyle(.checkbox)
                .font(.system(size: 11))
                .accessibilityIdentifier("models-show-companions")
                .help(l10n.tr("models.showCompanionsHelp"))
        }
    }

    private var downloadedSection: some View {
        let catalog = catalogSnapshot
        let allDownloaded = downloadedSnapshot
        let loaded = Set(inferenceService.listLoadedModels())
        let downloaded = allDownloaded.filter {
            !loaded.contains($0.id) && typeFilter.matches($0.modelType)
                && matchesQuery($0.id)
                && passesCompanionFilter($0.id, catalog: catalog)
                && passesDiskFilter($0)
                && passesStatusFacets(id: $0.id, isDownloaded: true, isLoaded: false)
        }
        let diskNarrowed = !resolvedDiskFilter.isEmpty
        let queryNarrowed = !modelQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                sectionHeader(l10n.tr("models.unloaded"), icon: "arrow.down.circle", count: downloaded.count)
                Spacer(minLength: 8)
                downloadedHeaderFilters
                    .layoutPriority(1)
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
                if queryNarrowed || diskNarrowed || statusFacetsNarrowed {
                    emptyState(l10n.tr("models.noFilterMatch"), subtitle: "")
                } else {
                    emptyState(l10n.tr("models.noInactiveModels"), subtitle: l10n.tr("models.noInactiveModelsSub"))
                }
            } else {
                ForEach(downloaded, id: \.id) { record in
                    modelRow(
                        record.id,
                        subtitle: "\(record.family.rawValue)  \(record.sizeBytes > 0 ? record.sizeBytes.bytesFormatted : "")",
                        isLoaded: false,
                        isDownloaded: true,
                        actions: {
                            if convertingModelId == record.id || appState.tieStatus[record.id]?.status == .converting {
                                tieConvertingLabel(for: record.id)
                            } else if loadingModelId == record.id {
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .controlSize(.small)
                                    Text(l10n.tr("models.loading") + " " + (record.id.components(separatedBy: "/").last ?? record.id))
                                        .font(.caption).foregroundColor(NovaTheme.Colors.accent)
                                        .lineLimit(1)
                                }
                            } else {
                                specCompanionToggle(for: record.id)
                                specBoostBadge(for: record.id)
                                addToCatalogButton(modelId: record.id)
                                tieActions(for: record.id, isLoaded: false)
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

    /// Verified catalog weights that are not on disk. Hidden when the
    /// Downloaded facet is "downloaded". A chosen disk has no not-downloaded
    /// rows; that combination still shows the empty filter state.
    private var showsNotDownloadedSection: Bool {
        guard downloadedFacet != .yes else { return false }
        if !resolvedDiskFilter.isEmpty { return downloadedFacet == .no }
        if downloadedFacet == .no || verifiedFacet != .any { return true }
        if !modelQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return true }
        return !filteredNotDownloadedWeights().isEmpty
    }

    private func filteredNotDownloadedWeights() -> [CatalogEntry] {
        guard resolvedDiskFilter.isEmpty else { return [] }
        let excluding = Set(
            downloadedSnapshot.map(\.id)
                + unmountedSnapshot.map(\.id)
                + appState.loadedModels
                + appState.restoringModels
                + inferenceService.listLoadedModels()
        )
        return LocalModelListFilter.notDownloadedWeights(
            catalog: catalogSnapshot,
            excluding: excluding
        ).filter { entry in
            typeFilter.matches(entry.category)
                && (matchesQuery(entry.id) || matchesQuery(entry.name) || matchesQuery(entry.quant ?? ""))
                && passesCompanionFilter(entry.id, catalog: catalogSnapshot)
                && passesStatusFacets(id: entry.id, isDownloaded: false, isLoaded: false)
        }
    }

    private var notDownloadedSection: some View {
        let weights = filteredNotDownloadedWeights()
        let queryNarrowed = !modelQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("models.notDownloadedSection"), icon: "square.and.arrow.down", count: weights.count)
            if weights.isEmpty {
                if queryNarrowed || statusFacetsNarrowed {
                    emptyState(l10n.tr("models.noFilterMatch"), subtitle: "")
                } else if catalogSnapshot.isEmpty {
                    emptyState(l10n.tr("models.notDownloadedSection"), subtitle: l10n.tr("models.notDownloadedEmpty"))
                } else {
                    emptyState(l10n.tr("models.notDownloadedEmpty"), subtitle: "")
                }
            } else {
                ForEach(weights, id: \.id) { entry in
                    let subtitle = [entry.family.rawValue, entry.quant, entry.size]
                        .compactMap { $0 }
                        .filter { !$0.isEmpty }
                        .joined(separator: "  ")
                    modelRow(
                        entry.id,
                        subtitle: subtitle,
                        isLoaded: false,
                        isDownloaded: false,
                        actions: {
                            notDownloadedActions(for: entry.id)
                        }
                    )
                }
            }
        }
        .sectionCard()
    }

    @ViewBuilder
    private func notDownloadedActions(for modelId: String) -> some View {
        if let task = appState.downloadTasks[modelId], task.isActive {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text(l10n.tr("models.downloading"))
                    .font(.caption)
                    .foregroundColor(NovaTheme.Colors.accent)
            }
        } else {
            Button(l10n.tr("models.download")) {
                appState.startDownload(repoId: modelId)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityIdentifier("models-download-\(modelId)")
        }
    }

    @ViewBuilder
    private func addToCatalogButton(modelId: String) -> some View {
        if appState.isCatalogAdmin,
           !modelManager.catalogStore.containsExactId(modelId),
           let record = modelManager.getRecord(modelId) {
            Button(l10n.tr("catalogAdmin.addVerified")) {
                appState.promoteToVerifiedCatalog(
                    id: record.id,
                    url: record.remoteURL,
                    category: record.modelType,
                    family: record.family,
                    sizeBytes: record.sizeBytes
                )
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private func playButton(modelId: String) -> some View {
        if !companionIds.contains(modelId),
           !CompanionModel.isAuxiliary(id: modelId, catalog: modelManager.catalogStore.models) {
            Button {
                openTypedDemo(modelId)
            } label: {
                Label("Play", systemImage: "play.fill")
                    .font(.system(size: 14, weight: .semibold))
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .help("Open web demo")
        }
    }

    /// One demo page per model type; the page itself switches same-type
    /// models. Local model ids open the matching demo on NovaMLX.
    private func openTypedDemo(_ modelId: String) {
        let path: String
        switch modelManager.getRecord(modelId)?.modelType {
        case .image: path = "/demo/image"
        case .decision: path = "/demo/decision"
        case .audio: path = "/demo/audio"
        case .embedding: path = "/demo/embed"
        case .vlm: path = "/demo/vlm"
        default: path = "/demo/chat"
        }
        openDemo(path: path, modelId: modelId)
    }

    private func openDemo(path: String, modelId: String) {
        var components = URLComponents(string: "http://127.0.0.1:\(appState.serverPort)\(path)")
        var items = [URLQueryItem(name: "model", value: modelId)]
        if let key = appState.apiKey, !key.isEmpty {
            items.append(URLQueryItem(name: "key", value: key))
        }
        components?.queryItems = items
        guard let url = components?.url else { return }
        NSWorkspace.shared.open(url)
    }

    private func modelRow(
        _ modelId: String,
        subtitle: String,
        isLoaded: Bool,
        isDownloaded: Bool,
        @ViewBuilder actions: () -> some View
    ) -> some View {
        HStack(spacing: 12) {
            Circle()
                .fill(isLoaded ? NovaTheme.Colors.statusOK : Color.clear)
                .frame(width: 8, height: 8)
                .overlay(Circle().stroke(Color.secondary.opacity(0.3), lineWidth: 1))

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Text(modelId).font(.system(size: 13, weight: .medium)).lineLimit(1)
                        .foregroundColor(NovaTheme.Colors.accent)
                        .help(l10n.tr("models.clickDetails"))
                        .onTapGesture { fetchModelCard(repoId: modelId) }
                    CopyIDButton(id: modelId)
                }
                if !subtitle.isEmpty {
                    Text(subtitle).font(.caption2).foregroundColor(.secondary)
                }
                LocalModelStatusBadges(
                    modelId: modelId,
                    isVerified: LocalModelListFilter.isVerifiedWeight(id: modelId, catalog: catalogSnapshot),
                    isDownloaded: isDownloaded,
                    isLoaded: isLoaded
                )
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 8)
            actions()
        }
        .id(modelId)
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

struct ModelTypeFilterBar: View {
    @Binding var filter: ModelsPageView.ModelTypeFilter

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(ModelsPageView.ModelTypeFilter.allCases, id: \.self) { item in
                    Button {
                        filter = item
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: item.icon)
                                .font(.system(size: 9))
                            Text(item.label)
                                .font(.system(size: 11, weight: filter == item ? .semibold : .regular))
                        }
                        .foregroundColor(filter == item ? .white : NovaTheme.Colors.accent)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(filter == item ? NovaTheme.Colors.accent : NovaTheme.Colors.accent.opacity(0.15))
                        .overlay(Capsule().stroke(NovaTheme.Colors.accent.opacity(filter == item ? 0 : 0.3), lineWidth: 0.5))
                        .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 8)
        }
    }
}

struct HFSearchResult {
    let id: String
    let tags: [String]
    /// True when the repo is in the NovaMLX verified catalog (tagged by the
    /// admin search route in allow-unlisted mode).
    var verified: Bool? = nil
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
