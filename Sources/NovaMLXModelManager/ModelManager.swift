import Foundation
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils
import Logging
import Hub

public enum DownloadState: String, Codable, Sendable {
    case notDownloaded
    case downloading
    case downloaded
    case failed
}

public struct ModelRecord: Codable, Sendable {
    public let id: String
    public let family: ModelFamily
    public let modelType: ModelType
    public let source: ModelSource
    public let localURL: URL
    public let remoteURL: String
    public let sizeBytes: UInt64
    public let downloadedAt: Date?
    public let version: String

    public init(
        id: String, family: ModelFamily, modelType: ModelType, source: ModelSource,
        localURL: URL, remoteURL: String, sizeBytes: UInt64 = 0,
        downloadedAt: Date? = nil, version: String = "1.0"
    ) {
        self.id = id; self.family = family; self.modelType = modelType
        self.source = source; self.localURL = localURL; self.remoteURL = remoteURL
        self.sizeBytes = sizeBytes; self.downloadedAt = downloadedAt; self.version = version
    }
}

public enum ModelSource: String, Codable, Sendable {
    case huggingFace = "huggingface"
    case local
    case custom
}

public struct DownloadProgress: Sendable {
    public let bytesDownloaded: UInt64
    public let totalBytes: UInt64
    public let bytesPerSecond: Double
    public let fractionCompleted: Double
    public init(bytesDownloaded: UInt64, totalBytes: UInt64, bytesPerSecond: Double = 0) {
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
        self.bytesPerSecond = bytesPerSecond
        self.fractionCompleted = totalBytes > 0 ? Double(bytesDownloaded) / Double(totalBytes) : 0
    }
}

public struct DownloadStatus: Codable, Sendable {
    public let modelId: String
    public let state: DownloadState
    public let progress: Double
    public let error: String?
    public let currentFile: String?
    public let filesCompleted: Int
    public let filesTotal: Int

    public init(modelId: String, state: DownloadState, progress: Double = 0,
                error: String? = nil, currentFile: String? = nil,
                filesCompleted: Int = 0, filesTotal: Int = 0) {
        self.modelId = modelId
        self.state = state
        self.progress = progress
        self.error = error
        self.currentFile = currentFile
        self.filesCompleted = filesCompleted
        self.filesTotal = filesTotal
    }
}

public final class ModelManager: @unchecked Sendable {
    public private(set) var modelsDirectory: URL
    /// Additional roots scanned for load (external disks), in priority order.
    public private(set) var extraModelDirectories: [URL]
    public let catalogStore: ModelCatalogStore
    private var registryFile: URL
    private var hubApi: HubApi
    private var _registry: [String: ModelRecord]
    private var _activeDownloads: Set<String>
    private var _downloadStates: [String: DownloadStatus]
    private var downloadedCache: [ModelRecord]?
    private var diskUsageCache: UInt64?
    private var catalogCacheKey: String?
    private var catalogCache: [CatalogEntry] = []
    private let lock = NovaMLXLock()

    public init(
        modelsDirectory: URL,
        extraModelDirectories: [URL] = [],
        catalogStore: ModelCatalogStore = ModelCatalogStore()
    ) {
        self.modelsDirectory = modelsDirectory
        self.extraModelDirectories = extraModelDirectories
        self.catalogStore = catalogStore
        self.registryFile = modelsDirectory.appendingPathComponent("registry.json")
        let hubDownloadBase = modelsDirectory.appendingPathComponent("hub")
        self.hubApi = HubApi(downloadBase: hubDownloadBase, cache: nil)
        self._registry = [:]
        self._activeDownloads = []
        self._downloadStates = [:]
        self.downloadedCache = nil
        self.diskUsageCache = nil
        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: hubDownloadBase, withIntermediateDirectories: true)
        loadRegistry()
    }

    /// Hot-apply a new ordered storage list (from `NovaMLXPaths.setModelsDirs`
    /// or the models-dirs admin API) without restarting the process.
    /// Existing registry records keep their absolute `localURL`s — models on
    /// disks that left the list stay loadable until the next discovery sync.
    public func updateDirectories(primary: URL, extras: [URL]) {
        lock.withLock {
            modelsDirectory = primary
            extraModelDirectories = extras.filter {
                $0.standardizedFileURL != primary.standardizedFileURL
            }
            registryFile = primary.appendingPathComponent("registry.json")
            hubApi = HubApi(
                downloadBase: primary.appendingPathComponent("hub"), cache: nil)
        }
        try? FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        invalidateListCaches()
        NovaMLXLog.info(
            "[ModelManager] Storage dirs updated: primary=\(primary.path) extras=\(extras.map(\.path))")
    }

    /// Download root for a new registration: strict priority order across the
    /// configured roots. Falls back to the primary when every root fails the
    /// space/writability preflight — the actual download re-checks and
    /// surfaces the error to the user.
    private func rootForNewModel(sizeBytes: UInt64) -> URL {
        let roots = [modelsDirectory] + extraModelDirectories
        if let chosen = try? NovaMLXPaths.downloadRoot(
            estimatedBytes: sizeBytes, roots: roots) {
            return chosen
        }
        return modelsDirectory
    }

    /// All on-disk locations for `modelId`, in storage-priority order:
    /// the registry's `localURL` plus every other configured root that holds
    /// a copy. Load paths try these in order so a corrupt copy on a
    /// higher-priority disk falls through to the next one.
    public func candidateLoadURLs(for modelId: String) -> [URL] {
        let record = lock.withLock { _registry[modelId] }
        var candidates = NovaMLXPaths.candidateDirectories(
            forModelId: modelId,
            roots: [modelsDirectory] + extraModelDirectories
        )
        if let record {
            let key = record.localURL.standardizedFileURL.path
            if !candidates.contains(where: { $0.standardizedFileURL.path == key }) {
                candidates.insert(record.localURL, at: 0)
            }
        }
        return candidates
    }

    /// Models that are downloaded and ready to use (complete weight files on disk).
    public func availableModels() -> [ModelRecord] {
        downloadedModels()
    }

    /// All records in the registry, including incomplete/failed downloads.
    /// Use this only for admin/management purposes.
    public func allRegisteredModels() -> [ModelRecord] {
        lock.withLock { Array(_registry.values) }
    }

    public func downloadedModels() -> [ModelRecord] {
        let cached = lock.withLock { downloadedCache }
        if let cached { return cached }
        let fresh = computeDownloadedModels()
        lock.withLock { downloadedCache = fresh }
        return fresh
    }

    /// Records whose files live on a removable volume that is currently
    /// unmounted. Hidden from `downloadedModels()`, but surfaced by the UI
    /// on demand so the user can re-download them to another disk.
    public func unmountedModels() -> [ModelRecord] {
        lock.withLock { Array(_registry.values) }.filter { record in
            guard record.downloadedAt != nil else { return false }
            return NovaMLXPaths.triggersRemovableVolumeTCC(record.localURL)
                && !NovaMLXPaths.isVolumeMounted(record.localURL)
        }
    }

    /// Reachability and directory size, off the caller. Views read the cache.
    public func refreshModelCaches() async {
        let records = lock.withLock { Array(_registry.values) }
        let dirs = lock.withLock { [modelsDirectory] + extraModelDirectories }
        let (fresh, size) = await Task.detached(priority: .utility) {
            let fresh = Self.reachableDownloaded(records)
            let size = dirs.reduce(UInt64(0)) { acc, dir in
                guard NovaMLXPaths.isVolumeMounted(dir) else { return acc }
                return acc + FileManager.default.directorySize(at: dir)
            }
            return (fresh, size)
        }.value
        lock.withLock {
            downloadedCache = fresh
            diskUsageCache = size
        }
    }

    private func computeDownloadedModels() -> [ModelRecord] {
        let records = lock.withLock { Array(_registry.values) }
        return Self.reachableDownloaded(records)
    }

    private static func reachableDownloaded(_ records: [ModelRecord]) -> [ModelRecord] {
        records.filter { record in
            guard record.downloadedAt != nil else { return false }
            // Removable-volume paths are never probed directly (TCC), but an
            // unmounted disk must not list as available — check the mount
            // table instead, which is probe-free.
            if NovaMLXPaths.triggersRemovableVolumeTCC(record.localURL) {
                return NovaMLXPaths.isVolumeMounted(record.localURL)
            }
            return (try? record.localURL.checkResourceIsReachable()) == true
        }
    }

    private func invalidateListCaches() {
        lock.withLock {
            downloadedCache = nil
            diskUsageCache = nil
        }
    }

    public func isDownloaded(_ modelId: String) -> Bool {
        lock.withLock {
            guard let record = _registry[modelId], record.downloadedAt != nil else { return false }
            if NovaMLXPaths.triggersRemovableVolumeTCC(record.localURL) {
                return NovaMLXPaths.isVolumeMounted(record.localURL)
            }
            return (try? record.localURL.checkResourceIsReachable()) == true
        }
    }

    public func getRecord(_ modelId: String) -> ModelRecord? {
        lock.withLock { _registry[modelId] }
    }

    public func getDownloadStatus(_ modelId: String) -> DownloadStatus? {
        lock.withLock { _downloadStates[modelId] }
    }

    @discardableResult
    public func register(
        id: String, family: ModelFamily = .other, modelType: ModelType = .llm,
        source: ModelSource = .huggingFace, remoteURL: String,
        sizeBytes: UInt64 = 0, version: String = "1.0"
    ) -> ModelRecord {
        let destRoot = rootForNewModel(sizeBytes: sizeBytes)
        let localURL = destRoot.appendingPathComponent(id, isDirectory: true)
        let record = ModelRecord(
            id: id, family: family, modelType: modelType, source: source,
            localURL: localURL, remoteURL: remoteURL, sizeBytes: sizeBytes, version: version
        )
        lock.withLock { _registry[id] = record }
        invalidateListCaches()
        saveRegistry()
        NovaMLXLog.info("[ModelManager] Registered model: \(id)")
        return record
    }

    public func unregister(_ modelId: String) {
        _ = lock.withLock { _registry.removeValue(forKey: modelId) }
        invalidateListCaches()
        saveRegistry()
    }

    public func startDownload(_ modelId: String, preferredRoot: URL? = nil) throws {
        let record: ModelRecord? = lock.withLock {
            guard let r = _registry[modelId] else { return nil }
            guard _activeDownloads.insert(modelId).inserted else { return nil }
            return r
        }

        guard let record = record else {
            let state = lock.withLock { _downloadStates[modelId] }
            if state?.state == .downloading {
                throw NovaMLXError.apiError("Download already in progress for \(modelId)")
            }
            throw NovaMLXError.modelNotFound(modelId)
        }

        // Preflight: pick the highest-priority root that is mounted, writable,
        // and has room. Throws ModelStorageError.insufficientSpace when none do.
        // A `preferredRoot` (user-chosen disk for re-downloads) goes through
        // the same validation, just as the only candidate.
        let roots = preferredRoot.map { [$0] }
            ?? lock.withLock { [modelsDirectory] + extraModelDirectories }
        let chosenRoot = try NovaMLXPaths.downloadRoot(
            estimatedBytes: record.sizeBytes, roots: roots)
        let primary = lock.withLock { modelsDirectory }
        let downloadHubApi: HubApi
        if chosenRoot.standardizedFileURL == primary.standardizedFileURL {
            downloadHubApi = lock.withLock { hubApi }
        } else {
            downloadHubApi = HubApi(
                downloadBase: chosenRoot.appendingPathComponent("hub"), cache: nil)
        }

        lock.withLock {
            _downloadStates[modelId] = DownloadStatus(modelId: modelId, state: .downloading)
        }

        Task { [downloadHubApi] in
            NovaMLXLog.info("[ModelManager] Download task started for \(modelId)")
            do {
                let repo = Hub.Repo(id: modelId)
                let destDir = downloadHubApi.localRepoLocation(repo)
                try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

                NovaMLXLog.info("[ModelManager] Fetching file list for \(modelId)...")
                let allFilenames = try await downloadHubApi.getFilenames(from: repo)
                NovaMLXLog.info("[ModelManager] Found \(allFilenames.count) files in \(modelId)")

                let totalFiles = allFilenames.count
                var remaining: [Aria2DownloadFile] = []
                var alreadyDone = 0
                for filename in allFilenames {
                    let fileDest = destDir.appendingPathComponent(filename)
                    if FileManager.default.fileExists(atPath: fileDest.path) {
                        alreadyDone += 1
                        continue
                    }
                    remaining.append(Aria2DownloadFile(
                        url: await hubFileURL(repo: repo, filename: filename),
                        relativePath: filename
                    ))
                }

                if !remaining.isEmpty {
                    let skipped = alreadyDone
                    try await Aria2Downloader.download(
                        files: remaining,
                        destination: destDir
                    ) { snapshot in
                        let done = skipped + snapshot.files.filter(\.isComplete).count
                        let progress: Double
                        if snapshot.totalBytes > 0 {
                            progress = Double(snapshot.downloadedBytes) / Double(snapshot.totalBytes)
                        } else {
                            progress = Double(done) / Double(max(totalFiles, 1))
                        }
                        let current = snapshot.files.first(where: { !$0.isComplete && $0.downloadedBytes > 0 })?.relativePath
                        self.lock.withLock {
                            self._downloadStates[modelId] = DownloadStatus(
                                modelId: modelId, state: .downloading,
                                progress: progress,
                                currentFile: current,
                                filesCompleted: done, filesTotal: totalFiles
                            )
                        }
                    }
                }

                let updatedRecord = ModelRecord(
                    id: record.id, family: record.family, modelType: record.modelType,
                    source: record.source, localURL: destDir, remoteURL: record.remoteURL,
                    sizeBytes: record.sizeBytes, downloadedAt: Date(), version: record.version
                )
                lock.withLock {
                    _registry[modelId] = updatedRecord
                    _downloadStates[modelId] = DownloadStatus(
                        modelId: modelId, state: .downloaded, progress: 1.0,
                        filesCompleted: totalFiles, filesTotal: totalFiles
                    )
                    _activeDownloads.remove(modelId)
                }
                invalidateListCaches()
                saveRegistry()
                NovaMLXLog.info("[ModelManager] Download complete: \(modelId)")
            } catch {
                NovaMLXLog.error("Download failed for \(modelId): \(error)")
                lock.withLock {
                    _downloadStates[modelId] = DownloadStatus(
                        modelId: modelId, state: .failed, error: error.localizedDescription
                    )
                    _activeDownloads.remove(modelId)
                }
            }
        }
    }

    public func deleteModel(_ modelId: String) throws {
        let record = lock.withLock { _registry[modelId] }
        guard let record = record else { throw NovaMLXError.modelNotFound(modelId) }

        if FileManager.default.fileExists(atPath: record.localURL.path) {
            try FileManager.default.removeItem(at: record.localURL)
        }

        let updated = ModelRecord(
            id: record.id, family: record.family, modelType: record.modelType,
            source: record.source, localURL: record.localURL, remoteURL: record.remoteURL,
            sizeBytes: record.sizeBytes, downloadedAt: nil, version: record.version
        )
        lock.withLock {
            _registry[modelId] = updated
            _downloadStates.removeValue(forKey: modelId)
        }
        invalidateListCaches()
        saveRegistry()
    }

    public func totalDiskUsage() -> UInt64 {
        if let cached = lock.withLock({ diskUsageCache }) { return cached }
        let dirs = lock.withLock { [modelsDirectory] + extraModelDirectories }
        let size = dirs.reduce(UInt64(0)) { acc, dir in
            guard NovaMLXPaths.isVolumeMounted(dir) else { return acc }
            return acc + FileManager.default.directorySize(at: dir)
        }
        lock.withLock { diskUsageCache = size }
        return size
    }

    // MARK: - Model Catalog

    public func fetchCatalog() async {
        await catalogStore.refresh()
    }

    public func catalogModels(forCategory category: ModelType?) -> [CatalogEntry] {
        let count = catalogStore.models.count
        let key = "\(category?.rawValue ?? "*")#\(count)"
        let hit = lock.withLock { () -> [CatalogEntry]? in
            guard catalogCacheKey == key else { return nil }
            return catalogCache
        }
        if let hit { return hit }
        let result = ModelCatalogPolicy.search(catalogStore.models, query: "", category: category)
        lock.withLock {
            catalogCacheKey = key
            catalogCache = result
        }
        return result
    }

    /// Registers every catalog model into the local registry (idempotent — skips ids
    /// already present). No-op until `fetchCatalog()` has populated the store.
    public func registerPopularModels() {
        for model in catalogStore.models {
            if ModelCatalogPolicy.isIdPattern(model.id) { continue }
            if model.status == .unsupported || ImageModelSupport.isUnsupported(model.id) { continue }
            if lock.withLock({ _registry[model.id] }) == nil {
                register(
                    id: model.id,
                    family: model.family,
                    modelType: model.category,
                    remoteURL: model.url,
                    sizeBytes: model.sizeBytes ?? 0
                )
            }
        }
    }

    // MARK: - Vault Moves

    /// A model directory found under one storage vault (root), for the
    /// two-pane move UI. `modelId` is the repo id relative to the vault
    /// (e.g. "mlx-community/Qwen3-0.6B-4bit").
    public struct VaultModelEntry: Identifiable, Sendable {
        public let modelId: String
        public let path: URL
        public let sizeBytes: UInt64
        public let isComplete: Bool
        public var id: String { path.path }
    }

    public enum ModelMoveError: LocalizedError {
        case sourceMissing(String)
        case destinationExists(String)
        case vaultUnavailable(String)
        case insufficientSpace(needed: UInt64, free: UInt64)

        public var errorDescription: String? {
            switch self {
            case .sourceMissing(let p): return "source not found: \(p)"
            case .destinationExists(let p): return "already exists at destination: \(p)"
            case .vaultUnavailable(let p): return "destination vault not mounted or not writable: \(p)"
            case .insufficientSpace(let needed, let free):
                return "insufficient space (need \(ByteCountFormatter.string(fromByteCount: Int64(needed), countStyle: .file)), have \(ByteCountFormatter.string(fromByteCount: Int64(free), countStyle: .file)))"
            }
        }
    }

    /// Models stored under one vault, scanning both layouts
    /// (`<root>/<id>` and `<root>/hub/models/<id>`). Sorted by id.
    public func modelsInVault(_ vault: URL) -> [VaultModelEntry] {
        let discovery = ModelDiscovery()
        var found: [VaultModelEntry] = []
        for base in [vault, vault.appendingPathComponent("hub/models", isDirectory: true)] {
            guard base.directoryExists else { continue }
            found.append(contentsOf: discovery.discover(in: base).map {
                VaultModelEntry(
                    modelId: $0.modelId, path: $0.modelPath,
                    sizeBytes: $0.estimatedSizeBytes, isComplete: $0.isComplete)
            })
        }
        // The same id under both layouts of one vault keeps the first hit.
        var seen = Set<String>()
        return found
            .filter { seen.insert($0.modelId).inserted }
            .sorted { $0.modelId < $1.modelId }
    }

    /// Move a model directory between vaults, preserving its layout
    /// (direct vs `hub/models/`). Cross-volume moves copy-then-delete via
    /// `FileManager.moveItem`. The registry record (when present) is
    /// repointed at the new location. Returns the new directory URL.
    @discardableResult
    public func moveModel(_ modelId: String, from sourceVault: URL, to destVault: URL) throws -> URL {
        guard let source = Self.modelDirectory(modelId, in: sourceVault) else {
            throw ModelMoveError.sourceMissing("\(sourceVault.path)/\(modelId)")
        }
        guard NovaMLXPaths.isVolumeMounted(destVault) else {
            throw ModelMoveError.vaultUnavailable(destVault.path)
        }
        // The vault dir itself may not exist yet (freshly added in Settings);
        // create it after the mounted check so we never materialize a bogus
        // mountpoint for an unmounted volume.
        if !destVault.directoryExists {
            try? FileManager.default.createDirectory(at: destVault, withIntermediateDirectories: true)
        }
        guard FileManager.default.isWritableFile(atPath: destVault.path) else {
            throw ModelMoveError.vaultUnavailable(destVault.path)
        }

        // Same vault or aliased roots: nothing to do.
        let sourceKey = sourceVault.standardizedFileURL.path
        let destKey = destVault.standardizedFileURL.path
        guard sourceKey != destKey else { return source }

        let inHubLayout = source.path.contains("/hub/models/")
        let relative = inHubLayout ? "hub/models/\(modelId)" : modelId
        let destination = destVault.appendingPathComponent(relative, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ModelMoveError.destinationExists(destination.path)
        }

        let needed = FileManager.default.directorySize(at: source)
        let free = (try? FileManager.default.attributesOfFileSystem(
            forPath: destVault.path)[.systemFreeSize] as? Int64) ?? 0
        guard UInt64(max(free, 0)) > needed else {
            throw ModelMoveError.insufficientSpace(needed: needed, free: UInt64(max(free, 0)))
        }

        // Create the parent chain (org dirs may not exist on the target yet).
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: source, to: destination)

        // Repoint the registry record; the moved copy is the live one.
        let record = lock.withLock { _registry[modelId] }
        if let record = record {
            lock.withLock {
                _registry[modelId] = ModelRecord(
                    id: record.id, family: record.family, modelType: record.modelType,
                    source: record.source, localURL: destination, remoteURL: record.remoteURL,
                    sizeBytes: max(record.sizeBytes, needed), downloadedAt: record.downloadedAt,
                    version: record.version)
            }
            saveRegistry()
        }
        invalidateListCaches()
        NovaMLXLog.info("[ModelManager] Moved \(modelId): \(source.path) → \(destination.path)")
        return destination
    }

    /// The on-disk directory for `modelId` under `vault`, either layout.
    private static func modelDirectory(_ modelId: String, in vault: URL) -> URL? {
        for candidate in [
            vault.appendingPathComponent(modelId, isDirectory: true),
            vault.appendingPathComponent("hub/models/\(modelId)", isDirectory: true),
        ] where candidate.directoryExists {
            return candidate
        }
        return nil
    }

    /// Delete the model directory at `path` (a vault move pane selection).
    /// When it is the registry's copy, the record is marked not-downloaded
    /// (like `deleteModel(_:)`); a copy remaining on another vault is
    /// re-registered by the next discovery pass.
    public func deleteModel(at path: URL) throws {
        guard path.directoryExists else {
            throw ModelMoveError.sourceMissing(path.path)
        }
        // Resolve both spellings BEFORE the delete — resolvingSymlinksInPath
        // degrades to the raw string once the target no longer exists, and
        // discovery stores /private/var spellings while callers pass /var.
        let key = path.resolvingSymlinksInPath().path
        let record = lock.withLock {
            _registry.values.first {
                $0.localURL.resolvingSymlinksInPath().path == key
            }
        }
        try FileManager.default.removeItem(at: path)
        if let record = record {
            lock.withLock {
                _registry[record.id] = ModelRecord(
                    id: record.id, family: record.family, modelType: record.modelType,
                    source: record.source, localURL: record.localURL, remoteURL: record.remoteURL,
                    sizeBytes: record.sizeBytes, downloadedAt: nil, version: record.version)
                _downloadStates.removeValue(forKey: record.id)
            }
            saveRegistry()
        }
        invalidateListCaches()
        NovaMLXLog.info("[ModelManager] Deleted model files at \(path.path)")
    }

    @discardableResult
    public func discoverModels() -> [DiscoveredModel] {
        let discovery = ModelDiscovery()
        var scanDirs: [URL] = [modelsDirectory]
        for extra in extraModelDirectories {
            if extra.standardizedFileURL != modelsDirectory.standardizedFileURL {
                scanDirs.append(extra)
            }
        }
        scanDirs.append(modelsDirectory.appendingPathComponent("hub/models", isDirectory: true))
        for extra in extraModelDirectories {
            scanDirs.append(extra.appendingPathComponent("hub/models", isDirectory: true))
        }
        var allDiscovered: [DiscoveredModel] = []

        #if DEBUG
        NovaMLXLog.info("[Discovery] Scanning dirs: \(scanDirs.map(\.path))")
        // List immediate subdirectories in modelsDirectory
        if let contents = try? FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil) {
            NovaMLXLog.info("[Discovery] modelsDirectory contents: \(contents.map(\.lastPathComponent))")
        }
        #endif

        for dir in scanDirs {
            if NovaMLXPaths.triggersRemovableVolumeTCC(dir),
               !NovaMLXPaths.isVolumeMounted(dir) {
                // Unmounted external disk — skip without probing (no TCC sheet).
                NovaMLXLog.info("[Discovery] Skipping unmounted volume: \(dir.path)")
                continue
            }
            guard dir.directoryExists else {
                #if DEBUG
                NovaMLXLog.info("[Discovery] Skipping non-existent dir: \(dir.path)")
                #endif
                continue
            }
            let found = discovery.discover(in: dir)
            allDiscovered.append(contentsOf: found)
        }

        // Same model id found under several roots: keep the copy on the
        // highest-priority root (roots were scanned in priority order).
        var seenIds = Set<String>()
        var prioritized: [DiscoveredModel] = []
        for model in allDiscovered where seenIds.insert(model.modelId).inserted {
            prioritized.append(model)
        }

        var registered = 0
        var updated = 0

        for model in prioritized {
            let existing = lock.withLock { _registry[model.modelId] }
            if existing == nil {
                let remoteURL = "https://huggingface.co/\(model.modelId)"
                register(
                    id: model.modelId,
                    family: model.family,
                    modelType: model.modelType,
                    source: .huggingFace,
                    remoteURL: remoteURL,
                    sizeBytes: model.estimatedSizeBytes
                )
                // Only mark as downloaded if model is complete (has valid weight files)
                let updatedRecord = ModelRecord(
                    id: model.modelId,
                    family: model.family,
                    modelType: model.modelType,
                    source: .huggingFace,
                    localURL: model.modelPath,
                    remoteURL: remoteURL,
                    sizeBytes: model.estimatedSizeBytes,
                    downloadedAt: model.isComplete ? Date() : nil,
                    version: "1.0"
                )
                lock.withLock { _registry[model.modelId] = updatedRecord }
                saveRegistry()
                registered += 1
            } else {
                // Existing record — sync localURL with actual disk path, update download status
                let currentRecord = existing!
                let needsURLUpdate = currentRecord.localURL != model.modelPath
                let needsDownloadUpdate = currentRecord.downloadedAt == nil && model.isComplete
                let needsCompletenessUpdate = currentRecord.downloadedAt != nil && !model.isComplete
                let needsModelTypeUpdate = currentRecord.modelType != model.modelType
                let needsFamilyUpdate = currentRecord.family != model.family

                if needsURLUpdate || needsDownloadUpdate || needsCompletenessUpdate || needsModelTypeUpdate || needsFamilyUpdate {
                    let updatedRecord = ModelRecord(
                        id: currentRecord.id,
                        family: model.family,
                        modelType: model.modelType,
                        source: currentRecord.source,
                        localURL: model.modelPath,
                        remoteURL: currentRecord.remoteURL,
                        sizeBytes: model.estimatedSizeBytes,
                        downloadedAt: model.isComplete
                            ? (currentRecord.downloadedAt ?? Date())
                            : nil,
                        version: currentRecord.version
                    )
                    lock.withLock { _registry[model.modelId] = updatedRecord }
                    saveRegistry()
                    updated += 1
                }
            }
        }

        if registered > 0 || updated > 0 { invalidateListCaches() }
        NovaMLXLog.info("[ModelManager] Model discovery: \(prioritized.count) found (\(allDiscovered.count) raw), \(registered) registered, \(updated) updated")
        return prioritized
    }

    /// Clean up empty directories left by failed/interrupted downloads,
    /// across every configured storage root.
    public func cleanupEmptyDirectories() {
        let roots = lock.withLock { [modelsDirectory] + extraModelDirectories }
        for root in roots {
            guard NovaMLXPaths.isVolumeMounted(root) else { continue }
            cleanupEmptyDirectories(under: root)
        }
    }

    private func cleanupEmptyDirectories(under modelsDirectory: URL) {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: modelsDirectory, includingPropertiesForKeys: nil
        ) else { return }

        for dir in contents {
            guard dir.directoryExists else { continue }
            let name = dir.lastPathComponent
            guard name != "hub" && !name.hasPrefix(".") && name != "registry.json" else { continue }

            if isEffectivelyEmpty(dir) {
                try? fm.removeItem(at: dir)
                NovaMLXLog.info("[ModelManager] Cleaned up empty directory: \(name)")
            }
        }
    }

    private func isEffectivelyEmpty(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil
        ) else { return true }
        if contents.isEmpty { return true }
        for item in contents {
            if item.directoryExists, isEffectivelyEmpty(item) { continue }
            return false
        }
        return true
    }

    private func hubFileURL(repo: Hub.Repo, filename: String) async -> URL {
        // Respect the configured HF endpoint (e.g. https://hf-mirror.com) —
        // hardcoded huggingface.co is unreachable from some networks.
        let base = (await NovaMLXConfiguration.shared.huggingfaceEndpoint)
            ?? "https://huggingface.co"
        var url = URL(string: base) ?? URL(string: "https://huggingface.co")!
        if repo.type != .models {
            url = url.appendingPathComponent(repo.type.rawValue)
        }
        url = url.appendingPathComponent(repo.id)
        url = url.appendingPathComponent("resolve")
        url = url.appendingPathComponent("main")
        url = url.appendingPathComponent(filename)
        return url
    }

    private func loadRegistry() {
        // SQLite is the source of truth (Phase D4). The legacy JSON file at
        // <modelsDir>/registry.json is imported once below on first run; after
        // that, the file is renamed to .migrated and we never read it again.
        importLegacyRegistryJSONIfNeeded()
        if let stored = try? NovaDB.shared.modelRegistryStore.listAsRegistry() {
            lock.withLock { _registry = stored }
        }
    }

    private func saveRegistry() {
        let snapshot = lock.withLock { _registry }
        let dir = lock.withLock { modelsDirectory }
        for (_, record) in snapshot {
            try? NovaDB.shared.modelRegistryStore.upsertRecord(record, modelsDirectory: dir)
        }
    }

    /// One-shot import of `<modelsDir>/registry.json` into the SQLite store.
    /// Idempotent: if the store already has rows, the file is left alone;
    /// otherwise we parse, upsert, and rename to .migrated.
    private func importLegacyRegistryJSONIfNeeded() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: registryFile.path) else { return }

        // Skip if store already populated — SQLite is source of truth.
        if let existing = try? NovaDB.shared.modelRegistryStore.list(), !existing.isEmpty {
            return
        }

        guard let data = try? Data(contentsOf: registryFile),
              let decoded = try? JSONDecoder().decode([String: ModelRecord].self, from: data) else {
            NovaMLXLog.warning("[ModelManager] Failed to parse legacy registry.json; leaving file in place")
            return
        }

        // Local URLs in the legacy file may point at an old directory; rewrite
        // them to the current modelsDirectory before persisting so we don't
        // carry stale paths into SQLite.
        for (id, var record) in decoded {
            if !record.localURL.path.hasPrefix(modelsDirectory.path) {
                record = ModelRecord(
                    id: record.id, family: record.family, modelType: record.modelType,
                    source: record.source,
                    localURL: modelsDirectory.appendingPathComponent(id.sanitized, isDirectory: true),
                    remoteURL: record.remoteURL, sizeBytes: record.sizeBytes,
                    downloadedAt: record.downloadedAt, version: record.version
                )
            }
            try? NovaDB.shared.modelRegistryStore.upsertRecord(record, modelsDirectory: modelsDirectory)
        }
        NovaMLXLog.info("[ModelManager] Imported \(decoded.count) entries from legacy registry.json")

        let migrated = registryFile.appendingPathExtension("migrated")
        if fm.fileExists(atPath: migrated.path) {
            try? fm.removeItem(at: registryFile)
        } else {
            try? fm.moveItem(at: registryFile, to: migrated)
        }
    }
}
