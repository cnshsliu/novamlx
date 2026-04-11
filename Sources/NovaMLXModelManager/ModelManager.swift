import Foundation
import NovaMLXCore
import NovaMLXUtils
import Logging
import Hub

private final class AtomicInt: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Int
    init(_ value: Int = 0) { self._value = value }
    func withLock<T>(_ body: (inout Int) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(&_value)
    }
}

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
    private let modelsDirectory: URL
    private let registryFile: URL
    private let hubApi: HubApi
    private var _registry: [String: ModelRecord]
    private var _activeDownloads: Set<String>
    private var _downloadStates: [String: DownloadStatus]
    private let lock = NovaMLXLock()

    public init(modelsDirectory: URL) {
        self.modelsDirectory = modelsDirectory
        self.registryFile = modelsDirectory.appendingPathComponent("registry.json")
        let hubDownloadBase = modelsDirectory.appendingPathComponent("hub")
        self.hubApi = HubApi(downloadBase: hubDownloadBase, cache: nil)
        self._registry = [:]
        self._activeDownloads = []
        self._downloadStates = [:]
        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: hubDownloadBase, withIntermediateDirectories: true)
        loadRegistry()
    }

    public func availableModels() -> [ModelRecord] {
        lock.withLock { Array(_registry.values) }
    }

    public func downloadedModels() -> [ModelRecord] {
        lock.withLock {
            _registry.values.filter {
                $0.downloadedAt != nil && FileManager.default.fileExists(atPath: $0.localURL.path)
            }
        }
    }

    public func isDownloaded(_ modelId: String) -> Bool {
        lock.withLock {
            guard let record = _registry[modelId] else { return false }
            return record.downloadedAt != nil && FileManager.default.fileExists(atPath: record.localURL.path)
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
        let localURL = modelsDirectory.appendingPathComponent(id.sanitized, isDirectory: true)
        let record = ModelRecord(
            id: id, family: family, modelType: modelType, source: source,
            localURL: localURL, remoteURL: remoteURL, sizeBytes: sizeBytes, version: version
        )
        lock.withLock { _registry[id] = record }
        saveRegistry()
        NovaMLXLog.info("Registered model: \(id)")
        return record
    }

    public func unregister(_ modelId: String) {
        _ = lock.withLock { _registry.removeValue(forKey: modelId) }
        saveRegistry()
    }

    public func startDownload(_ modelId: String) throws {
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

        lock.withLock {
            _downloadStates[modelId] = DownloadStatus(modelId: modelId, state: .downloading)
        }

        Task { [hubApi] in
            NovaMLXLog.info("Download task started for \(modelId)")
            do {
                let repo = Hub.Repo(id: modelId)
                let destDir = hubApi.localRepoLocation(repo)
                try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

                NovaMLXLog.info("Fetching file list for \(modelId)...")
                let allFilenames = try await hubApi.getFilenames(from: repo)
                NovaMLXLog.info("Found \(allFilenames.count) files in \(modelId)")

                let totalFiles = allFilenames.count
                let completedCount = AtomicInt(0)

                for filename in allFilenames {
                    let fileDest = destDir.appendingPathComponent(filename)
                    let fileDir = fileDest.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: fileDir, withIntermediateDirectories: true)

                    if FileManager.default.fileExists(atPath: fileDest.path) {
                        NovaMLXLog.info("Skip (exists): \(filename)")
                        completedCount.withLock { $0 += 1 }
                        continue
                    }

                    let sourceURL = hubFileURL(repo: repo, filename: filename)
                    NovaMLXLog.info("Downloading \(filename)...")

                    let cc = completedCount.withLock { $0 }
                    self.lock.withLock {
                        self._downloadStates[modelId] = DownloadStatus(
                            modelId: modelId, state: .downloading,
                            progress: Double(cc) / Double(totalFiles),
                            currentFile: filename,
                            filesCompleted: cc, filesTotal: totalFiles
                        )
                    }

                    try await downloadFile(url: sourceURL, to: fileDest) { bytesWritten, totalBytes in
                        let fileProgress = totalBytes > 0 ? Double(bytesWritten) / Double(totalBytes) : 0
                        let cc2 = completedCount.withLock { $0 }
                        let overallProgress = (Double(cc2) + fileProgress) / Double(totalFiles)
                        self.lock.withLock {
                            self._downloadStates[modelId] = DownloadStatus(
                                modelId: modelId, state: .downloading,
                                progress: overallProgress,
                                currentFile: filename,
                                filesCompleted: cc2, filesTotal: totalFiles
                            )
                        }
                    }

                    let fileSize = fileSizeString(at: fileDest)
                    NovaMLXLog.info("Done: \(filename) (\(fileSize))")
                    completedCount.withLock { $0 += 1 }
                }

                _ = completedCount.withLock { $0 }

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
                saveRegistry()
                NovaMLXLog.info("Download complete: \(modelId)")
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
        saveRegistry()
    }

    public func totalDiskUsage() -> UInt64 {
        FileManager.default.directorySize(at: modelsDirectory)
    }

    public func registerPopularModels() {
        let models: [(String, ModelFamily, ModelType, String, UInt64)] = [
            ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", .llama, .llm, "https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", 4_800_000_000),
            ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", .mistral, .llm, "https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit", 4_100_000_000),
            ("mlx-community/Phi-3.5-mini-instruct-4bit", .phi, .llm, "https://huggingface.co/mlx-community/Phi-3.5-mini-instruct-4bit", 2_200_000_000),
            ("mlx-community/Qwen2.5-7B-Instruct-4bit", .qwen, .llm, "https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit", 4_400_000_000),
            ("mlx-community/gemma-2-9b-it-4bit", .gemma, .llm, "https://huggingface.co/mlx-community/gemma-2-9b-it-4bit", 5_200_000_000),
        ]
        for (id, family, type, url, size) in models {
            if lock.withLock({ _registry[id] }) == nil {
                register(id: id, family: family, modelType: type, remoteURL: url, sizeBytes: size)
            }
        }
    }

    @discardableResult
    public func discoverModels() -> [DiscoveredModel] {
        let discovery = ModelDiscovery()
        let hubDir = modelsDirectory.appendingPathComponent("hub/models", isDirectory: true)
        let scanDirs = [modelsDirectory, hubDir]
        var allDiscovered: [DiscoveredModel] = []

        for dir in scanDirs {
            guard dir.directoryExists else { continue }
            let found = discovery.discover(in: dir)
            allDiscovered.append(contentsOf: found)
        }

        var registered = 0
        var updated = 0

        for model in allDiscovered {
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
                let updatedRecord = ModelRecord(
                    id: model.modelId,
                    family: model.family,
                    modelType: model.modelType,
                    source: .huggingFace,
                    localURL: model.modelPath,
                    remoteURL: remoteURL,
                    sizeBytes: model.estimatedSizeBytes,
                    downloadedAt: Date(),
                    version: "1.0"
                )
                lock.withLock { _registry[model.modelId] = updatedRecord }
                saveRegistry()
                registered += 1
            } else if existing!.downloadedAt == nil && model.modelPath.directoryExists {
                let updatedRecord = ModelRecord(
                    id: existing!.id,
                    family: existing!.family,
                    modelType: existing!.modelType,
                    source: existing!.source,
                    localURL: model.modelPath,
                    remoteURL: existing!.remoteURL,
                    sizeBytes: model.estimatedSizeBytes,
                    downloadedAt: Date(),
                    version: existing!.version
                )
                lock.withLock { _registry[model.modelId] = updatedRecord }
                saveRegistry()
                updated += 1
            }
        }

        NovaMLXLog.info("Model discovery: \(allDiscovered.count) found, \(registered) registered, \(updated) updated")
        return allDiscovered
    }

    private func hubFileURL(repo: Hub.Repo, filename: String) -> URL {
        var url = URL(string: "https://huggingface.co")!
        if repo.type != .models {
            url = url.appendingPathComponent(repo.type.rawValue)
        }
        url = url.appendingPathComponent(repo.id)
        url = url.appendingPathComponent("resolve")
        url = url.appendingPathComponent("main")
        url = url.appendingPathComponent(filename)
        return url
    }

    private func downloadFile(
        url: URL, to destination: URL,
        onProgress: @Sendable @escaping (UInt64, UInt64) -> Void
    ) async throws {
        let request = URLRequest(url: url)

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NovaMLXError.downloadFailed("Invalid response", underlying: NSError(domain: "NovaMLX", code: -1))
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw NovaMLXError.downloadFailed("HTTP \(httpResponse.statusCode) for \(url.lastPathComponent)", underlying: NSError(domain: "NovaMLX", code: httpResponse.statusCode))
        }

        let totalBytes = UInt64(httpResponse.value(forHTTPHeaderField: "Content-Length") ?? "0") ?? 0
        let tmpDest = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).tmp")

        NovaMLXLog.info("Downloading \(url.lastPathComponent) (\(totalBytes.bytesFormatted)) from HTTP \(httpResponse.statusCode)")

        try? FileManager.default.removeItem(at: tmpDest)
        FileManager.default.createFile(atPath: tmpDest.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tmpDest)
        defer { try? handle.close() }

        var bytesWritten: UInt64 = 0
        var lastReportTime = ContinuousClock.now
        var buffer = Data()
        let flushSize = 1024 * 1024

        for try await byte in asyncBytes {
            buffer.append(byte)
            bytesWritten += 1

            if buffer.count >= flushSize {
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
            }

            let now = ContinuousClock.now
            if now - lastReportTime > .milliseconds(500) {
                lastReportTime = now
                onProgress(bytesWritten, totalBytes)
            }
        }

        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
        }

        onProgress(bytesWritten, totalBytes)
        try? handle.close()

        try FileManager.default.moveItem(at: tmpDest, to: destination)
    }

    private func fileSizeString(at url: URL) -> String {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64) ?? 0
        return size.bytesFormatted
    }

    private func loadRegistry() {
        guard let data = try? Data(contentsOf: registryFile),
              let decoded = try? JSONDecoder().decode([String: ModelRecord].self, from: data) else { return }
        lock.withLock { _registry = decoded }
    }

    private func saveRegistry() {
        let data = lock.withLock { (try? JSONEncoder().encode(_registry)) ?? Data() }
        try? data.write(to: registryFile, options: .atomic)
    }
}
