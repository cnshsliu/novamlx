import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct HFModelInfo: Codable, Sendable {
    public let id: String
    public let author: String?
    public let downloads: Int?
    public let likes: Int?
    public let trendingScore: Double?
    public let tags: [String]?
    public let pipelineTag: String?
    public let createdAt: String?
    public let lastModified: String?
    public let privateRepo: Bool?
    public let gated: Bool?
}

public struct HFSearchResult: Codable, Sendable {
    public let models: [HFModelInfo]
    public let total: Int?
}

public struct HFModelDetail: Codable, Sendable {
    public let id: String
    public let author: String?
    public let downloads: Int?
    public let likes: Int?
    public let tags: [String]?
    public let siblings: [HFFile]?
    public let cardData: HFCardData?
    public let config: HFConfig?

    public struct HFFile: Codable, Sendable {
        public let rfilename: String
        public let size: Int?
    }
    public struct HFCardData: Codable, Sendable {
        public let language: [String]?
        public let license: String?
        public let libraryName: String?
        public let tags: [String]?
    }
    public struct HFConfig: Codable, Sendable {
        public let modelType: String?
        public let architectures: [String]?
        private enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case architectures
        }
    }
}

// MARK: - Download Types

public struct FileProgress: Codable, Sendable {
    public let filename: String
    public var downloadedBytes: Int64
    public var totalBytes: Int64
    public var status: String  // "downloading", "completed", "failed", "waiting"
    public init(filename: String, downloadedBytes: Int64 = 0, totalBytes: Int64 = 0, status: String = "waiting") {
        self.filename = filename
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.status = status
    }
}

public struct HFDownloadTask: Codable, Sendable {
    public let id: String
    public let repoId: String
    public var status: String
    public var progress: Double
    public var downloadedBytes: Int64
    public var totalBytes: Int64
    public var error: String?
    public let startedAt: Date
    public var completedAt: Date?
    public var fileProgresses: [FileProgress]
    public init(repoId: String) {
        self.id = UUID().uuidString
        self.repoId = repoId
        self.status = "pending"
        self.progress = 0
        self.downloadedBytes = 0
        self.totalBytes = 0
        self.startedAt = Date()
        self.fileProgresses = []
    }
}

// MARK: - Internal

private struct FileToDownload: Sendable {
    let filename: String
    let expectedSize: Int64
    let sourceURL: URL
}

private struct FileResult: Sendable {
    let filename: String
    let success: Bool
    let downloadedBytes: Int64
    let errorMessage: String?
}

/// Thread-safe byte counter for real-time progress across parallel downloads
private final class SharedProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var _total: Int64 = 0
    private var _fileBytes: [String: Int64] = [:]
    private var _fileTotals: [String: Int64] = [:]

    func addBytes(_ bytes: Int64, forFile filename: String) {
        lock.lock()
        _total += bytes
        _fileBytes[filename, default: 0] += bytes
        lock.unlock()
    }
    func setFile(filename: String, bytes: Int64) {
        lock.lock()
        _fileBytes[filename] = bytes
        lock.unlock()
    }
    func setFileTotal(filename: String, total: Int64) {
        lock.lock()
        _fileTotals[filename] = total
        lock.unlock()
    }
    func getTotal() -> Int64 {
        lock.lock(); defer { lock.unlock() }; return _total
    }
    func getFileBytes(_ filename: String) -> Int64 {
        lock.lock(); defer { lock.unlock() }; return _fileBytes[filename] ?? 0
    }
    func getFileTotal(_ filename: String) -> Int64 {
        lock.lock(); defer { lock.unlock() }; return _fileTotals[filename] ?? 0
    }
}

// MARK: - HuggingFaceService

public final class HuggingFaceService: @unchecked Sendable {
    private let session: URLSession
    private let baseURL: String
    private let lock = NovaMLXLock()
    private var activeTasks: [String: HFDownloadTask] = [:]
    private var downloadTask: Task<Void, Never>?
    private let modelDirectory: URL
    public var onModelDownloaded: ((String) -> Void)?

    private static let maxConcurrent = 4
    private static let maxRetries = 20
    private static let initialBackoff: Double = 2.0
    private static let maxBackoff: Double = 120.0

    public init(modelDirectory: URL, endpoint: String? = nil) {
        self.modelDirectory = modelDirectory
        self.baseURL = endpoint ?? "https://huggingface.co"
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 7200
        self.session = URLSession(configuration: config)
    }

    // MARK: - Public API

    public func searchModels(query: String, sort: String = "trending", limit: Int = 50, mlxOnly: Bool = true) async throws -> HFSearchResult {
        var components = URLComponents(string: "\(baseURL)/api/models")!
        var items = [URLQueryItem(name: "limit", value: String(limit))]
        if !query.isEmpty { items.append(URLQueryItem(name: "search", value: query)) }
        if mlxOnly { items.append(URLQueryItem(name: "filter", value: "mlx")) }
        components.queryItems = items
        let request = URLRequest(url: components.url!)
        let (data, _) = try await session.data(for: request)
        let models = try JSONDecoder().decode([HFModelInfo].self, from: data)
        return HFSearchResult(models: models, total: models.count)
    }

    public func getModelDetail(repoId: String) async throws -> HFModelDetail {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        let url = URL(string: "\(baseURL)/api/models/\(encoded)")!
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(HFModelDetail.self, from: data)
    }

    public func startDownload(repoId: String, hfToken: String? = nil) async throws -> HFDownloadTask {
        var task = HFDownloadTask(repoId: repoId)
        task.status = "downloading"
        lock.withLock { activeTasks[task.id] = task }

        #if DEBUG
        NovaMLXLog.info("[DL] Starting download for \(repoId)")
        #endif

        let taskCopy = task
        downloadTask = Task { [weak self] in
            await self?.runDownload(task: taskCopy, hfToken: hfToken) { [weak self] _ in
                self?.onModelDownloaded?(repoId)
            }
        }
        return task
    }

    public func getTasks() -> [HFDownloadTask] {
        lock.withLock { Array(activeTasks.values) }
    }

    public func cancelTask(id: String) -> Bool {
        lock.withLock {
            guard var task = activeTasks[id] else { return false }
            task.status = "cancelled"
            task.completedAt = Date()
            activeTasks[id] = task
            return true
        }
    }

    public func removeTask(id: String) -> Bool {
        lock.withLock { activeTasks.removeValue(forKey: id) != nil }
    }

    // MARK: - Core Download Engine (instance method — accesses self.lock/activeTasks directly)

    private func runDownload(
        task: HFDownloadTask,
        hfToken: String?,
        onModelDownloaded: @Sendable @escaping (String) -> Void
    ) async {
        let targetDir = modelDirectory.appendingPathComponent(
            task.repoId.replacingOccurrences(of: ":", with: "_"), isDirectory: true)
        var currentTask = task

        func save(_ t: HFDownloadTask) {
            lock.withLock { activeTasks[t.id] = t }
        }

        do {
            // Step 1: File list
            let detail = try await Self.getModelDetailStatic(repoId: task.repoId, baseURL: baseURL, session: session, hfToken: hfToken)
            let allFiles = Self.filterDownloadable(detail.siblings ?? [])
            let totalFiles = allFiles.count
            guard totalFiles > 0 else {
                currentTask.status = "failed"
                currentTask.error = "No downloadable files for \(task.repoId)"
                save(currentTask); return
            }
            try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

            // Step 2: HEAD each file for size
            var fileInfos: [FileToDownload] = []
            var estimatedTotal: Int64 = 0
            for file in allFiles {
                let url = URL(string: "\(baseURL)/\(task.repoId)/resolve/main/\(file.rfilename)")!
                let size = await Self.headFileSize(url: url, session: session, hfToken: hfToken)
                fileInfos.append(.init(filename: file.rfilename, expectedSize: size, sourceURL: url))
                if size > 0 { estimatedTotal += size }
            }
            currentTask.totalBytes = estimatedTotal
            currentTask.fileProgresses = fileInfos.map { .init(filename: $0.filename, totalBytes: $0.expectedSize, status: "waiting") }
            save(currentTask)

            #if DEBUG
            NovaMLXLog.info("[DL] \(task.repoId): \(totalFiles) files, ~\(estimatedTotal) bytes")
            #endif

            // Step 3: Partition complete vs remaining
            var completedCount = 0
            var totalDownloaded: Int64 = 0
            var remaining: [FileToDownload] = []
            for (i, file) in fileInfos.enumerated() {
                let dest = targetDir.appendingPathComponent(file.filename)
                if FileManager.default.fileExists(atPath: dest.path) {
                    let size = FileManager.default.fileSize(at: dest) ?? 0
                    if file.expectedSize == 0 || size == UInt64(file.expectedSize) {
                        completedCount += 1; totalDownloaded += Int64(size)
                        currentTask.fileProgresses[i].status = "completed"
                        currentTask.fileProgresses[i].downloadedBytes = Int64(size)
                        continue
                    }
                }
                remaining.append(file)
            }
            currentTask.downloadedBytes = totalDownloaded
            save(currentTask)

            #if DEBUG
            NovaMLXLog.info("[DL] \(task.repoId): \(completedCount) done, \(remaining.count) to download")
            #endif

            // Step 4: Parallel streaming download
            let shared = SharedProgress()
            let taskId = task.id

            if !remaining.isEmpty {
                // Sync timer: shared progress → activeTasks every 300ms for polling
                let syncTask = Task { [weak self] in
                    while !Task.isCancelled {
                        try? await Task.sleep(for: .milliseconds(300))
                        guard let self else { return }
                        self.lock.withLock {
                            guard var t = self.activeTasks[taskId] else { return }
                            t.downloadedBytes = shared.getTotal()
                            var calculatedTotal: Int64 = 0
                            for i in t.fileProgresses.indices {
                                let live = shared.getFileBytes(t.fileProgresses[i].filename)
                                if live > 0 {
                                    t.fileProgresses[i].downloadedBytes = live
                                    // Auto-detect: file has bytes → it's downloading
                                    if t.fileProgresses[i].status == "waiting" {
                                        t.fileProgresses[i].status = "downloading"
                                    }
                                }
                                // Update totalBytes from actual response if HEAD failed
                                let liveTotal = shared.getFileTotal(t.fileProgresses[i].filename)
                                if liveTotal > 0 && t.fileProgresses[i].totalBytes == 0 {
                                    t.fileProgresses[i].totalBytes = liveTotal
                                }
                                calculatedTotal += t.fileProgresses[i].totalBytes > 0
                                    ? t.fileProgresses[i].totalBytes
                                    : t.fileProgresses[i].downloadedBytes
                            }
                            // Recalculate task total from file totals (may exceed original HEAD estimate)
                            if calculatedTotal > t.totalBytes { t.totalBytes = calculatedTotal }
                            if t.totalBytes > 0 {
                                t.progress = min(Double(t.downloadedBytes) / Double(t.totalBytes) * 99.0, 99.0)
                            }
                            self.activeTasks[taskId] = t
                        }
                    }
                }
                defer { syncTask.cancel() }

                let sess = session; let tok = hfToken

                await withTaskGroup(of: FileResult.self) { group in
                    var iter = remaining.makeIterator()
                    for _ in 0..<min(Self.maxConcurrent, remaining.count) {
                        if let f = iter.next() {
                            group.addTask { await Self.streamFile(file: f, target: targetDir, session: sess, token: tok, progress: shared) }
                        }
                    }
                    for await result in group {
                        completedCount += 1
                        if result.success { totalDownloaded = shared.getTotal() }
                        lock.withLock {
                            guard var t = activeTasks[taskId] else { return }
                            for i in t.fileProgresses.indices where t.fileProgresses[i].filename == result.filename {
                                t.fileProgresses[i].status = result.success ? "completed" : "failed"
                                if result.success { t.fileProgresses[i].downloadedBytes = result.downloadedBytes }
                            }
                            t.downloadedBytes = shared.getTotal()
                            if t.totalBytes > 0 { t.progress = min(Double(t.downloadedBytes) / Double(t.totalBytes) * 99.0, 99.0) }
                            activeTasks[taskId] = t
                        }
                        #if DEBUG
                        NovaMLXLog.info("[DL] \(result.filename): \(result.success ? "OK" : "FAIL") (\(completedCount)/\(totalFiles))")
                        #endif
                        if let f = iter.next() {
                            group.addTask { await Self.streamFile(file: f, target: targetDir, session: sess, token: tok, progress: shared) }
                        }
                    }
                }
            }

            // Step 5: Verify all files
            var failed: [String] = []
            for file in fileInfos {
                let dest = targetDir.appendingPathComponent(file.filename)
                guard FileManager.default.fileExists(atPath: dest.path) else { failed.append(file.filename); continue }
                if file.expectedSize > 0 {
                    let actual = FileManager.default.fileSize(at: dest) ?? 0
                    if actual != UInt64(file.expectedSize) { failed.append("\(file.filename) [size mismatch]") }
                }
            }
            if !failed.isEmpty {
                currentTask.status = "failed"
                currentTask.error = "Missing/corrupt: \(failed.prefix(5).joined(separator: ", "))"
                save(currentTask); return
            }

            // Step 6: Complete
            currentTask.status = "completed"
            currentTask.progress = 100
            currentTask.totalBytes = shared.getTotal()
            currentTask.downloadedBytes = currentTask.totalBytes
            currentTask.completedAt = Date()
            for i in currentTask.fileProgresses.indices { currentTask.fileProgresses[i].status = "completed" }
            save(currentTask)
            #if DEBUG
            NovaMLXLog.info("[DL] COMPLETED \(task.repoId): \(totalFiles) files, \(currentTask.totalBytes) bytes")
            #endif
            onModelDownloaded(task.repoId)

        } catch {
            currentTask.status = "failed"
            currentTask.error = error.localizedDescription
            currentTask.completedAt = Date()
            save(currentTask)
            #if DEBUG
            NovaMLXLog.error("[DL] FAILED \(task.repoId): \(error.localizedDescription)")
            #endif
        }
    }

    // MARK: - Streaming Single File (resume + retry + real-time progress)

    private static func streamFile(
        file: FileToDownload,
        target: URL,
        session: URLSession,
        token: String?,
        progress: SharedProgress
    ) async -> FileResult {
        let dest = target.appendingPathComponent(file.filename)
        let temp = dest.appendingPathExtension("download")
        let fm = FileManager.default

        if let dir = dest.deletingLastPathComponent().path.removingPercentEncoding {
            try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }

        // Already complete?
        if fm.fileExists(atPath: dest.path) {
            let size = fm.fileSize(at: dest) ?? 0
            if file.expectedSize == 0 || size == UInt64(file.expectedSize) {
                progress.setFile(filename: file.filename, bytes: Int64(size))
                return .init(filename: file.filename, success: true, downloadedBytes: Int64(size), errorMessage: nil)
            }
            try? fm.removeItem(at: dest)
        }

        var backoff = initialBackoff
        for attempt in 0...maxRetries {
            do {
                let existing: Int64 = fm.fileExists(atPath: temp.path) ? Int64(fm.fileSize(at: temp) ?? 0) : 0

                // Temp already complete?
                if existing > 0 && file.expectedSize > 0 && existing == file.expectedSize {
                    try moveFile(temp: temp, dest: dest)
                    progress.setFile(filename: file.filename, bytes: existing)
                    return .init(filename: file.filename, success: true, downloadedBytes: existing, errorMessage: nil)
                }

                var request = URLRequest(url: file.sourceURL)
                request.timeoutInterval = 600
                if let t = token { request.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }

                var resumeFrom: Int64 = 0
                if existing > 0 {
                    request.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range")
                    resumeFrom = existing
                }

                let (asyncBytes, response) = try await session.bytes(for: request)
                guard let http = response as? HTTPURLResponse else { throw NSError(domain: "NovaMLX", code: -1) }

                let isResume: Bool
                if http.statusCode == 206 && resumeFrom > 0 {
                    isResume = true
                } else if http.statusCode == 200 {
                    isResume = false
                    if resumeFrom > 0 { try? fm.removeItem(at: temp); resumeFrom = 0 }
                } else {
                    throw NSError(domain: "NovaMLX", code: http.statusCode,
                                  userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"])
                }

                // Discover actual file size from response if HEAD didn't give it to us
                if file.expectedSize == 0 {
                    if let cl = http.value(forHTTPHeaderField: "Content-Length"), let contentLength = Int64(cl) {
                        let actualTotal = isResume ? resumeFrom + contentLength : contentLength
                        progress.setFileTotal(filename: file.filename, total: actualTotal)
                    }
                }

                if !fm.fileExists(atPath: temp.path) { fm.createFile(atPath: temp.path, contents: nil) }
                let handle = try FileHandle(forWritingTo: temp)
                if isResume { try handle.seek(toOffset: UInt64(resumeFrom)) }
                else { try handle.truncate(atOffset: 0) }

                var written: Int64 = resumeFrom
                var buffer = Data()
                let flushSize = 512 * 1024
                var sinceLastReport: Int64 = 0

                for try await byte in asyncBytes {
                    buffer.append(byte)
                    written += 1
                    sinceLastReport += 1
                    if buffer.count >= flushSize {
                        try handle.write(contentsOf: buffer)
                        progress.addBytes(sinceLastReport, forFile: file.filename)
                        sinceLastReport = 0
                        buffer.removeAll(keepingCapacity: true)
                    }
                }
                if !buffer.isEmpty {
                    try handle.write(contentsOf: buffer)
                    progress.addBytes(sinceLastReport, forFile: file.filename)
                }
                try handle.close()

                // Verify
                if file.expectedSize > 0 {
                    let finalSize = fm.fileSize(at: temp) ?? 0
                    if finalSize != UInt64(file.expectedSize) {
                        try? fm.removeItem(at: temp)
                        progress.setFile(filename: file.filename, bytes: 0)
                        throw NSError(domain: "NovaMLX", code: -2,
                                      userInfo: [NSLocalizedDescriptionKey: "Size mismatch: \(finalSize) != \(file.expectedSize)"])
                    }
                }

                try moveFile(temp: temp, dest: dest)
                progress.setFile(filename: file.filename, bytes: written)
                return .init(filename: file.filename, success: true, downloadedBytes: written, errorMessage: nil)

            } catch {
                if attempt < maxRetries {
                    #if DEBUG
                    NovaMLXLog.warning("[DL] Retry \(attempt+1)/\(maxRetries) \(file.filename): \(error.localizedDescription)")
                    #endif
                    try? await Task.sleep(for: .seconds(backoff))
                    backoff = min(backoff * 2, maxBackoff)
                } else {
                    #if DEBUG
                    NovaMLXLog.error("[DL] GAVE UP \(file.filename) after \(maxRetries) retries")
                    #endif
                    return .init(filename: file.filename, success: false, downloadedBytes: 0, errorMessage: error.localizedDescription)
                }
            }
        }
        return .init(filename: file.filename, success: false, downloadedBytes: 0, errorMessage: "Max retries")
    }

    // MARK: - Helpers

    private static func moveFile(temp: URL, dest: URL) throws {
        if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
        try FileManager.default.moveItem(at: temp, to: dest)
    }

    private static func filterDownloadable(_ siblings: [HFModelDetail.HFFile]) -> [HFModelDetail.HFFile] {
        siblings.filter { f in
            f.rfilename.hasSuffix(".safetensors") ||
            f.rfilename.hasSuffix(".json") ||
            f.rfilename.hasSuffix(".model") ||
            f.rfilename.hasSuffix(".txt") ||
            f.rfilename.hasSuffix(".tiktoken") ||
            f.rfilename == "tokenizer.model" ||
            f.rfilename.hasSuffix(".py")
        }
    }

    private static func headFileSize(url: URL, session: URLSession, hfToken: String?) async -> Int64 {
        var req = URLRequest(url: url)
        req.httpMethod = "HEAD"
        if let t = hfToken { req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }
        do {
            let (_, resp) = try await session.data(for: req)
            if let http = resp as? HTTPURLResponse,
               let cl = http.value(forHTTPHeaderField: "Content-Length"),
               let size = Int64(cl) { return size }
        } catch {}
        return 0
    }

    private static func getModelDetailStatic(repoId: String, baseURL: String, session: URLSession, hfToken: String?) async throws -> HFModelDetail {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        let url = URL(string: "\(baseURL)/api/models/\(encoded)")!
        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let t = hfToken { req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }
        let (data, _) = try await session.data(for: req)
        return try JSONDecoder().decode(HFModelDetail.self, from: data)
    }
}
