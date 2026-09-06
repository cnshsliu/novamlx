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

    public init(
        id: String,
        author: String? = nil,
        downloads: Int? = nil,
        likes: Int? = nil,
        trendingScore: Double? = nil,
        tags: [String]? = nil,
        pipelineTag: String? = nil,
        createdAt: String? = nil,
        lastModified: String? = nil,
        privateRepo: Bool? = false,
        gated: Bool? = false
    ) {
        self.id = id
        self.author = author
        self.downloads = downloads
        self.likes = likes
        self.trendingScore = trendingScore
        self.tags = tags
        self.pipelineTag = pipelineTag
        self.createdAt = createdAt
        self.lastModified = lastModified
        self.privateRepo = privateRepo
        self.gated = gated
    }
}

public struct HFSearchResult: Codable, Sendable {
    public let models: [HFModelInfo]
    public let total: Int?

    public init(models: [HFModelInfo], total: Int? = nil) {
        self.models = models
        self.total = total
    }
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

    public var currentURL: String?
    public var retryCount: Int = 0
    public var isResuming: Bool = false
    public var speed: Double = 0
    /// Seconds since the last byte was received for this file. nil if the
    /// file has never received any data (still in Connecting phase). Drives
    /// the UI's "Stalled (Ns)" indicator.
    public var secondsSinceLastByte: Double? = nil

    public init(
        filename: String,
        downloadedBytes: Int64 = 0,
        totalBytes: Int64 = 0,
        status: String = "waiting",
        currentURL: String? = nil,
        retryCount: Int = 0,
        isResuming: Bool = false,
        speed: Double = 0,
        secondsSinceLastByte: Double? = nil
    ) {
        self.filename = filename
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.status = status
        self.currentURL = currentURL
        self.retryCount = retryCount
        self.isResuming = isResuming
        self.speed = speed
        self.secondsSinceLastByte = secondsSinceLastByte
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

/// Thread-safe byte counter for real-time progress across parallel downloads.
/// Also tracks per-file download speed (bytes/sec) computed over the last
/// sample window. The speed sample is updated whenever addBytes is called —
/// we record (timestamp, cumulative bytes) and compute the delta-time-weighted
/// rate on read. This gives the UI a live "X MB/s" indicator without needing
/// a separate timer.
private final class SharedProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var _total: Int64 = 0
    private var _fileBytes: [String: Int64] = [:]
    private var _fileTotals: [String: Int64] = [:]
    /// Per-file last-sample timestamp. Used with _fileBytes to compute speed.
    private var _fileLastSampleAt: [String: Date] = [:]
    /// Per-file last computed speed in bytes/sec. Updated on each addBytes
    /// call so callers reading via getFileSpeed see fresh data.
    private var _fileSpeed: [String: Double] = [:]
    /// Per-file timestamp of the last time we observed >0 bytes flowing.
    /// Drives the UI's "Stalled (Ns since last byte)" indicator when an
    /// established connection goes quiet.
    private var _fileLastByteAt: [String: Date] = [:]

    func addBytes(_ bytes: Int64, forFile filename: String) {
        let now = Date()
        lock.lock()
        _total += bytes
        let prevBytes = _fileBytes[filename, default: 0]
        let prevTime = _fileLastSampleAt[filename]
        _fileBytes[filename] = prevBytes + bytes
        _fileLastByteAt[filename] = now
        // Compute instantaneous speed. Only update if the sample gap is
        // meaningfulful (>= 50ms) to avoid div-by-near-zero blowups when
        // multiple flushes land in the same millisecond.
        if let prevTime = prevTime {
            let dt = now.timeIntervalSince(prevTime)
            if dt >= 0.05 {
                // Exponential moving average — smooths jitter without
                // keeping a ring buffer per file.
                let instantaneous = Double(bytes) / dt
                let prev = _fileSpeed[filename, default: 0]
                _fileSpeed[filename] = prev == 0 ? instantaneous : (prev * 0.5 + instantaneous * 0.5)
                _fileLastSampleAt[filename] = now
            }
        } else {
            _fileLastSampleAt[filename] = now
        }
        lock.unlock()
    }
    func setFile(filename: String, bytes: Int64) {
        let now = Date()
        lock.lock()
        let prev = _fileBytes[filename, default: 0]
        _fileBytes[filename] = bytes
        _total = _fileBytes.values.reduce(0, +)
        if bytes > prev {
            _fileLastByteAt[filename] = now
            if let prevTime = _fileLastSampleAt[filename] {
                let dt = now.timeIntervalSince(prevTime)
                if dt >= 0.05 {
                    let instantaneous = Double(bytes - prev) / dt
                    let old = _fileSpeed[filename, default: 0]
                    _fileSpeed[filename] = old == 0 ? instantaneous : (old * 0.5 + instantaneous * 0.5)
                    _fileLastSampleAt[filename] = now
                }
            } else {
                _fileLastSampleAt[filename] = now
            }
        }
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
    func getFileSpeed(_ filename: String) -> Double {
        lock.lock(); defer { lock.unlock() }; return _fileSpeed[filename] ?? 0
    }
    /// Seconds since the last byte was received for this file. Returns nil
    /// if the file has never received data (still connecting).
    func secondsSinceLastByte(_ filename: String) -> Double? {
        lock.lock(); defer { lock.unlock() }
        guard let d = _fileLastByteAt[filename] else { return nil }
        return Date().timeIntervalSince(d)
    }
}

// MARK: - Mirror Adapter (for different HF-compatible sources)

enum MirrorKind: String, Sendable {
    case huggingface
    case modelscope
}

protocol MirrorAdapter: Sendable {
    var kind: MirrorKind { get }
    var endpoint: String { get }
    var defaultRevision: String { get }

    func searchURL(query: String, limit: Int, mlxOnly: Bool) -> URL
    func modelDetailURL(repoId: String) -> URL
    func fileListURL(repoId: String) -> URL
    func resolveURL(repoId: String, filename: String, revision: String?) -> URL
}

// HF-compatible mirrors (official huggingface.co, custom HF-style hosts)
private struct HFMirrorAdapter: MirrorAdapter {
    let kind: MirrorKind = .huggingface
    let endpoint: String
    let defaultRevision: String = "main"

    init(endpoint: String) {
        self.endpoint = endpoint
    }

    func searchURL(query: String, limit: Int, mlxOnly: Bool) -> URL {
        var components = URLComponents(string: "\(endpoint)/api/models")!
        var items: [URLQueryItem] = [URLQueryItem(name: "limit", value: String(limit))]
        if !query.isEmpty { items.append(URLQueryItem(name: "search", value: query)) }

        if mlxOnly {
            items.append(URLQueryItem(name: "filter", value: "mlx"))
        }

        components.queryItems = items
        return components.url!
    }

    func modelDetailURL(repoId: String) -> URL {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        return URL(string: "\(endpoint)/api/models/\(encoded)")!
    }

    func fileListURL(repoId: String) -> URL {
        // HF style uses model detail which includes siblings
        return modelDetailURL(repoId: repoId)
    }

    func resolveURL(repoId: String, filename: String, revision: String?) -> URL {
        let rev = revision ?? defaultRevision
        return URL(string: "\(endpoint)/\(repoId)/resolve/\(rev)/\(filename)")!
    }
}

// Alibaba ModelScope adapter
private struct ModelScopeAdapter: MirrorAdapter {
    let kind: MirrorKind = .modelscope
    let endpoint: String
    let defaultRevision: String = "master"

    init(endpoint: String) {
        self.endpoint = endpoint.hasSuffix("/") ? String(endpoint.dropLast()) : endpoint
    }

    func searchURL(query: String, limit: Int, mlxOnly: Bool) -> URL {
        // Real endpoint is PUT /api/v1/dolphin/models with JSON body {Name, PageSize, ...}
        // We keep a representative URL here for logging / future use.
        return URL(string: "\(endpoint)/api/v1/dolphin/models")!
    }

    func modelDetailURL(repoId: String) -> URL {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        return URL(string: "\(endpoint)/api/v1/models/\(encoded)")!
    }

    func fileListURL(repoId: String) -> URL {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        return URL(string: "\(endpoint)/api/v1/models/\(encoded)/repo/files?Revision=\(defaultRevision)&Recursive=true")!
    }

    func resolveURL(repoId: String, filename: String, revision: String?) -> URL {
        let rev = revision ?? defaultRevision
        return URL(string: "\(endpoint)/models/\(repoId)/resolve/\(rev)/\(filename)")!
    }
}

// Factory
private extension HuggingFaceService {
    static func makeAdapter(for endpoint: String?) -> any MirrorAdapter {
        let ep = endpoint ?? "https://huggingface.co"
        if ep.contains("modelscope") {
            return ModelScopeAdapter(endpoint: ep)
        } else {
            return HFMirrorAdapter(endpoint: ep)
        }
    }
}

// MARK: - HuggingFaceService

public final class HuggingFaceService: @unchecked Sendable {
    private let session: URLSession
    private let adapter: any MirrorAdapter
    private let lock = NovaMLXLock()
    private var activeTasks: [String: HFDownloadTask] = [:]
    /// Live Swift `Task` handles keyed by task ID. We keep these separate from
    /// `activeTasks` (which is Codable metadata only) so cancellation actually
    /// propagates into the in-flight download instead of just flipping a flag.
    /// Guarded by `lock` to stay consistent with `activeTasks` mutations.
    private var activeTaskHandles: [String: Task<Void, Never>] = [:]
    private var downloadTask: Task<Void, Never>?
    private let modelDirectory: URL
    public var onModelDownloaded: ((String) -> Void)?

    private static let modelscopeUserAgent =
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    public init(modelDirectory: URL, endpoint: String? = nil) {
        self.modelDirectory = modelDirectory
        self.adapter = Self.makeAdapter(for: endpoint)
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 7200
        self.session = URLSession(configuration: config)

        if adapter.kind != .huggingface || endpoint != nil {
            NovaMLXLog.info("[HF] Using mirror \(adapter.kind.rawValue) @ \(adapter.endpoint) (defaultRevision=\(adapter.defaultRevision))")
        }
    }

    // MARK: - Public API

    public func searchModels(query: String, sort: String = "trending", limit: Int = 50, mlxOnly: Bool = false) async throws -> HFSearchResult {
        if adapter.kind == .modelscope {
            // Real ModelScope search uses PUT /api/v1/dolphin/models (discovered via live traffic)
            // The old /api/v1/models?search=... returned 404 and was never the correct endpoint.
            let searchURL = URL(string: "\(adapter.endpoint)/api/v1/dolphin/models")!
            var req = URLRequest(url: searchURL)
            req.httpMethod = "PUT"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("application/json, text/plain, */*", forHTTPHeaderField: "Accept")
            let body: [String: Any] = [
                "PageSize": limit,
                "PageNumber": 1,
                "Name": query,
                "SortBy": "Default",
                "Criterion": [] as [Any],
                "SingleCriterion": [] as [Any],
                "Target": ""
            ]
            req.httpBody = try JSONSerialization.data(withJSONObject: body)

            NovaMLXLog.info("[HF][ModelScope] Real search: PUT \(searchURL) Name=\(query) PageSize=\(limit) mlxOnly=\(mlxOnly)")
            let (data, resp) = try await session.data(for: req)
            if let http = resp as? HTTPURLResponse {
                NovaMLXLog.info("[HF][ModelScope] Upstream HTTP \(http.statusCode)")
            }
            let preview = String(data: data.prefix(1200), encoding: .utf8) ?? "<binary/\(data.count)b>"
            NovaMLXLog.info("[HF][ModelScope] Raw body preview: \(preview)")

            return try parseModelScopeSearchResponse(data, limit: limit, mlxOnly: mlxOnly)
        }

        // HF-style (official huggingface.co, custom HF-compatible hosts)
        let url = adapter.searchURL(query: query, limit: limit, mlxOnly: mlxOnly)
        let request = URLRequest(url: url)
        let (data, _) = try await session.data(for: request)
        let parsed = try parseHuggingFaceStyleSearchResponse(data)
        guard mlxOnly else { return parsed }
        let kept = parsed.models.filter { ModelCatalogPolicy.looksLikeMLXRepo(id: $0.id, tags: $0.tags ?? []) }
        return HFSearchResult(models: kept, total: kept.count)
    }

    // MARK: - 不同镜像的搜索解析

    private func parseHuggingFaceStyleSearchResponse(_ data: Data) throws -> HFSearchResult {
        // Try 1: Direct array
        if let models = try? JSONDecoder().decode([HFModelInfo].self, from: data) {
            return HFSearchResult(models: models, total: models.count)
        }

        // Try 2: Object with "models"
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let modelsArray = json["models"] as? [[String: Any]] {

            let models: [HFModelInfo] = modelsArray.compactMap { dict in
                guard let data = try? JSONSerialization.data(withJSONObject: dict) else { return nil }
                return try? JSONDecoder().decode(HFModelInfo.self, from: data)
            }
            let total = json["total"] as? Int ?? models.count
            return HFSearchResult(models: models, total: total)
        }

        // Try 3: Full object
        if let result = try? JSONDecoder().decode(HFSearchResult.self, from: data) {
            return result
        }

        throw NSError(domain: "HuggingFaceService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unexpected HF-style response"])
    }

    private func parseModelScopeSearchResponse(_ data: Data, limit: Int, mlxOnly: Bool) throws -> HFSearchResult {
        // Real shape (from live reverse engineering of www.modelscope.cn):
        // { "Code": 200, "Data": { "Model": { "Models": [ { "Id": num, "Name": "...", "Path": "Org", "Organization": {...}, "Downloads": N, "Stars": N, "Libraries": [...], "Tags": [...], "ModelType": "...", ... } ], "TotalCount": N }, "FiledAgg": {...} }, ... }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            NovaMLXLog.error("[HF][ModelScope] parse: not a JSON object")
            throw NSError(domain: "HuggingFaceService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unexpected ModelScope response (not JSON)"])
        }

        let dataObj = (json["Data"] as? [String: Any]) ?? (json["data"] as? [String: Any]) ?? [:]
        let modelContainer = (dataObj["Model"] as? [String: Any]) ?? (dataObj["model"] as? [String: Any]) ?? [:]
        var modelsArray = (modelContainer["Models"] as? [[String: Any]]) ?? (modelContainer["models"] as? [[String: Any]]) ?? []

        // Legacy fallback (old guessed API)
        if modelsArray.isEmpty {
            modelsArray = (dataObj["models"] as? [[String: Any]]) ?? []
        }

        NovaMLXLog.info("[HF][ModelScope] parse: raw models in response = \(modelsArray.count), totalCount=\(modelContainer["TotalCount"] ?? dataObj["total"] ?? "n/a")")

        var models: [HFModelInfo] = []
        var skippedNoId = 0, skippedMlx = 0, skippedDecode = 0

        for dict in modelsArray {
            // Construct stable repoId: "Path/Name" (e.g. "Qwen/Qwen3.6-27B") used by fileList + downloads
            let path = (dict["Path"] as? String)
                ?? ((dict["Organization"] as? [String: Any])?["Name"] as? String)
                ?? (dict["Owner"] as? String) ?? ""
            let name = (dict["Name"] as? String) ?? (dict["name"] as? String) ?? ""
            let fullId = path.isEmpty ? name : "\(path)/\(name)"
            if fullId.isEmpty {
                skippedNoId += 1
                continue
            }

            // MLX filter using the fields that actually exist on ModelScope items (Libraries + Tags + ModelType + name)
            if mlxOnly {
                let libs = (dict["Libraries"] as? [String]) ?? []
                let tags = (dict["Tags"] as? [String]) ?? []
                let modelType = ((dict["ModelType"] as? String) ?? (dict["model_type"] as? String) ?? "").lowercased()
                let hasMlx = libs.contains("mlx") || tags.contains("mlx") || modelType.contains("mlx") || fullId.lowercased().contains("mlx")
                if !hasMlx {
                    skippedMlx += 1
                    continue
                }
            }

            // Normalize to something the HFModelInfo decoder can understand (it expects "id", "author", "downloads" etc.)
            var norm: [String: Any] = [
                "id": fullId,
                "author": path,
                "downloads": dict["Downloads"] ?? 0,
                "likes": dict["Stars"] ?? 0,
                "tags": Array(Set(((dict["Tags"] as? [String]) ?? []) + ((dict["Libraries"] as? [String]) ?? []))),
                "pipelineTag": (dict["Tasks"] as? [String])?.first ?? (dict["ModelType"] as? String) ?? "" as String?
            ]
            if let t = dict["CreatedTime"] ?? dict["LastUpdatedTime"] { norm["createdAt"] = "\(t)" }

            var info: HFModelInfo?
            if let itemData = try? JSONSerialization.data(withJSONObject: norm),
               let decoded = try? JSONDecoder().decode(HFModelInfo.self, from: itemData) {
                info = decoded
            } else if let itemData2 = try? JSONSerialization.data(withJSONObject: dict),
                      let decoded2 = try? JSONDecoder().decode(HFModelInfo.self, from: itemData2) {
                info = decoded2
            }

            if let info = info {
                models.append(info)
            } else {
                skippedDecode += 1
            }

            if models.count >= limit { break }
        }

        NovaMLXLog.info("[HF][ModelScope] parse done: kept=\(models.count) skippedNoId=\(skippedNoId) skippedMlx=\(skippedMlx) skippedDecode=\(skippedDecode)")

        let total = (modelContainer["TotalCount"] as? Int) ?? (dataObj["total"] as? Int) ?? models.count
        return HFSearchResult(models: models, total: total)
    }

    public func getModelDetail(repoId: String) async throws -> HFModelDetail {
        let url = adapter.modelDetailURL(repoId: repoId)
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(HFModelDetail.self, from: data)
    }

    /// Adapter-aware file list fetch.
    /// For HF mirrors: uses /api/models/{repo} which returns siblings.
    /// For ModelScope: uses the special /repo/files endpoint and converts the response.
    private func fetchFileList(
        repoId: String,
        hfToken: String?,
        adapter: any MirrorAdapter
    ) async throws -> [HFModelDetail.HFFile] {
        if adapter.kind == .huggingface {
            let detail = try await Self.getModelDetailStatic(
                repoId: repoId,
                baseURL: adapter.endpoint,
                session: session,
                hfToken: hfToken
            )
            return Self.filterDownloadable(detail.siblings ?? [])
        } else {
            // ModelScope path — delegate to dedicated implementation
            NovaMLXLog.info("[HF][Download] ModelScope branch taken for \(repoId), adapter.endpoint=\(adapter.endpoint), kind=\(adapter.kind)")
            let ms = ModelScopeService(endpoint: adapter.endpoint, session: session)
            let files = try await ms.listFiles(repoId: repoId, revision: adapter.defaultRevision)

            NovaMLXLog.info("[HF][Download] ModelScope file list returned \(files.count) files for \(repoId)")
            return files.map { f in
                HFModelDetail.HFFile(rfilename: f.path, size: f.size)
            }
        }
    }

    public func startDownload(
        repoId: String,
        hfToken: String? = nil,
        mirrorEndpoint: String? = nil,
        revision: String? = nil
    ) async throws -> HFDownloadTask {
        // Idempotent Resume: kill any in-flight tasks for the SAME repoId
        // before minting a new one. Without this, a user clicking Resume
        // repeatedly would spawn N concurrent downloads all writing to the
        // same `.download` temp files — guaranteed corruption + wasted bandwidth.
        // We can't rely on the client-side dedup guard alone; restart crashes,
        // race windows, and rapid clicks all leak through.
        cancelTasksForRepo(repoId: repoId)

        // If the caller supplied a live mirror (ModelScope, custom HF host),
        // build a one-off adapter for this download only.
        // The HFDownloadTask is still registered in *this* service's activeTasks
        // (the one that the /tasks API reads), so the UI always sees it.
        let effectiveAdapter = mirrorEndpoint != nil
            ? Self.makeAdapter(for: mirrorEndpoint)
            : self.adapter

        var task = HFDownloadTask(repoId: repoId)
        task.status = "downloading"
        lock.withLock { activeTasks[task.id] = task }

        #if DEBUG
        NovaMLXLog.info("[DL] Starting download for \(repoId) (task=\(task.id.prefix(8)))")
        #endif

        let taskCopy = task
        let handle: Task<Void, Never> = Task { [weak self] in
            guard let self else { return }
            await self.runDownload(
                task: taskCopy,
                hfToken: hfToken,
                adapter: effectiveAdapter,
                revision: revision
            ) { [weak self] _ in
                self?.onModelDownloaded?(repoId)
            }
        }
        lock.withLock { activeTaskHandles[task.id] = handle }
        downloadTask = handle
        return task
    }

    public func getTasks() -> [HFDownloadTask] {
        lock.withLock { Array(activeTasks.values) }
    }

    /// Cancel one task by id. Marks status AND signals the live Swift `Task`
    /// to stop. The aria2 poll loop checks `Task.isCancelled` and SIGTERMs
    /// the child process.
    public func cancelTask(id: String) -> Bool {
        var handle: Task<Void, Never>?
        var didMark = false
        lock.withLock() {
            handle = activeTaskHandles[id]
            if var task = activeTasks[id] {
                task.status = "cancelled"
                task.completedAt = Date()
                activeTasks[id] = task
                didMark = true
            }
        }
        handle?.cancel()
        return didMark
    }

    /// Cancel every active task whose repoId matches. Returns the count of
    /// tasks signalled. Used by `startDownload` to guarantee single-flight
    /// per repo — see the comment there for why.
    @discardableResult
    public func cancelTasksForRepo(repoId: String) -> Int {
        var idsToCancel: [String] = []
        lock.withLock() {
            idsToCancel = activeTasks
                .filter { $0.value.repoId == repoId && $0.value.status == "downloading" }
                .map { $0.key }
        }
        guard !idsToCancel.isEmpty else { return 0 }

        #if DEBUG
        NovaMLXLog.info("[DL] Cancel \(idsToCancel.count) in-flight task(s) for \(repoId) before starting new one")
        #endif

        // Signal each task to stop. We don't wait for them to actually exit —
        // aria2c is SIGTERM'd when the Swift Task is cancelled. Straggler
        // writes after this point are harmless because the new task resumes
        // with aria2 `-c`.
        var handles: [Task<Void, Never>] = []
        lock.withLock() {
            for id in idsToCancel {
                if var t = activeTasks[id] { t.status = "cancelled"; activeTasks[id] = t }
                if let h = activeTaskHandles.removeValue(forKey: id) { handles.append(h) }
            }
        }
        for h in handles { h.cancel() }
        return idsToCancel.count
    }

    public func removeTask(id: String) -> Bool {
        lock.withLock { activeTasks.removeValue(forKey: id) != nil }
    }

    // MARK: - Core Download Engine (instance method — accesses self.lock/activeTasks directly)

    private func runDownload(
        task: HFDownloadTask,
        hfToken: String?,
        adapter: any MirrorAdapter,
        revision: String?,
        onModelDownloaded: @Sendable @escaping (String) -> Void
    ) async {
        let targetDir = modelDirectory.appendingPathComponent(
            task.repoId.replacingOccurrences(of: ":", with: "_"), isDirectory: true)
        var currentTask = task

        func save(_ t: HFDownloadTask) {
            lock.withLock {
                activeTasks[t.id] = t
                // Drop the live Task handle once we've reached a terminal
                // status. Keeps the dict from leaking handles for finished
                // downloads; cancellation on a finished task is a no-op anyway.
                if t.status == "completed" || t.status == "failed" || t.status == "cancelled" {
                    activeTaskHandles.removeValue(forKey: t.id)
                }
            }
        }

        do {
            // Step 1: File list (adapter-aware)
            let allFiles = try await self.fetchFileList(repoId: task.repoId, hfToken: hfToken, adapter: adapter)
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
                let url = adapter.resolveURL(repoId: task.repoId, filename: file.rfilename, revision: revision)
                if adapter.kind == .modelscope {
                    NovaMLXLog.info("[HF][ModelScope] Resolve URL for \(file.rfilename): \(url)")
                }
                let size: Int64
                if adapter.kind == .modelscope && file.size ?? 0 > 0 {
                    // Trust the size we already received from ModelScope's /repo/files API
                    size = Int64(file.size ?? 0)
                } else {
                    size = await Self.headFileSize(url: url, session: session, hfToken: hfToken)
                }
                fileInfos.append(.init(filename: file.rfilename, expectedSize: size, sourceURL: url))
                if size > 0 { estimatedTotal += size }
            }

            // Per-file URL, speed, and resume state for the Downloads activity list
            currentTask.fileProgresses = fileInfos.map {
                FileProgress(
                    filename: $0.filename,
                    totalBytes: $0.expectedSize,
                    status: "waiting",
                    currentURL: $0.sourceURL.absoluteString
                )
            }
            save(currentTask)
            currentTask.totalBytes = estimatedTotal
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

            // Step 4: spawn aria2c. NovaMLX does not stream model bytes.
            let shared = SharedProgress()
            let taskId = task.id
            for file in fileInfos {
                let dest = targetDir.appendingPathComponent(file.filename)
                if FileManager.default.fileExists(atPath: dest.path) {
                    let size = Int64(FileManager.default.fileSize(at: dest) ?? 0)
                    if file.expectedSize == 0 || size == file.expectedSize {
                        shared.setFile(filename: file.filename, bytes: size)
                        if file.expectedSize > 0 {
                            shared.setFileTotal(filename: file.filename, total: file.expectedSize)
                        }
                    }
                }
            }

            if !remaining.isEmpty {
                let syncTask = Task { [weak self] in
                    while !Task.isCancelled {
                        try? await Task.sleep(for: .milliseconds(300))
                        guard let self else { return }
                        self.lock.withLock {
                            guard var t = self.activeTasks[taskId] else { return }
                            t.downloadedBytes = shared.getTotal()
                            var calculatedTotal: Int64 = 0
                            for i in t.fileProgresses.indices {
                                let fname = t.fileProgresses[i].filename
                                let live = shared.getFileBytes(fname)
                                if live > 0 {
                                    t.fileProgresses[i].downloadedBytes = live
                                    if t.fileProgresses[i].status == "waiting" {
                                        t.fileProgresses[i].status = "downloading"
                                    }
                                }
                                let liveTotal = shared.getFileTotal(fname)
                                if liveTotal > 0 && t.fileProgresses[i].totalBytes == 0 {
                                    t.fileProgresses[i].totalBytes = liveTotal
                                }
                                t.fileProgresses[i].speed = shared.getFileSpeed(fname)
                                t.fileProgresses[i].secondsSinceLastByte = shared.secondsSinceLastByte(fname)
                                calculatedTotal += t.fileProgresses[i].totalBytes > 0
                                    ? t.fileProgresses[i].totalBytes
                                    : t.fileProgresses[i].downloadedBytes
                            }
                            if calculatedTotal > t.totalBytes { t.totalBytes = calculatedTotal }
                            if t.totalBytes > 0 {
                                t.progress = min(Double(t.downloadedBytes) / Double(t.totalBytes) * 99.0, 99.0)
                            }
                            self.activeTasks[taskId] = t
                        }
                    }
                }
                defer { syncTask.cancel() }

                let tok = hfToken
                let ua: String? = adapter.kind == .modelscope ? Self.modelscopeUserAgent : nil
                try await Aria2Downloader.download(
                    files: remaining.map {
                        Aria2DownloadFile(
                            url: $0.sourceURL,
                            relativePath: $0.filename,
                            expectedSize: $0.expectedSize
                        )
                    },
                    destination: targetDir,
                    authorization: tok.map { "Bearer \($0)" },
                    userAgent: ua
                ) { snapshot in
                    for file in snapshot.files {
                        shared.setFile(filename: file.relativePath, bytes: file.downloadedBytes)
                        if file.totalBytes > 0 {
                            shared.setFileTotal(filename: file.relativePath, total: file.totalBytes)
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

        } catch is CancellationError {
            currentTask.status = "cancelled"
            currentTask.error = "Cancelled"
            currentTask.completedAt = Date()
            save(currentTask)
            NovaMLXLog.info("[DL] CANCELLED \(task.repoId)")
        } catch {
            currentTask.status = "failed"
            currentTask.error = error.localizedDescription
            currentTask.completedAt = Date()
            save(currentTask)
            NovaMLXLog.error("[DL] FAILED \(task.repoId): \(error.localizedDescription)")
        }
    }

    // MARK: - Helpers

    private static func filterDownloadable(_ siblings: [HFModelDetail.HFFile]) -> [HFModelDetail.HFFile] {
        siblings.filter { f in
            let name = f.rfilename
            if name == ".gitattributes" { return false }
            return name.hasSuffix(".safetensors")
                || name.hasSuffix(".json")
                || name.hasSuffix(".jinja")
                || name.hasSuffix(".model")
                || name.hasSuffix(".txt")
                || name.hasSuffix(".tiktoken")
                || name.hasSuffix(".py")
                || name == "tokenizer.model"
                || name == "README.md"
        }
    }

    private static func headFileSize(url: URL, session: URLSession, hfToken: String?) async -> Int64 {
        var req = URLRequest(url: url)
        req.httpMethod = "HEAD"
        if let t = hfToken { req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }

        if url.host?.contains("modelscope") == true {
            req.setValue(modelscopeUserAgent, forHTTPHeaderField: "User-Agent")
        }

        do {
            let (_, resp) = try await session.data(for: req)
            if let http = resp as? HTTPURLResponse {
                if http.statusCode != 200 && http.statusCode != 206 {
                    NovaMLXLog.error("[HF] HEAD failed for \(url) → HTTP \(http.statusCode)")
                }
                if let cl = http.value(forHTTPHeaderField: "Content-Length"),
                   let size = Int64(cl) { return size }
            }
        } catch {
            NovaMLXLog.error("[HF] HEAD error for \(url): \(error.localizedDescription)")
        }
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
