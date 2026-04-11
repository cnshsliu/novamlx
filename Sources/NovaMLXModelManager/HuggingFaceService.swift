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

    public init(repoId: String) {
        self.id = UUID().uuidString
        self.repoId = repoId
        self.status = "pending"
        self.progress = 0
        self.downloadedBytes = 0
        self.totalBytes = 0
        self.startedAt = Date()
    }
}

public final class HuggingFaceService: @unchecked Sendable {
    private let session: URLSession
    private let baseURL: String
    private let lock = NovaMLXLock()
    private var activeTasks: [String: HFDownloadTask] = [:]
    private var downloadTask: Task<Void, Never>?
    private let modelDirectory: URL
    public var onModelDownloaded: ((String) -> Void)?

    public init(modelDirectory: URL, endpoint: String? = nil) {
        self.modelDirectory = modelDirectory
        self.baseURL = endpoint ?? "https://huggingface.co"
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 3600
        self.session = URLSession(configuration: config)
    }

    public func searchModels(query: String, sort: String = "trending", limit: Int = 50, mlxOnly: Bool = true) async throws -> HFSearchResult {
        var components = URLComponents(string: "\(baseURL)/api/models")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "expand", value: "safetensors,downloads,likes,trendingScore"),
        ]
        if !query.isEmpty {
            items.append(URLQueryItem(name: "search", value: query))
        }
        if mlxOnly {
            items.append(URLQueryItem(name: "filter", value: "mlx"))
        }
        components.queryItems = items

        let request = URLRequest(url: components.url!)
        let (data, _) = try await session.data(for: request)
        let models = try JSONDecoder().decode([HFModelInfo].self, from: data)
        return HFSearchResult(models: models, total: models.count)
    }

    public func getModelDetail(repoId: String) async throws -> HFModelDetail {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        let url = URL(string: "\(baseURL)/api/models/\(encoded)")!
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "expand", value: "safetensors")]

        var request = URLRequest(url: components.url!)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(HFModelDetail.self, from: data)
    }

    public func startDownload(repoId: String, hfToken: String? = nil) async throws -> HFDownloadTask {
        var task = HFDownloadTask(repoId: repoId)
        task.status = "downloading"

        let detail = try await getModelDetail(repoId: repoId)
        let safetensorsFiles = (detail.siblings ?? []).filter { $0.rfilename.hasSuffix(".safetensors") }
        let totalSize: Int64 = safetensorsFiles.reduce(0) { $0 + Int64($1.size ?? 0) }
        task.totalBytes = totalSize > 0 ? totalSize : 1

        lock.withLock { activeTasks[task.id] = task }

        let taskCopy = task
        let modelDir = modelDirectory
        let baseURL = self.baseURL
        let session = self.session
        let _ = task.id

        downloadTask = Task {
            await Self.performDownload(
                task: taskCopy,
                modelDir: modelDir,
                baseURL: baseURL,
                session: session,
                hfToken: hfToken,
                lock: lock,
                activeTasks: &activeTasks,
                onProgress: { _ in },
                onModelDownloaded: { [weak self] _ in self?.onModelDownloaded?(repoId) }
            )
        }

        return task
    }

    private static func performDownload(
        task: HFDownloadTask,
        modelDir: URL,
        baseURL: String,
        session: URLSession,
        hfToken: String?,
        lock: NovaMLXLock,
        activeTasks: inout [String: HFDownloadTask],
        onProgress: (HFDownloadTask) -> Void,
        onModelDownloaded: (String) -> Void
    ) async {
        let repoName = task.repoId.components(separatedBy: "/").last ?? task.repoId
        let targetDir = modelDir.appendingPathComponent(repoName)

        var currentTask = task

        do {
            let detail = try await getModelDetailStatic(repoId: task.repoId, baseURL: baseURL, session: session, hfToken: hfToken)
            let files = (detail.siblings ?? []).filter { file in
                file.rfilename.hasSuffix(".safetensors") ||
                file.rfilename.hasSuffix(".json") ||
                file.rfilename.hasSuffix(".model") ||
                file.rfilename.hasSuffix(".txt") ||
                file.rfilename.hasSuffix(".tiktoken") ||
                file.rfilename == "tokenizer.model" ||
                file.rfilename.hasSuffix(".py")
            }

            try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

            for file in files {
                if Task.isCancelled { break }
                let fileURL = URL(string: "\(baseURL)/\(task.repoId)/resolve/main/\(file.rfilename)")!
                var request = URLRequest(url: fileURL)
                if let token = hfToken {
                    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }

                let destPath = targetDir.appendingPathComponent(file.rfilename)
                if let dir = destPath.deletingLastPathComponent().path.removingPercentEncoding {
                    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
                }

                let (tempURL, response) = try await session.download(for: request)
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    if FileManager.default.fileExists(atPath: destPath.path) {
                        try FileManager.default.removeItem(at: destPath)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destPath)
                    if let attrs = try? FileManager.default.attributesOfItem(atPath: destPath.path),
                       let size = attrs[.size] as? Int64 {
                        currentTask.downloadedBytes += size
                        currentTask.progress = min(Double(currentTask.downloadedBytes) / Double(currentTask.totalBytes) * 100, 99.0)
                        lock.withLock { activeTasks[currentTask.id] = currentTask }
                        onProgress(currentTask)
                    }
                }
            }

            currentTask.status = "completed"
            currentTask.progress = 100
            currentTask.completedAt = Date()
            lock.withLock { activeTasks[currentTask.id] = currentTask }
            onModelDownloaded(task.repoId)
        } catch {
            currentTask.status = "failed"
            currentTask.error = error.localizedDescription
            currentTask.completedAt = Date()
            lock.withLock { activeTasks[currentTask.id] = currentTask }
        }
    }

    private static func getModelDetailStatic(repoId: String, baseURL: String, session: URLSession, hfToken: String?) async throws -> HFModelDetail {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        let url = URL(string: "\(baseURL)/api/models/\(encoded)")!
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "expand", value: "safetensors")]

        var request = URLRequest(url: components.url!)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token = hfToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(HFModelDetail.self, from: data)
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
}
