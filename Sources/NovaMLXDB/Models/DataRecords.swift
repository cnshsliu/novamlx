import GRDB
import Foundation

// MARK: - Model Registry Record

public struct ModelRegistryRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "model_registry"
    public var modelId: String
    public var family: String?
    public var modelType: String?
    public var source: String?
    public var localPath: String?
    public var remoteUrl: String?
    public var sizeBytes: Int64?
    public var downloadedAt: Date?
    public var version: String?
    public var architecture: String?

    public init(
        modelId: String,
        family: String? = nil,
        modelType: String? = nil,
        source: String? = nil,
        localPath: String? = nil,
        remoteUrl: String? = nil,
        sizeBytes: Int64? = nil,
        downloadedAt: Date? = nil,
        version: String? = nil,
        architecture: String? = nil
    ) {
        self.modelId = modelId
        self.family = family
        self.modelType = modelType
        self.source = source
        self.localPath = localPath
        self.remoteUrl = remoteUrl
        self.sizeBytes = sizeBytes
        self.downloadedAt = downloadedAt
        self.version = version
        self.architecture = architecture
    }

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case family
        case modelType = "model_type"
        case source
        case localPath = "local_path"
        case remoteUrl = "remote_url"
        case sizeBytes = "size_bytes"
        case downloadedAt = "downloaded_at"
        case version, architecture
    }
}

extension ModelRegistryRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Loaded Model Record

public struct LoadedModelRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "loaded_models"
    public var modelId: String
    public var loadedAt: Date

    public init(modelId: String, loadedAt: Date = Date()) {
        self.modelId = modelId
        self.loadedAt = loadedAt
    }

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case loadedAt = "loaded_at"
    }
}

extension LoadedModelRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Metrics Record

public struct MetricsRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "metrics"
    var id: Int = 1
    var totalRequests: Int64
    var totalTokens: Int64
    var totalInferenceTimeMs: Int64
    var cacheHits: Int64
    var cacheMisses: Int64
    var evictions: Int64
    var perModelStats: String?
    var perModelCache: String?
    var updatedAt: Date?

    enum CodingKeys: String, CodingKey {
        case id
        case totalRequests = "total_requests"
        case totalTokens = "total_tokens"
        case totalInferenceTimeMs = "total_inference_time_ms"
        case cacheHits = "cache_hits"
        case cacheMisses = "cache_misses"
        case evictions
        case perModelStats = "per_model_stats"
        case perModelCache = "per_model_cache"
        case updatedAt = "updated_at"
    }
}

extension MetricsRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Worker Deployment Record

public struct WorkerDeploymentRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "worker_deployments"
    var hostname: String
    var phase: String
    var username: String?
    var version: String?
    var startedAt: Date?
    var updatedAt: Date?
    var extraJson: String?

    enum CodingKeys: String, CodingKey {
        case hostname, phase, username, version
        case startedAt = "started_at"
        case updatedAt = "updated_at"
        case extraJson = "extra_json"
    }
}

extension WorkerDeploymentRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Chat Record

public struct ChatRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "chats"
    var id: String
    var title: String?
    var model: String
    var systemPrompt: String?
    var createdAt: Date
    var updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case id, title, model
        case systemPrompt = "system_prompt"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

extension ChatRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Chat Message Record

public struct ChatMessageRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "chat_messages"
    var id: String
    var chatId: String
    var role: String
    var content: String?
    var thinkingContent: String?
    var createdAt: Date
    var sortOrder: Int

    enum CodingKeys: String, CodingKey {
        case id
        case chatId = "chat_id"
        case role, content
        case thinkingContent = "thinking_content"
        case createdAt = "created_at"
        case sortOrder = "sort_order"
    }
}

extension ChatMessageRecord: FetchableRecord, MutablePersistableRecord {}
