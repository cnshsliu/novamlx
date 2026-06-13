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
    public var id: Int = 1
    public var totalRequests: Int64
    public var totalTokens: Int64
    public var totalInferenceTimeMs: Int64
    public var cacheHits: Int64
    public var cacheMisses: Int64
    public var evictions: Int64
    public var perModelStats: String?
    public var perModelCache: String?
    public var updatedAt: Date?
    // v2 expand: 4 columns added to match PersistentMetrics exactly.
    public var modelsLoaded: Int64 = 0
    public var modelsUnloaded: Int64 = 0
    public var ttlEvictions: Int64 = 0
    public var memoryPressureEvictions: Int64 = 0

    public init(
        id: Int = 1,
        totalRequests: Int64,
        totalTokens: Int64,
        totalInferenceTimeMs: Int64,
        cacheHits: Int64,
        cacheMisses: Int64,
        evictions: Int64,
        perModelStats: String? = "{}",
        perModelCache: String? = "{}",
        updatedAt: Date? = nil,
        modelsLoaded: Int64 = 0,
        modelsUnloaded: Int64 = 0,
        ttlEvictions: Int64 = 0,
        memoryPressureEvictions: Int64 = 0
    ) {
        self.id = id
        self.totalRequests = totalRequests
        self.totalTokens = totalTokens
        self.totalInferenceTimeMs = totalInferenceTimeMs
        self.cacheHits = cacheHits
        self.cacheMisses = cacheMisses
        self.evictions = evictions
        self.perModelStats = perModelStats
        self.perModelCache = perModelCache
        self.updatedAt = updatedAt
        self.modelsLoaded = modelsLoaded
        self.modelsUnloaded = modelsUnloaded
        self.ttlEvictions = ttlEvictions
        self.memoryPressureEvictions = memoryPressureEvictions
    }

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
        case modelsLoaded = "models_loaded"
        case modelsUnloaded = "models_unloaded"
        case ttlEvictions = "ttl_evictions"
        case memoryPressureEvictions = "memory_pressure_evictions"
    }
}

extension MetricsRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Worker Deployment Record

public struct WorkerDeploymentRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "worker_deployments"
    public var hostname: String
    public var phase: String
    public var username: String?
    public var version: String?
    public var startedAt: Date?
    public var updatedAt: Date?
    public var extraJson: String?

    public init(
        hostname: String,
        phase: String,
        username: String? = nil,
        version: String? = nil,
        startedAt: Date? = nil,
        updatedAt: Date? = nil,
        extraJson: String? = nil
    ) {
        self.hostname = hostname
        self.phase = phase
        self.username = username
        self.version = version
        self.startedAt = startedAt
        self.updatedAt = updatedAt
        self.extraJson = extraJson
    }

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
    public var id: String
    public var title: String?
    public var model: String
    public var systemPrompt: String?
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String,
        title: String? = nil,
        model: String,
        systemPrompt: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date? = nil
    ) {
        self.id = id
        self.title = title
        self.model = model
        self.systemPrompt = systemPrompt
        self.createdAt = createdAt
        self.updatedAt = updatedAt ?? createdAt
    }

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
    public var id: String
    public var chatId: String
    public var role: String
    public var content: String?
    public var thinkingContent: String?
    public var createdAt: Date
    public var sortOrder: Int

    public init(
        id: String,
        chatId: String,
        role: String,
        content: String? = nil,
        thinkingContent: String? = nil,
        createdAt: Date = Date(),
        sortOrder: Int = 0
    ) {
        self.id = id
        self.chatId = chatId
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.createdAt = createdAt
        self.sortOrder = sortOrder
    }

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
