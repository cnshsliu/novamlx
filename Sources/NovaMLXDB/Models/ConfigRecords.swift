import GRDB
import Foundation

// MARK: - Config Record

public struct ConfigRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "config"
    public var id: Int = 1
    public var host: String
    public var port: Int
    public var adminPort: Int
    public var tlsEnabled: Bool
    public var tlsCertPath: String?
    public var tlsKeyPath: String?
    public var defaultModel: String?
    public var modelsDir: String?
    public var hfEndpoint: String
    public var authUrl: String?
    public var tknetApiKey: String?
    public var clusterConfig: String?
    public var autoLoad: String?
    public var logLevel: String?
    // v2: server fields that fully back ServerConfig (Types.swift).
    // Defaults satisfy memberwise init sites in ConfigStore without an
    // explicit init declaration.
    public var maxConcurrentRequests: Int = 16
    public var requestTimeout: Double = 300
    public var contextScalingTarget: Int? = nil
    public var tlsKeyPassword: String? = nil
    public var maxRequestSizeMB: Double = 100
    public var maxProcessMemory: String = "auto"
    public var prefixCacheEnabled: Bool = true

    /// Explicit public memberwise initializer so cross-module callers
    /// (NovaMLXCore/Configuration.swift) can construct records. The
    /// synthesized memberwise init is internal-only for public structs.
    public init(
        id: Int = 1,
        host: String,
        port: Int,
        adminPort: Int,
        tlsEnabled: Bool,
        tlsCertPath: String? = nil,
        tlsKeyPath: String? = nil,
        defaultModel: String? = nil,
        modelsDir: String? = nil,
        hfEndpoint: String,
        authUrl: String? = nil,
        tknetApiKey: String? = nil,
        clusterConfig: String? = nil,
        autoLoad: String? = nil,
        logLevel: String? = nil,
        maxConcurrentRequests: Int = 16,
        requestTimeout: Double = 300,
        contextScalingTarget: Int? = nil,
        tlsKeyPassword: String? = nil,
        maxRequestSizeMB: Double = 100,
        maxProcessMemory: String = "auto",
        prefixCacheEnabled: Bool = true
    ) {
        self.id = id
        self.host = host
        self.port = port
        self.adminPort = adminPort
        self.tlsEnabled = tlsEnabled
        self.tlsCertPath = tlsCertPath
        self.tlsKeyPath = tlsKeyPath
        self.defaultModel = defaultModel
        self.modelsDir = modelsDir
        self.hfEndpoint = hfEndpoint
        self.authUrl = authUrl
        self.tknetApiKey = tknetApiKey
        self.clusterConfig = clusterConfig
        self.autoLoad = autoLoad
        self.logLevel = logLevel
        self.maxConcurrentRequests = maxConcurrentRequests
        self.requestTimeout = requestTimeout
        self.contextScalingTarget = contextScalingTarget
        self.tlsKeyPassword = tlsKeyPassword
        self.maxRequestSizeMB = maxRequestSizeMB
        self.maxProcessMemory = maxProcessMemory
        self.prefixCacheEnabled = prefixCacheEnabled
    }

    enum CodingKeys: String, CodingKey {
        case id, host, port
        case adminPort = "admin_port"
        case tlsEnabled = "tls_enabled"
        case tlsCertPath = "tls_cert_path"
        case tlsKeyPath = "tls_key_path"
        case defaultModel = "default_model"
        case modelsDir = "models_dir"
        case hfEndpoint = "hf_endpoint"
        case authUrl = "auth_url"
        case tknetApiKey = "tknet_api_key"
        case clusterConfig = "cluster_config"
        case autoLoad = "auto_load"
        case logLevel = "log_level"
        case maxConcurrentRequests = "max_concurrent_requests"
        case requestTimeout = "request_timeout"
        case contextScalingTarget = "context_scaling_target"
        case tlsKeyPassword = "tls_key_password"
        case maxRequestSizeMB = "max_request_size_mb"
        case maxProcessMemory = "max_process_memory"
        case prefixCacheEnabled = "prefix_cache_enabled"
    }
}

extension ConfigRecord: FetchableRecord, MutablePersistableRecord {
    public mutating func didInsert(_ inserted: InsertionSuccess) {
        id = Int(inserted.rowID)
    }
}

// MARK: - API Key Record

public struct APIKeyRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "api_keys"
    public var id: String
    public var name: String
    public var keyHash: String
    public var rawKey: String
    public var keyPrefix: String
    public var keySuffix: String
    public var createdAt: Date
    public var expiresAt: Date?
    public var isEnabled: Bool
    public var rateLimitPerSecond: Double?
    public var rateLimitBurst: Int?
    public var allowedModels: String? // JSON array
    public var allowedEndpoints: String? // JSON array
    public var maxTokensPerPeriod: Int64?
    public var maxRequestsPerPeriod: Int64?
    public var usageResetPeriod: String
    public var totalTokensUsed: Int64
    public var totalRequests: Int64
    public var lastUsedAt: Date?
    public var periodTokens: Int64
    public var periodRequests: Int64
    public var periodResetDate: String?
    public var perModelTokens: String? // JSON object

    public init(
        id: String,
        name: String,
        keyHash: String,
        rawKey: String,
        keyPrefix: String,
        keySuffix: String,
        createdAt: Date,
        expiresAt: Date? = nil,
        isEnabled: Bool = true,
        rateLimitPerSecond: Double? = nil,
        rateLimitBurst: Int? = nil,
        allowedModels: String? = nil,
        allowedEndpoints: String? = nil,
        maxTokensPerPeriod: Int64? = nil,
        maxRequestsPerPeriod: Int64? = nil,
        usageResetPeriod: String = "daily",
        totalTokensUsed: Int64 = 0,
        totalRequests: Int64 = 0,
        lastUsedAt: Date? = nil,
        periodTokens: Int64 = 0,
        periodRequests: Int64 = 0,
        periodResetDate: String? = nil,
        perModelTokens: String? = nil
    ) {
        self.id = id
        self.name = name
        self.keyHash = keyHash
        self.rawKey = rawKey
        self.keyPrefix = keyPrefix
        self.keySuffix = keySuffix
        self.createdAt = createdAt
        self.expiresAt = expiresAt
        self.isEnabled = isEnabled
        self.rateLimitPerSecond = rateLimitPerSecond
        self.rateLimitBurst = rateLimitBurst
        self.allowedModels = allowedModels
        self.allowedEndpoints = allowedEndpoints
        self.maxTokensPerPeriod = maxTokensPerPeriod
        self.maxRequestsPerPeriod = maxRequestsPerPeriod
        self.usageResetPeriod = usageResetPeriod
        self.totalTokensUsed = totalTokensUsed
        self.totalRequests = totalRequests
        self.lastUsedAt = lastUsedAt
        self.periodTokens = periodTokens
        self.periodRequests = periodRequests
        self.periodResetDate = periodResetDate
        self.perModelTokens = perModelTokens
    }

    enum CodingKeys: String, CodingKey {
        case id, name
        case keyHash = "key_hash"
        case rawKey = "raw_key"
        case keyPrefix = "key_prefix"
        case keySuffix = "key_suffix"
        case createdAt = "created_at"
        case expiresAt = "expires_at"
        case isEnabled = "is_enabled"
        case rateLimitPerSecond = "rate_limit_per_second"
        case rateLimitBurst = "rate_limit_burst"
        case allowedModels = "allowed_models"
        case allowedEndpoints = "allowed_endpoints"
        case maxTokensPerPeriod = "max_tokens_per_period"
        case maxRequestsPerPeriod = "max_requests_per_period"
        case usageResetPeriod = "usage_reset_period"
        case totalTokensUsed = "total_tokens_used"
        case totalRequests = "total_requests"
        case lastUsedAt = "last_used_at"
        case periodTokens = "period_tokens"
        case periodRequests = "period_requests"
        case periodResetDate = "period_reset_date"
        case perModelTokens = "per_model_tokens"
    }
}

extension APIKeyRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Model Settings Record

public struct ModelSettingsRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "model_settings"
    var modelId: String
    var alias: String?
    var isDefault: Bool
    var isPinned: Bool
    var samplingParams: String? // JSON
    var ttlSeconds: Int?
    var contextWindow: Int?
    var draftModel: String?
    var updatedAt: Date?

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case alias
        case isDefault = "is_default"
        case isPinned = "is_pinned"
        case samplingParams = "sampling_params"
        case ttlSeconds = "ttl_seconds"
        case contextWindow = "context_window"
        case draftModel = "draft_model"
        case updatedAt = "updated_at"
    }
}

extension ModelSettingsRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Tokenhub Provider Record

public struct TokenhubProviderRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "tokenhub_providers"
    var name: String
    var endpoint: String
    var apiKey: String?
    var remoteModel: String?
    var isEnabled: Bool
    var isManaged: Bool
    var loadBalanceWeight: Double
    var totalRequests: Int64
    var totalTokens: Int64
    var avgLatencyMs: Double?
    var lastUsedAt: Date?
    var extraConfig: String?

    enum CodingKeys: String, CodingKey {
        case name, endpoint
        case apiKey = "api_key"
        case remoteModel = "remote_model"
        case isEnabled = "is_enabled"
        case isManaged = "is_managed"
        case loadBalanceWeight = "load_balance_weight"
        case totalRequests = "total_requests"
        case totalTokens = "total_tokens"
        case avgLatencyMs = "avg_latency_ms"
        case lastUsedAt = "last_used_at"
        case extraConfig = "extra_config"
    }
}

extension TokenhubProviderRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Modelfile Record

public struct ModelfileRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "modelfiles"
    var name: String
    var baseModel: String?
    var systemPrompt: String?
    var parameters: String?
    var tools: String?
    var createdAt: Date
    var updatedAt: Date?

    enum CodingKeys: String, CodingKey {
        case name
        case baseModel = "base_model"
        case systemPrompt = "system_prompt"
        case parameters, tools
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

extension ModelfileRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Auth Session Record

public struct AuthSessionRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "auth_session"
    var id: Int = 1
    var sessionToken: String
    var authValid: Bool?
    var authPlan: String?
    var authStatus: String?
    var authCancelAtPeriodEnd: Bool?
    var authExpiresAt: Date?
    var authCachedAt: Date?
    var userEmail: String?

    enum CodingKeys: String, CodingKey {
        case id
        case sessionToken = "session_token"
        case authValid = "auth_valid"
        case authPlan = "auth_plan"
        case authStatus = "auth_status"
        case authCancelAtPeriodEnd = "auth_cancel_at_period_end"
        case authExpiresAt = "auth_expires_at"
        case authCachedAt = "auth_cached_at"
        case userEmail = "user_email"
    }
}

extension AuthSessionRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Cluster Policy Record

public struct ClusterPolicyRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "cluster_policy"
    var id: Int = 1
    var policyJSON: String
    var updatedAt: Date?

    enum CodingKeys: String, CodingKey {
        case id
        case policyJSON = "policy_json"
        case updatedAt = "updated_at"
    }
}

extension ClusterPolicyRecord: FetchableRecord, MutablePersistableRecord {}
