import GRDB
import Foundation

// MARK: - Config Record

public struct ConfigRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "config"
    var id: Int = 1
    var host: String
    var port: Int
    var adminPort: Int
    var tlsEnabled: Bool
    var tlsCertPath: String?
    var tlsKeyPath: String?
    var defaultModel: String?
    var modelsDir: String?
    var hfEndpoint: String
    var authUrl: String?
    var tknetApiKey: String?
    var clusterConfig: String?
    var autoLoad: String?
    var logLevel: String?

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
    var id: String
    var name: String
    var keyHash: String
    var rawKey: String
    var keyPrefix: String
    var keySuffix: String
    var createdAt: Date
    var expiresAt: Date?
    var isEnabled: Bool
    var rateLimitPerSecond: Double?
    var rateLimitBurst: Int?
    var allowedModels: String? // JSON array
    var allowedEndpoints: String? // JSON array
    var maxTokensPerPeriod: Int64?
    var maxRequestsPerPeriod: Int64?
    var usageResetPeriod: String
    var totalTokensUsed: Int64
    var totalRequests: Int64
    var lastUsedAt: Date?
    var periodTokens: Int64
    var periodRequests: Int64
    var periodResetDate: String?
    var perModelTokens: String? // JSON object

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
