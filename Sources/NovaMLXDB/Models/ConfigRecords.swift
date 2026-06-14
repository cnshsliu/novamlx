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
    public var modelId: String
    public var alias: String?
    public var isDefault: Bool
    public var isPinned: Bool
    public var samplingParams: String? // JSON
    public var ttlSeconds: Int?
    public var contextWindow: Int?
    public var draftModel: String?
    public var updatedAt: Date?

    public init(
        modelId: String,
        alias: String? = nil,
        isDefault: Bool = false,
        isPinned: Bool = false,
        samplingParams: String? = nil,
        ttlSeconds: Int? = nil,
        contextWindow: Int? = nil,
        draftModel: String? = nil,
        updatedAt: Date? = nil
    ) {
        self.modelId = modelId
        self.alias = alias
        self.isDefault = isDefault
        self.isPinned = isPinned
        self.samplingParams = samplingParams
        self.ttlSeconds = ttlSeconds
        self.contextWindow = contextWindow
        self.draftModel = draftModel
        self.updatedAt = updatedAt
    }

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
    public var name: String
    public var endpoint: String
    public var apiKey: String?
    public var remoteModel: String?
    public var isEnabled: Bool
    public var loadBalanceWeight: Double
    public var totalRequests: Int64
    public var totalTokens: Int64
    public var avgLatencyMs: Double?
    public var lastUsedAt: Date?
    public var extraConfig: String?
    // v2 tokenhub expand: fields that bring this record to parity with the
    // JSON-shape `TokenhubProvider` struct (Sources/NovaMLXCore/TokenhubTypes.swift).
    // Non-optional fields carry defaults so the synthesized memberwise init
    // only requires the original v1 args; existing call sites keep compiling.
    //
    // Task 6 (TokenHub cleanup): dropped `is_managed`, `include_in_load_balance`,
    // and `is_local` from both the struct and the runtime schema (v4 migration).
    // Cloud/nova/local distinction is now via the `tags` column. The historical
    // schema definitions in ConfigDBSchema.swift are left untouched as records
    // of what the v1/v2 migrations did.
    public var providerId: String? = nil
    public var tags: String? = nil
    public var isFree: Bool = false
    public var supportsResponsesAPI: Bool = false
    public var supportsVision: Bool = false
    public var visionStrategy: String? = nil
    public var anthropicEndpoint: String? = nil
    public var visionCompanionModel: String? = nil
    public var requestCount: Int = 0
    public var successCount: Int = 0
    public var lastTestedAt: Date? = nil
    public var lastStatus: String? = nil
    public var contextWindowOverride: Int? = nil

    /// Explicit public memberwise initializer (Swift does not synthesize
    /// public inits for public structs). Only the original v1 args are
    /// required; v2 fields all carry defaults so existing v1-era call sites
    /// keep compiling. Callers wanting v2 values assign them post-construction
    /// (e.g. `var rec = TokenhubProviderRecord(...); rec.tags = "[...]"`).
    public init(
        name: String,
        endpoint: String,
        apiKey: String? = nil,
        remoteModel: String? = nil,
        isEnabled: Bool = true,
        loadBalanceWeight: Double = 1.0,
        totalRequests: Int64 = 0,
        totalTokens: Int64 = 0,
        avgLatencyMs: Double? = nil,
        lastUsedAt: Date? = nil,
        extraConfig: String? = nil,
        providerId: String? = nil,
        tags: String? = nil,
        isFree: Bool = false,
        supportsResponsesAPI: Bool = false,
        supportsVision: Bool = false,
        visionStrategy: String? = nil,
        anthropicEndpoint: String? = nil,
        visionCompanionModel: String? = nil,
        requestCount: Int = 0,
        successCount: Int = 0,
        lastTestedAt: Date? = nil,
        lastStatus: String? = nil,
        contextWindowOverride: Int? = nil
    ) {
        self.name = name
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.remoteModel = remoteModel
        self.isEnabled = isEnabled
        self.loadBalanceWeight = loadBalanceWeight
        self.totalRequests = totalRequests
        self.totalTokens = totalTokens
        self.avgLatencyMs = avgLatencyMs
        self.lastUsedAt = lastUsedAt
        self.extraConfig = extraConfig
        self.providerId = providerId
        self.tags = tags
        self.isFree = isFree
        self.supportsResponsesAPI = supportsResponsesAPI
        self.supportsVision = supportsVision
        self.visionStrategy = visionStrategy
        self.anthropicEndpoint = anthropicEndpoint
        self.visionCompanionModel = visionCompanionModel
        self.requestCount = requestCount
        self.successCount = successCount
        self.lastTestedAt = lastTestedAt
        self.lastStatus = lastStatus
        self.contextWindowOverride = contextWindowOverride
    }

    enum CodingKeys: String, CodingKey {
        case name, endpoint
        case apiKey = "api_key"
        case remoteModel = "remote_model"
        case isEnabled = "is_enabled"
        case loadBalanceWeight = "load_balance_weight"
        case totalRequests = "total_requests"
        case totalTokens = "total_tokens"
        case avgLatencyMs = "avg_latency_ms"
        case lastUsedAt = "last_used_at"
        case extraConfig = "extra_config"
        case providerId = "provider_id"
        case tags
        case isFree = "is_free"
        case supportsResponsesAPI = "supports_responses_api"
        case supportsVision = "supports_vision"
        case visionStrategy = "vision_strategy"
        case anthropicEndpoint = "anthropic_endpoint"
        case visionCompanionModel = "vision_companion_model"
        case requestCount = "request_count"
        case successCount = "success_count"
        case lastTestedAt = "last_tested_at"
        case lastStatus = "last_status"
        case contextWindowOverride = "context_window_override"
    }
}

extension TokenhubProviderRecord: FetchableRecord, MutablePersistableRecord {}

// MARK: - Modelfile Record

public struct ModelfileRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "modelfiles"
    public var name: String
    public var baseModel: String?
    public var systemPrompt: String?
    public var parameters: String?
    public var tools: String?
    public var description: String?
    public var createdAt: Date
    public var updatedAt: Date?

    public init(
        name: String,
        baseModel: String? = nil,
        systemPrompt: String? = nil,
        parameters: String? = nil,
        tools: String? = nil,
        description: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date? = nil
    ) {
        self.name = name
        self.baseModel = baseModel
        self.systemPrompt = systemPrompt
        self.parameters = parameters
        self.tools = tools
        self.description = description
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case name
        case baseModel = "base_model"
        case systemPrompt = "system_prompt"
        case parameters, tools
        case description
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

extension ModelfileRecord: FetchableRecord, MutablePersistableRecord {
    public mutating func didInsert(_ inserted: InsertionSuccess) {
        // name is the primary key, no rowid to capture.
    }
}

// MARK: - Auth Session Record

public struct AuthSessionRecord: Codable, Sendable, PersistableRecord {
    public static let databaseTableName = "auth_session"
    public var id: Int = 1
    public var sessionToken: String
    public var authValid: Bool?
    public var authPlan: String?
    public var authStatus: String?
    public var authCancelAtPeriodEnd: Bool?
    public var authExpiresAt: Date?
    public var authCachedAt: Date?
    public var userEmail: String?

    public init(
        id: Int = 1,
        sessionToken: String,
        authValid: Bool? = nil,
        authPlan: String? = nil,
        authStatus: String? = nil,
        authCancelAtPeriodEnd: Bool? = nil,
        authExpiresAt: Date? = nil,
        authCachedAt: Date? = nil,
        userEmail: String? = nil
    ) {
        self.id = id
        self.sessionToken = sessionToken
        self.authValid = authValid
        self.authPlan = authPlan
        self.authStatus = authStatus
        self.authCancelAtPeriodEnd = authCancelAtPeriodEnd
        self.authExpiresAt = authExpiresAt
        self.authCachedAt = authCachedAt
        self.userEmail = userEmail
    }

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
    public var id: Int = 1
    public var policyJSON: String
    public var updatedAt: Date?

    public init(id: Int = 1, policyJSON: String, updatedAt: Date? = nil) {
        self.id = id
        self.policyJSON = policyJSON
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case id
        case policyJSON = "policy_json"
        case updatedAt = "updated_at"
    }
}

extension ClusterPolicyRecord: FetchableRecord, MutablePersistableRecord {}
