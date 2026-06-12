import Foundation
import os.log

public actor NovaMLXConfiguration {
    public static let shared = NovaMLXConfiguration()

    private let log = Logger(subsystem: "com.novamlx", category: "Configuration")
    private var _modelsDirectory: URL
    private var _serverConfig: ServerConfig
    private var _defaultModel: String?
    private var _huggingfaceEndpoint: String?
    private var _apiKeys: [APIKey] = []
    private var _apiKeysMigrated = false

    private init() {
        _modelsDirectory = NovaMLXPaths.modelsDir
        _serverConfig = ServerConfig()
    }

    public var modelsDirectory: URL {
        get async { _modelsDirectory }
    }

    public var serverConfig: ServerConfig {
        get async { _serverConfig }
    }

    public var defaultModel: String? {
        get async { _defaultModel }
    }

    public var huggingfaceEndpoint: String? {
        get async { _huggingfaceEndpoint }
    }

    public var apiKeys: [APIKey] {
        get async { _apiKeys }
    }

    public func setModelsDirectory(_ url: URL) {
        _modelsDirectory = url
    }

    public func setServerConfig(_ config: ServerConfig) {
        _serverConfig = config
    }

    public func setDefaultModel(_ model: String?) {
        _defaultModel = model
    }

    public func setHuggingfaceEndpoint(_ endpoint: String?) {
        _huggingfaceEndpoint = endpoint
    }

    public func initializeDirectories() throws {
        let fm = FileManager.default
        try fm.createDirectory(at: _modelsDirectory, withIntermediateDirectories: true)
    }

    public func loadFromFile(_ url: URL) throws {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(PersistedConfig.self, from: data)
        _serverConfig = config.server
        _defaultModel = config.defaultModel
        if let modelsDir = config.modelsDirectory {
            _modelsDirectory = URL(fileURLWithPath: modelsDir)
        }
        _huggingfaceEndpoint = config.huggingfaceEndpoint
    }

    public func saveToFile(_ url: URL) throws {
        let config = PersistedConfig(
            server: _serverConfig,
            defaultModel: _defaultModel,
            modelsDirectory: _modelsDirectory.path,
            huggingfaceEndpoint: _huggingfaceEndpoint,
            language: nil
        )
        let data = try JSONEncoder().encode(config)
        try data.write(to: url, options: .atomic)
    }

    public func updateApiKeys(_ keys: [String], file url: URL) throws {
        _serverConfig = ServerConfig(
            host: _serverConfig.host,
            port: _serverConfig.port,
            adminPort: _serverConfig.adminPort,
            apiKeys: keys,
            maxConcurrentRequests: _serverConfig.maxConcurrentRequests,
            requestTimeout: _serverConfig.requestTimeout,
            contextScalingTarget: _serverConfig.contextScalingTarget,
            tlsCertPath: _serverConfig.tlsCertPath,
            tlsKeyPath: _serverConfig.tlsKeyPath,
            tlsKeyPassword: _serverConfig.tlsKeyPassword,
            maxRequestSizeMB: _serverConfig.maxRequestSizeMB,
            maxProcessMemory: _serverConfig.maxProcessMemory,
            prefixCacheEnabled: _serverConfig.prefixCacheEnabled
        )
        try saveToFile(url)
    }

    public var configFileURL: URL {
        NovaMLXPaths.configFile
    }
}

// MARK: - API Key Store

extension NovaMLXConfiguration {

    /// Load API keys from `api_keys.json`, migrating from flat config if needed.
    public func loadAPIKeys() {
        let file = NovaMLXPaths.apiKeysFile
        let fm = FileManager.default

        if fm.fileExists(atPath: file.path) {
            do {
                let data = try Data(contentsOf: file)
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let keys = try decoder.decode([APIKey].self, from: data)
                _apiKeys = keys
                log.info("[APIKeys] Loaded \(keys.count) keys from \(file.path)")
            } catch {
                log.error("[APIKeys] Failed to load \(file.path): \(error)")
                _apiKeys = []
            }
        } else {
            // Migration: convert flat apiKeys from config.json
            migrateFlatKeys()
        }
    }

    /// Save current keys to disk.
    public func saveAPIKeys() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(_apiKeys)
        try data.write(to: NovaMLXPaths.apiKeysFile, options: .atomic)
    }

    /// Create a new API key. Returns the raw key string (shown only once).
    @discardableResult
    public func createAPIKey(name: String, rateLimitPerSecond: Double? = nil, rateLimitBurst: Int? = nil, allowedModels: [String]? = nil, allowedEndpoints: [String]? = nil, maxTokensPerPeriod: Int64? = nil, maxRequestsPerPeriod: Int64? = nil, usageResetPeriod: UsageResetPeriod = .daily) throws -> (apiKey: APIKey, rawKey: String) {
        let raw = APIKey.generateRawKey()
        let hash = APIKey.hashRawKey(raw)
        let prefix = String(raw.prefix(19))
        let suffix = String(raw.suffix(4))

        let key = APIKey(
            name: name,
            keyHash: hash,
            keyPrefix: prefix,
            keySuffix: suffix,
            rateLimitPerSecond: rateLimitPerSecond,
            rateLimitBurst: rateLimitBurst,
            allowedModels: allowedModels,
            allowedEndpoints: allowedEndpoints,
            maxTokensPerPeriod: maxTokensPerPeriod,
            maxRequestsPerPeriod: maxRequestsPerPeriod,
            usageResetPeriod: usageResetPeriod
        )
        _apiKeys.append(key)
        try saveAPIKeys()
        log.info("[APIKeys] Created key '\(name)' (\(prefix)...\(suffix))")
        return (key, raw)
    }

    /// Look up a key by its raw value (hashes and searches).
    public func findAPIKeyByRaw(_ raw: String) -> APIKey? {
        let hash = APIKey.hashRawKey(raw)
        return _apiKeys.first { $0.keyHash == hash }
    }

    /// Look up a key by ID.
    public func findAPIKeyById(_ id: String) -> APIKey? {
        _apiKeys.first { $0.id == id }
    }

    /// Update a key's mutable fields.
    public func updateAPIKey(id: String, _ updates: @Sendable (inout APIKey) -> Void) throws {
        guard let idx = _apiKeys.firstIndex(where: { $0.id == id }) else {
            throw NovaMLXError.apiError("API key not found: \(id)")
        }
        updates(&_apiKeys[idx])
        try saveAPIKeys()
    }

    /// Delete a key by ID.
    public func deleteAPIKey(id: String) throws {
        _apiKeys.removeAll { $0.id == id }
        try saveAPIKeys()
        log.info("[APIKeys] Deleted key \(id)")
    }

    /// Rotate a key — returns new raw key (shown only once).
    @discardableResult
    public func rotateAPIKey(id: String) throws -> (apiKey: APIKey, rawKey: String) {
        guard let idx = _apiKeys.firstIndex(where: { $0.id == id }) else {
            throw NovaMLXError.apiError("API key not found: \(id)")
        }
        let raw = APIKey.generateRawKey()
        let hash = APIKey.hashRawKey(raw)
        let prefix = String(raw.prefix(19))
        let suffix = String(raw.suffix(4))
        let old = _apiKeys[idx]
        _apiKeys[idx] = APIKey(
            id: old.id,
            name: old.name,
            keyHash: hash,
            keyPrefix: prefix,
            keySuffix: suffix,
            createdAt: old.createdAt,
            expiresAt: old.expiresAt,
            isEnabled: old.isEnabled,
            rateLimitPerSecond: old.rateLimitPerSecond,
            rateLimitBurst: old.rateLimitBurst,
            allowedModels: old.allowedModels,
            allowedEndpoints: old.allowedEndpoints,
            maxTokensPerPeriod: old.maxTokensPerPeriod,
            maxRequestsPerPeriod: old.maxRequestsPerPeriod,
            usageResetPeriod: old.usageResetPeriod,
            usage: old.usage
        )
        try saveAPIKeys()
        log.info("[APIKeys] Rotated key '\(old.name)' (\(prefix))")
        return (_apiKeys[idx], raw)
    }

    /// Record usage for a key, including model breakdown.
    public func recordUsage(keyId: String, tokens: Int64, model: String? = nil) throws {
        guard let idx = _apiKeys.firstIndex(where: { $0.id == keyId }) else { return }
        let periodKey = Self.periodDate(for: _apiKeys[idx].usageResetPeriod)

        // Reset period counters if period changed
        if _apiKeys[idx].usage.periodResetDate != periodKey {
            _apiKeys[idx].usage.periodTokens = 0
            _apiKeys[idx].usage.periodRequests = 0
            _apiKeys[idx].usage.periodResetDate = periodKey
        }

        _apiKeys[idx].usage.totalTokensUsed += tokens
        _apiKeys[idx].usage.totalRequests += 1
        _apiKeys[idx].usage.periodTokens += tokens
        _apiKeys[idx].usage.periodRequests += 1
        _apiKeys[idx].usage.lastUsedAt = Date()

        // Per-model breakdown
        if let model {
            _apiKeys[idx].usage.perModelTokens[model, default: 0] += tokens
        }

        try saveAPIKeys()
    }

    /// Check if a key has exceeded its period limits.
    public func isWithinLimits(keyId: String) -> Bool {
        guard let key = _apiKeys.first(where: { $0.id == keyId }) else { return false }

        // "never" means no limits
        if key.usageResetPeriod == .never { return true }

        let periodKey = Self.periodDate(for: key.usageResetPeriod)
        var periodTokens = key.usage.periodTokens
        var periodRequests = key.usage.periodRequests
        if key.usage.periodResetDate != periodKey {
            periodTokens = 0
            periodRequests = 0
        }

        if let maxTokens = key.maxTokensPerPeriod, periodTokens >= maxTokens { return false }
        if let maxRequests = key.maxRequestsPerPeriod, periodRequests >= maxRequests { return false }
        return true
    }

    /// Get the period usage as a fraction (0.0 - 1.0) for progress display.
    public func periodUsageFraction(keyId: String) -> Double {
        guard let key = _apiKeys.first(where: { $0.id == keyId }) else { return 0 }
        guard let max = key.maxTokensPerPeriod, max > 0 else { return 0 }

        let periodKey = Self.periodDate(for: key.usageResetPeriod)
        var tokens = key.usage.periodTokens
        if key.usage.periodResetDate != periodKey { tokens = 0 }

        return min(1.0, Double(tokens) / Double(max))
    }

    // MARK: - Migration

    private func migrateFlatKeys() {
        let flatKeys = _serverConfig.apiKeys
        guard !flatKeys.isEmpty else {
            _apiKeys = []
            return
        }

        log.info("[APIKeys] Migrating \(flatKeys.count) flat keys from config.json → api_keys.json")
        _apiKeys = flatKeys.map { raw in
            let hash = APIKey.hashRawKey(raw)
            let prefix = String(raw.prefix(19))
            let suffix = String(raw.suffix(4))
            return APIKey(
                name: "Migrated (\(prefix)...\(suffix))",
                keyHash: hash,
                keyPrefix: prefix,
                keySuffix: suffix
            )
        }

        do {
            try saveAPIKeys()
            // Clear flat keys from config.json after successful migration
            let current = _serverConfig
            _serverConfig = ServerConfig(
                host: current.host,
                port: current.port,
                adminPort: current.adminPort,
                apiKeys: [],
                maxConcurrentRequests: current.maxConcurrentRequests,
                requestTimeout: current.requestTimeout,
                contextScalingTarget: current.contextScalingTarget,
                tlsCertPath: current.tlsCertPath,
                tlsKeyPath: current.tlsKeyPath,
                tlsKeyPassword: current.tlsKeyPassword,
                maxRequestSizeMB: current.maxRequestSizeMB,
                maxProcessMemory: current.maxProcessMemory,
                prefixCacheEnabled: current.prefixCacheEnabled,
                autoLoad: current.autoLoad,
                cluster: current.cluster
            )
            try saveToFile(configFileURL)
            log.info("[APIKeys] Migration complete, flat keys cleared from config.json")
        } catch {
            log.error("[APIKeys] Migration failed: \(error)")
        }
    }

    private static func periodDate(for period: UsageResetPeriod) -> String {
        let fmt = DateFormatter()
        let date = Date()
        switch period {
        case .daily:
            fmt.dateFormat = "yyyy-MM-dd"
        case .weekly:
            fmt.dateFormat = "yyyy-ww"
        case .monthly:
            fmt.dateFormat = "yyyy-MM"
        case .never:
            return "never"
        }
        return fmt.string(from: date)
    }
}

private struct PersistedConfig: Codable {
    let server: ServerConfig
    let defaultModel: String?
    let modelsDirectory: String?
    let huggingfaceEndpoint: String?
    let language: String?
}
