import Foundation
import NovaMLXDB
import os.log

/// Encode an optional Codable value to a JSON string. Returns nil if the
/// value is nil or if encoding fails. Used for the Bridge Phase to store
/// complex sub-objects (cluster, autoLoad) in ConfigRecord string columns.
private func encodeJSONString<T: Encodable>(_ value: T?) -> String? {
    guard let value else { return nil }
    guard let data = try? JSONEncoder().encode(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

/// Decode a JSON string back to a Decodable value. Returns `fallback` if
/// the string is nil/empty or if decoding fails. Used in `loadFromStore`
/// during the B3 cutover.
private func decodeJSONString<T: Decodable>(_ value: String?, _ fallback: T) -> T {
    guard let value, let data = value.data(using: .utf8) else { return fallback }
    return (try? JSONDecoder().decode(T.self, from: data)) ?? fallback
}

public actor NovaMLXConfiguration {
    public static let shared = NovaMLXConfiguration()

    private let log = Logger(subsystem: "com.novamlx", category: "Configuration")
    private var _modelsDirectory: URL
    private var _serverConfig: ServerConfig
    private var _defaultModel: String?
    private var _huggingfaceEndpoint: String?

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
        syncToStore()  // Bridge: shadow-write to DB after JSON load
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
        syncToStore()  // Bridge: shadow-write to DB after JSON save
    }

    /// Bridge Phase: shadow-write current state into the SQLite configStore.
    /// JSON remains authoritative; this only populates the DB so future Phase B
    /// cutover tasks can validate reads against it. Errors are tolerated via
    /// `try?` semantics so a DB issue doesn't break the primary JSON code path.
    public func syncToStore() {
        let server = _serverConfig
        let record: ConfigRecord = ConfigRecord(
            host: server.host,
            port: server.port,
            adminPort: server.adminPort,
            tlsEnabled: server.tlsCertPath != nil,
            tlsCertPath: server.tlsCertPath,
            tlsKeyPath: server.tlsKeyPath,
            defaultModel: _defaultModel,
            modelsDir: _modelsDirectory.path,
            hfEndpoint: _huggingfaceEndpoint ?? "https://huggingface.co",
            authUrl: nil,
            tknetApiKey: nil,
            clusterConfig: encodeJSONString(server.cluster),
            autoLoad: encodeJSONString(server.autoLoad),
            logLevel: nil,
            maxConcurrentRequests: server.maxConcurrentRequests,
            requestTimeout: server.requestTimeout,
            contextScalingTarget: server.contextScalingTarget,
            tlsKeyPassword: server.tlsKeyPassword,
            maxRequestSizeMB: server.maxRequestSizeMB,
            maxProcessMemory: server.maxProcessMemory,
            prefixCacheEnabled: server.prefixCacheEnabled
        )
        do {
            try NovaDB.shared.configStore.update { existing in
                existing = record
                existing.id = 1
            }
        } catch {
            // Bridge phase: tolerate DB failures so JSON path stays authoritative.
            log.warning("[Configuration] syncToStore failed (Bridge tolerated): \(String(describing: error))")
        }
    }

    /// Read state from the SQLite configStore into in-memory properties.
    /// Used by Phase B cutover as the JSON replacement. Throws on DB errors
    /// (unlike syncToStore) so callers can decide policy.
    public func loadFromStore() throws {
        let record = try NovaDB.shared.configStore.get()
        let cluster: ServerConfig.ClusterSettings? = decodeJSONString(record.clusterConfig, nil as ServerConfig.ClusterSettings?)
        let autoLoad: AutoLoadConfig = decodeJSONString(record.autoLoad, .init())

        _serverConfig = ServerConfig(
            host: record.host,
            port: record.port,
            adminPort: record.adminPort,
            maxConcurrentRequests: record.maxConcurrentRequests,
            requestTimeout: record.requestTimeout,
            contextScalingTarget: record.contextScalingTarget,
            tlsCertPath: record.tlsCertPath,
            tlsKeyPath: record.tlsKeyPath,
            tlsKeyPassword: record.tlsKeyPassword,
            maxRequestSizeMB: record.maxRequestSizeMB,
            maxProcessMemory: record.maxProcessMemory,
            prefixCacheEnabled: record.prefixCacheEnabled,
            autoLoad: autoLoad,
            cluster: cluster
        )
        if let model = record.defaultModel, !model.isEmpty { _defaultModel = model }
        if let dir = record.modelsDir, !dir.isEmpty { _modelsDirectory = URL(fileURLWithPath: dir) }
        if !record.hfEndpoint.isEmpty { _huggingfaceEndpoint = record.hfEndpoint }
    }

    public var configFileURL: URL {
        NovaMLXPaths.configFile
    }
}

private struct PersistedConfig: Codable {
    let server: ServerConfig
    let defaultModel: String?
    let modelsDirectory: String?
    let huggingfaceEndpoint: String?
    let language: String?
}
