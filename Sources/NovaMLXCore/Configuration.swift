import Foundation

public actor NovaMLXConfiguration {
    public static let shared = NovaMLXConfiguration()

    private var _modelsDirectory: URL
    private var _cacheDirectory: URL
    private var _serverConfig: ServerConfig
    private var _defaultModel: String?

    private init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let base = appSupport.appendingPathComponent("NovaMLX", isDirectory: true)
        _modelsDirectory = base.appendingPathComponent("models", isDirectory: true)
        _cacheDirectory = base.appendingPathComponent("cache", isDirectory: true)
        _serverConfig = ServerConfig()
    }

    public var modelsDirectory: URL {
        get async { _modelsDirectory }
    }

    public var cacheDirectory: URL {
        get async { _cacheDirectory }
    }

    public var serverConfig: ServerConfig {
        get async { _serverConfig }
    }

    public var defaultModel: String? {
        get async { _defaultModel }
    }

    public func setModelsDirectory(_ url: URL) {
        _modelsDirectory = url
    }

    public func setCacheDirectory(_ url: URL) {
        _cacheDirectory = url
    }

    public func setServerConfig(_ config: ServerConfig) {
        _serverConfig = config
    }

    public func setDefaultModel(_ model: String?) {
        _defaultModel = model
    }

    public func initializeDirectories() throws {
        let fm = FileManager.default
        try fm.createDirectory(at: _modelsDirectory, withIntermediateDirectories: true)
        try fm.createDirectory(at: _cacheDirectory, withIntermediateDirectories: true)
    }

    public func loadFromFile(_ url: URL) throws {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(PersistedConfig.self, from: data)
        _serverConfig = config.server
        _defaultModel = config.defaultModel
        if let modelsDir = config.modelsDirectory {
            _modelsDirectory = URL(fileURLWithPath: modelsDir)
        }
    }

    public func saveToFile(_ url: URL) throws {
        let config = PersistedConfig(
            server: _serverConfig,
            defaultModel: _defaultModel,
            modelsDirectory: _modelsDirectory.path
        )
        let data = try JSONEncoder().encode(config)
        try data.write(to: url, options: .atomic)
    }
}

private struct PersistedConfig: Codable {
    let server: ServerConfig
    let defaultModel: String?
    let modelsDirectory: String?
}
