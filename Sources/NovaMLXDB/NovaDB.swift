import Foundation
import GRDB
import Logging

public final class NovaDB: @unchecked Sendable {
    public static let shared = NovaDB()

    public private(set) var configDB: DatabasePool!
    public private(set) var dataDB: DatabasePool!

    public private(set) var apiKeyStore: APIKeyStore!
    public private(set) var configStore: ConfigStore!
    public private(set) var modelSettingsStore: ModelSettingsStore!
    public private(set) var tokenhubStore: TokenhubStore!
    public private(set) var modelfileStore: ModelfileStore!
    public private(set) var authStore: AuthStore!
    public private(set) var clusterPolicyStore: ClusterPolicyStore!
    public private(set) var modelRegistryStore: ModelRegistryStore!
    public private(set) var loadedModelsStore: LoadedModelsStore!
    public private(set) var metricsStore: MetricsDBStore!
    public private(set) var chatStore: ChatStore!
    public private(set) var workerDeploymentStore: WorkerDeploymentStore!

    private let log = Logger(label: "NovaMLXDB")
    private let queue = DispatchQueue(label: "com.novamlx.db.setup")
    private var _isSetup = false

    private init() {}

    /// Whether `setup(baseDir:)` has successfully completed at least once.
    public var isSetup: Bool { queue.sync { _isSetup } }

    /// Idempotent setup. Subsequent calls are no-ops (the existing DBs are
    /// preserved). This lets eager property initializers in callers (e.g.
    /// `MLXEngine` constructed as a stored property on AppDelegate) trigger
    /// setup without coordinating with the init-body's explicit setup call.
    public func setup(baseDir: URL) throws {
        try queue.sync {
            guard !_isSetup else { return }
            let fm = FileManager.default
            try fm.createDirectory(at: baseDir, withIntermediateDirectories: true)

            let configDBURL = baseDir.appendingPathComponent("nova_config.db")
            let dataDBURL = baseDir.appendingPathComponent("nova_data.db")

            var config = Configuration()
            config.prepareDatabase { db in
                try db.execute(sql: "PRAGMA journal_mode=WAL")
                try db.execute(sql: "PRAGMA synchronous=NORMAL")
                try db.execute(sql: "PRAGMA foreign_keys=ON")
            }

            self.configDB = try DatabasePool(path: configDBURL.path, configuration: config)
            self.dataDB = try DatabasePool(path: dataDBURL.path, configuration: config)

            log.info("[NovaDB] Opened databases at \(baseDir.path)")

            try runMigrations()
            initStores()
            try importLegacyJSON(baseDir: baseDir)
            cleanupOrphanedLegacyFiles(baseDir: baseDir)
            _isSetup = true
        }
    }

    /// Rename any legacy JSON file that is still on disk but whose store
    /// already has data (i.e. the import was skipped because the table had
    /// rows from a prior run). This is the post-cutover cleanup pass — the
    /// file is inert garbage and can be safely moved aside.
    ///
    /// Called after `importLegacyJSON`. Safe to call repeatedly.
    private func cleanupOrphanedLegacyFiles(baseDir: URL) {
        let fm = FileManager.default
        // Only files NovaDB itself imports are safe to clean up here. Files
        // owned by their manager (MetricsStore, ModelSettingsManager,
        // AuthClient, WorkerDeployer, etc.) handle their own rename inside
        // their importer; touching them here would race and lose data.
        let novaDBOwned: [(file: String, table: String, db: DatabasePool)] = [
            ("config.json", "config", configDB),
            ("loaded_models.json", "loaded_models", dataDB),
            ("cluster-policy.json", "cluster_policy", configDB)
        ]
        for entry in novaDBOwned {
            let file = baseDir.appendingPathComponent(entry.file)
            guard fm.fileExists(atPath: file.path) else { continue }
            guard let count = try? entry.db.read({ db in
                try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM \(entry.table)")
            }), count > 0 else { continue }
            let migrated = file.appendingPathExtension("migrated")
            if fm.fileExists(atPath: migrated.path) {
                try? fm.removeItem(at: file)
            } else {
                try? fm.moveItem(at: file, to: migrated)
            }
            log.info("[NovaDB] Cleaned up orphan legacy file: \(entry.file)")
        }
    }

    private func runMigrations() throws {
        var configMigrator = DatabaseMigrator()
        configMigrator.registerMigration("v1_config_schema") { db in
            try ConfigDBSchema.v1.createAll(in: db)
        }
        configMigrator.registerMigration("v2_config_add_server_fields") { db in
            try ConfigDBSchema.v2AddServerFields(in: db)
        }
        configMigrator.registerMigration("v2_tokenhub_expand_columns") { db in
            try ConfigDBSchema.v2ExpandTokenhubColumns(in: db)
        }
        configMigrator.registerMigration("v3_modelfile_add_description") { db in
            try ConfigDBSchema.v3ModelfileAddDescription(in: db)
        }
        try configMigrator.migrate(configDB)
        log.info("[NovaDB] Config DB migrations complete")

        var dataMigrator = DatabaseMigrator()
        dataMigrator.registerMigration("v1_data_schema") { db in
            try DataDBSchema.v1.createAll(in: db)
        }
        dataMigrator.registerMigration("v2_expand_metrics_columns") { db in
            try DataDBSchema.v2ExpandMetricsColumns(in: db)
        }
        try dataMigrator.migrate(dataDB)
        log.info("[NovaDB] Data DB migrations complete")
    }

    private func initStores() {
        self.apiKeyStore = APIKeyStore(db: configDB)
        self.configStore = ConfigStore(db: configDB)
        self.modelSettingsStore = ModelSettingsStore(db: configDB)
        self.tokenhubStore = TokenhubStore(db: configDB)
        self.modelfileStore = ModelfileStore(db: configDB)
        self.authStore = AuthStore(db: configDB)
        self.clusterPolicyStore = ClusterPolicyStore(db: configDB)
        self.modelRegistryStore = ModelRegistryStore(db: dataDB)
        self.loadedModelsStore = LoadedModelsStore(db: dataDB)
        self.metricsStore = MetricsDBStore(db: dataDB)
        self.chatStore = ChatStore(db: dataDB)
        self.workerDeploymentStore = WorkerDeploymentStore(db: dataDB)
    }

    // MARK: - Legacy Import
    // Only import tables whose stores are ACTIVELY WIRED to replace JSON I/O.
    // Adding more imports here BEFORE wiring the stores causes the JSON file to be
    // renamed to .migrated, breaking the old code path that still reads it.

    private func importLegacyJSON(baseDir: URL) throws {
        // API keys — Phase 1, store is wired for plaintext display
        try maybeImportLegacy(
            file: baseDir.appendingPathComponent("api_keys.json"),
            tableName: "api_keys",
            into: configDB
        ) { data in
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            if let keys = try? decoder.decode([LegacyAPIKeyImport].self, from: data) {
                try self.configDB.write { db in
                    for key in keys {
                        let record = APIKeyRecord(
                            id: key.id,
                            name: key.name,
                            keyHash: key.keyHash,
                            rawKey: "sk-novamlx-" + String(repeating: "0", count: 64),
                            keyPrefix: key.keyPrefix,
                            keySuffix: key.keySuffix ?? "",
                            createdAt: key.createdAt,
                            expiresAt: key.expiresAt,
                            isEnabled: key.isEnabled,
                            rateLimitPerSecond: key.rateLimitPerSecond,
                            rateLimitBurst: key.rateLimitBurst,
                            allowedModels: encodeJSON(key.allowedModels),
                            allowedEndpoints: encodeJSON(key.allowedEndpoints),
                            maxTokensPerPeriod: key.maxTokensPerPeriod,
                            maxRequestsPerPeriod: key.maxRequestsPerPeriod,
                            usageResetPeriod: key.usageResetPeriod ?? "daily",
                            totalTokensUsed: key.usage?.totalTokensUsed ?? 0,
                            totalRequests: key.usage?.totalRequests ?? 0,
                            lastUsedAt: key.usage?.lastUsedAt,
                            periodTokens: key.usage?.periodTokens ?? 0,
                            periodRequests: key.usage?.periodRequests ?? 0,
                            periodResetDate: key.usage?.periodResetDate,
                            perModelTokens: encodeJSON(key.usage?.perModelTokens ?? [:]) ?? "{}"
                        )
                        try record.insert(db, onConflict: .ignore)
                    }
                }
            }
        }

        // config.json — Phase B, store is wired and Configuration.loadFromStore is sole reader
        try maybeImportLegacy(
            file: baseDir.appendingPathComponent("config.json"),
            tableName: "config",
            into: configDB
        ) { data in
            // Best-effort parse using a permissive shape. We only extract the
            // fields that ConfigRecord stores; unknown keys are ignored.
            struct LegacyConfigImport: Decodable {
                struct Server: Decodable {
                    let host: String?
                    let port: Int?
                    let adminPort: Int?
                    let maxConcurrentRequests: Int?
                    let requestTimeout: Double?
                    let contextScalingTarget: Int?
                    let tlsCertPath: String?
                    let tlsKeyPath: String?
                    let tlsKeyPassword: String?
                    let maxRequestSizeMB: Double?
                    let maxProcessMemory: String?
                    let prefixCacheEnabled: Bool?
                }
                let server: Server?
                let defaultModel: String?
                let modelsDirectory: String?
                let huggingfaceEndpoint: String?
            }
            guard let parsed = try? JSONDecoder().decode(LegacyConfigImport.self, from: data) else { return }
            try self.configDB.write { db in
                let existing = try ConfigRecord.fetchOne(db, key: 1)
                var record = existing ?? ConfigRecord(
                    host: "127.0.0.1", port: 6590, adminPort: 6591, tlsEnabled: false,
                    hfEndpoint: "https://huggingface.co"
                )
                if let s = parsed.server {
                    if let v = s.host { record.host = v }
                    if let v = s.port { record.port = v }
                    if let v = s.adminPort { record.adminPort = v }
                    if let v = s.maxConcurrentRequests { record.maxConcurrentRequests = v }
                    if let v = s.requestTimeout { record.requestTimeout = v }
                    if let v = s.contextScalingTarget { record.contextScalingTarget = v }
                    if let v = s.tlsCertPath { record.tlsCertPath = v; record.tlsEnabled = !v.isEmpty }
                    if let v = s.tlsKeyPath { record.tlsKeyPath = v }
                    if let v = s.tlsKeyPassword { record.tlsKeyPassword = v }
                    if let v = s.maxRequestSizeMB { record.maxRequestSizeMB = v }
                    if let v = s.maxProcessMemory { record.maxProcessMemory = v }
                    if let v = s.prefixCacheEnabled { record.prefixCacheEnabled = v }
                }
                if let v = parsed.defaultModel { record.defaultModel = v }
                if let v = parsed.modelsDirectory { record.modelsDir = v }
                if let v = parsed.huggingfaceEndpoint { record.hfEndpoint = v }
                try record.save(db)
            }
        }

        // providers.json — Phase C, TokenhubManager is wired to tokenhubStore
        try maybeImportLegacy(
            file: baseDir.appendingPathComponent("tokenhub/providers.json"),
            tableName: "tokenhub_providers",
            into: configDB
        ) { data in
            struct LegacyProvider: Decodable {
                let id: String?
                let name: String
                let endpoint: String
                let apiKey: String?
                let remoteModel: String?
                let isEnabled: Bool?
                let includeInLoadBalance: Bool?
                let tags: [String]?
                let isLocal: Bool?
                let isFree: Bool?
                let isManaged: Bool?
                let supportsResponsesAPI: Bool?
                let supportsVision: Bool?
                let visionStrategy: String?
                let anthropicEndpoint: String?
                let visionCompanionModel: String?
                let requestCount: Int?
                let successCount: Int?
                let avgLatencyMs: Double?
                let lastTestedAt: Date?
                let lastStatus: String?
                let contextWindowOverride: Int?
            }
            guard let parsed = try? JSONDecoder().decode([LegacyProvider].self, from: data) else { return }
            try self.configDB.write { db in
                for p in parsed {
                    var record = TokenhubProviderRecord(
                        name: p.name,
                        endpoint: p.endpoint,
                        apiKey: p.apiKey?.isEmpty == false ? p.apiKey : nil,
                        remoteModel: p.remoteModel?.isEmpty == false ? p.remoteModel : nil,
                        isEnabled: p.isEnabled ?? true,
                        isManaged: p.isManaged ?? false,
                        loadBalanceWeight: (p.includeInLoadBalance ?? true) ? 1.0 : 0.0,
                        totalRequests: Int64(p.requestCount ?? 0),
                        totalTokens: 0,
                        avgLatencyMs: p.avgLatencyMs,
                        lastUsedAt: p.lastTestedAt,
                        extraConfig: nil
                    )
                    record.providerId = p.id
                    record.includeInLoadBalance = p.includeInLoadBalance ?? true
                    record.tags = (p.tags ?? []).isEmpty ? nil : (try? String(data: JSONEncoder().encode(p.tags), encoding: .utf8))
                    record.isLocal = p.isLocal ?? false
                    record.isFree = p.isFree ?? false
                    record.supportsResponsesAPI = p.supportsResponsesAPI ?? false
                    record.supportsVision = p.supportsVision ?? false
                    record.visionStrategy = p.visionStrategy
                    record.anthropicEndpoint = p.anthropicEndpoint
                    record.visionCompanionModel = p.visionCompanionModel
                    record.requestCount = p.requestCount ?? 0
                    record.successCount = p.successCount ?? 0
                    record.lastTestedAt = p.lastTestedAt
                    record.lastStatus = p.lastStatus
                    record.contextWindowOverride = p.contextWindowOverride
                    try record.save(db)
                }
            }
        }

        // Phase 2+: model settings, model registry, loaded models, metrics,
        // chat history, worker deployments, auth, cluster policy will be imported
        // here once their stores replace the old JSON code paths.

        // loaded_models.json — Phase D1, InferenceService.saveLoadedModelsList is wired to store
        try maybeImportLegacy(
            file: baseDir.appendingPathComponent("loaded_models.json"),
            tableName: "loaded_models",
            into: dataDB
        ) { data in
            guard let ids = try? JSONDecoder().decode([String].self, from: data) else { return }
            try self.dataDB.write { db in
                for id in ids {
                    let record = LoadedModelRecord(modelId: id, loadedAt: Date())
                    try record.insert(db, onConflict: .ignore)
                }
            }
        }

        // model_settings.json — Phase D2
        // Importer lives in ModelSettingsManager (NovaMLXModelManager) where the
        // ModelSettings domain type is visible. NovaMLXDB cannot import
        // NovaMLXCore without creating a circular dependency, so this table's
        // legacy import is handled by the manager on first init.

        // cluster-policy.json — Phase F, clusterPolicyStore is the sole reader.
        try maybeImportLegacy(
            file: baseDir.appendingPathComponent("cluster-policy.json"),
            tableName: "cluster_policy",
            into: configDB
        ) { data in
            // Re-serialize through JSONSerialization to canonicalise whitespace
            // and ignore trailing junk. The schema only stores the raw string;
            // we don't need a typed Decodable here.
            guard let obj = try? JSONSerialization.jsonObject(with: data),
                  let canonical = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else { return }
            let json = String(data: canonical, encoding: .utf8) ?? "{}"
            try self.configDB.write { db in
                var record = try ClusterPolicyRecord.fetchOne(db, key: 1) ?? ClusterPolicyRecord(policyJSON: "{}")
                // Only overwrite if the DB still holds the placeholder.
                if record.policyJSON == "{}" {
                    record.policyJSON = json
                    record.updatedAt = Date()
                    try record.save(db)
                }
            }
        }

        log.info("[NovaDB] Legacy JSON import complete")
    }

    private func maybeImportLegacy(
        file: URL,
        tableName: String,
        into db: DatabasePool,
        import: (Data) throws -> Void
    ) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: file.path) else { return }

        let count = try db.read { db in
            try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM \(tableName)") ?? 0
        }
        guard count == 0 else {
            // Table already has rows from a prior import — leave the file on
            // disk untouched. The SQLite store is the sole source of truth
            // post-cutover; the JSON file is inert (and typically already
            // renamed to .migrated on the first successful import).
            return
        }

        let data = try Data(contentsOf: file)
        try `import`(data)
        try migrateFile(file)
        log.info("[NovaDB] Migrated \(file.lastPathComponent) → \(tableName)")
    }

    private func migrateFile(_ originalURL: URL) throws {
        let fm = FileManager.default
        let migratedURL = originalURL.appendingPathExtension("migrated")
        guard fm.fileExists(atPath: originalURL.path) else { return }
        guard !fm.fileExists(atPath: migratedURL.path) else {
            try? fm.removeItem(at: originalURL)
            return
        }
        try fm.moveItem(at: originalURL, to: migratedURL)
    }
}

private func encodeJSON<T: Encodable>(_ value: T?) -> String? {
    guard let value else { return nil }
    guard let data = try? JSONEncoder().encode(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

// Lightweight import struct matching the original APIKey JSON format
private struct LegacyAPIKeyImport: Codable {
    var id: String
    var name: String
    var keyHash: String
    var keyPrefix: String
    var keySuffix: String?
    var createdAt: Date
    var expiresAt: Date?
    var isEnabled: Bool
    var rateLimitPerSecond: Double?
    var rateLimitBurst: Int?
    var allowedModels: [String]?
    var allowedEndpoints: [String]?
    var maxTokensPerPeriod: Int64?
    var maxRequestsPerPeriod: Int64?
    var usageResetPeriod: String?
    var usage: LegacyAPIKeyUsage?

    struct LegacyAPIKeyUsage: Codable {
        var totalTokensUsed: Int64?
        var totalRequests: Int64?
        var lastUsedAt: Date?
        var periodTokens: Int64?
        var periodRequests: Int64?
        var periodResetDate: String?
        var perModelTokens: [String: Int64]?
    }
}
