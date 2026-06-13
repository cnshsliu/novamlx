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

    private init() {}

    public func setup(baseDir: URL) throws {
        try queue.sync {
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
        try configMigrator.migrate(configDB)
        log.info("[NovaDB] Config DB migrations complete")

        var dataMigrator = DatabaseMigrator()
        dataMigrator.registerMigration("v1_data_schema") { db in
            try DataDBSchema.v1.createAll(in: db)
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

        // Phase 2+: Config, model settings, tokenhub providers, model registry,
        // loaded models, metrics, chat history, worker deployments, auth, cluster policy
        // will be imported here once their stores replace the old JSON code paths.

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

    private func importChatHistory(_ data: Data) throws {
        if let record = try? JSONDecoder().decode(LegacyChatRecord.self, from: data) {
            try dataDB.write { db in
                let chat = ChatRecord(
                    id: record.id,
                    title: record.title,
                    model: record.model,
                    systemPrompt: record.systemPrompt,
                    createdAt: record.createdAt,
                    updatedAt: record.updatedAt
                )
                try chat.insert(db, onConflict: .ignore)

                for (idx, msg) in record.messages.enumerated() {
                    let message = ChatMessageRecord(
                        id: msg.id ?? UUID().uuidString,
                        chatId: record.id,
                        role: msg.role,
                        content: msg.content,
                        thinkingContent: msg.thinkingContent,
                        createdAt: msg.createdAt,
                        sortOrder: idx
                    )
                    try message.insert(db, onConflict: .ignore)
                }
            }
        }
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
