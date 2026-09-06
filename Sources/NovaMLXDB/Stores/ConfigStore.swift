import Foundation
import GRDB
import Logging

public final class ConfigStore: Sendable {
    private let db: DatabasePool
    private let log = Logger(label: "ConfigStore")

    public init(db: DatabasePool) {
        self.db = db
    }

    public func get() throws -> ConfigRecord {
        if let record = try db.read({ db in try ConfigRecord.fetchOne(db, key: 1) }) {
            return record
        }
        // Create default
        let record = ConfigRecord(
            host: "0.0.0.0",
            port: 6590,
            adminPort: 6591,
            tlsEnabled: false,
            hfEndpoint: "https://huggingface.co"
        )
        try db.write { db in
            try record.insert(db)
        }
        return record
    }

    public func update(_ updates: @Sendable (inout ConfigRecord) -> Void) throws {
        try db.write { db in
            var record = try ConfigRecord.fetchOne(db, key: 1) ?? ConfigRecord(
                host: "0.0.0.0", port: 6590, adminPort: 6591, tlsEnabled: false, hfEndpoint: "https://huggingface.co"
            )
            updates(&record)
            try record.save(db)
        }
    }

    public func updateValue(column: String, value: String?) throws {
        try db.write { db in
            var record = try ConfigRecord.fetchOne(db, key: 1) ?? ConfigRecord(
                host: "0.0.0.0", port: 6590, adminPort: 6591, tlsEnabled: false, hfEndpoint: "https://huggingface.co"
            )
            switch column {
            case "host": record.host = value ?? "0.0.0.0"
            case "default_model": record.defaultModel = value
            case "hf_endpoint": record.hfEndpoint = value ?? "https://huggingface.co"
            case "auth_url": record.authUrl = value
            case "tknet_api_key": record.tknetApiKey = value
            case "cluster_config": record.clusterConfig = value
            case "auto_load": record.autoLoad = value
            case "log_level": record.logLevel = value
            default: break
            }
            try record.save(db)
        }
    }
}
