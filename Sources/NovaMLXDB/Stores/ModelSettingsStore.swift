import Foundation
import GRDB

public final class ModelSettingsStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list() throws -> [ModelSettingsRecord] {
        try db.read { db in
            try ModelSettingsRecord.fetchAll(db)
        }
    }

    public func get(modelId: String) throws -> ModelSettingsRecord? {
        try db.read { db in
            try ModelSettingsRecord.fetchOne(db, key: modelId)
        }
    }

    public func upsert(_ record: ModelSettingsRecord) throws {
        try db.write { db in
            try record.save(db)
        }
    }

    public func delete(modelId: String) throws {
        try db.write { db in
            try ModelSettingsRecord.deleteOne(db, key: modelId)
        }
    }
}
