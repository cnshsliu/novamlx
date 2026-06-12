import Foundation
import GRDB

public final class ModelRegistryStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list() throws -> [ModelRegistryRecord] {
        try db.read { db in
            try ModelRegistryRecord.fetchAll(db)
        }
    }

    public func get(modelId: String) throws -> ModelRegistryRecord? {
        try db.read { db in
            try ModelRegistryRecord.fetchOne(db, key: modelId)
        }
    }

    public func upsert(_ record: ModelRegistryRecord) throws {
        try db.write { db in
            try record.save(db)
        }
    }

    public func delete(modelId: String) throws {
        try db.write { db in
            try ModelRegistryRecord.deleteOne(db, key: modelId)
        }
    }

    public func findByType(_ modelType: String) throws -> [ModelRegistryRecord] {
        try db.read { db in
            try ModelRegistryRecord
                .filter(Column("modelType") == modelType)
                .fetchAll(db)
        }
    }
}
