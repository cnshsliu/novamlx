import Foundation
import GRDB

public final class LoadedModelsStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list() throws -> [String] {
        try db.read { db in
            try LoadedModelRecord
                .order(Column("loaded_at"))
                .fetchAll(db)
                .map { $0.modelId }
        }
    }

    public func add(modelId: String) throws {
        try db.write { db in
            let record = LoadedModelRecord(modelId: modelId, loadedAt: Date())
            try record.insert(db, onConflict: .ignore)
        }
    }

    public func remove(modelId: String) throws {
        try db.write { db in
            try LoadedModelRecord.deleteOne(db, key: modelId)
        }
    }

    public func replaceAll(with modelIds: [String]) throws {
        try db.write { db in
            try LoadedModelRecord.deleteAll(db)
            for modelId in modelIds {
                let record = LoadedModelRecord(modelId: modelId, loadedAt: Date())
                try record.insert(db)
            }
        }
    }
}
