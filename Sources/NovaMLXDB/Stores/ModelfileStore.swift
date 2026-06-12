import Foundation
import GRDB

public final class ModelfileStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list() throws -> [ModelfileRecord] {
        try db.read { db in
            try ModelfileRecord.fetchAll(db)
        }
    }

    public func get(name: String) throws -> ModelfileRecord? {
        try db.read { db in
            try ModelfileRecord.fetchOne(db, key: name)
        }
    }

    public func upsert(_ record: ModelfileRecord) throws {
        try db.write { db in
            try record.save(db)
        }
    }

    public func delete(name: String) throws {
        try db.write { db in
            try ModelfileRecord.deleteOne(db, key: name)
        }
    }
}
