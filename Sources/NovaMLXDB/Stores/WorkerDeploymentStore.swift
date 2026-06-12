import Foundation
import GRDB

public final class WorkerDeploymentStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list() throws -> [WorkerDeploymentRecord] {
        try db.read { db in
            try WorkerDeploymentRecord.fetchAll(db)
        }
    }

    public func get(hostname: String) throws -> WorkerDeploymentRecord? {
        try db.read { db in
            try WorkerDeploymentRecord.fetchOne(db, key: hostname)
        }
    }

    public func upsert(_ record: WorkerDeploymentRecord) throws {
        try db.write { db in
            try record.save(db)
        }
    }

    public func delete(hostname: String) throws {
        try db.write { db in
            try WorkerDeploymentRecord.deleteOne(db, key: hostname)
        }
    }
}
