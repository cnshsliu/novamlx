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

    /// Atomically replace every row with the provided snapshot. Used by
    /// the cutover path that holds an in-memory cache and writes the full
    /// picture back to disk on each mutation.
    public func replaceAll(_ records: [WorkerDeploymentRecord]) throws {
        try db.write { db in
            try WorkerDeploymentRecord.deleteAll(db)
            for record in records {
                try record.save(db)
            }
        }
    }
}
