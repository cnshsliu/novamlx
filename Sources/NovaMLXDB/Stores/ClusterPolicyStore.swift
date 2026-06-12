import Foundation
import GRDB

public final class ClusterPolicyStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func get() throws -> String {
        try db.read { db in
            if let record = try ClusterPolicyRecord.fetchOne(db, key: 1) {
                return record.policyJSON
            }
            return "{}"
        }
    }

    public func set(_ json: String) throws {
        try db.write { db in
            var record = try ClusterPolicyRecord.fetchOne(db, key: 1) ?? ClusterPolicyRecord(policyJSON: "{}")
            record.policyJSON = json
            record.updatedAt = Date()
            try record.save(db)
        }
    }
}
