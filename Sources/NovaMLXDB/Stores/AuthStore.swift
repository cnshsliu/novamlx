import Foundation
import GRDB

public final class AuthStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func getSession() throws -> AuthSessionRecord? {
        try db.read { db in
            try AuthSessionRecord.fetchOne(db, key: 1)
        }
    }

    public func saveSession(token: String) throws {
        try db.write { db in
            var record = try AuthSessionRecord.fetchOne(db, key: 1) ?? AuthSessionRecord(sessionToken: "")
            record.sessionToken = token
            try record.save(db)
        }
    }

    public func saveAuthCache(valid: Bool, plan: String?, status: String?, cancelAtPeriodEnd: Bool?, expiresAt: Date?, cachedAt: Date, userEmail: String?) throws {
        try db.write { db in
            var record = try AuthSessionRecord.fetchOne(db, key: 1) ?? AuthSessionRecord(sessionToken: "")
            record.authValid = valid
            record.authPlan = plan
            record.authStatus = status
            record.authCancelAtPeriodEnd = cancelAtPeriodEnd
            record.authExpiresAt = expiresAt
            record.authCachedAt = cachedAt
            record.userEmail = userEmail
            try record.save(db)
        }
    }

    public func clear() throws {
        try db.write { db in
            _ = try AuthSessionRecord.deleteOne(db, key: 1)
        }
    }
}
