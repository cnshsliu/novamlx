import Foundation
import NovaMLXDB
import os.log

// MARK: - Auth Client

private let authLog = Logger(subsystem: "com.novamlx", category: "Auth")

public struct AuthClient: Sendable {
    public let baseURL: String

    public init(baseURL: String = AuthClient.defaultBaseURL) {
        self.baseURL = baseURL
    }

    /// Cached auth URL — computed once, logs once.
    public static let defaultBaseURL: String = {
        // 1. Environment variable (CLI or dev override)
        if let env = ProcessInfo.processInfo.environment["NOVA_AUTH_URL"], !env.isEmpty {
            authLog.info("[Auth] URL from env: \(env)")
            return env
        }
        // 2. Config file (~/.nova/config.json → auth.authURL)
        if let configData = FileManager.default.contents(atPath: NovaMLXPaths.configFile.path) {
            if let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let auth = json["auth"] as? [String: Any],
               let url = auth["authURL"] as? String, !url.isEmpty {
                authLog.info("[Auth] URL from config: \(url)")
                return url
            }
        }
        // 3. Production default
        authLog.info("[Auth] Using production default: https://novamlx.ai")
        return "https://novamlx.ai"
    }()

    public func login(email: String, password: String) async throws -> LoginResponse {
        var request = URLRequest(url: URL(string: "\(baseURL)/api/v1/auth/login")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["email": email, "password": password])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw AuthError.networkError }

        switch http.statusCode {
        case 200:
            return try JSONDecoder().decode(LoginResponse.self, from: data)
        case 401:
            throw AuthError.invalidCredentials
        case 429:
            throw AuthError.rateLimited
        default:
            let body = String(data: data, encoding: .utf8) ?? ""
            throw AuthError.unexpectedStatus(http.statusCode, body)
        }
    }

    public func checkSession(_ session: String) async throws -> CheckResponse {
        var request = URLRequest(url: URL(string: "\(baseURL)/api/v1/auth/check")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["session": session])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw AuthError.networkError }

        switch http.statusCode {
        case 200:
            return try JSONDecoder().decode(CheckResponse.self, from: data)
        case 401:
            throw AuthError.sessionExpired
        case 403:
            if let body = try? JSONDecoder().decode(CheckErrorBody.self, from: data) {
                throw AuthError.noSubscription(body.subscribe_url ?? "/cloud")
            }
            throw AuthError.noSubscription("/cloud")
        default:
            throw AuthError.unexpectedStatus(http.statusCode, "")
        }
    }
}

// MARK: - Data Models

public struct LoginResponse: Decodable, Sendable {
    public let session: String
    public let user: LoginUser
}

public struct LoginUser: Decodable, Sendable {
    public let id: Int
    public let email: String
    public let name: String?
    public let plan: String
}

public struct CheckResponse: Decodable, Sendable {
    public let valid: Bool
    public let plan: String?
    public let status: String?
    public let cancelAtPeriodEnd: Bool?
    public let expiresAt: String?
    public let user: CheckUser?

    private enum CodingKeys: String, CodingKey {
        case valid, plan, status, user
        case cancelAtPeriodEnd = "cancel_at_period_end"
        case expiresAt = "expires_at"
    }
}

public struct CheckUser: Decodable, Sendable {
    public let email: String
    public let name: String?
}

struct CheckErrorBody: Decodable {
    let valid: Bool
    let error: String
    let subscribe_url: String?
    let plan: String?
}

// MARK: - Auth Error

public enum AuthError: LocalizedError, Sendable {
    case invalidCredentials
    case sessionExpired
    case noSubscription(String)
    case rateLimited
    case networkError
    case unexpectedStatus(Int, String)

    public var errorDescription: String? {
        switch self {
        case .invalidCredentials:
            return "Invalid email or password"
        case .sessionExpired:
            return "Session expired. Please run: nova login"
        case .noSubscription(let url):
            return "No active subscription. Visit \(url) to subscribe"
        case .rateLimited:
            return "Too many login attempts. Please try again in 1 minute"
        case .networkError:
            return "Network connection failed"
        case .unexpectedStatus(let code, _):
            return "Server error (\(code))"
        }
    }
}

// MARK: - Auth Cache (local persistence)

public struct AuthCache: Codable, Sendable {
    public let valid: Bool
    public let plan: String
    public let status: String
    public let cancelAtPeriodEnd: Bool
    public let expiresAt: String?
    public let cachedAt: TimeInterval
    public let userEmail: String

    private static let ttl: TimeInterval = 300 // 5 minutes

    public var isExpired: Bool {
        Date().timeIntervalSince1970 - cachedAt > Self.ttl
    }

    public init(
        valid: Bool, plan: String, status: String,
        cancelAtPeriodEnd: Bool, expiresAt: String?,
        cachedAt: TimeInterval, userEmail: String
    ) {
        self.valid = valid
        self.plan = plan
        self.status = status
        self.cancelAtPeriodEnd = cancelAtPeriodEnd
        self.expiresAt = expiresAt
        self.cachedAt = cachedAt
        self.userEmail = userEmail
    }

    public static func loadSession() -> String? {
        Self.importLegacyAuthFilesIfNeeded()
        let record: AuthSessionRecord?
        do { record = try NovaDB.shared.authStore.getSession() } catch { return nil }
        let token = record?.sessionToken ?? ""
        return token.isEmpty ? nil : token
    }

    public static func saveSession(_ token: String) throws {
        try NovaDB.shared.authStore.saveSession(token: token)
    }

    public static func load() -> AuthCache? {
        Self.importLegacyAuthFilesIfNeeded()
        let record: AuthSessionRecord?
        do { record = try NovaDB.shared.authStore.getSession() } catch { return nil }
        guard let record, record.authValid != nil else { return nil }
        let cachedAt = record.authCachedAt ?? Date()
        return AuthCache(
            valid: record.authValid ?? false,
            plan: record.authPlan ?? "free",
            status: record.authStatus ?? "inactive",
            cancelAtPeriodEnd: record.authCancelAtPeriodEnd ?? false,
            expiresAt: record.authExpiresAt.map { Self.formatISO8601($0) },
            cachedAt: cachedAt.timeIntervalSince1970,
            userEmail: record.userEmail ?? ""
        )
    }

    public func save() throws {
        let expiresDate = expiresAt.flatMap { Self.parseISO8601($0) }
        try NovaDB.shared.authStore.saveAuthCache(
            valid: valid,
            plan: plan,
            status: status,
            cancelAtPeriodEnd: cancelAtPeriodEnd,
            expiresAt: expiresDate,
            cachedAt: Date(timeIntervalSince1970: cachedAt),
            userEmail: userEmail
        )
    }

    public static func clearAll() {
        try? NovaDB.shared.authStore.clear()
    }

    /// ISO8601 parse + format strategy. Sendable so it's safe to hold in a
    /// static let. AuthCache.expiresAt is a server-provided ISO string; we
    /// round-trip it through this style when persisting to/from SQLite.
    private static let iso8601 = Date.ISO8601FormatStyle()

    private static func parseISO8601(_ s: String) -> Date? {
        try? Date(s, strategy: iso8601)
    }

    private static func formatISO8601(_ d: Date) -> String {
        d.formatted(iso8601)
    }

    /// One-shot import of the legacy `~/.nova/session` + `~/.nova/auth_cache.json`
    /// pair into authStore. Idempotent: skipped if the store already has a row
    /// with a non-empty token; otherwise we read the files (if present),
    /// upsert them as a single auth_session row, and rename both files to
    /// `.migrated` so we never run again.
    private static func importLegacyAuthFilesIfNeeded() {
        let existing: AuthSessionRecord?
        do { existing = try NovaDB.shared.authStore.getSession() } catch { return }
        let token = existing?.sessionToken ?? ""
        guard token.isEmpty else {
            // Store already populated — leave any legacy files alone.
            return
        }

        let fm = FileManager.default
        let sessionToken = (try? String(contentsOf: NovaMLXPaths.sessionFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let cache: AuthCache? = {
            guard let data = try? Data(contentsOf: NovaMLXPaths.authCacheFile) else { return nil }
            return try? JSONDecoder().decode(AuthCache.self, from: data)
        }()

        // Only persist if we have something to import.
        guard !sessionToken.isEmpty || cache != nil else { return }

        // Always ensure a row exists with the token, then layer cache fields on top.
        if !sessionToken.isEmpty {
            try? NovaDB.shared.authStore.saveSession(token: sessionToken)
        }
        if let cache {
            try? cache.save()
        }

        // Rename legacy files to .migrated so the import never runs again.
        for file in [NovaMLXPaths.sessionFile, NovaMLXPaths.authCacheFile] {
            guard fm.fileExists(atPath: file.path) else { continue }
            let migrated = file.appendingPathExtension("migrated")
            if fm.fileExists(atPath: migrated.path) {
                try? fm.removeItem(at: file)
            } else {
                try? fm.moveItem(at: file, to: migrated)
            }
        }
    }
}

// MARK: - Subscription Validation Gate

public enum CloudAuth {
    /// Synchronous subscription check (disk cache only, no network).
    public static func isSubscribed() -> Bool {
        guard let cache = AuthCache.load(), !cache.isExpired, cache.valid else { return false }
        return true
    }

    public static func validate() async throws -> AuthCache {
        guard let session = AuthCache.loadSession() else {
            throw AuthError.sessionExpired
        }

        // Fast path: cache hit within TTL
        if let cache = AuthCache.load(), !cache.isExpired, cache.valid {
            return cache
        }

        // Slow path: call check API
        let client = AuthClient()
        authLog.info("[Auth] Checking session against \(client.baseURL)...")
        do {
            let response = try await client.checkSession(session)

            guard response.valid else {
                throw AuthError.noSubscription("/cloud")
            }

            let cache = AuthCache(
                valid: response.valid,
                plan: response.plan ?? "free",
                status: response.status ?? "unknown",
                cancelAtPeriodEnd: response.cancelAtPeriodEnd ?? false,
                expiresAt: response.expiresAt,
                cachedAt: Date().timeIntervalSince1970,
                userEmail: response.user?.email ?? ""
            )
            try cache.save()

            return cache
        } catch let error as AuthError {
            authLog.warning("[Auth] Check failed: \(error.localizedDescription)")
            throw error
        } catch {
            authLog.warning("[Auth] Network error: \(error.localizedDescription)")
            throw AuthError.networkError
        }
    }
}
