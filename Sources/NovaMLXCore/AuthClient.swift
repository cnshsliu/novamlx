import Foundation

// MARK: - Auth Client

private func authLog(_ msg: String) {
    // Write directly to novamlx.log for debugging
    let line = "[\(Date().formatted(.iso8601))] [AUTH] \(msg)\n"
    if let handle = try? FileHandle(forWritingTo: NovaMLXPaths.logFile) {
        handle.seekToEndOfFile()
        handle.write(Data(line.utf8))
        handle.closeFile()
    }
}

public struct AuthClient: Sendable {
    public let baseURL: String

    public init(baseURL: String = AuthClient.defaultBaseURL) {
        self.baseURL = baseURL
    }

    public static var defaultBaseURL: String {
        // 1. Environment variable (CLI or dev override)
        if let env = ProcessInfo.processInfo.environment["NOVA_AUTH_URL"], !env.isEmpty {
            authLog("Auth URL from env: \(env)")
            return env
        }
        // 2. Config file (~/.nova/config.json → auth.authURL)
        if let configData = FileManager.default.contents(atPath: NovaMLXPaths.configFile.path) {
            authLog("Config file size: \(configData.count) bytes")
            if let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let auth = json["auth"] as? [String: Any],
               let url = auth["authURL"] as? String, !url.isEmpty {
                authLog("Auth URL from config: \(url)")
                return url
            } else {
                authLog("Config parsed but no auth.authURL found")
            }
        } else {
            authLog("Config file not found at \(NovaMLXPaths.configFile.path)")
        }
        // 3. Production default
        authLog("Using production default: https://novamlx.ai")
        return "https://novamlx.ai"
    }

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
        try? String(contentsOf: NovaMLXPaths.sessionFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func saveSession(_ token: String) throws {
        let dir = NovaMLXPaths.sessionFile.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try token.write(to: NovaMLXPaths.sessionFile, atomically: true, encoding: .utf8)
        // Set file permission to 600 (owner read/write only)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: NovaMLXPaths.sessionFile.path
        )
    }

    public static func load() -> AuthCache? {
        guard let data = try? Data(contentsOf: NovaMLXPaths.authCacheFile) else { return nil }
        return try? JSONDecoder().decode(AuthCache.self, from: data)
    }

    public func save() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(self)
        try data.write(to: NovaMLXPaths.authCacheFile, options: .atomic)
    }

    public static func clearAll() {
        try? FileManager.default.removeItem(at: NovaMLXPaths.authCacheFile)
        try? FileManager.default.removeItem(at: NovaMLXPaths.sessionFile)
    }
}

// MARK: - Subscription Validation Gate

public enum CloudAuth {
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
        authLog("Checking session against \(client.baseURL)...")
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
            authLog("Check failed: \(error.localizedDescription)")
            throw error
        } catch {
            authLog("Network error: \(error.localizedDescription)")
            throw AuthError.networkError
        }
    }
}
