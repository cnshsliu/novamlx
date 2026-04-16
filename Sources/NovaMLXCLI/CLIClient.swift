import Foundation

enum CLIClient {
    private static let baseURL = "http://127.0.0.1:8080"
    private static let adminURL = "http://127.0.0.1:8081"

    // Read server config from ~/.nova/config.json (same file the server uses)
    private static func configFromFile() -> [String: Any]? {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let configURL = homeDir.appendingPathComponent(".nova/config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json
    }

    private static func portFromFile() -> (inference: Int, admin: Int) {
        guard let json = configFromFile(),
              let server = json["server"] as? [String: Any]
        else { return (6590, 6591) }
        return (server["port"] as? Int ?? 6590, server["adminPort"] as? Int ?? 6591)
    }

    /// Resolve API key: env var > config file > nil
    private static func resolveAPIKey() -> String? {
        // 1. Environment variable takes priority
        if let envKey = ProcessInfo.processInfo.environment["NOVA_API_KEY"], !envKey.isEmpty {
            return envKey
        }
        // 2. Read from config file (first key in apiKeys array)
        if let json = configFromFile(),
           let server = json["server"] as? [String: Any],
           let keys = server["apiKeys"] as? [String],
           let firstKey = keys.first, !firstKey.isEmpty {
            return firstKey
        }
        return nil
    }

    private static var effectiveBaseURL: String {
        "http://127.0.0.1:\(portFromFile().inference)"
    }

    private static var effectiveAdminURL: String {
        "http://127.0.0.1:\(portFromFile().admin)"
    }

    struct Response {
        let statusCode: Int
        let body: String
    }

    static func get(_ path: String, admin: Bool = false) async throws -> Response {
        let base = admin ? effectiveAdminURL : effectiveBaseURL
        let url = URL(string: "\(base)\(path)")!
        var request = URLRequest(url: url)
        if let key = resolveAPIKey() {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        return Response(statusCode: http.statusCode, body: String(data: data, encoding: .utf8) ?? "")
    }

    static func post(_ path: String, body: String? = nil, admin: Bool = false) async throws -> Response {
        let base = admin ? effectiveAdminURL : effectiveBaseURL
        let url = URL(string: "\(base)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = resolveAPIKey() {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        if let body { request.httpBody = body.data(using: .utf8) }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        return Response(statusCode: http.statusCode, body: String(data: data, encoding: .utf8) ?? "")
    }

    static func delete(_ path: String, admin: Bool = false) async throws -> Response {
        let base = admin ? effectiveAdminURL : effectiveBaseURL
        let url = URL(string: "\(base)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        if let key = resolveAPIKey() {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        return Response(statusCode: http.statusCode, body: String(data: data, encoding: .utf8) ?? "")
    }

    static func put(_ path: String, body: String, admin: Bool = false) async throws -> Response {
        let base = admin ? effectiveAdminURL : effectiveBaseURL
        let url = URL(string: "\(base)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = resolveAPIKey() {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = body.data(using: .utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        return Response(statusCode: http.statusCode, body: String(data: data, encoding: .utf8) ?? "")
    }

    static func printJSON(_ body: String) {
        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
              let str = String(data: pretty, encoding: .utf8)
        else {
            print(body)
            return
        }
        print(str)
    }
}
