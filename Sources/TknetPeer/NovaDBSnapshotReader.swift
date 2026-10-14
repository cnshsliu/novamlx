import Foundation

public enum NovaDBReadError: Error, Equatable, CustomStringConvertible {
    case sqliteMissing
    case sqliteFailed(String)
    case unreadableJSON(String)

    public var description: String {
        switch self {
        case .sqliteMissing:
            return "sqlite3 is not installed; cannot read nova_config.db"
        case .sqliteFailed(let message):
            return message
        case .unreadableJSON(let message):
            return message
        }
    }
}

/// Reads NovaMLX `nova_config.db` load-balancer and Tokenhub tables via the
/// `sqlite3` CLI. The peer library does not link NovaMLXDB or GRDB.
public enum NovaDBSnapshotReader {
    public static func load(database url: URL) throws -> NovaMigrationInput {
        let providers = try query(url, sql: """
            SELECT name, endpoint, api_key, remote_model, is_enabled,
                   provider_id, anthropic_endpoint
            FROM tokenhub_providers
            ORDER BY name
            """)
        let balancers = try query(url, sql: """
            SELECT id, name, slug, strategy, max_retries, is_enabled
            FROM load_balancers
            ORDER BY rowid
            """)
        let members = try query(url, sql: """
            SELECT m.lb_id, lb.slug AS lb_slug, m.kind, m.ref, m.weight, m.is_enabled
            FROM lb_members AS m
            LEFT JOIN load_balancers AS lb ON lb.id = m.lb_id
            ORDER BY m.rowid
            """)
        return try decode(providersJSON: providers, loadBalancersJSON: balancers, membersJSON: members)
    }

    public static func decode(providersJSON: Data, loadBalancersJSON: Data, membersJSON: Data) throws -> NovaMigrationInput {
        let decoder = JSONDecoder()
        do {
            let providers = try decoder.decode([ProviderRow].self, from: providersJSON).map { $0.snapshot }
            let balancers = try decoder.decode([BalancerRow].self, from: loadBalancersJSON).map { $0.snapshot }
            let members = try decoder.decode([MemberRow].self, from: membersJSON).map { $0.snapshot }
            return NovaMigrationInput(providers: providers, loadBalancers: balancers, members: members)
        } catch {
            throw NovaDBReadError.unreadableJSON("nova_config.db JSON could not be read: \(error)")
        }
    }

    // MARK: - sqlite3

    static func query(_ database: URL, sql: String) throws -> Data {
        guard let binary = sqlite3Binary() else { throw NovaDBReadError.sqliteMissing }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["-json", database.path, sql]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        if process.terminationStatus != 0 {
            let trimmed = err.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.contains("no such table") {
                throw NovaDBReadError.sqliteFailed(
                    "nova_config.db is missing load_balancers or tokenhub_providers (\(database.path)). \(trimmed)")
            }
            throw NovaDBReadError.sqliteFailed(trimmed.isEmpty ? "sqlite3 exited \(process.terminationStatus)" : trimmed)
        }
        if out.isEmpty { return Data("[]".utf8) }
        return out
    }

    private static func sqlite3Binary() -> String? {
        let candidates = ["/usr/bin/sqlite3", "/usr/local/bin/sqlite3", "/opt/homebrew/bin/sqlite3"]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return nil
    }
}

// MARK: - Rows

private struct FlexBool: Decodable {
    let value: Bool
    init(from decoder: Decoder) throws {
        let box = try decoder.singleValueContainer()
        if let flag = try? box.decode(Bool.self) {
            value = flag
        } else if let number = try? box.decode(Int.self) {
            value = number != 0
        } else if let text = try? box.decode(String.self) {
            value = text == "1" || text.lowercased() == "true"
        } else {
            value = false
        }
    }
}

private struct FlexInt: Decodable {
    let value: Int?
    init(from decoder: Decoder) throws {
        let box = try decoder.singleValueContainer()
        if box.decodeNil() {
            value = nil
        } else if let number = try? box.decode(Int.self) {
            value = number
        } else if let text = try? box.decode(String.self) {
            value = Int(text)
        } else {
            value = nil
        }
    }
}

private struct ProviderRow: Decodable {
    let name: String
    let endpoint: String?
    let apiKey: String?
    let remoteModel: String?
    let isEnabled: FlexBool?
    let providerId: String?
    let anthropicEndpoint: String?

    enum CodingKeys: String, CodingKey {
        case name, endpoint
        case apiKey = "api_key"
        case remoteModel = "remote_model"
        case isEnabled = "is_enabled"
        case providerId = "provider_id"
        case anthropicEndpoint = "anthropic_endpoint"
    }

    var snapshot: NovaProviderSnapshot {
        NovaProviderSnapshot(
            id: providerId ?? "",
            name: name,
            endpoint: endpoint ?? "",
            apiKey: apiKey ?? "",
            remoteModel: remoteModel ?? "",
            isEnabled: isEnabled?.value ?? true,
            anthropicEndpoint: anthropicEndpoint)
    }
}

private struct BalancerRow: Decodable {
    let id: String
    let name: String
    let slug: String
    let strategy: String?
    let maxRetries: FlexInt?
    let isEnabled: FlexBool?

    enum CodingKeys: String, CodingKey {
        case id, name, slug, strategy
        case maxRetries = "max_retries"
        case isEnabled = "is_enabled"
    }

    var snapshot: NovaLBSnapshot {
        NovaLBSnapshot(
            id: id, name: name, slug: slug,
            strategy: strategy ?? "round_robin",
            maxRetries: maxRetries?.value ?? 3,
            isEnabled: isEnabled?.value ?? true)
    }
}

private struct MemberRow: Decodable {
    let lbId: String?
    let lbSlug: String?
    let kind: String
    let ref: String
    let weight: FlexInt?
    let isEnabled: FlexBool?

    enum CodingKeys: String, CodingKey {
        case lbId = "lb_id"
        case lbSlug = "lb_slug"
        case kind, ref, weight
        case isEnabled = "is_enabled"
    }

    var snapshot: NovaLBMemberSnapshot {
        NovaLBMemberSnapshot(
            lbSlug: lbSlug, lbId: lbId, kind: kind, ref: ref,
            weight: weight?.value, isEnabled: isEnabled?.value ?? true)
    }
}
