import Foundation

// Legacy JSON structures used ONLY for importing old data.

struct LegacyConfig: Codable {
    var server: LegacyServer?
    var huggingfaceEndpoint: String?
    var modelsDirectory: String?
    var apiKeys: [String]?
    var authUrl: String?
    var tknetApiKey: String?

    var host: String { server?.host ?? "0.0.0.0" }
    var port: Int { server?.port ?? 6590 }
    var adminPort: Int { server?.adminPort ?? 6591 }
    var tlsEnabled: Bool? { server?.tlsEnabled }
    var tlsCertPath: String? { server?.tlsCertPath }
    var tlsKeyPath: String? { server?.tlsKeyPath }
    var defaultModel: String? { server?.defaultModel }
    var modelsDir: String? { modelsDirectory }
    var hfEndpoint: String? { huggingfaceEndpoint }
    var clusterConfig: String? { server?.clusterJSON }
    var autoLoad: String? { server?.autoLoadJSON }
    var logLevel: String? { server?.logLevel }
}

struct LegacyServer: Codable {
    var host: String?
    var port: Int?
    var adminPort: Int?
    var tlsEnabled: Bool?
    var tlsCertPath: String?
    var tlsKeyPath: String?
    var defaultModel: String?
    var apiKey: String?
    var cluster: String?
    var autoLoad: AutoLoadValue?
    var logLevel: String?

    var clusterJSON: String? { cluster }
    var autoLoadJSON: String? {
        if let autoLoad {
            return (try? JSONEncoder().encode(autoLoad)).flatMap { String(data: $0, encoding: .utf8) }
        }
        return nil
    }

    // Accept both String and Dict for autoLoad
    enum AutoLoadValue: Codable {
        case string(String)
        case dict([String: AnyCodable])

        func encode(to encoder: Encoder) throws {
            switch self {
            case .string(let s): try s.encode(to: encoder)
            case .dict(let d): try d.encode(to: encoder)
            }
        }

        init(from decoder: Decoder) throws {
            if let s = try? decoder.singleValueContainer().decode(String.self) {
                self = .string(s)
            } else if let d = try? decoder.singleValueContainer().decode([String: AnyCodable].self) {
                self = .dict(d)
            } else {
                self = .string("{}")
            }
        }
    }
}

// Type-erased Codable wrapper
struct AnyCodable: Codable, @unchecked Sendable {
    let value: Any

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Bool.self) { value = v }
        else if let v = try? container.decode(Int.self) { value = v }
        else if let v = try? container.decode(Double.self) { value = v }
        else if let v = try? container.decode(String.self) { value = v }
        else if let v = try? container.decode([AnyCodable].self) { value = v.map { $0.value } }
        else if let v = try? container.decode([String: AnyCodable].self) { value = v.mapValues { $0.value } }
        else { value = NSNull() }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let v = value as? Bool { try container.encode(v) }
        else if let v = value as? Int { try container.encode(v) }
        else if let v = value as? Double { try container.encode(v) }
        else if let v = value as? String { try container.encode(v) }
        else if let v = value as? [Any] { try container.encode(v.map { AnyCodable(value: $0) }) }
        else if let v = value as? [String: Any] { try container.encode(v.mapValues { AnyCodable(value: $0) }) }
        else { try container.encodeNil() }
    }

    init(value: Any) { self.value = value }
}

struct LegacyModelSettingsContainer: Codable {
    var settings: [String: LegacyModelSettings]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        settings = try container.decode([String: LegacyModelSettings].self)
    }
}

struct LegacyModelSettings: Codable {
    var alias: String?
    var isDefault: Bool?
    var isPinned: Bool?
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var ttlSeconds: Int?
    var contextWindow: Int?
    var draftModel: String?
}

struct LegacyModelRecord: Codable {
    var family: String?
    var modelType: String?
    var source: String?
    var localPath: String?
    var remoteUrl: String?
    var sizeBytes: Int64?
    var downloadedAt: Date?
    var version: String?
    var architecture: String?
}

struct LegacyMetrics: Codable {
    var totalRequests: Int64 = 0
    var totalTokens: Int64 = 0
    var totalInferenceTimeMs: Int64 = 0
    var cacheHits: Int64 = 0
    var cacheMisses: Int64 = 0
    var evictions: Int64 = 0
    var perModelStats: [String: ModelStatEntry]?
    var perModelCache: [String: CacheEntry]?

    struct ModelStatEntry: Codable {
        var requests: Int64?
        var tokens: Int64?
        var inferenceTimeMs: Int64?
    }

    struct CacheEntry: Codable {
        var hits: Int64?
        var misses: Int64?
    }
}

struct LegacyWorkerDeployment: Codable {
    var phase: String?
    var username: String?
    var version: String?
    var startedAt: Date?
    var updatedAt: Date?
}

struct LegacyAuthCache: Codable {
    var valid: Bool?
    var plan: String?
    var status: String?
    var cancelAtPeriodEnd: Bool?
    var expiresAt: Date?
    var cachedAt: Date?
    var userEmail: String?
}

struct LegacyChatRecord: Codable {
    var id: String
    var title: String?
    var model: String
    var systemPrompt: String?
    var createdAt: Date
    var updatedAt: Date
    var messages: [LegacyChatMessage]
}

struct LegacyChatMessage: Codable {
    var id: String?
    var role: String
    var content: String?
    var thinkingContent: String?
    var createdAt: Date
}
