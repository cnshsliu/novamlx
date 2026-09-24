import Foundation

/// An upstream the node can forward to. `localNovaMLX` is a loopback preset
/// over the app's own OpenAI-compatible API (spec Phase-1 deviation).
public struct SourceConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var type: SourceType
    public var endpoint: URL
    /// Reference into the SecretStore — the key itself never lives here.
    public var apiKeyRef: String
    /// Model name the source expects for capabilities bound to this source.
    public var upstreamModel: String

    public init(id: String, name: String, type: SourceType, endpoint: URL,
                apiKeyRef: String, upstreamModel: String) {
        self.id = id; self.name = name; self.type = type
        self.endpoint = endpoint; self.apiKeyRef = apiKeyRef; self.upstreamModel = upstreamModel
    }
}

public struct NodeConfig: Codable, Equatable, Sendable {
    public var nodeId: String?
    public var serverURL: URL
    public var sources: [SourceConfig]
    public var capabilities: [Capability]
    public var concurrencyLimit: Int
    public var requestTimeoutSeconds: Double
    public var idleMinutesBeforeSlowPoll: Double
    /// DemandIds the server stopped listing. Declarations are kept (UI greying);
    /// nothing is deleted. Persisted so retirement survives restarts.
    private var retiredDemandIds: Set<String> = []

    public init(nodeId: String? = nil, serverURL: URL,
                sources: [SourceConfig] = [], capabilities: [Capability] = [],
                concurrencyLimit: Int = 1, requestTimeoutSeconds: Double = 300,
                idleMinutesBeforeSlowPoll: Double = 5) {
        self.nodeId = nodeId; self.serverURL = serverURL; self.sources = sources
        self.capabilities = capabilities; self.concurrencyLimit = concurrencyLimit
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.idleMinutesBeforeSlowPoll = idleMinutesBeforeSlowPoll
    }

    private enum CodingKeys: String, CodingKey {
        case nodeId, serverURL, sources, capabilities, concurrencyLimit
        case requestTimeoutSeconds, idleMinutesBeforeSlowPoll, retiredDemandIds
    }

    /// Custom decoding so config files written before retirement tracking
    /// (and hand-edited files) still load: every key is optional with defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        nodeId = try c.decodeIfPresent(String.self, forKey: .nodeId)
        serverURL = try c.decode(URL.self, forKey: .serverURL)
        sources = try c.decodeIfPresent([SourceConfig].self, forKey: .sources) ?? []
        capabilities = try c.decodeIfPresent([Capability].self, forKey: .capabilities) ?? []
        concurrencyLimit = try c.decodeIfPresent(Int.self, forKey: .concurrencyLimit) ?? 1
        requestTimeoutSeconds = try c.decodeIfPresent(Double.self, forKey: .requestTimeoutSeconds) ?? 300
        idleMinutesBeforeSlowPoll = try c.decodeIfPresent(Double.self, forKey: .idleMinutesBeforeSlowPoll) ?? 5
        retiredDemandIds = try c.decodeIfPresent(Set<String>.self, forKey: .retiredDemandIds) ?? []
    }

    public static func defaultConfig() -> NodeConfig {
        NodeConfig(serverURL: URL(string: "wss://tknet.ai")!)
    }

    /// Capabilities whose demandId is still in the server's demand list.
    public var activeCapabilities: [Capability] {
        capabilities.filter { !retiredDemandIds.contains($0.demandId) }
    }

    /// Marks entries whose demand retired. Returns the retired demandIds.
    /// Nothing is deleted: sources are reusable assets (spec: Demand lifecycle).
    @discardableResult
    public mutating func markRetired(demandIds: Set<String>) -> [String] {
        // Phase 1: server simply stops dispatching retired demands; the node
        // keeps declarations for UI greying. The demand list itself is the
        // source of truth, exposed via NodeService.
        let mine = Set(capabilities.map(\.demandId)).intersection(demandIds)
        retiredDemandIds.formUnion(mine)
        return mine.sorted()
    }

    /// Two-way reconciliation against the server's live demand list: ids this
    /// node declares but the list no longer carries are retired, and ids that
    /// come back are un-retired (retirement is a latch, not a tombstone).
    /// Returns the ids newly retired by this call. Declarations are never
    /// deleted (spec: Demand lifecycle — sources are reusable assets).
    @discardableResult
    public mutating func reconcile(demandIds: Set<String>) -> [String] {
        let mine = Set(capabilities.map(\.demandId))
        retiredDemandIds.subtract(mine.intersection(demandIds))
        let newlyRetired = mine.subtracting(demandIds).subtracting(retiredDemandIds)
        retiredDemandIds.formUnion(newlyRetired)
        return newlyRetired.sorted()
    }
}

public enum ConfigStore {
    public static func load(from url: URL) throws -> NodeConfig {
        try NodeConfig(from: Data(contentsOf: url))
    }

    /// Atomic write with 0600 so a shared machine can't read node secrets refs.
    public static func save(_ config: NodeConfig, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try data.write(to: url, options: [.atomic, .completeFileProtection])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}

extension NodeConfig {
    fileprivate init(from data: Data) throws {
        self = try JSONDecoder().decode(NodeConfig.self, from: data)
    }
}
