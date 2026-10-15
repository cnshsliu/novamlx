import Foundation

/// Selection strategy for a pool. Raw values match the remaining NovaMLX
/// `LBStrategy` strings so a migration can read the same names.
public enum PoolStrategy: String, Codable, Sendable, CaseIterable, Equatable {
    /// Equal rotation across enabled members.
    case roundRobin = "round_robin"
    /// Repeat each member by weight, then rotate.
    case weighted
    /// Lowest average latency first. Cold start (no successes) sorts as 0.
    case lowestLatency = "lowest_latency"
    /// Order by a stable hash of the member id.
    case random

    /// `tiered` and any unknown name load as round robin. Tiered ranked a
    /// loopback member above everyone else, which does not match a gateway
    /// whose inference sits on another machine.
    public static func resolved(_ raw: String?) -> PoolStrategy {
        guard let raw, let known = PoolStrategy(rawValue: raw) else { return .roundRobin }
        return known
    }
}

/// One upstream the gateway can forward to. Evolved from `SourceConfig`:
/// the key itself never lives here, only `apiKeyRef`.
public struct MemberConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var type: SourceType
    public var endpoint: URL
    /// Reference into the SecretStore — the key itself never lives here.
    public var apiKeyRef: String
    /// Model name this member expects on the wire.
    public var upstreamModel: String
    public var weight: Int?
    public var isEnabled: Bool

    public init(id: String, name: String, type: SourceType, endpoint: URL,
                apiKeyRef: String, upstreamModel: String,
                weight: Int? = nil, isEnabled: Bool = true) {
        self.id = id
        self.name = name
        self.type = type
        self.endpoint = endpoint
        self.apiKeyRef = apiKeyRef
        self.upstreamModel = upstreamModel
        self.weight = weight
        self.isEnabled = isEnabled
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, type, endpoint, apiKeyRef, upstreamModel, weight, isEnabled
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        type = try c.decode(SourceType.self, forKey: .type)
        endpoint = try c.decode(URL.self, forKey: .endpoint)
        apiKeyRef = try c.decode(String.self, forKey: .apiKeyRef)
        upstreamModel = try c.decode(String.self, forKey: .upstreamModel)
        weight = try c.decodeIfPresent(Int.self, forKey: .weight)
        isEnabled = try c.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(type, forKey: .type)
        try c.encode(endpoint, forKey: .endpoint)
        try c.encode(apiKeyRef, forKey: .apiKeyRef)
        try c.encode(upstreamModel, forKey: .upstreamModel)
        try c.encodeIfPresent(weight, forKey: .weight)
        try c.encode(isEnabled, forKey: .isEnabled)
    }
}

/// A named set of members plus the strategy used to order them.
/// `api` is the interface this pool serves. Absent means chat only
/// (legacy configs and pathless tunnel frames).
public struct PoolConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var strategy: PoolStrategy
    public var memberIds: [String]
    public var maxRetries: Int
    public var isEnabled: Bool
    /// Bumped once per accepted request, not once per retry.
    public var requestCount: Int
    /// Interface paths this pool accepts. Nil → chat completions only.
    public var api: APIContract?

    public init(id: String, name: String, strategy: PoolStrategy = .roundRobin,
                memberIds: [String], maxRetries: Int = 3, isEnabled: Bool = true,
                requestCount: Int = 0, api: APIContract? = nil) {
        self.id = id
        self.name = name
        self.strategy = strategy
        self.memberIds = memberIds
        self.maxRetries = maxRetries
        self.isEnabled = isEnabled
        self.requestCount = requestCount
        self.api = api
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, strategy, memberIds, maxRetries, isEnabled, requestCount, api
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
        strategy = PoolStrategy.resolved(try c.decodeIfPresent(String.self, forKey: .strategy))
        memberIds = try c.decodeIfPresent([String].self, forKey: .memberIds) ?? []
        maxRetries = try c.decodeIfPresent(Int.self, forKey: .maxRetries) ?? 3
        isEnabled = try c.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
        requestCount = try c.decodeIfPresent(Int.self, forKey: .requestCount) ?? 0
        api = try c.decodeIfPresent(APIContract.self, forKey: .api)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(strategy, forKey: .strategy)
        try c.encode(memberIds, forKey: .memberIds)
        try c.encode(maxRetries, forKey: .maxRetries)
        try c.encode(isEnabled, forKey: .isEnabled)
        try c.encode(requestCount, forKey: .requestCount)
        try c.encodeIfPresent(api, forKey: .api)
    }
}

/// Loopback detection for `X-No-Autoload`.
public enum MemberAddress {
    /// `127.0.0.1`, `localhost`, and `::1` only — the addresses NovaMLX binds.
    public static func isLoopback(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased(), !host.isEmpty else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    /// NovaMLX on this machine: an explicit preset, or any loopback endpoint.
    /// Used only to send `X-No-Autoload`, not to rank the member.
    public static func isLocalTier(_ member: MemberConfig) -> Bool {
        member.type == .localNovaMLX || isLoopback(member.endpoint)
    }
}

/// Legacy `sources` + `capabilities` → one pool per published model name.
///
/// Lucas (2026-10-11): one `capability.model` → one pool. Pools may share a
/// member id when they use the same upstream. When the preferred pool id
/// (`first member id`) is already taken, the id becomes
/// `memberId-model` so each published name still gets its own pool.
public enum PoolMigration {
    public static func build(_ capabilities: [Capability]) -> (pools: [PoolConfig], poolIds: [String]) {
        var modelOrder: [String] = []
        var memberIdsByModel: [String: [String]] = [:]
        var indexesByModel: [String: [Int]] = [:]
        for (index, capability) in capabilities.enumerated() {
            if memberIdsByModel[capability.model] == nil {
                modelOrder.append(capability.model)
                memberIdsByModel[capability.model] = []
                indexesByModel[capability.model] = []
            }
            indexesByModel[capability.model, default: []].append(index)
            if !memberIdsByModel[capability.model, default: []].contains(capability.sourceId) {
                memberIdsByModel[capability.model, default: []].append(capability.sourceId)
            }
        }

        var used: Set<String> = []
        var pools: [PoolConfig] = []
        var poolIds = Array(repeating: "", count: capabilities.count)
        for model in modelOrder {
            let memberIds = memberIdsByModel[model] ?? []
            guard let first = memberIds.first else { continue }
            var id = first
            if used.contains(id) {
                id = "\(first)-\(model)"
                var suffix = 2
                while used.contains(id) {
                    id = "\(first)-\(model)-\(suffix)"
                    suffix += 1
                }
            }
            used.insert(id)
            pools.append(PoolConfig(id: id, name: model, memberIds: memberIds))
            for index in indexesByModel[model] ?? [] {
                poolIds[index] = id
            }
        }
        return (pools, poolIds)
    }
}
