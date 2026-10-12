import Foundation

/// Selection strategy for a pool. Raw values match NovaMLX `LBStrategy`
/// (`Sources/NovaMLXCore/LoadBalancerTypes.swift`) so a later migration can
/// read the same strings.
public enum PoolStrategy: String, Codable, Sendable, CaseIterable, Equatable {
    /// Loopback / local NovaMLX, then free, then paid. Rotate inside a tier.
    case tiered
    /// Equal rotation across enabled members.
    case roundRobin = "round_robin"
    /// Repeat each member by weight, then rotate.
    case weighted
    /// Lowest average latency first. Cold start (no successes) sorts as 0.
    case lowestLatency = "lowest_latency"
    /// Order by a stable hash of the member id.
    case random
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
    public var isFree: Bool

    public init(id: String, name: String, type: SourceType, endpoint: URL,
                apiKeyRef: String, upstreamModel: String,
                weight: Int? = nil, isEnabled: Bool = true, isFree: Bool = false) {
        self.id = id
        self.name = name
        self.type = type
        self.endpoint = endpoint
        self.apiKeyRef = apiKeyRef
        self.upstreamModel = upstreamModel
        self.weight = weight
        self.isEnabled = isEnabled
        self.isFree = isFree
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, type, endpoint, apiKeyRef, upstreamModel, weight, isEnabled, isFree
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
        isFree = try c.decodeIfPresent(Bool.self, forKey: .isFree) ?? false
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
        try c.encode(isFree, forKey: .isFree)
    }
}

/// A named set of members plus the strategy used to order them.
public struct PoolConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var strategy: PoolStrategy
    public var memberIds: [String]
    public var maxRetries: Int
    public var isEnabled: Bool
    /// Bumped once per accepted request, not once per retry.
    public var requestCount: Int

    public init(id: String, name: String, strategy: PoolStrategy = .tiered,
                memberIds: [String], maxRetries: Int = 3, isEnabled: Bool = true,
                requestCount: Int = 0) {
        self.id = id
        self.name = name
        self.strategy = strategy
        self.memberIds = memberIds
        self.maxRetries = maxRetries
        self.isEnabled = isEnabled
        self.requestCount = requestCount
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, strategy, memberIds, maxRetries, isEnabled, requestCount
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
        strategy = try c.decodeIfPresent(PoolStrategy.self, forKey: .strategy) ?? .tiered
        memberIds = try c.decodeIfPresent([String].self, forKey: .memberIds) ?? []
        maxRetries = try c.decodeIfPresent(Int.self, forKey: .maxRetries) ?? 3
        isEnabled = try c.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
        requestCount = try c.decodeIfPresent(Int.self, forKey: .requestCount) ?? 0
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
    }
}

/// Loopback detection for tiered ranking and `X-No-Autoload`.
public enum MemberAddress {
    /// `127.0.0.1`, `localhost`, and `::1` only — the addresses NovaMLX binds.
    public static func isLoopback(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased(), !host.isEmpty else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    /// Local tier: an explicit NovaMLX preset, or any loopback endpoint.
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
