import Foundation

public struct PeerAdminError: Error, CustomStringConvertible, Equatable {
    public var message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

/// Pure member/pool edits shared by `tknet-peer` and tests. Secrets are written
/// to the store and never included in list text.
public enum PeerAdmin {
    public static func addMember(
        _ config: inout PeerConfig, secrets: any SecretStore,
        id: String, name: String, endpoint: URL, upstreamModel: String,
        apiKey: String?, weight: Int?, type: SourceType,
        isEnabled: Bool = true
    ) throws {
        try validateIdentity(id)
        try validateEndpoint(endpoint)
        try validateWeight(weight)
        let upstream = upstreamModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !upstream.isEmpty else { throw PeerAdminError("Upstream model is required.") }
        let ref = "src/\(id)"
        config.upsertMember(MemberConfig(
            id: id, name: name.isEmpty ? id : name, type: type, endpoint: endpoint,
            apiKeyRef: ref, upstreamModel: upstream, weight: weight,
            isEnabled: isEnabled))
        if let apiKey { try secrets.save(apiKey, for: ref) }
    }

    public static func updateMember(
        _ config: inout PeerConfig, secrets: any SecretStore, id: String,
        name: String? = nil, endpoint: URL? = nil, upstreamModel: String? = nil,
        apiKey: String? = nil, weight: Int? = nil,
        type: SourceType? = nil, isEnabled: Bool? = nil
    ) throws {
        guard var member = config.member(id: id) else {
            throw PeerAdminError("No member \(id).")
        }
        if let name { member.name = name }
        if let endpoint {
            try validateEndpoint(endpoint)
            member.endpoint = endpoint
        }
        if let upstreamModel {
            let upstream = upstreamModel.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !upstream.isEmpty else { throw PeerAdminError("Upstream model is required.") }
            member.upstreamModel = upstream
        }
        if let weight {
            try validateWeight(weight)
            member.weight = weight
        }
        if let type { member.type = type }
        if let isEnabled { member.isEnabled = isEnabled }
        config.upsertMember(member)
        if let apiKey { try secrets.save(apiKey, for: member.apiKeyRef) }
    }

    public static func removeMember(_ config: inout PeerConfig, id: String) throws {
        guard config.member(id: id) != nil else { throw PeerAdminError("No member \(id).") }
        config.removeMember(id: id)
    }

    /// Name clients see. A stored name that is empty or just the internal id
    /// falls back to the upstream model string.
    public static func publicModel(_ member: MemberConfig) -> String {
        let name = member.name.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty || name == member.id { return member.upstreamModel }
        return name
    }

    /// Puts `memberId` in the pool named `model`, creating that pool when
    /// it does not exist. A second endpoint for the same model joins it.
    public static func ensureModelPool(_ config: inout PeerConfig, model: String, memberId: String) throws {
        let poolName = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !poolName.isEmpty else { throw PeerAdminError("Model name is required.") }
        guard config.member(id: memberId) != nil else { throw PeerAdminError("No member \(memberId).") }
        if var pool = config.pool(named: poolName) {
            if !pool.memberIds.contains(memberId) {
                pool.memberIds.append(memberId)
            }
            config.upsertPool(pool)
            return
        }
        var poolId = poolName
        if config.pool(id: poolId) != nil {
            var suffix = 2
            while config.pool(id: "\(poolName)-\(suffix)") != nil {
                suffix += 1
            }
            poolId = "\(poolName)-\(suffix)"
        }
        config.upsertPool(PoolConfig(id: poolId, name: poolName, memberIds: [memberId]))
    }

    /// Drops the member from the pool with this published name. Other pools stay.
    public static func detachMember(_ config: inout PeerConfig, memberId: String, fromPoolNamed name: String) {
        let poolName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var pool = config.pool(named: poolName) else { return }
        guard pool.memberIds.contains(memberId) else { return }
        pool.memberIds.removeAll { $0 == memberId }
        config.upsertPool(pool)
    }

    public static func addPool(
        _ config: inout PeerConfig, id: String?, name: String,
        strategy: PoolStrategy = .roundRobin, maxRetries: Int = 3, memberIds: [String],
        api: APIContract? = nil
    ) throws {
        let poolName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !poolName.isEmpty else { throw PeerAdminError("Pool name is required.") }
        let poolId = (id?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 } ?? poolName
        guard maxRetries >= 1 else { throw PeerAdminError("max-retries must be at least 1.") }
        guard !memberIds.isEmpty else { throw PeerAdminError("A pool needs at least one --member.") }
        for memberId in memberIds where config.member(id: memberId) == nil {
            throw PeerAdminError("No member \(memberId).")
        }
        config.upsertPool(PoolConfig(
            id: poolId, name: poolName, strategy: strategy,
            memberIds: memberIds, maxRetries: maxRetries, api: api))
    }

    public static func updatePool(
        _ config: inout PeerConfig, id: String, name: String? = nil,
        strategy: PoolStrategy? = nil, maxRetries: Int? = nil,
        addMembers: [String] = [], removeMembers: [String] = [], isEnabled: Bool? = nil,
        api: APIContract? = nil, setApi: Bool = false
    ) throws {
        guard var pool = config.pool(id: id) else { throw PeerAdminError("No pool \(id).") }
        if let name {
            let poolName = name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !poolName.isEmpty else { throw PeerAdminError("Pool name is required.") }
            pool.name = poolName
        }
        if let strategy { pool.strategy = strategy }
        if let maxRetries {
            guard maxRetries >= 1 else { throw PeerAdminError("max-retries must be at least 1.") }
            pool.maxRetries = maxRetries
        }
        if let isEnabled { pool.isEnabled = isEnabled }
        if setApi { pool.api = api }
        for memberId in addMembers {
            guard config.member(id: memberId) != nil else { throw PeerAdminError("No member \(memberId).") }
            if !pool.memberIds.contains(memberId) { pool.memberIds.append(memberId) }
        }
        for memberId in removeMembers {
            pool.memberIds.removeAll { $0 == memberId }
        }
        config.upsertPool(pool)
    }

    public static func removePool(_ config: inout PeerConfig, id: String) throws {
        guard config.pool(id: id) != nil else { throw PeerAdminError("No pool \(id).") }
        config.removePool(id: id)
    }

    public static func strategy(_ raw: String) throws -> PoolStrategy {
        guard let strategy = PoolStrategy(rawValue: raw) else {
            throw PeerAdminError(
                "Unknown strategy \(raw). Use round_robin, weighted, lowest_latency, or random.")
        }
        return strategy
    }

    public static func apiContract(_ raw: String) throws -> APIContract {
        let kind = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let api = SupplyAPI.contract(kind: kind) else {
            let kinds = SupplyAPI.allContracts.map(\.kind).joined(separator: ", ")
            throw PeerAdminError("Unknown interface \(raw). Use one of: \(kinds).")
        }
        return api
    }

    public static func sourceType(_ raw: String) throws -> SourceType {
        guard let type = SourceType(rawValue: raw) else {
            throw PeerAdminError("Unknown member type \(raw). Use openaiCompatible, anthropic, or localNovaMLX.")
        }
        return type
    }

    /// Operator-facing text. Never includes secret values.
    public static func memberList(_ config: PeerConfig) -> String {
        if config.members.isEmpty { return "(no members)" }
        return config.members.map { member in
            let weight = member.weight.map(String.init) ?? "-"
            return "\(member.id)\t\(member.name)\t\(member.endpoint.absoluteString)\tupstream=\(member.upstreamModel)\tenabled=\(member.isEnabled)\tweight=\(weight)\ttype=\(member.type.rawValue)"
        }.joined(separator: "\n")
    }

    public static func poolList(_ config: PeerConfig) -> String {
        if config.pools.isEmpty { return "(no pools)" }
        return config.pools.map { pool in
            "\(pool.id)\t\(pool.name)\tstrategy=\(pool.strategy.rawValue)\tretries=\(pool.maxRetries)\tenabled=\(pool.isEnabled)\tapi=\(pool.api?.kind ?? "openai.chat")\tmembers=\(pool.memberIds.joined(separator: ","))"
        }.joined(separator: "\n")
    }

    private static func validateIdentity(_ id: String) throws {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed == id, !id.contains("/") else {
            throw PeerAdminError("Member id must be a non-empty token without '/'.")
        }
    }

    private static func validateEndpoint(_ endpoint: URL) throws {
        guard let scheme = endpoint.scheme?.lowercased(), scheme == "http" || scheme == "https",
              endpoint.host != nil else {
            throw PeerAdminError("Endpoint must be an http(s) URL.")
        }
    }

    private static func validateWeight(_ weight: Int?) throws {
        if let weight, weight < 1 { throw PeerAdminError("Weight must be at least 1.") }
    }
}
