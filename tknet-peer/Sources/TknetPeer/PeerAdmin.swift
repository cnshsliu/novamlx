import Foundation

public struct PeerAdminError: Error, CustomStringConvertible, Equatable {
    public var message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

/// Pure member/interface/pool edits shared by `tknet-peer` and tests. Secrets
/// are written to the store and never included in list text. A member is an
/// (interface, model) pair — URLs resolve through the interface's factory.
public enum PeerAdmin {

    // MARK: - Members

    public static func addMember(
        _ config: inout PeerConfig, secrets: any SecretStore = NullSecretStore(),
        id: String, name: String,
        interfaceId: String, upstreamModel: String,
        apiKey: String? = nil,
        weight: Int? = nil, isEnabled: Bool = true
    ) throws {
        try validateIdentity(id)
        try validateWeight(weight)
        guard config.interface(id: interfaceId) != nil else {
            throw PeerAdminError("No interface \(interfaceId).")
        }
        let upstream = upstreamModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !upstream.isEmpty else { throw PeerAdminError("Upstream model is required.") }
        let ref: String? = (apiKey?.isEmpty == false) ? "src/\(id)" : nil
        config.upsertMember(MemberConfig(
            id: id, name: name.isEmpty ? id : name, interfaceId: interfaceId,
            upstreamModel: upstream, apiKeyRef: ref, weight: weight, isEnabled: isEnabled))
        if let ref, let apiKey {
            try secrets.save(apiKey, for: ref)
        }
    }

    public static func updateMember(
        _ config: inout PeerConfig, secrets: any SecretStore = NullSecretStore(),
        id: String,
        name: String? = nil, interfaceId: String? = nil,
        upstreamModel: String? = nil, apiKey: String? = nil,
        weight: Int? = nil, isEnabled: Bool? = nil
    ) throws {
        guard var member = config.member(id: id) else {
            throw PeerAdminError("No member \(id).")
        }
        if let name { member.name = name }
        if let interfaceId {
            guard config.interface(id: interfaceId) != nil else {
                throw PeerAdminError("No interface \(interfaceId).")
            }
            member.interfaceId = interfaceId
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
        if let isEnabled { member.isEnabled = isEnabled }
        if let apiKey {
            // nil = leave unchanged; "" = back to the factory key; value = own.
            let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                member.apiKeyRef = nil
            } else {
                let ref = member.apiKeyRef ?? "src/\(member.id)"
                member.apiKeyRef = ref
                try secrets.save(trimmed, for: ref)
            }
        }
        config.upsertMember(member)
    }

    public static func removeMember(_ config: inout PeerConfig, id: String) throws {
        guard config.member(id: id) != nil else { throw PeerAdminError("No member \(id).") }
        config.removeMember(id: id)
    }

    // MARK: - Factories and interfaces

    /// Create or update a factory. `baseUrl` keeps scheme + host + port only;
    /// a pasted path or query is dropped. An optional `apiKey` seeds/updates
    /// the shared factory key (an ALL-UPPERCASE value names an env variable).
    /// Member URLs always derive from the base, so a base change needs no
    /// member rewrite.
    public static func saveFactory(
        _ config: inout PeerConfig, secrets: any SecretStore,
        id: String?, name: String, baseUrl: URL, apiKey: String?
    ) throws {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { throw PeerAdminError("Factory name is required.") }
        guard let base = normalizeFactoryBase(baseUrl) else {
            throw PeerAdminError("Factory URL must be an http(s) host[:port].")
        }
        let factoryId = id ?? UUID().uuidString
        for other in config.factories where other.id != factoryId {
            if PeerConfig.normalizedBase(other.baseUrl) == PeerConfig.normalizedBase(base) {
                throw PeerAdminError("A factory with URL \(other.baseUrl.absoluteString) already exists.")
            }
        }
        let existing = config.factory(id: factoryId)
        let ref = existing?.apiKeyRef ?? "factory/\(factoryId)"
        config.upsertFactory(FactoryConfig(
            id: factoryId, name: trimmedName, baseUrl: base, apiKeyRef: ref,
            interfaces: existing?.interfaces ?? []))
        if let apiKey { try secrets.save(apiKey, for: ref) }
    }

    /// Drop a factory, its interfaces, and every member that relied on them.
    public static func removeFactory(_ config: inout PeerConfig, id: String) throws {
        guard config.factory(id: id) != nil else { throw PeerAdminError("No factory \(id).") }
        config.removeFactory(id: id)
    }

    /// Attach an interface (calling template) to a factory. Idempotent per
    /// kind: the existing interface id is returned.
    @discardableResult
    public static func ensureInterface(
        _ config: inout PeerConfig, factoryId: String, kind: InterfaceKind
    ) throws -> String {
        guard config.factory(id: factoryId) != nil else {
            throw PeerAdminError("No factory \(factoryId).")
        }
        guard let interfaceId = config.ensureInterface(kind: kind, on: factoryId) else {
            throw PeerAdminError("No factory \(factoryId).")
        }
        return interfaceId
    }

    public static func removeInterface(
        _ config: inout PeerConfig, factoryId: String, kind: InterfaceKind
    ) throws {
        guard var factory = config.factory(id: factoryId),
              let interface = factory.interface(of: kind) else {
            throw PeerAdminError("No \(kind.rawValue) interface on factory \(factoryId).")
        }
        for member in config.members(ofInterface: interface.id) {
            config.removeMember(id: member.id)
        }
        factory.interfaces.removeAll { $0.id == interface.id }
        config.upsertFactory(factory)
    }

    // MARK: - Pools

    /// Name clients see. A stored name that is empty or just the internal id
    /// falls back to the upstream model string.
    public static func publicModel(_ member: MemberConfig) -> String {
        let name = member.name.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty || name == member.id { return member.upstreamModel }
        return name
    }

    /// The interface kind a member resolves through, or nil when dangling.
    public static func interfaceKind(
        of member: MemberConfig, in config: PeerConfig
    ) -> InterfaceKind? {
        config.memberInterface(member)?.kind
    }

    /// Puts `memberId` in the pool named `model`, creating that pool when it
    /// does not exist. The pool carries the member's interface kind; joining
    /// members must share it — cross-factory is fine, cross-template is not.
    public static func ensureModelPool(_ config: inout PeerConfig, model: String, memberId: String) throws {
        let poolName = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !poolName.isEmpty else { throw PeerAdminError("Model name is required.") }
        guard let member = config.member(id: memberId) else { throw PeerAdminError("No member \(memberId).") }
        guard let kind = interfaceKind(of: member, in: config) else {
            throw PeerAdminError("Member \(memberId) has no interface.")
        }
        if var pool = config.pool(named: poolName) {
            if let poolKind = pool.kind, poolKind != kind {
                throw PeerAdminError(
                    "Pool \(poolName) is \(poolKind.displayName); member \(memberId) is \(kind.displayName).")
            }
            pool.kind = kind
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
        config.upsertPool(PoolConfig(id: poolId, name: poolName, kind: kind, memberIds: [memberId]))
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
        kind: InterfaceKind? = nil
    ) throws {
        let poolName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !poolName.isEmpty else { throw PeerAdminError("Pool name is required.") }
        let poolId = (id?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 } ?? poolName
        guard maxRetries >= 1 else { throw PeerAdminError("max-retries must be at least 1.") }
        guard !memberIds.isEmpty else { throw PeerAdminError("A pool needs at least one --member.") }
        var resolvedKind = kind
        for memberId in memberIds {
            guard let member = config.member(id: memberId) else {
                throw PeerAdminError("No member \(memberId).")
            }
            guard let memberKind = interfaceKind(of: member, in: config) else {
                throw PeerAdminError("Member \(memberId) has no interface.")
            }
            if let resolvedKind, resolvedKind != memberKind {
                throw PeerAdminError(
                    "Pool members must share one interface: \(resolvedKind.displayName) vs \(memberKind.displayName).")
            }
            resolvedKind = memberKind
        }
        config.upsertPool(PoolConfig(
            id: poolId, name: poolName, strategy: strategy, kind: resolvedKind ?? .openaiChat,
            memberIds: memberIds, maxRetries: maxRetries))
    }

    public static func updatePool(
        _ config: inout PeerConfig, id: String, name: String? = nil,
        strategy: PoolStrategy? = nil, maxRetries: Int? = nil,
        addMembers: [String] = [], removeMembers: [String] = [], isEnabled: Bool? = nil
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
        let poolKind = pool.kind ?? .openaiChat
        for memberId in addMembers {
            guard let member = config.member(id: memberId) else {
                throw PeerAdminError("No member \(memberId).")
            }
            guard let memberKind = interfaceKind(of: member, in: config) else {
                throw PeerAdminError("Member \(memberId) has no interface.")
            }
            guard memberKind == poolKind else {
                throw PeerAdminError(
                    "Pool is \(poolKind.displayName); member \(memberId) is \(memberKind.displayName).")
            }
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

    public static func interfaceKind(_ raw: String) throws -> InterfaceKind {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let kind = InterfaceKind(rawValue: trimmed) else {
            let kinds = InterfaceKind.allCases.map(\.rawValue).joined(separator: ", ")
            throw PeerAdminError("Unknown interface \(raw). Use one of: \(kinds).")
        }
        return kind
    }

    /// Operator-facing text. Never includes secret values.
    public static func memberList(_ config: PeerConfig) -> String {
        if config.members.isEmpty { return "(no members)" }
        return config.members.map { member in
            let weight = member.weight.map(String.init) ?? "-"
            let interface = config.memberInterface(member)
            let factory = interface.flatMap { config.factory(ofInterface: $0.id) }
            let kind = interface?.kind ?? InterfaceKind.openaiChat
            let url = config.memberEndpointURL(member)?.absoluteString ?? "-"
            return "\(member.id)\t\(member.name)\t\(url)\tinterface=\(kind.rawValue)\tupstream=\(member.upstreamModel)\tenabled=\(member.isEnabled)\tweight=\(weight)\tfactory=\(factory?.name ?? "-")"
        }.joined(separator: "\n")
    }

    public static func poolList(_ config: PeerConfig) -> String {
        if config.pools.isEmpty { return "(no pools)" }
        return config.pools.map { pool in
            "\(pool.id)\t\(pool.name)\tstrategy=\(pool.strategy.rawValue)\tretries=\(pool.maxRetries)\tenabled=\(pool.isEnabled)\tinterface=\(pool.kind?.rawValue ?? "openai.chat")\tmembers=\(pool.memberIds.joined(separator: ","))"
        }.joined(separator: "\n")
    }

    // MARK: - Validation helpers

    /// scheme://host[:port], any path/query stripped. nil when not http(s).
    public static func normalizeFactoryBase(_ url: URL) -> URL? {
        guard var parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let scheme = parts.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = parts.host, !host.isEmpty else { return nil }
        parts.scheme = scheme
        parts.path = ""
        parts.query = nil
        parts.fragment = nil
        return parts.url
    }

    private static func validateIdentity(_ id: String) throws {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed == id, !id.contains("/") else {
            throw PeerAdminError("Member id must be a non-empty token without '/'.")
        }
    }

    private static func validateWeight(_ weight: Int?) throws {
        if let weight, weight < 1 { throw PeerAdminError("Weight must be at least 1.") }
    }
}
