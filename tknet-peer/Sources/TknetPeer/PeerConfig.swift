import Foundation

/// One server this Mac has registered with. The identity is `peerId` in
/// `PeerConfig.registrations` plus the local connection key. `peerName` is
/// only the label last typed for that server.
public struct ServerMemory: Codable, Equatable, Sendable, Identifiable {
    public var server: String
    public var peerName: String
    public var members: [MemberConfig]
    public var factories: [FactoryConfig]
    public var pools: [PoolConfig]
    public var capabilities: [Capability]
    public var retiredDemandIds: [String]
    public var id: String { server }

    public init(server: String, peerName: String, members: [MemberConfig] = [],
                factories: [FactoryConfig] = [],
                pools: [PoolConfig] = [], capabilities: [Capability] = [],
                retiredDemandIds: [String] = []) {
        self.server = server
        self.peerName = peerName
        self.members = members
        self.factories = factories
        self.pools = pools
        self.capabilities = capabilities
        self.retiredDemandIds = retiredDemandIds
    }

    private enum CodingKeys: String, CodingKey {
        case server, peerName, sources, members, factories, pools, capabilities, retiredDemandIds
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        server = try c.decode(String.self, forKey: .server)
        peerName = try c.decodeIfPresent(String.self, forKey: .peerName) ?? ""
        if let members = try c.decodeIfPresent([MemberConfig].self, forKey: .members) {
            self.members = members
        } else {
            self.members = try c.decodeIfPresent([MemberConfig].self, forKey: .sources) ?? []
        }
        factories = try c.decodeIfPresent([FactoryConfig].self, forKey: .factories) ?? []
        capabilities = try c.decodeIfPresent([Capability].self, forKey: .capabilities) ?? []
        retiredDemandIds = try c.decodeIfPresent([String].self, forKey: .retiredDemandIds) ?? []
        pools = try c.decodeIfPresent([PoolConfig].self, forKey: .pools) ?? []
        migrateLegacyPoolsIfNeeded()
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(server, forKey: .server)
        try c.encode(peerName, forKey: .peerName)
        try c.encode(members, forKey: .members)
        try c.encode(factories, forKey: .factories)
        try c.encode(pools, forKey: .pools)
        try c.encode(capabilities, forKey: .capabilities)
        try c.encode(retiredDemandIds, forKey: .retiredDemandIds)
    }

    /// One-time upgrade for files written before factories existed: group
    /// members by endpoint `scheme://host:port`. Local NovaMLX members stay
    /// ungrouped (nil factoryId) — they are special-cased elsewhere and only
    /// appear under "All". Idempotent: skipped once any factory is present.
    public mutating func migrateFactoriesIfNeeded() {
        guard factories.isEmpty, !members.isEmpty else { return }
        factories = PeerConfig.deriveFactories(from: &members)
    }

    /// Fill `pools` from capabilities when a row still has only members.
    /// Rewrites `capability.sourceId` to the pool id (unchanged when the
    /// published model had the source to itself).
    public mutating func migrateLegacyPoolsIfNeeded() {
        guard pools.isEmpty, !members.isEmpty else { return }
        let built = PoolMigration.build(capabilities)
        pools = built.pools
        for index in capabilities.indices where built.poolIds.indices.contains(index) {
            let poolId = built.poolIds[index]
            if !poolId.isEmpty { capabilities[index].sourceId = poolId }
        }
    }
}

public struct PeerConfig: Codable, Equatable, Sendable {
    public var peerId: String?
    public var serverURL: URL
    /// Per-server identity registry (Lucas 2026-10-06): a machine holds ONE
    /// identity PER SERVER. The key is the tunnel URL. The value is the
    /// server-issued peerId. The name is not the identity.
    public var registrations: [String: String]
    /// Saved servers for the picker. A row exists after a successful
    /// registration, or after migrating an older registrations map.
    public var serverMemory: [ServerMemory] = []
    public var members: [MemberConfig]
    public var factories: [FactoryConfig] = []
    public var pools: [PoolConfig]
    public var capabilities: [Capability]
    public var concurrencyLimit: Int
    public var requestTimeoutSeconds: Double
    public var idleMinutesBeforeSlowPoll: Double
    /// OpenAI-compatible bind address for the local gateway.
    public var gatewayHost: String = "127.0.0.1"
    /// `0` asks the OS for an ephemeral port (tests). Production default is 6600.
    public var gatewayPort: Int = 6600
    /// Set when `tknet-peer migrate-from-novamlx` wrote this file. A later run
    /// is a no-op until `--force`.
    public var migratedFromNovaAt: Date? = nil
    /// DemandIds the server stopped listing. Declarations are kept (UI greying);
    /// nothing is deleted. Persisted so retirement survives restarts.
    private var retiredDemandIds: Set<String> = []

    public init(peerId: String? = nil, serverURL: URL, registrations: [String: String] = [:],
                members: [MemberConfig] = [], factories: [FactoryConfig] = [],
                pools: [PoolConfig] = [],
                capabilities: [Capability] = [],
                concurrencyLimit: Int = 1, requestTimeoutSeconds: Double = 300,
                idleMinutesBeforeSlowPoll: Double = 5) {
        self.peerId = peerId
        self.serverURL = serverURL
        self.registrations = registrations
        self.members = members
        self.factories = factories
        self.pools = pools
        self.capabilities = capabilities
        self.concurrencyLimit = concurrencyLimit
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.idleMinutesBeforeSlowPoll = idleMinutesBeforeSlowPoll
    }

    private enum CodingKeys: String, CodingKey {
        case peerId, serverURL, sources, members, factories, pools, capabilities, concurrencyLimit, registrations
        case requestTimeoutSeconds, idleMinutesBeforeSlowPoll, retiredDemandIds, serverMemory
        case gatewayHost, gatewayPort, migratedFromNovaAt
    }

    /// Custom decoding so config files written before retirement tracking
    /// (and hand-edited files) still load: every key is optional with defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        peerId = try c.decodeIfPresent(String.self, forKey: .peerId)
        serverURL = try c.decode(URL.self, forKey: .serverURL)
        registrations = (try? c.decode([String: String].self, forKey: .registrations)) ?? [:]
        serverMemory = (try? c.decode([ServerMemory].self, forKey: .serverMemory)) ?? []
        if let members = try c.decodeIfPresent([MemberConfig].self, forKey: .members) {
            self.members = members
        } else {
            members = try c.decodeIfPresent([MemberConfig].self, forKey: .sources) ?? []
        }
        capabilities = try c.decodeIfPresent([Capability].self, forKey: .capabilities) ?? []
        pools = try c.decodeIfPresent([PoolConfig].self, forKey: .pools) ?? []
        concurrencyLimit = try c.decodeIfPresent(Int.self, forKey: .concurrencyLimit) ?? 1
        requestTimeoutSeconds = try c.decodeIfPresent(Double.self, forKey: .requestTimeoutSeconds) ?? 300
        idleMinutesBeforeSlowPoll = try c.decodeIfPresent(Double.self, forKey: .idleMinutesBeforeSlowPoll) ?? 5
        retiredDemandIds = try c.decodeIfPresent(Set<String>.self, forKey: .retiredDemandIds) ?? []
        gatewayHost = try c.decodeIfPresent(String.self, forKey: .gatewayHost) ?? "127.0.0.1"
        gatewayPort = try c.decodeIfPresent(Int.self, forKey: .gatewayPort) ?? 6600
        migratedFromNovaAt = Self.decodeMigrationStamp(c)
        // One-time upgrade: pre-factory files carry no `factories` key, so
        // derive factories from endpoint host:port groups on first load. The
        // key is always written after that (even empty), so a deliberate
        // delete-the-last-factory is never resurrected.
        if let decoded = try? c.decodeIfPresent([FactoryConfig].self, forKey: .factories) {
            factories = decoded
        } else {
            migrateFactoriesIfNeeded()
            for index in serverMemory.indices {
                serverMemory[index].migrateFactoriesIfNeeded()
            }
        }
        migrateLegacyPoolsIfNeeded()
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(peerId, forKey: .peerId)
        try c.encode(serverURL, forKey: .serverURL)
        try c.encode(registrations, forKey: .registrations)
        try c.encode(serverMemory, forKey: .serverMemory)
        try c.encode(members, forKey: .members)
        try c.encode(factories, forKey: .factories)
        try c.encode(pools, forKey: .pools)
        try c.encode(capabilities, forKey: .capabilities)
        try c.encode(concurrencyLimit, forKey: .concurrencyLimit)
        try c.encode(requestTimeoutSeconds, forKey: .requestTimeoutSeconds)
        try c.encode(idleMinutesBeforeSlowPoll, forKey: .idleMinutesBeforeSlowPoll)
        try c.encode(retiredDemandIds, forKey: .retiredDemandIds)
        try c.encode(gatewayHost, forKey: .gatewayHost)
        try c.encode(gatewayPort, forKey: .gatewayPort)
        if let migratedFromNovaAt {
            try c.encode(Self.migrationStampString(migratedFromNovaAt), forKey: .migratedFromNovaAt)
        }
    }

    private static func migrationStampString(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    private static func decodeMigrationStamp(_ c: KeyedDecodingContainer<CodingKeys>) -> Date? {
        if let text = try? c.decodeIfPresent(String.self, forKey: .migratedFromNovaAt) {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            return formatter.date(from: text)
        }
        if let seconds = try? c.decodeIfPresent(Double.self, forKey: .migratedFromNovaAt) {
            return Date(timeIntervalSinceReferenceDate: seconds)
        }
        return nil
    }

    /// When `pools` is missing, derive one pool per published model and point
    /// each capability's `sourceId` at that pool. No-op once pools exist.
    public mutating func migrateLegacyPoolsIfNeeded() {
        if pools.isEmpty, !members.isEmpty {
            let built = PoolMigration.build(capabilities)
            pools = built.pools
            for index in capabilities.indices where built.poolIds.indices.contains(index) {
                let poolId = built.poolIds[index]
                if !poolId.isEmpty { capabilities[index].sourceId = poolId }
            }
        }
        for index in serverMemory.indices {
            serverMemory[index].migrateLegacyPoolsIfNeeded()
        }
    }

    /// One-time upgrade for files written before factories existed: group
    /// members by endpoint `scheme://host:port`. Local NovaMLX members stay
    /// ungrouped (nil factoryId) — they are special-cased elsewhere and only
    /// appear under "All". Idempotent: skipped once any factory is present.
    public mutating func migrateFactoriesIfNeeded() {
        guard factories.isEmpty, !members.isEmpty else { return }
        factories = Self.deriveFactories(from: &members)
    }

    /// Group `members` by endpoint base, assigning `factoryId` in place.
    /// Local NovaMLX members are left ungrouped.
    public static func deriveFactories(from members: inout [MemberConfig]) -> [FactoryConfig] {
        var byBase: [String: FactoryConfig] = [:]
        for index in members.indices {
            guard members[index].type != .localNovaMLX,
                  let base = factoryBase(of: members[index].endpoint) else { continue }
            let factory = byBase[base] ?? FactoryConfig(
                id: UUID().uuidString, name: factoryName(for: base),
                baseUrl: members[index].endpoint)
            byBase[base] = factory
            members[index].factoryId = factory.id
        }
        return byBase.values.sorted { $0.name < $1.name }
    }

    /// Canonical `scheme://host[:port]` of an endpoint, or nil when not http(s).
    public static func factoryBase(of url: URL) -> String? {
        guard var parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let scheme = parts.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = parts.host, !host.isEmpty else { return nil }
        parts.path = ""
        parts.query = nil
        parts.fragment = nil
        return parts.url?.absoluteString
    }

    private static func factoryName(for base: String) -> String {
        guard let parts = URLComponents(string: base), let host = parts.host else { return base }
        if let port = parts.port,
           !((parts.scheme?.lowercased() == "https" && port == 443)
             || (parts.scheme?.lowercased() == "http" && port == 80)) {
            return "\(host):\(port)"
        }
        return host
    }

    public static func defaultConfig() -> PeerConfig {
        PeerConfig(serverURL: URL(string: "wss://tknet.ai")!)
    }

    /// Capabilities whose demandId is still in the server's demand list.
    public var activeCapabilities: [Capability] {
        capabilities.filter { !retiredDemandIds.contains($0.demandId) }
    }

    /// Capabilities for hello / capabilities.update: pool.api is the
    /// truth source. A missing pool api means chat (legacy configs).
    public var wireCapabilities: [Capability] {
        activeCapabilities.map { cap in
            var next = cap
            if let pool = pool(id: cap.sourceId) {
                next.api = pool.api ?? SupplyAPI.chat
            } else if next.api == nil {
                next.api = SupplyAPI.chat
            }
            return next
        }
    }

    /// Marks entries whose demand retired. Returns the retired demandIds.
    /// Nothing is deleted: sources are reusable assets (spec: Demand lifecycle).
    @discardableResult
    public mutating func markRetired(demandIds: Set<String>) -> [String] {
        // Phase 1: server simply stops dispatching retired demands; the peer
        // keeps declarations for UI greying. The demand list itself is the
        // source of truth, exposed via PeerService.
        let mine = Set(capabilities.map(\.demandId)).intersection(demandIds)
        retiredDemandIds.formUnion(mine)
        return mine.sorted()
    }

    /// Two-way reconciliation against the server's live demand list: ids this
    /// peer declares but the list no longer carries are retired, and ids that
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

    public func retiredDemandIdList() -> [String] { retiredDemandIds.sorted() }

    public mutating func setBindings(members: [MemberConfig], factories: [FactoryConfig] = [],
                                     pools: [PoolConfig] = [],
                                     capabilities: [Capability], retiredDemandIds: [String]) {
        self.members = members
        self.factories = factories
        self.pools = pools
        self.capabilities = capabilities
        self.retiredDemandIds = Set(retiredDemandIds)
    }

    public func member(id: String) -> MemberConfig? {
        members.first { $0.id == id }
    }

    public func factory(id: String) -> FactoryConfig? {
        factories.first { $0.id == id }
    }

    public func members(in factory: FactoryConfig) -> [MemberConfig] {
        members.filter { $0.factoryId == factory.id }
    }

    public mutating func upsertFactory(_ factory: FactoryConfig) {
        if let index = factories.firstIndex(where: { $0.id == factory.id }) {
            factories[index] = factory
        } else {
            factories.append(factory)
        }
    }

    /// Drop the factory. `deleteMembers` false (default) keeps its members,
    /// orphaned (nil factoryId); true removes them and strips pool ids.
    public mutating func removeFactory(id: String, deleteMembers: Bool = false) {
        guard factories.contains(where: { $0.id == id }) else { return }
        factories.removeAll { $0.id == id }
        if deleteMembers {
            for member in members where member.factoryId == id {
                removeMember(id: member.id)
            }
        } else {
            for index in members.indices where members[index].factoryId == id {
                members[index].factoryId = nil
            }
        }
    }

    public func pool(id: String) -> PoolConfig? {
        pools.first { $0.id == id }
    }

    /// First pool whose published name matches. Local gateway model lookup.
    public func pool(named name: String) -> PoolConfig? {
        pools.first { $0.name == name }
    }

    public func members(in pool: PoolConfig) -> [MemberConfig] {
        pool.memberIds.compactMap { memberId in members.first { $0.id == memberId } }
    }

    public mutating func upsertMember(_ member: MemberConfig) {
        if let index = members.firstIndex(where: { $0.id == member.id }) {
            members[index] = member
        } else {
            members.append(member)
        }
    }

    /// Drops the member and strips its id from every pool.
    public mutating func removeMember(id: String) {
        members.removeAll { $0.id == id }
        for index in pools.indices {
            pools[index].memberIds.removeAll { $0 == id }
        }
    }

    public mutating func upsertPool(_ pool: PoolConfig) {
        if let index = pools.firstIndex(where: { $0.id == pool.id }) {
            pools[index] = pool
        } else {
            pools.append(pool)
        }
    }

    public mutating func removePool(id: String) {
        pools.removeAll { $0.id == id }
    }

    public mutating func upsertServerMemory(_ memory: ServerMemory) {
        if let index = serverMemory.firstIndex(where: { $0.server == memory.server }) {
            serverMemory[index] = memory
        } else {
            serverMemory.append(memory)
        }
    }

    /// Copy each existing registration into the saved-server list. Older
    /// files have a peerId and no name. Returns true when a row was added.
    @discardableResult
    public mutating func adoptRegistrationsIntoMemory() -> Bool {
        var changed = false
        for key in registrations.keys.sorted() {
            guard let tunnel = URL(string: key), let rest = Self.normalizedREST(tunnel) else { continue }
            if serverMemory.contains(where: { $0.server == rest }) { continue }
            let active = Self.normalizedREST(serverURL) == rest
            upsertServerMemory(ServerMemory(
                server: rest, peerName: "",
                members: active ? members : [],
                factories: active ? factories : [],
                pools: active ? pools : [],
                capabilities: active ? capabilities : [],
                retiredDemandIds: active ? retiredDemandIdList() : []
            ))
            changed = true
        }
        return changed
    }

    /// Save this server's bindings, then make `restURL` the active server.
    /// Restores that server's peerId when this Mac already has one.
    /// Returns false when the URL is invalid or already active.
    @discardableResult
    public mutating func activateServer(_ restURL: URL) -> Bool {
        guard let rest = Self.normalizedREST(restURL), rest != Self.normalizedREST(serverURL) else { return false }
        snapshotActiveBindings()
        guard let asURL = URL(string: rest) else { return false }
        let tunnel = Self.tunnelURL(fromREST: asURL)
        serverURL = tunnel
        peerId = registrations[Self.tunnelKey(tunnel)]
        if let saved = serverMemory.first(where: { $0.server == rest }) {
            setBindings(members: saved.members, factories: saved.factories, pools: saved.pools,
                        capabilities: saved.capabilities,
                        retiredDemandIds: saved.retiredDemandIds)
        } else {
            setBindings(members: [], factories: [], pools: [], capabilities: [], retiredDemandIds: [])
        }
        return true
    }

    /// Drop one saved server. When it was active, switch to another saved
    /// server, or back to an unregistered tknet.ai.
    public mutating func forgetServer(_ rest: String) {
        guard let canonical = Self.normalizedREST(URL(string: rest) ?? URL(fileURLWithPath: "/")) else { return }
        let wasActive = Self.normalizedREST(serverURL) == canonical
        serverMemory.removeAll { $0.server == canonical }
        if let url = URL(string: canonical) {
            registrations.removeValue(forKey: Self.tunnelKey(Self.tunnelURL(fromREST: url)))
        }
        guard wasActive else { return }
        if let next = serverMemory.first, let nextURL = URL(string: next.server) {
            let tunnel = Self.tunnelURL(fromREST: nextURL)
            serverURL = tunnel
            peerId = registrations[Self.tunnelKey(tunnel)]
            setBindings(members: next.members, factories: next.factories, pools: next.pools,
                        capabilities: next.capabilities,
                        retiredDemandIds: next.retiredDemandIds)
        } else {
            peerId = nil
            serverURL = URL(string: "wss://tknet.ai")!
            setBindings(members: [], factories: [], pools: [], capabilities: [], retiredDemandIds: [])
        }
    }

    /// Store the label for a server. Does not change peerId.
    public mutating func rememberPeerName(_ rest: String, name: String) {
        guard let canonical = Self.normalizedREST(URL(string: rest) ?? URL(fileURLWithPath: "/")) else { return }
        var row = serverMemory.first { $0.server == canonical }
            ?? ServerMemory(server: canonical, peerName: name)
        row.peerName = name
        if Self.normalizedREST(serverURL) == canonical {
            row.members = members
            row.factories = factories
            row.pools = pools
            row.capabilities = capabilities
            row.retiredDemandIds = retiredDemandIdList()
        }
        upsertServerMemory(row)
    }

    private mutating func snapshotActiveBindings() {
        guard let current = Self.normalizedREST(serverURL),
              var row = serverMemory.first(where: { $0.server == current }) else { return }
        row.members = members
        row.factories = factories
        row.pools = pools
        row.capabilities = capabilities
        row.retiredDemandIds = retiredDemandIdList()
        upsertServerMemory(row)
    }

    /// `https://host` and `wss://host/` are the same saved server.
    public static func normalizedREST(_ url: URL) -> String? {
        guard var parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let scheme = parts.scheme?.lowercased(),
              let host = parts.host, !host.isEmpty else { return nil }
        switch scheme {
        case "https", "wss": parts.scheme = "https"
        case "http", "ws": parts.scheme = "http"
        default: return nil
        }
        parts.query = nil
        parts.fragment = nil
        parts.user = nil
        parts.password = nil
        if parts.path == "/" { parts.path = "" }
        guard var text = parts.url?.absoluteString else { return nil }
        while text.hasSuffix("/") { text.removeLast() }
        return text
    }

    public static func tunnelURL(fromREST base: URL) -> URL {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false),
              let scheme = components.scheme?.lowercased() else { return base }
        switch scheme {
        case "https": components.scheme = "wss"
        case "http": components.scheme = "ws"
        default: return base
        }
        return components.url ?? base
    }

    public static func restURL(fromTunnel base: URL) -> URL? {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false),
              let scheme = components.scheme?.lowercased() else { return nil }
        switch scheme {
        case "wss": components.scheme = "https"
        case "ws": components.scheme = "http"
        default: return base
        }
        return components.url
    }

    public static func tunnelKey(_ tunnel: URL) -> String {
        var key = tunnel.absoluteString
        while key.count > 1, key.hasSuffix("/") { key.removeLast() }
        return key
    }
}

/// The product has one server. localhost exists only in a novamlx checkout.
public enum TknetServers {
    public static let official = "https://tknet.ai"
    public static let localDev = "http://localhost:6188"

    /// True when this app bundle sits inside a `dev/novamlx` checkout.
    /// An installed copy, such as `/Applications/NovaMLX.app`, is false.
    public static func allowsLocalServer(bundlePath: String) -> Bool {
        bundlePath.contains("/dev/novamlx/")
    }

    public static func choices(allowsLocal: Bool) -> [String] {
        allowsLocal ? [official, localDev] : [official]
    }
}

public enum ConfigStore {
    public static func load(from url: URL) throws -> PeerConfig {
        try PeerConfig(from: Data(contentsOf: url))
    }

    /// Atomic write with 0600 so a shared machine can't read peer secrets refs.
    /// `.completeFileProtection` is deliberately absent: it fails EPERM on
    /// volumes that don't support FileVault-style protection (e.g. $TMPDIR),
    /// and the POSIX 0600 chmod is the binding spec requirement.
    public static func save(_ config: PeerConfig, to url: URL) throws {
        var config = config
        config.migrateLegacyPoolsIfNeeded()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}

extension PeerConfig {
    fileprivate init(from data: Data) throws {
        self = try JSONDecoder().decode(PeerConfig.self, from: data)
    }
}
