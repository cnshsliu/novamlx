import Foundation

/// An upstream the peer can forward to. `localNovaMLX` is a loopback preset
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

/// One server this Mac has registered with. The identity is `peerId` in
/// `PeerConfig.registrations` plus the local connection key. `peerName` is
/// only the label last typed for that server.
public struct ServerMemory: Codable, Equatable, Sendable, Identifiable {
    public var server: String
    public var peerName: String
    public var sources: [SourceConfig]
    public var capabilities: [Capability]
    public var retiredDemandIds: [String]
    public var id: String { server }

    public init(server: String, peerName: String, sources: [SourceConfig] = [],
                capabilities: [Capability] = [], retiredDemandIds: [String] = []) {
        self.server = server
        self.peerName = peerName
        self.sources = sources
        self.capabilities = capabilities
        self.retiredDemandIds = retiredDemandIds
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
    public var sources: [SourceConfig]
    public var capabilities: [Capability]
    public var concurrencyLimit: Int
    public var requestTimeoutSeconds: Double
    public var idleMinutesBeforeSlowPoll: Double
    /// DemandIds the server stopped listing. Declarations are kept (UI greying);
    /// nothing is deleted. Persisted so retirement survives restarts.
    private var retiredDemandIds: Set<String> = []

    public init(peerId: String? = nil, serverURL: URL, registrations: [String: String] = [:],
                sources: [SourceConfig] = [], capabilities: [Capability] = [],
                concurrencyLimit: Int = 1, requestTimeoutSeconds: Double = 300,
                idleMinutesBeforeSlowPoll: Double = 5) {
        self.peerId = peerId; self.serverURL = serverURL; self.registrations = registrations; self.sources = sources
        self.capabilities = capabilities; self.concurrencyLimit = concurrencyLimit
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.idleMinutesBeforeSlowPoll = idleMinutesBeforeSlowPoll
    }

    private enum CodingKeys: String, CodingKey {
        case peerId, serverURL, sources, capabilities, concurrencyLimit, registrations
        case requestTimeoutSeconds, idleMinutesBeforeSlowPoll, retiredDemandIds, serverMemory
    }

    /// Custom decoding so config files written before retirement tracking
    /// (and hand-edited files) still load: every key is optional with defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        peerId = try c.decodeIfPresent(String.self, forKey: .peerId)
        serverURL = try c.decode(URL.self, forKey: .serverURL)
        registrations = (try? c.decode([String: String].self, forKey: .registrations)) ?? [:]
        serverMemory = (try? c.decode([ServerMemory].self, forKey: .serverMemory)) ?? []
        sources = try c.decodeIfPresent([SourceConfig].self, forKey: .sources) ?? []
        capabilities = try c.decodeIfPresent([Capability].self, forKey: .capabilities) ?? []
        concurrencyLimit = try c.decodeIfPresent(Int.self, forKey: .concurrencyLimit) ?? 1
        requestTimeoutSeconds = try c.decodeIfPresent(Double.self, forKey: .requestTimeoutSeconds) ?? 300
        idleMinutesBeforeSlowPoll = try c.decodeIfPresent(Double.self, forKey: .idleMinutesBeforeSlowPoll) ?? 5
        retiredDemandIds = try c.decodeIfPresent(Set<String>.self, forKey: .retiredDemandIds) ?? []
    }

    public static func defaultConfig() -> PeerConfig {
        PeerConfig(serverURL: URL(string: "wss://tknet.ai")!)
    }

    /// Capabilities whose demandId is still in the server's demand list.
    public var activeCapabilities: [Capability] {
        capabilities.filter { !retiredDemandIds.contains($0.demandId) }
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

    public mutating func setBindings(sources: [SourceConfig], capabilities: [Capability],
                                     retiredDemandIds: [String]) {
        self.sources = sources
        self.capabilities = capabilities
        self.retiredDemandIds = Set(retiredDemandIds)
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
                sources: active ? sources : [],
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
            setBindings(sources: saved.sources, capabilities: saved.capabilities,
                        retiredDemandIds: saved.retiredDemandIds)
        } else {
            setBindings(sources: [], capabilities: [], retiredDemandIds: [])
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
            setBindings(sources: next.sources, capabilities: next.capabilities,
                        retiredDemandIds: next.retiredDemandIds)
        } else {
            peerId = nil
            serverURL = URL(string: "wss://tknet.ai")!
            setBindings(sources: [], capabilities: [], retiredDemandIds: [])
        }
    }

    /// Store the label for a server. Does not change peerId.
    public mutating func rememberPeerName(_ rest: String, name: String) {
        guard let canonical = Self.normalizedREST(URL(string: rest) ?? URL(fileURLWithPath: "/")) else { return }
        var row = serverMemory.first { $0.server == canonical }
            ?? ServerMemory(server: canonical, peerName: name)
        row.peerName = name
        if Self.normalizedREST(serverURL) == canonical {
            row.sources = sources
            row.capabilities = capabilities
            row.retiredDemandIds = retiredDemandIdList()
        }
        upsertServerMemory(row)
    }

    private mutating func snapshotActiveBindings() {
        guard let current = Self.normalizedREST(serverURL),
              var row = serverMemory.first(where: { $0.server == current }) else { return }
        row.sources = sources
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
