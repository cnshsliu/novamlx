import Foundation
import NovaMLXDB
import NovaMLXTknetPeer
import NovaMLXUtils

/// Bridges the cross-platform PeerService into the Mac UI. Owned by the
/// app, not the peer page, so switching pages or closing the window does
/// not tear the tunnel down. `stop()`/`deinit` still stop the service —
/// PeerService must never be dropped without `stop()` (the relay's
/// AsyncHTTPClient traps in debug).
@MainActor
public final class TknetPeerState: ObservableObject {
    @Published private(set) var config: PeerConfig

    /// Per-server token storage key — one identity per server.
    private func tokenKey(forTunnel tunnel: URL) -> String {
        let host = tunnel.host ?? "default"
        let port = tunnel.port.map { ":\($0)" } ?? ""
        return "peer/token/\(host)\(port)"
    }

    /// Account session for one server. Separate from the peer connection key.
    private func accountRef(_ rest: URL, _ kind: String) -> String {
        let host = rest.host ?? "default"
        let port = rest.port.map { ":\($0)" } ?? ""
        return "user/\(kind)/\(host)\(port)"
    }

    /// Email last used to sign in on the active server. Empty when none.
    var savedLoginEmail: String {
        guard let server = URL(string: activeREST) else { return "" }
        return secrets.load(accountRef(server, "email")) ?? ""
    }

    /// The peer's connection key, for the "claim this machine" flow on
    /// tknet.ai (paste Peer ID + key into the dashboard). Named 连接密钥 /
    /// connection key in the UI — NEVER "token", which collides with the
    /// LLM token billing everywhere else in the product (Lucas 2026-10-06).
    public var machineKey: String? {
        guard config.peerId != nil else { return nil }
        let v = secrets.load(tokenKey(forTunnel: config.serverURL)) ?? secrets.load("peer/token") ?? ""
        return v.isEmpty ? nil : v
    }
    @Published private(set) var status: PeerStatus?
    @Published private(set) var demand: [DemandEntry] = []
    @Published private(set) var running = false
    @Published private(set) var lastError: String?
    @Published private(set) var earnings: PeerEarningsSummary?
    /// Signed-in tknet account for the active server. Nil means signed out.
    /// The session token itself stays in the secret store.
    @Published private(set) var accountEmail: String?
    @Published private(set) var accountName: String?
    @Published private(set) var accountError: String?
    /// Peer id last claimed on the active server by the signed-in account.
    @Published private(set) var claimedPeerId: String?

    /// Same location the `tknet-peer` CLI uses, so the app and CLI share one
    /// peer identity and source set.
    private let configURL: URL
    private let secrets: FileSecretStore
    private var service: PeerService?
    /// Consumes `service.statusStream` — the stream NEVER finishes, so the
    /// task's lifetime is tied to this object: cancelled in `stop()` (and
    /// `deinit`), otherwise it would capture the service forever.
    private var statusTask: Task<Void, Never>?
    /// Launch auto-connect only. Manual Start keeps the client's own reconnect loop.
    private var watchingLaunch = false
    /// Failed dials during this launch. The first one is the initial try;
    /// the next three are the retries the user asked for.
    private var launchFailures = 0
    private static let maxLaunchRetries = 3
    private static let tunnelWantedKey = "tknetPeer.tunnelWanted"
    /// Single TknetREST held for the object's lifetime; `shutdown()` is
    /// single-use (throws on a second call) so it is called exactly once in
    /// `deinit`.
    private let rest = TknetREST()

    public init() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/tknet-peer")
        self.configURL = dir.appendingPathComponent("peer.json")
        self.secrets = FileSecretStore(directory: dir.appendingPathComponent("secrets"))
        var loaded = (try? ConfigStore.load(from: configURL)) ?? .defaultConfig()
        if loaded.adoptRegistrationsIntoMemory() {
            try? ConfigStore.save(loaded, to: configURL)
        }
        self.config = loaded
    }

    /// REST form of the active server, without a trailing slash.
    var activeREST: String {
        PeerConfig.normalizedREST(config.serverURL) ?? "https://tknet.ai"
    }

    /// Label saved for this server. Empty when this Mac never stored one.
    func peerName(for rest: String) -> String {
        guard let url = URL(string: rest),
              let canonical = PeerConfig.normalizedREST(url) else { return "" }
        return config.serverMemory.first { $0.server == canonical }?.peerName ?? ""
    }

    /// Registered means this Mac holds the server-issued peerId AND the
    /// connection key for that server. The peer name is not part of the check.
    func isRegistered(_ rest: String) -> Bool {
        guard let url = URL(string: rest),
              let canonical = PeerConfig.normalizedREST(url),
              let canonicalURL = URL(string: canonical) else { return false }
        let tunnel = PeerConfig.tunnelURL(fromREST: canonicalURL)
        guard let peerId = config.registrations[PeerConfig.tunnelKey(tunnel)], !peerId.isEmpty else {
            return false
        }
        let token = secrets.load(tokenKey(forTunnel: tunnel)) ?? ""
        return !token.isEmpty
    }

    deinit {
        statusTask?.cancel()
        // PeerService must be stopped before being dropped; the REST client
        // must be shut down exactly once. Both are Sendable, so hand them to
        // a detached task (deinit cannot await).
        let service = self.service
        let rest = self.rest
        Task {
            if let service { await service.stop() }
            try? await rest.shutdown()
        }
    }

    // MARK: - Registration

    /// Register this Mac with the tknet server, storing peerId + token
    /// locally. The token only ever travels to `FileSecretStore` — it is
    /// never logged and never appears in UI state.
    @discardableResult
    func register(server: URL, peerName: String) async -> Bool {
        guard let scheme = server.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = server.host, !host.isEmpty else {
            lastError = "Invalid server URL (expected http/https): \(server.absoluteString)"
            return false
        }
        let trimmedName = peerName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            lastError = "Peer name is empty"
            return false
        }
        await tearDown()
        do {
            var config = self.config
            let tunnel = Self.tunnelURL(fromREST: server)
            // Previous identity on THIS server. The server keeps the same
            // peerId only when this name matches. A different name retires
            // the old peer and issues a new peerId.
            let prevId = config.registrations[PeerConfig.tunnelKey(tunnel)]
            let prevToken = prevId != nil ? (secrets.load(tokenKey(forTunnel: tunnel)) ?? "") : ""
            let (peerId, token) = try await rest.register(server: server, peerName: trimmedName,
                                                          previousPeerId: prevId, previousToken: prevToken.isEmpty ? nil : prevToken)
            config.peerId = peerId
            config.serverURL = tunnel
            config.registrations[PeerConfig.tunnelKey(tunnel)] = peerId
            if let rest = PeerConfig.normalizedREST(server) {
                config.rememberPeerName(rest, name: trimmedName)
            }
            secrets.save(token, for: tokenKey(forTunnel: tunnel))
            secrets.save(token, for: "peer/token")
            persist(config)
            await fetchDemand()
            return lastError == nil
        } catch {
            NovaMLXLog.error("[TknetPeer] register failed for \(server.absoluteString): \(error)")
            lastError = Self.userMessage(error)
            return false
        }
    }

    // MARK: - Demand list

    /// Supplier money view (fetched on demand — Refresh on the earnings card).
    func fetchEarnings() async {
        guard config.peerId != nil,
              let token = secrets.load("peer/token"), !token.isEmpty else { return }
        guard let restBase = Self.restURL(fromTunnel: config.serverURL) else { return }
        do {
            earnings = try await rest.fetchEarnings(server: restBase, token: token)
        } catch {
            NovaMLXLog.error("[TknetPeer] fetchEarnings failed: \(error)")
            lastError = Self.userMessage(error)
        }
    }

    /// One-shot REST fetch (used before the tunnel is up); the running
    /// service keeps `demand` fresh via `statusStream` afterwards.
    func fetchDemand() async {
        guard config.peerId != nil,
              let token = secrets.load("peer/token"), !token.isEmpty else { return }
        guard let restBase = Self.restURL(fromTunnel: config.serverURL) else { return }
        do {
            demand = try await rest.fetchDemand(server: restBase, token: token)
        } catch {
            lastError = Self.userMessage(error)
        }
    }

    // MARK: - Account session

    /// Check the stored login. A dead session is dropped. The password is never stored.
    func refreshAccount() async {
        guard let server = URL(string: activeREST) else { return }
        claimedPeerId = secrets.load(accountRef(server, "claimed"))
        guard let session = secrets.load(accountRef(server, "session")), !session.isEmpty else {
            accountEmail = nil
            accountName = nil
            accountError = nil
            return
        }
        do {
            let account = try await rest.currentUser(server: server, session: session)
            accountEmail = account.email
            accountName = account.name
            secrets.save(account.email, for: accountRef(server, "email"))
            accountError = nil
        } catch let error as RESTError {
            if case .badStatus(401, _) = error {
                secrets.delete(accountRef(server, "session"))
                accountEmail = nil
                accountName = nil
            }
            accountError = Self.userMessage(error)
        } catch {
            accountError = Self.userMessage(error)
        }
    }

    /// Sign in and keep the session for this server. Returns false on failure.
    @discardableResult
    func signIn(email: String, password: String) async -> Bool {
        guard let server = URL(string: activeREST) else {
            accountError = "Invalid server URL (expected http/https)"
            return false
        }
        let trimmed = email.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !password.isEmpty else {
            accountError = "Email and password required"
            return false
        }
        do {
            let login = try await rest.login(server: server, email: trimmed, password: password)
            secrets.save(login.session, for: accountRef(server, "session"))
            secrets.save(login.account.email, for: accountRef(server, "email"))
            accountEmail = login.account.email
            accountName = login.account.name
            accountError = nil
            return true
        } catch {
            NovaMLXLog.error("[TknetPeer] sign-in failed for \(server.absoluteString): \(error)")
            accountError = Self.userMessage(error)
            return false
        }
    }

    func signOut() async {
        guard let server = URL(string: activeREST) else { return }
        if let session = secrets.load(accountRef(server, "session")), !session.isEmpty {
            try? await rest.logout(server: server, session: session)
        }
        secrets.delete(accountRef(server, "session"))
        accountEmail = nil
        accountName = nil
        accountError = nil
    }

    /// Bind this Mac's peer to the signed-in account.
    @discardableResult
    func claimThisMac() async -> Bool {
        guard let server = URL(string: activeREST) else {
            accountError = "Invalid server URL (expected http/https)"
            return false
        }
        guard accountEmail != nil,
              let session = secrets.load(accountRef(server, "session")), !session.isEmpty else {
            accountError = "Sign in before claiming this Mac."
            return false
        }
        guard let peerId = config.peerId, !peerId.isEmpty, let key = machineKey else {
            accountError = "Register this Mac before claiming it."
            return false
        }
        do {
            _ = try await rest.claimPeer(server: server, session: session, peerId: peerId, connectionKey: key)
            secrets.save(peerId, for: accountRef(server, "claimed"))
            claimedPeerId = peerId
            accountError = nil
            return true
        } catch {
            NovaMLXLog.error("[TknetPeer] claim failed for \(server.absoluteString): \(error)")
            accountError = Self.userMessage(error)
            return false
        }
    }

    // MARK: - Start / stop

    /// After a process restart, dial the saved peer without a Start click.
    /// Skipped when the user pressed Stop. If that dial never reaches
    /// `connected`, retry at most 3 times, then stop until the next launch
    /// or a manual Start.
    public func restoreTunnelAfterLaunch() async {
        guard tunnelWanted else {
            NovaMLXLog.info("[TknetPeer] auto-connect skipped (tunnel was stopped)")
            return
        }
        guard config.peerId != nil else { return }
        NovaMLXLog.info("[TknetPeer] auto-connect after launch")
        await start(limitLaunchAttempts: true)
    }

    func start(limitLaunchAttempts: Bool = false) async {
        guard service == nil, config.peerId != nil else { return }
        guard let token = secrets.load("peer/token"), !token.isEmpty else {
            lastError = "No peer token stored — register again."
            return
        }
        setTunnelWanted(true)
        watchingLaunch = limitLaunchAttempts
        launchFailures = 0
        let service = PeerService(
            config: config, secrets: secrets,
            transportFactory: WSTransport.factory(
                server: config.serverURL, tokenRef: "peer/token", secrets: secrets),
            onConfigChange: { [configURL] updated in
                // Demand retire/revive reconciliations must survive restarts.
                try? ConfigStore.save(updated, to: configURL)
            })
        self.service = service
        statusTask = Task { [weak self] in
            for await status in service.statusStream {
                guard let self else { break }
                self.status = status
                self.demand = status.demand
                // Mirror service-side config edits (retire/revive) into the UI.
                self.config = service.currentConfig
                self.noteLaunchConnection(status.connection)
            }
        }
        do {
            try await service.start()
            running = true
            lastError = nil
        } catch {
            watchingLaunch = false
            statusTask?.cancel()
            statusTask = nil
            self.service = nil
            await service.stop()
            running = false
            lastError = Self.userMessage(error)
        }
    }

    func stop() async {
        setTunnelWanted(false)
        await tearDown()
    }

    /// UserDefaults absent means "this install predates the flag": a registered
    /// Mac should come back after a restart. Stop writes false.
    private var tunnelWanted: Bool {
        if UserDefaults.standard.object(forKey: Self.tunnelWantedKey) == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: Self.tunnelWantedKey)
    }

    private func setTunnelWanted(_ wanted: Bool) {
        UserDefaults.standard.set(wanted, forKey: Self.tunnelWantedKey)
    }

    private func noteLaunchConnection(_ connection: TunnelStatus) {
        guard watchingLaunch else { return }
        switch connection {
        case .connected:
            watchingLaunch = false
            launchFailures = 0
            NovaMLXLog.info("[TknetPeer] tunnel connected")
        case .backingOff:
            launchFailures += 1
            let retriesUsed = launchFailures - 1
            guard retriesUsed >= Self.maxLaunchRetries else {
                NovaMLXLog.info("[TknetPeer] connect failed, retry \(retriesUsed + 1)/\(Self.maxLaunchRetries)")
                return
            }
            watchingLaunch = false
            NovaMLXLog.info("[TknetPeer] connect failed, stopped after \(Self.maxLaunchRetries) retries")
            Task { await self.abandonLaunch() }
        case .upgradeRequired:
            watchingLaunch = false
        default:
            break
        }
    }

    private func abandonLaunch() async {
        await tearDown()
        lastError = "Could not open the tunnel after \(Self.maxLaunchRetries) retries."
        NovaMLXLog.error("[TknetPeer] \(lastError ?? "")")
    }

    private func tearDown() async {
        watchingLaunch = false
        statusTask?.cancel()
        statusTask = nil
        guard let service else {
            running = false
            return
        }
        self.service = nil
        await service.stop()
        running = false
        status = nil
    }

    // MARK: - Config edits

    /// Add a local-NovaMLX source bound to a demand entry (loopback preset).
    /// Auto-fills the source key with the first key in the app's APIKeyStore
    /// so the loopback relay is authorized against this app's own server.
    func addLocalSource(for entry: DemandEntry, upstreamModel: String, apiKeyStore: APIKeyStore, api: APIContract?) async {
        let upstream = upstreamModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !upstream.isEmpty else {
            lastError = "This model is not on this Mac."
            return
        }
        var primaryKey = ""
        if let first = (try? apiKeyStore.list())?.first,
           let raw = try? apiKeyStore.getRawKey(id: first.id) {
            primaryKey = raw
        }
        var config = self.config
        let sourceId = "src-\(entry.demandId)"
        config.sources.removeAll { $0.id == sourceId }
        config.sources.append(SourceConfig(
            id: sourceId, name: "NovaMLX (local)", type: .localNovaMLX,
            endpoint: URL(string: "http://127.0.0.1:6590/v1")!,
            apiKeyRef: "src/\(sourceId)", upstreamModel: upstream))
        secrets.save(primaryKey, for: "src/\(sourceId)")
        config.capabilities.removeAll { $0.demandId == entry.demandId }
        config.capabilities.append(Capability(
            demandId: entry.demandId, model: entry.model, sourceId: sourceId,
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0, api: api))
        await commit(config)
    }

    /// Fill a missing interface on supplies already being served, using the
    /// upstream model. A model we cannot place is left blank.
    func fillMissingAPI(modelType: (String) -> String?, family: (String) -> String?) async {
        var config = self.config
        var changed = false
        for index in config.capabilities.indices where config.capabilities[index].api == nil {
            let upstream = config.sources.first { $0.id == config.capabilities[index].sourceId }?.upstreamModel
                ?? config.capabilities[index].model
            let type = modelType(upstream) ?? "unknown"
            let fam = family(upstream) ?? ""
            guard let api = SupplyAPI.contract(modelId: upstream, modelType: type, family: fam) else { continue }
            config.capabilities[index].api = api
            changed = true
        }
        if changed { await commit(config) }
    }

    /// Upstream model name the bound source expects for this capability.
    func upstreamModel(for capability: Capability) -> String {
        config.sources.first { $0.id == capability.sourceId }?.upstreamModel
            ?? capability.model
    }

    func updateUpstreamModel(for demandId: String, to model: String) async {
        guard !model.isEmpty,
              let cap = config.capabilities.first(where: { $0.demandId == demandId }) else { return }
        var config = self.config
        guard let index = config.sources.firstIndex(where: { $0.id == cap.sourceId }) else { return }
        config.sources[index].upstreamModel = model
        await commit(config)
    }

    /// Persist an edited config and, while running, propagate it to the live
    /// relay (capability edits ride the next frame set without a reconnect).
    private func commit(_ config: PeerConfig) async {
        do {
            try ConfigStore.save(config, to: configURL)
            self.config = config
            lastError = nil
        } catch {
            lastError = Self.userMessage(error)
            return
        }
        if let service {
            await service.applyConfig(config)
        }
    }

    /// Make a saved server active. Restores its peerId and connection key.
    /// Does nothing when this server is already active.
    func selectServer(_ restBase: URL) async {
        guard let scheme = restBase.scheme?.lowercased(),
              scheme == "http" || scheme == "https", restBase.host != nil else {
            lastError = "Invalid server URL (expected http/https)"
            return
        }
        guard PeerConfig.normalizedREST(restBase) != activeREST else { return }
        await tearDown()
        var config = self.config
        guard config.activateServer(restBase) else { return }
        applyActiveToken(config)
        persist(config)
        demand = []
        if config.peerId != nil { await fetchDemand() }
    }

    /// Forget one saved server on this Mac. Deletes its peerId and connection
    /// key locally. The account on the server stays.
    func forgetServer(_ rest: String) async {
        guard let url = URL(string: rest), let canonical = PeerConfig.normalizedREST(url),
              let canonicalURL = URL(string: canonical) else {
            lastError = "Invalid server URL (expected http/https)"
            return
        }
        let wasActive = activeREST == canonical
        if wasActive { await tearDown() }
        secrets.delete(tokenKey(forTunnel: PeerConfig.tunnelURL(fromREST: canonicalURL)))
        var config = self.config
        config.forgetServer(canonical)
        if wasActive { applyActiveToken(config) }
        persist(config)
        demand = []
        earnings = nil
        if config.peerId != nil { await fetchDemand() }
    }

    /// Store the peer name on this Mac. Does not call the server, so it
    /// cannot retire the peer or issue a new peerId.
    func rememberName(_ rest: String, name: String) async {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var config = self.config
        config.rememberPeerName(rest, name: trimmed)
        await commit(config)
    }

    private func applyActiveToken(_ config: PeerConfig) {
        if config.peerId != nil {
            let perServer = secrets.load(tokenKey(forTunnel: config.serverURL)) ?? ""
            if perServer.isEmpty {
                secrets.delete("peer/token")
            } else {
                secrets.save(perServer, for: "peer/token")
            }
        } else {
            secrets.delete("peer/token")
        }
    }

    private func persist(_ config: PeerConfig) {
        do {
            try FileManager.default.createDirectory(at: configURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try ConfigStore.save(config, to: configURL)
            self.config = config
            lastError = nil
        } catch {
            lastError = Self.userMessage(error)
        }
    }

    /// Server text when the error has one. The debug dump stays in the log.
    private static func userMessage(_ error: Error) -> String {
        if let text = (error as? LocalizedError)?.errorDescription, !text.isEmpty {
            return text
        }
        return String(describing: error)
    }

    // MARK: - URL scheme helpers (mirror TknetPeerCLI)

    private static func tunnelURL(fromREST base: URL) -> URL {
        PeerConfig.tunnelURL(fromREST: base)
    }

    public static func restURL(fromTunnel base: URL) -> URL? {
        PeerConfig.restURL(fromTunnel: base)
    }
}
