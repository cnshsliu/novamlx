import TknetPeer
import Combine
import Foundation

/// Latest tunnel URL, read on every dial. Registration can switch servers
/// without tearing the local gateway down.
final class TunnelServerBox: @unchecked Sendable {
    private let lock = NSLock()
    private var url: URL

    init(_ url: URL) { self.url = url }

    var current: URL {
        lock.lock()
        defer { lock.unlock() }
        return url
    }

    func update(_ url: URL) {
        lock.lock()
        defer { lock.unlock() }
        self.url = url
    }
}

/// Menu-bar owner of the local gateway and the tknet tunnel.
///
/// Secrets are read from `~/.config/tknet-peer/secrets`, the same files
/// `tknet-peer serve` uses. Quit stops the gateway and the tunnel and
/// releases the process lock.
/// Stop stops only the tunnel — the port stays up, and `tunnelWanted` stays
/// false so the next launch does not dial. Retry exhaustion does not clear
/// `tunnelWanted`.
@MainActor
final class PeerAppState: ObservableObject {
    @Published private(set) var config: PeerConfig
    @Published private(set) var status: PeerStatus?
    @Published private(set) var running = false
    @Published private(set) var gatewayPort: Int?
    @Published private(set) var lastError: String?
    @Published private(set) var lockError: String?
    @Published private(set) var tunnelWanted: Bool
    @Published private(set) var isBusy = false
    @Published private(set) var isAccountBusy = false
    @Published private(set) var isEarningsBusy = false
    @Published private(set) var earnings: PeerEarningsSummary?
    @Published private(set) var earningsError: String?
    /// Signed-in tknet account. Nil means signed out. The session stays in the secret store.
    @Published private(set) var accountEmail: String?
    @Published private(set) var accountName: String?
    @Published private(set) var accountError: String?
    /// Peer id last claimed on the active server by the signed-in account.
    @Published private(set) var claimedPeerId: String?

    private let configURL: URL
    private let secrets: any SecretStore
    private let defaults: UserDefaults
    private let serverBox: TunnelServerBox
    private var service: PeerService?
    private var statusTask: Task<Void, Never>?
    private var processLock: PeerProcessLock?
    private var launchPolicy = TunnelLaunchPolicy()
    private var launchGeneration = 0
    private var didLaunch = false
    private var isShutdown = false
    private var restClosed = false
    private let rest = TknetREST()

    init(defaults: UserDefaults = .standard) {
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/tknet-peer", isDirectory: true)
        let configURL = directory.appendingPathComponent("peer.json")
        self.configURL = configURL
        let secretsResult = Self.makeSecrets(in: directory)
        self.secrets = secretsResult.store
        self.defaults = defaults
        var loaded = (try? ConfigStore.load(from: configURL)) ?? .defaultConfig()
        if loaded.adoptRegistrationsIntoMemory() {
            try? ConfigStore.save(loaded, to: configURL)
        }
        loaded.migrateLegacyPoolsIfNeeded()
        self.config = loaded
        self.tunnelWanted = TunnelWantedPreference.read(from: defaults)
        self.serverBox = TunnelServerBox(loaded.serverURL)
        self.lastError = secretsResult.migrationError
    }

    /// Email last used to sign in on the active server. Empty when none.
    var savedLoginEmail: String {
        readSecret(accountRef("email")) ?? ""
    }

    var activeREST: String {
        PeerConfig.normalizedREST(config.serverURL) ?? "https://tknet.ai"
    }

    /// Connection key for the active server. Nil when this peer is not registered.
    var connectionKey: String? {
        guard config.peerId != nil else { return nil }
        let value = readSecret(tokenKey(for: config.serverURL)) ?? readSecret("peer/token") ?? ""
        return value.isEmpty ? nil : value
    }

    var isGatewayUp: Bool {
        status?.gatewayListening == true
    }

    // MARK: - Traffic animation

    /// Owns the animation clock. Deliberately a separate ObservableObject —
    /// see TrafficAnimator.
    let trafficAnimator = TrafficAnimator()

    /// Pass-through for callers that only read the phase.
    var trafficPhase: Int? { trafficAnimator.phase }

    func noteTraffic() {
        trafficAnimator.noteTraffic()
    }

    /// Fakes a token stream so the menu-bar animation can be previewed
    /// without a live tunnel. Toggles: start on the first call, stop on the
    /// second.
    @Published private(set) var isSimulatingTraffic = false
    private var trafficSimulation: Task<Void, Never>?

    func toggleTrafficSimulation() {
        if let task = trafficSimulation {
            task.cancel()
            trafficSimulation = nil
            isSimulatingTraffic = false
            return
        }
        isSimulatingTraffic = true
        trafficSimulation = Task { @MainActor in
            while !Task.isCancelled {
                noteTraffic()
                try? await Task.sleep(for: .milliseconds(90))
            }
        }
    }

    /// Dial in progress or the session is up. Idle after retry exhaustion is not active,
    /// so Start can dial again without waiting for the next launch.
    var isTunnelActive: Bool {
        switch status?.connection {
        case .connecting, .connected, .backingOff, .stopping:
            return true
        case .idle, .upgradeRequired, .none:
            return false
        }
    }

    var gatewayLabel: String {
        guard isGatewayUp, let port = gatewayPort ?? status?.gatewayPort else {
            return "not listening"
        }
        return "\(config.gatewayHost):\(port)"
    }

    var connectionLabel: String {
        switch status?.connection {
        case .connected: return "Connected"
        case .connecting: return "Connecting"
        case .backingOff(let seconds): return "Retrying (\(Int(seconds.rounded()))s)"
        case .upgradeRequired: return "Upgrade required"
        case .stopping: return "Stopping"
        case .idle, .none: return "Idle"
        }
    }

    /// In-window status glyph. The menu bar uses `MenuBarIcon` instead.
    var connectionSymbol: String {
        switch status?.connection {
        case .connected: return "circle.fill"
        case .connecting: return "circle.dotted"
        case .backingOff: return "clock.arrow.circlepath"
        case .upgradeRequired: return "exclamationmark.triangle.fill"
        case .stopping: return "circle.lefthalf.filled"
        case .idle, .none: return "circle"
        }
    }

    // MARK: - Launch / quit

    func launch() async {
        guard !didLaunch else { return }
        didLaunch = true
        guard !isShutdown else { return }
        do {
            processLock = try PeerProcessLock.acquire(at: PeerProcessLock.url(besideConfig: configURL))
        } catch {
            let message = Self.userMessage(error)
            lockError = message
            lastError = message
            return
        }
        guard !isShutdown else {
            releaseLock()
            return
        }
        syncActiveToken()
        serverBox.update(config.serverURL)
        let service = PeerService(
            config: config,
            secrets: secrets,
            transportFactory: WSTransport.factory(
                server: { [serverBox] in serverBox.current },
                tokenRef: "peer/token",
                secrets: secrets),
            onConfigChange: { [configURL] updated in
                try? ConfigStore.save(updated, to: configURL)
            })
        self.service = service
        statusTask = Task { [weak self] in
            for await status in service.statusStream {
                guard let self else { break }
                self.apply(status, from: service)
            }
        }
        do {
            try await service.start(tunnel: false)
        } catch {
            lastError = Self.userMessage(error)
            return
        }
        guard !isShutdown else { return }
        running = true
        gatewayPort = service.gatewayListenPort
        Task { await self.refreshAccount() }
        await dialIfWanted()
    }

    /// Full teardown. Quit calls this; Stop does not.
    func shutdown() async {
        guard !isShutdown else { return }
        isShutdown = true
        launchGeneration += 1
        launchPolicy.stopWatching()
        statusTask?.cancel()
        statusTask = nil
        let service = self.service
        self.service = nil
        if let service {
            await service.stop()
        }
        releaseLock()
        if !restClosed {
            restClosed = true
            try? await rest.shutdown()
        }
        running = false
    }

    // MARK: - Tunnel

    func startTunnelFromUser() async {
        guard lockError == nil, !isBusy else { return }
        guard let service else {
            lastError = "Gateway is not running."
            return
        }
        isBusy = true
        defer { isBusy = false }
        launchGeneration += 1
        launchPolicy.stopWatching()
        TunnelWantedPreference.write(true, to: defaults)
        tunnelWanted = true
        do {
            try await service.startTunnel()
            lastError = nil
        } catch {
            lastError = Self.userMessage(error)
        }
    }

    func stopTunnelFromUser() async {
        // Flip preference first so the footer updates even if teardown awaits
        // a stuck dial. Do not hold isBusy across stopTunnel — that left Stop
        // permanently grey while status stayed Connecting.
        launchGeneration += 1
        launchPolicy.stopWatching()
        TunnelWantedPreference.write(false, to: defaults)
        tunnelWanted = false
        await service?.stopTunnel()
    }

    // MARK: - Registration

    @discardableResult
    func register(serverText: String, peerName: String) async -> Bool {
        guard lockError == nil, !isShutdown else { return false }
        let trimmedServer = serverText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let server = URL(string: trimmedServer),
              let scheme = server.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              server.host?.isEmpty == false else {
            lastError = "Invalid server URL (expected http/https)."
            return false
        }
        let trimmedName = peerName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            lastError = "Peer name is empty."
            return false
        }
        do {
            var config = self.config
            let tunnel = PeerConfig.tunnelURL(fromREST: server)
            let previousId = config.registrations[PeerConfig.tunnelKey(tunnel)]
            let stored = previousId == nil ? "" : (readSecret(tokenKey(for: tunnel)) ?? "")
            let (peerId, token) = try await rest.register(
                server: server,
                peerName: trimmedName,
                previousPeerId: previousId,
                previousToken: stored.isEmpty ? nil : stored)
            guard !isShutdown else { return false }
            config.peerId = peerId
            config.serverURL = tunnel
            config.registrations[PeerConfig.tunnelKey(tunnel)] = peerId
            if let restBase = PeerConfig.normalizedREST(server) {
                config.rememberPeerName(restBase, name: trimmedName)
            }
            try secrets.save(token, for: tokenKey(for: tunnel))
            try secrets.save(token, for: "peer/token")
            earnings = nil
            earningsError = nil
            serverBox.update(tunnel)
            guard await persist(config) else { return false }
            if tunnelWanted {
                launchGeneration += 1
                launchPolicy.stopWatching()
                try await service?.startTunnel()
            }
            return true
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
    }

    // MARK: - Members and pools

    /// Saves a member under one public model name. The internal id is kept on
    /// edit and generated on create. The member joins the pool of that name.
    func saveMember(
        existingId: String?, model: String, upstreamOverride: String, type: SourceType,
        endpoint: URL, apiKey: String?, weight: Int?, isEnabled: Bool, isFree: Bool
    ) async -> Bool {
        guard lockError == nil else { return false }
        let publicModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !publicModel.isEmpty else {
            lastError = "Model name is required."
            return false
        }
        let override = upstreamOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        let upstream = override.isEmpty ? publicModel : override
        var config = self.config
        let previous = existingId.flatMap { config.member(id: $0) }
        let id = previous?.id ?? Self.freshMemberId(in: config)
        do {
            try PeerAdmin.addMember(
                &config, secrets: secrets, id: id, name: publicModel, endpoint: endpoint,
                upstreamModel: upstream, apiKey: apiKey, weight: weight,
                isFree: isFree, type: type, isEnabled: isEnabled)
            if let previous {
                let oldModel = PeerAdmin.publicModel(previous)
                if oldModel != publicModel {
                    PeerAdmin.detachMember(&config, memberId: id, fromPoolNamed: oldModel)
                }
            }
            try PeerAdmin.ensureModelPool(&config, model: publicModel, memberId: id)
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
        return await persist(config)
    }

    private static func freshMemberId(in config: PeerConfig) -> String {
        for _ in 0..<8 {
            let id = "m-" + UUID().uuidString.lowercased().prefix(8)
            if config.member(id: id) == nil { return id }
        }
        return "m-" + UUID().uuidString.lowercased()
    }

    func deleteMember(id: String) async -> Bool {
        guard lockError == nil else { return false }
        var config = self.config
        do {
            try PeerAdmin.removeMember(&config, id: id)
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
        return await persist(config)
    }

    func savePool(
        id: String, name: String, strategy: PoolStrategy, memberIds: [String],
        maxRetries: Int, isEnabled: Bool, isNew: Bool
    ) async -> Bool {
        guard lockError == nil else { return false }
        guard !memberIds.isEmpty else {
            lastError = "A pool needs at least one member."
            return false
        }
        var config = self.config
        do {
            if isNew || config.pool(id: id) == nil {
                try PeerAdmin.addPool(
                    &config, id: id, name: name, strategy: strategy,
                    maxRetries: maxRetries, memberIds: memberIds)
                if !isEnabled {
                    try PeerAdmin.updatePool(&config, id: id, isEnabled: false)
                }
            } else {
                let existing = config.pool(id: id)?.memberIds ?? []
                let add = memberIds.filter { !existing.contains($0) }
                let remove = existing.filter { !memberIds.contains($0) }
                try PeerAdmin.updatePool(
                    &config, id: id, name: name, strategy: strategy, maxRetries: maxRetries,
                    addMembers: add, removeMembers: remove, isEnabled: isEnabled)
            }
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
        return await persist(config)
    }

    func deletePool(id: String) async -> Bool {
        guard lockError == nil else { return false }
        var config = self.config
        do {
            try PeerAdmin.removePool(&config, id: id)
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
        return await persist(config)
    }

    // MARK: - Account and earnings

    /// Check the stored login. A dead session is dropped. The password is never stored.
    func refreshAccount() async {
        guard !isShutdown else { return }
        guard let server = URL(string: activeREST) else { return }
        claimedPeerId = readSecret(accountRef("claimed"))
        guard let session = readSecret(accountRef("session")) else {
            accountEmail = nil
            accountName = nil
            return
        }
        do {
            let account = try await rest.currentUser(server: server, session: session)
            guard !isShutdown else { return }
            accountEmail = account.email
            accountName = account.name
            try? secrets.save(account.email, for: accountRef("email"))
            accountError = nil
        } catch let error as RESTError {
            if case .badStatus(401, _) = error {
                try? secrets.delete(accountRef("session"))
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
        guard !isAccountBusy, !isShutdown else { return false }
        guard let server = URL(string: activeREST) else {
            accountError = "Invalid server URL (expected http/https)."
            return false
        }
        let trimmed = email.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !password.isEmpty else {
            accountError = "Email and password are required."
            return false
        }
        isAccountBusy = true
        defer { isAccountBusy = false }
        do {
            let login = try await rest.login(server: server, email: trimmed, password: password)
            guard !isShutdown else { return false }
            try secrets.save(login.session, for: accountRef("session"))
            try secrets.save(login.account.email, for: accountRef("email"))
            accountEmail = login.account.email
            accountName = login.account.name
            accountError = nil
            return true
        } catch {
            accountError = Self.userMessage(error)
            return false
        }
    }

    func signOut() async {
        guard !isAccountBusy else { return }
        isAccountBusy = true
        defer { isAccountBusy = false }
        if let server = URL(string: activeREST), let session = readSecret(accountRef("session")) {
            try? await rest.logout(server: server, session: session)
        }
        try? secrets.delete(accountRef("session"))
        accountEmail = nil
        accountName = nil
        claimedPeerId = nil
        accountError = nil
    }

    /// Link this token factory to the signed-in account. The connection key proves possession.
    @discardableResult
    func claimTokenFactory() async -> Bool {
        guard !isAccountBusy, !isShutdown else { return false }
        guard let server = URL(string: activeREST) else {
            accountError = "Invalid server URL (expected http/https)."
            return false
        }
        guard accountEmail != nil, let session = readSecret(accountRef("session")) else {
            accountError = "Sign in before claiming this token factory."
            return false
        }
        guard let peerId = config.peerId, !peerId.isEmpty, let key = connectionKey else {
            accountError = "Register this token factory before claiming it."
            return false
        }
        isAccountBusy = true
        defer { isAccountBusy = false }
        do {
            _ = try await rest.claimPeer(
                server: server, session: session, peerId: peerId, connectionKey: key)
            guard !isShutdown else { return false }
            try secrets.save(peerId, for: accountRef("claimed"))
            claimedPeerId = peerId
            accountError = nil
            return true
        } catch {
            accountError = Self.userMessage(error)
            return false
        }
    }

    /// Supplier money view. No-op until this token factory has a connection key.
    func fetchEarnings() async {
        guard !isEarningsBusy, !isShutdown else { return }
        guard let token = connectionKey, let server = URL(string: activeREST) else { return }
        isEarningsBusy = true
        defer { isEarningsBusy = false }
        do {
            earnings = try await rest.fetchEarnings(server: server, token: token)
            guard !isShutdown else { return }
            earningsError = nil
        } catch {
            earningsError = Self.userMessage(error)
        }
    }

    // MARK: - Internals

    /// Shared file store only. An ad-hoc signed build asks for the login
    /// keychain password once per item, and Always Allow does not survive
    /// the next codesign, so the prompt repeats. `tknet-peer serve` reads
    /// this same directory.
    private static func makeSecrets(in directory: URL) -> (store: any SecretStore, migrationError: String?) {
        let file = FileSecretStore(directory: directory.appendingPathComponent("secrets"))
        return (file, nil)
    }

    private func readSecret(_ ref: String) -> String? {
        guard let value = try? secrets.load(ref), !value.isEmpty else { return nil }
        return value
    }

    private func accountRef(_ kind: String) -> String {
        guard let server = URL(string: activeREST) else { return "user/\(kind)/default" }
        let host = server.host ?? "default"
        let port = server.port.map { ":\($0)" } ?? ""
        return "user/\(kind)/\(host)\(port)"
    }

    private func apply(_ status: PeerStatus, from service: PeerService) {
        self.status = status
        gatewayPort = status.gatewayPort
        running = status.gatewayListening
        let latest = service.currentConfig
        config = latest
        serverBox.update(latest.serverURL)
        noteLaunch(status.connection)
    }

    private func dialIfWanted() async {
        guard !isShutdown, tunnelWanted else { return }
        guard config.peerId != nil else { return }
        guard connectionKey != nil else {
            lastError = "No connection key stored. Register this peer again."
            return
        }
        launchGeneration += 1
        launchPolicy.begin()
        do {
            try await service?.startTunnel()
        } catch {
            guard !isShutdown else { return }
            launchPolicy.stopWatching()
            lastError = Self.userMessage(error)
        }
    }

    private func noteLaunch(_ connection: TunnelStatus) {
        let generation = launchGeneration
        switch launchPolicy.note(connection) {
        case .keepWatching, .connected:
            break
        case .abandon:
            let retries = TunnelLaunchPolicy.maxLaunchRetries
            Task { await self.abandonLaunch(generation: generation, retries: retries) }
        case .upgradeRequired:
            lastError = "This build's peer protocol is too old. Download an update from \(TknetPeer.downloadURL)."
        }
    }

    /// Stops the tunnel after a failed launch. Does not clear `tunnelWanted`.
    private func abandonLaunch(generation: Int, retries: Int) async {
        guard generation == launchGeneration, !isShutdown else { return }
        await service?.stopTunnel()
        guard generation == launchGeneration, !isShutdown else { return }
        lastError = "Could not open the tunnel after \(retries) retries. The gateway is still listening."
    }

    @discardableResult
    private func persist(_ config: PeerConfig) async -> Bool {
        do {
            try FileManager.default.createDirectory(
                at: configURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try ConfigStore.save(config, to: configURL)
            self.config = config
            serverBox.update(config.serverURL)
            lastError = nil
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
        if let service {
            await service.applyConfig(config)
            self.config = service.currentConfig
        }
        return true
    }

    private func syncActiveToken() {
        guard config.peerId != nil else { return }
        guard let perServer = readSecret(tokenKey(for: config.serverURL)) else { return }
        try? secrets.save(perServer, for: "peer/token")
    }

    private func tokenKey(for tunnel: URL) -> String {
        let host = tunnel.host ?? "default"
        let port = tunnel.port.map { ":\($0)" } ?? ""
        return "peer/token/\(host)\(port)"
    }

    private func releaseLock() {
        processLock?.unlock()
        processLock = nil
    }

    private static func userMessage(_ error: Error) -> String {
        if let text = (error as? LocalizedError)?.errorDescription, !text.isEmpty {
            return text
        }
        return String(describing: error)
    }
}
