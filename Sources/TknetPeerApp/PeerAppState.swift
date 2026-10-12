import NovaMLXTknetPeer
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
/// Secrets stay in `FileSecretStore` next to the CLI (`~/.config/tknet-peer/secrets`)
/// so both tools share one peer identity. Keychain is a later hardening step.
/// Quit stops the gateway and the tunnel and releases the process lock.
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

    private let configURL: URL
    private let secrets: FileSecretStore
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
        self.secrets = FileSecretStore(directory: directory.appendingPathComponent("secrets"))
        self.defaults = defaults
        var loaded = (try? ConfigStore.load(from: configURL)) ?? .defaultConfig()
        if loaded.adoptRegistrationsIntoMemory() {
            try? ConfigStore.save(loaded, to: configURL)
        }
        loaded.migrateLegacyPoolsIfNeeded()
        self.config = loaded
        self.tunnelWanted = TunnelWantedPreference.read(from: defaults)
        self.serverBox = TunnelServerBox(loaded.serverURL)
    }

    var activeREST: String {
        PeerConfig.normalizedREST(config.serverURL) ?? "https://tknet.ai"
    }

    /// Connection key for the active server. Nil when this peer is not registered.
    var connectionKey: String? {
        guard config.peerId != nil else { return nil }
        let value = secrets.load(tokenKey(for: config.serverURL)) ?? secrets.load("peer/token") ?? ""
        return value.isEmpty ? nil : value
    }

    var isGatewayUp: Bool {
        status?.gatewayListening == true
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
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
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
            let stored = previousId == nil ? "" : (secrets.load(tokenKey(for: tunnel)) ?? "")
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
            secrets.save(token, for: tokenKey(for: tunnel))
            secrets.save(token, for: "peer/token")
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

    func saveMember(
        id: String, name: String, type: SourceType, endpoint: URL, upstreamModel: String,
        apiKey: String?, weight: Int?, isEnabled: Bool, isFree: Bool
    ) async -> Bool {
        guard lockError == nil else { return false }
        var config = self.config
        do {
            try PeerAdmin.addMember(
                &config, secrets: secrets, id: id, name: name, endpoint: endpoint,
                upstreamModel: upstreamModel, apiKey: apiKey, weight: weight,
                isFree: isFree, type: type, isEnabled: isEnabled)
        } catch {
            lastError = Self.userMessage(error)
            return false
        }
        return await persist(config)
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

    // MARK: - Internals

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
        let perServer = secrets.load(tokenKey(for: config.serverURL)) ?? ""
        if !perServer.isEmpty {
            secrets.save(perServer, for: "peer/token")
        }
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
