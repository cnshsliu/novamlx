import Foundation
import NovaMLXDB
import NovaMLXTknetPeer

/// Bridges the cross-platform PeerService into the Mac UI. Owns the service
/// for the page's lifetime (the app keeps every page alive via the opacity
/// switch in `NovaAppView.detailView`, so this object lives as long as the
/// app) and tears it down on `stop()`/`deinit` — PeerService must never be
/// dropped without `stop()` (the relay's AsyncHTTPClient traps in debug).
@MainActor
final class TknetPeerState: ObservableObject {
    @Published private(set) var config: PeerConfig
    @Published private(set) var status: PeerStatus?
    @Published private(set) var demand: [DemandEntry] = []
    @Published private(set) var running = false
    @Published private(set) var lastError: String?

    /// Same location the `tknet-peer` CLI uses, so the app and CLI share one
    /// peer identity and source set.
    private let configURL: URL
    private let secrets: FileSecretStore
    private var service: PeerService?
    /// Consumes `service.statusStream` — the stream NEVER finishes, so the
    /// task's lifetime is tied to this object: cancelled in `stop()` (and
    /// `deinit`), otherwise it would capture the service forever.
    private var statusTask: Task<Void, Never>?
    /// Single TknetREST held for the object's lifetime; `shutdown()` is
    /// single-use (throws on a second call) so it is called exactly once in
    /// `deinit`.
    private let rest = TknetREST()

    init() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/tknet-peer")
        self.configURL = dir.appendingPathComponent("peer.json")
        self.secrets = FileSecretStore(directory: dir.appendingPathComponent("secrets"))
        self.config = (try? ConfigStore.load(from: configURL)) ?? .defaultConfig()
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
    func register(server: URL, peerName: String) async {
        guard let scheme = server.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = server.host, !host.isEmpty else {
            lastError = "Invalid server URL (expected http/https): \(server.absoluteString)"
            return
        }
        do {
            var config = self.config
            let (peerId, token) = try await rest.register(server: server, peerName: peerName)
            config.peerId = peerId
            config.serverURL = Self.tunnelURL(fromREST: server)
            secrets.save(token, for: "peer/token")
            try FileManager.default.createDirectory(at: configURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try ConfigStore.save(config, to: configURL)
            self.config = config
            lastError = nil
            await fetchDemand()
        } catch {
            lastError = String(describing: error)
        }
    }

    // MARK: - Demand list

    /// One-shot REST fetch (used before the tunnel is up); the running
    /// service keeps `demand` fresh via `statusStream` afterwards.
    func fetchDemand() async {
        guard config.peerId != nil,
              let token = secrets.load("peer/token"), !token.isEmpty else { return }
        guard let restBase = Self.restURL(fromTunnel: config.serverURL) else { return }
        do {
            demand = try await rest.fetchDemand(server: restBase, token: token)
        } catch {
            lastError = String(describing: error)
        }
    }

    // MARK: - Start / stop

    func start() async {
        guard service == nil, config.peerId != nil else { return }
        guard let token = secrets.load("peer/token"), !token.isEmpty else {
            lastError = "No peer token stored — register again."
            return
        }
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
            }
        }
        do {
            try await service.start()
            running = true
            lastError = nil
        } catch {
            statusTask?.cancel()
            statusTask = nil
            self.service = nil
            await service.stop()
            running = false
            lastError = String(describing: error)
        }
    }

    func stop() async {
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
    func addLocalSource(for entry: DemandEntry, apiKeyStore: APIKeyStore) async {
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
            apiKeyRef: "src/\(sourceId)", upstreamModel: entry.model))
        secrets.save(primaryKey, for: "src/\(sourceId)")
        config.capabilities.removeAll { $0.demandId == entry.demandId }
        config.capabilities.append(Capability(
            demandId: entry.demandId, model: entry.model, sourceId: sourceId,
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0))
        await commit(config)
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
            lastError = String(describing: error)
            return
        }
        if let service {
            await service.applyConfig(config)
        }
    }

    // MARK: - URL scheme helpers (mirror TknetPeerCLI)

    /// REST base (`https://…`) → tunnel base (`wss://…`).
    private static func tunnelURL(fromREST base: URL) -> URL {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false),
              let scheme = components.scheme?.lowercased() else { return base }
        switch scheme {
        case "https": components.scheme = "wss"
        case "http": components.scheme = "ws"
        default: return base
        }
        return components.url ?? base
    }

    /// Tunnel base (`wss://…`) → REST base (`https://…`).
    private static func restURL(fromTunnel base: URL) -> URL? {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false),
              let scheme = components.scheme?.lowercased() else { return nil }
        switch scheme {
        case "wss": components.scheme = "https"
        case "ws": components.scheme = "http"
        default: return base
        }
        return components.url
    }
}
