import ArgumentParser
import Foundation
import NovaMLXTknetPeer

@main
struct TknetPeerCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tknet-peer",
        abstract: "Tknet inference supply peer",
        version: TknetPeer.version,
        subcommands: [Setup.self, Serve.self, Earnings.self]
    )
}

// MARK: - Shared helpers

extension TknetPeerCLI {
    struct Earnings: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Show what this peer has earned (supplier money view)")

        @Option(help: "Config file path")
        var config: String = CLIConfig.defaultConfigPath

        func run() async throws {
            let peerConfig = try TknetPeerCLI.loadConfig(config)
            guard var comps = URLComponents(url: peerConfig.serverURL, resolvingAgainstBaseURL: false) else {
                throw ValidationError("Invalid server URL in config: \(peerConfig.serverURL)")
            }
            switch comps.scheme?.lowercased() {
            case "wss": comps.scheme = "https"
            case "ws": comps.scheme = "http"
            default: break
            }
            guard let restBase = comps.url else {
                throw ValidationError("Invalid server URL in config: \(peerConfig.serverURL)")
            }
            let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
            guard let token = secrets.load("peer/token"), !token.isEmpty else {
                throw ValidationError("Missing peer token — run `tknet-peer setup` first.")
            }
            let rest = TknetREST()
            do {
                let summary = try await rest.fetchEarnings(server: restBase, token: token)
                let a = summary.availability
                print("Total earned:   $\(a.totalEarned)")
                print("Available:      $\(a.available)  (on hold $\(a.onHold), paid out $\(a.paidOut))")
                if !summary.byModel.isEmpty {
                    print("")
                    print("By model:")
                    for m in summary.byModel {
                        print("  \(m.model): \(m.requests) requests, \(m.tokens) tokens, $\(m.earned)")
                    }
                }
                if !summary.recent.isEmpty {
                    print("")
                    print("Recent:")
                    for e in summary.recent.prefix(10) {
                        let paid = e.payoutId != nil ? " [paid]" : ""
                        print("  \(e.createdAt) \(e.model) in=\(e.promptTokens) out=\(e.completionTokens) $\(e.grossAmount)\(paid)")
                    }
                }
                try await rest.shutdown()
            } catch {
                try? await rest.shutdown()
                throw error
            }
        }
    }

    /// Portable default display name. `Host.current()` is AppKit-adjacent and
    /// unavailable off macOS; `hostName` is Foundation-portable.
    static var defaultPeerName: String {
        let host = ProcessInfo.processInfo.hostName
        return host.isEmpty ? "tknet-peer" : "tknet-peer-\(host.prefix(20))"
    }

    static func configURL(_ path: String) -> URL {
        URL(fileURLWithPath: CLIConfig.expandTilde(path))
    }

    /// Loads the config at `path`, falling back to defaults when the file is
    /// absent so a fresh machine can run `setup` without a pre-seeded file.
    static func loadConfig(_ path: String) throws -> PeerConfig {
        let url = configURL(path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return PeerConfig.defaultConfig()
        }
        do {
            return try ConfigStore.load(from: url)
        } catch {
            throw ValidationError("Config file is unreadable (\(url.path)): \(error)")
        }
    }

    static func saveConfig(_ config: PeerConfig, path: String) throws {
        let url = configURL(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ConfigStore.save(config, to: url)
    }

    /// Secrets live next to the config file, in a 0700 directory.
    static func secretsDirectory(for configPath: String) -> URL {
        configURL(configPath).deletingLastPathComponent().appendingPathComponent("secrets")
    }

    /// REST base (`https://…`) → tunnel base (`wss://…`): both endpoints live
    /// on the same host. Other schemes (raw `ws`/`wss`) pass through.
    static func tunnelURL(fromREST base: URL) -> URL {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false),
              let scheme = components.scheme?.lowercased() else { return base }
        switch scheme {
        case "https": components.scheme = "wss"
        case "http": components.scheme = "ws"
        default: return base
        }
        return components.url ?? base
    }
}

// MARK: - setup

extension TknetPeerCLI {
    struct Setup: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Register and configure this peer")

        @Option(help: "Config file path")
        var config: String = CLIConfig.defaultConfigPath

        @Option(help: "tknet.ai base URL (https)")
        var server: String = "https://tknet.ai"

        @Option(help: "Display name for this peer")
        var name: String = TknetPeerCLI.defaultPeerName

        func run() async throws {
            guard let restBase = URL(string: server),
                  let scheme = restBase.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  let host = restBase.host, !host.isEmpty else {
                throw ValidationError("Invalid server URL: \(server)")
            }

            // TknetREST.shutdown() is single-use (throws on a second call);
            // shut it down exactly once on both the success and error paths.
            let rest = TknetREST()
            do {
                try await performSetup(rest: rest, restBase: restBase)
                try await rest.shutdown()
            } catch {
                try? await rest.shutdown()
                throw error
            }
        }

        private func performSetup(rest: TknetREST, restBase: URL) async throws {
            var peerConfig = try TknetPeerCLI.loadConfig(config)
            let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
            peerConfig.serverURL = TknetPeerCLI.tunnelURL(fromREST: restBase)

            if peerConfig.peerId == nil {
                print("Registering with \(server) …")
                let (peerId, token) = try await rest.register(server: restBase, peerName: name)
                peerConfig.peerId = peerId
                secrets.save(token, for: "peer/token")
                print("Registered: \(peerId)")
            }
            // The token is a secret: loaded here, never printed or logged.
            guard let token = secrets.load("peer/token"), !token.isEmpty else {
                throw ValidationError(
                    "No peer token in \(TknetPeerCLI.secretsDirectory(for: config).path) — " +
                    "delete the config file and re-run `tknet-peer setup` to re-register.")
            }

            print("Fetching demand list …")
            let demand = try await rest.fetchDemand(server: restBase, token: token)
            if demand.isEmpty {
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("The server has no demand right now. Saved \(CLIConfig.expandTilde(config)); re-run setup later.")
                return
            }
            for (index, entry) in demand.enumerated() {
                let note = entry.note.map { " — \($0)" } ?? ""
                print("[\(index)] \(entry.model) (\(entry.modality))\(note)")
            }
            print("Enter the numbers to serve (comma-separated), then per pick: source endpoint, API key, upstream model, prices.")
            guard let line = readLine(), !line.isEmpty else {
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("Nothing selected. Saved \(CLIConfig.expandTilde(config)).")
                return
            }
            let picks = line.split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

            for pick in picks {
                guard demand.indices.contains(pick) else {
                    print("Skipping \(pick): out of range.")
                    continue
                }
                let entry = demand[pick]
                let endpoint = Self.prompt(
                    "Source endpoint for \(entry.model) [http://127.0.0.1:6590/v1]: ",
                    default: "http://127.0.0.1:6590/v1")
                guard let endpointURL = URL(string: endpoint),
                      let endpointScheme = endpointURL.scheme,
                      endpointScheme.hasPrefix("http"),
                      endpointURL.host != nil else {
                    print("Invalid endpoint — skipping \(entry.model).")
                    continue
                }
                // Keys are interactive stdin only; never echoed by the tool.
                let key = Self.prompt("API key (empty for none): ", default: "")
                let upstream = Self.prompt(
                    "Upstream model name [\(entry.model)]: ", default: entry.model)
                let prices = Self.prompt(
                    "Price per 1k tokens (input output), e.g. 0 0: ", default: "0 0")
                    .split(whereSeparator: { $0 == " " || $0 == "," || $0 == "\t" })
                    .compactMap { Double($0) }

                let sourceId = "src-\(entry.demandId)"
                peerConfig.sources.removeAll { $0.id == sourceId }
                peerConfig.sources.append(SourceConfig(
                    id: sourceId, name: entry.model, type: .openaiCompatible,
                    endpoint: endpointURL, apiKeyRef: "src/\(sourceId)",
                    upstreamModel: upstream))
                secrets.save(key, for: "src/\(sourceId)")
                peerConfig.capabilities.removeAll { $0.demandId == entry.demandId }
                peerConfig.capabilities.append(Capability(
                    demandId: entry.demandId, model: entry.model, sourceId: sourceId,
                    sourceType: .openaiCompatible,
                    priceIn: prices.first ?? 0, priceOut: prices.last ?? 0))
            }
            try TknetPeerCLI.saveConfig(peerConfig, path: config)
            print("Saved \(CLIConfig.expandTilde(config)). Run `tknet-peer serve`.")
        }

        /// Reads one line from stdin; empty input falls back to `default`.
        private static func prompt(_ text: String, default fallback: String) -> String {
            print(text, terminator: "")
            let line = readLine()?.trimmingCharacters(in: .whitespaces) ?? ""
            return line.isEmpty ? fallback : line
        }
    }
}

// MARK: - serve

extension TknetPeerCLI {
    struct Serve: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Run the peer")

        @Option(help: "Config file path")
        var config: String = CLIConfig.defaultConfigPath

        func run() async throws {
            let configURL = TknetPeerCLI.configURL(config)
            let peerConfig = try TknetPeerCLI.loadConfig(config)
            guard peerConfig.peerId != nil else {
                throw ValidationError("Not registered — run `tknet-peer setup` first.")
            }
            let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
            guard let token = secrets.load("peer/token"), !token.isEmpty else {
                throw ValidationError("Missing peer token — re-run `tknet-peer setup`.")
            }
            let service = PeerService(
                config: peerConfig,
                secrets: secrets,
                transportFactory: WSTransport.factory(
                    server: peerConfig.serverURL, tokenRef: "peer/token", secrets: secrets),
                onConfigChange: { updated in
                    // Demand reconciliations (retire/revive) must survive restarts.
                    try? ConfigStore.save(updated, to: configURL)
                })

            let statusTask = Task {
                for await status in service.statusStream {
                    let error = status.lastError.map { " err=\($0)" } ?? ""
                    print("status: \(status.connection) req=\(status.totalRequests) tok=\(status.totalCompletionTokens)\(error)")
                    if status.connection == .upgradeRequired {
                        // Terminal: the server refuses this protocol version.
                        // One clear line, then exit — retrying would hammer
                        // the server's bcrypt forever.
                        print("Upgrade required: this build's peer protocol is below the server minimum. Download the latest tknet-peer / NovaMLX from \(TknetPeer.downloadURL)")
                        await service.stop()
                        Darwin.exit(2)
                    }
                }
            }

            // SIGINT/SIGTERM → clean teardown. PeerService must never be
            // dropped without stop(): the relay's AsyncHTTPClient traps in
            // debug builds otherwise.
            let stopSignal = ShutdownSignal()
            do {
                try await service.start()
            } catch {
                statusTask.cancel()
                throw error
            }
            await stopSignal.wait()
            print("\nShutting down …")
            statusTask.cancel()
            await service.stop()
        }
    }
}

/// Watches SIGINT and SIGTERM; `wait()` resumes exactly once, whichever
/// signal fires first (repeats are swallowed). Portable Foundation/Darwin —
/// no ServiceLifecycle dependency.
private final class ShutdownSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var fired = false
    private var sources: [DispatchSourceSignal] = []

    init() {
        for sig in [SIGINT, SIGTERM] {
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            signal(sig, SIG_IGN)
            source.setEventHandler { [weak self] in self?.fire() }
            source.resume()
            sources.append(source)
        }
    }

    private func fire() {
        lock.lock()
        defer { lock.unlock() }
        guard !fired else { return }
        fired = true
        continuation?.resume()
        continuation = nil
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if fired {
                lock.unlock()
                continuation.resume()
                return
            }
            self.continuation = continuation
            lock.unlock()
        }
    }
}
