import ArgumentParser
import Foundation
import NovaMLXTknetPeer
#if canImport(Glibc)
import Glibc
#endif

@main
struct TknetPeerCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tknet-peer",
        abstract: "Tknet inference supply peer",
        version: TknetPeer.version,
        subcommands: [Setup.self, Serve.self, Earnings.self, Member.self, Pool.self, MigrateFromNova.self]
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

    static func terminate(_ code: Int32) -> Never {
        #if canImport(Darwin)
        Darwin.exit(code)
        #elseif canImport(Glibc)
        Glibc.exit(code)
        #else
        fatalError("no process exit")
        #endif
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
                peerConfig.upsertMember(MemberConfig(
                    id: sourceId, name: entry.model, type: .openaiCompatible,
                    endpoint: endpointURL, apiKeyRef: "src/\(sourceId)",
                    upstreamModel: upstream))
                peerConfig.upsertPool(PoolConfig(
                    id: sourceId, name: entry.model, memberIds: [sourceId]))
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
        static let configuration = CommandConfiguration(abstract: "Run the local gateway and, when registered, the tunnel")

        @Option(help: "Config file path")
        var config: String = CLIConfig.defaultConfigPath

        @Option(name: .customLong("listen-host"), help: "Gateway bind host (default 127.0.0.1, or the config file)")
        var listenHost: String?

        @Option(name: .customLong("listen-port"), help: "Gateway bind port (default 6600, or the config file)")
        var listenPort: Int?

        @Flag(name: .customLong("tunnel"), help: "Dial the tknet tunnel")
        var tunnel: Bool = false

        @Flag(name: .customLong("no-tunnel"), help: "Serve the local gateway without dialing tknet")
        var noTunnel: Bool = false

        func run() async throws {
            if tunnel && noTunnel {
                throw ValidationError("Pass only one of --tunnel and --no-tunnel.")
            }
            let configURL = TknetPeerCLI.configURL(config)
            var peerConfig = try TknetPeerCLI.loadConfig(config)
            if let listenHost { peerConfig.gatewayHost = listenHost }
            if let listenPort { peerConfig.gatewayPort = listenPort }
            let dial = tunnel || (!noTunnel && peerConfig.peerId != nil)
            let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
            if dial {
                guard peerConfig.peerId != nil else {
                    throw ValidationError("Not registered — run `tknet-peer setup` first.")
                }
                guard let token = secrets.load("peer/token"), !token.isEmpty else {
                    throw ValidationError("Missing peer token — re-run `tknet-peer setup`.")
                }
                _ = token
            }
            let factory: TransportFactory
            if dial {
                factory = WSTransport.factory(server: peerConfig.serverURL, tokenRef: "peer/token", secrets: secrets)
            } else {
                factory = { @Sendable in throw TunnelError.notRegistered }
            }
            let processLock: PeerProcessLock
            do {
                processLock = try PeerProcessLock.acquire(
                    at: PeerProcessLock.url(besideConfig: configURL))
            } catch let error as PeerProcessLockError {
                throw ValidationError(error.description)
            }
            defer { processLock.unlock() }

            let service = PeerService(
                config: peerConfig,
                secrets: secrets,
                transportFactory: factory,
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
                        TknetPeerCLI.terminate(2)
                    }
                }
            }

            // SIGINT/SIGTERM → clean teardown. PeerService must never be
            // dropped without stop(): the relay's AsyncHTTPClient traps in
            // debug builds otherwise.
            let stopSignal = ShutdownSignal()
            do {
                try await service.start(tunnel: dial)
            } catch {
                statusTask.cancel()
                throw error
            }
            if let port = service.gatewayListenPort {
                print("gateway listening on http://\(peerConfig.gatewayHost):\(port)")
            }
            await stopSignal.wait()
            print("\nShutting down …")
            statusTask.cancel()
            await service.stop()
        }
    }
}

// MARK: - member / pool

extension TknetPeerCLI {
    struct Member: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Add, remove, list, and update upstream members",
            subcommands: [Add.self, Remove.self, List.self, Update.self])

        struct Add: AsyncParsableCommand {
            static let configuration = CommandConfiguration(abstract: "Add or replace a member")
            @Option var config: String = CLIConfig.defaultConfigPath
            @Option var id: String
            @Option var name: String?
            @Option var endpoint: String
            @Option(name: .customLong("upstream-model")) var upstreamModel: String
            @Option(name: .customLong("api-key")) var apiKey: String?
            @Option var weight: Int?
            @Flag var free: Bool = false
            @Flag var paid: Bool = false
            @Option var type: String = "openaiCompatible"

            func run() async throws {
                if free && paid { throw ValidationError("Pass only one of --free and --paid.") }
                guard let endpointURL = URL(string: endpoint) else {
                    throw ValidationError("Invalid endpoint: \(endpoint)")
                }
                var peerConfig = try TknetPeerCLI.loadConfig(config)
                let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
                do {
                    try PeerAdmin.addMember(
                        &peerConfig, secrets: secrets, id: id, name: name ?? id,
                        endpoint: endpointURL, upstreamModel: upstreamModel, apiKey: apiKey,
                        weight: weight, isFree: free, type: try PeerAdmin.sourceType(type))
                } catch let error as PeerAdminError {
                    throw ValidationError(error.message)
                }
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("member \(id)")
            }
        }

        struct Remove: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "Remove a member")
            @Option var config: String = CLIConfig.defaultConfigPath
            @Option var id: String

            func run() throws {
                var peerConfig = try TknetPeerCLI.loadConfig(config)
                do { try PeerAdmin.removeMember(&peerConfig, id: id) }
                catch let error as PeerAdminError { throw ValidationError(error.message) }
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("removed \(id)")
            }
        }

        struct List: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "List members")
            @Option var config: String = CLIConfig.defaultConfigPath

            func run() throws {
                let peerConfig = try TknetPeerCLI.loadConfig(config)
                print(PeerAdmin.memberList(peerConfig))
            }
        }

        struct Update: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "Update a member")
            @Option var config: String = CLIConfig.defaultConfigPath
            @Option var id: String
            @Option var name: String?
            @Option var endpoint: String?
            @Option(name: .customLong("upstream-model")) var upstreamModel: String?
            @Option(name: .customLong("api-key")) var apiKey: String?
            @Option var weight: Int?
            @Flag var free: Bool = false
            @Flag var paid: Bool = false
            @Option var type: String?
            @Flag var enable: Bool = false
            @Flag var disable: Bool = false

            func run() throws {
                if free && paid { throw ValidationError("Pass only one of --free and --paid.") }
                if enable && disable { throw ValidationError("Pass only one of --enable and --disable.") }
                var peerConfig = try TknetPeerCLI.loadConfig(config)
                let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
                let endpointURL = try endpoint.map { raw -> URL in
                    guard let url = URL(string: raw) else { throw ValidationError("Invalid endpoint: \(raw)") }
                    return url
                }
                let memberType = try type.map { raw -> SourceType in
                    do { return try PeerAdmin.sourceType(raw) }
                    catch let error as PeerAdminError { throw ValidationError(error.message) }
                }
                let isFree: Bool? = free ? true : (paid ? false : nil)
                let isEnabled: Bool? = enable ? true : (disable ? false : nil)
                do {
                    try PeerAdmin.updateMember(
                        &peerConfig, secrets: secrets, id: id, name: name, endpoint: endpointURL,
                        upstreamModel: upstreamModel, apiKey: apiKey, weight: weight,
                        isFree: isFree, type: memberType, isEnabled: isEnabled)
                } catch let error as PeerAdminError {
                    throw ValidationError(error.message)
                }
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("member \(id)")
            }
        }
    }

    struct Pool: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Add, remove, list, and update pools",
            subcommands: [Add.self, Remove.self, List.self, Update.self])

        struct Add: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "Add or replace a pool")
            @Option var config: String = CLIConfig.defaultConfigPath
            @Option var id: String?
            @Option var name: String
            @Option var strategy: String = "tiered"
            @Option(name: .customLong("max-retries")) var maxRetries: Int = 3
            @Option(name: .customLong("member")) var member: [String] = []

            func run() throws {
                var peerConfig = try TknetPeerCLI.loadConfig(config)
                do {
                    try PeerAdmin.addPool(
                        &peerConfig, id: id, name: name,
                        strategy: try PeerAdmin.strategy(strategy),
                        maxRetries: maxRetries, memberIds: member)
                } catch let error as PeerAdminError {
                    throw ValidationError(error.message)
                }
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("pool \(id ?? name)")
            }
        }

        struct Remove: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "Remove a pool")
            @Option var config: String = CLIConfig.defaultConfigPath
            @Option var id: String

            func run() throws {
                var peerConfig = try TknetPeerCLI.loadConfig(config)
                do { try PeerAdmin.removePool(&peerConfig, id: id) }
                catch let error as PeerAdminError { throw ValidationError(error.message) }
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("removed \(id)")
            }
        }

        struct List: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "List pools")
            @Option var config: String = CLIConfig.defaultConfigPath

            func run() throws {
                print(PeerAdmin.poolList(try TknetPeerCLI.loadConfig(config)))
            }
        }

        struct Update: ParsableCommand {
            static let configuration = CommandConfiguration(abstract: "Update a pool")
            @Option var config: String = CLIConfig.defaultConfigPath
            @Option var id: String
            @Option var name: String?
            @Option var strategy: String?
            @Option(name: .customLong("max-retries")) var maxRetries: Int?
            @Option(name: .customLong("add-member")) var addMember: [String] = []
            @Option(name: .customLong("remove-member")) var removeMember: [String] = []
            @Flag var enable: Bool = false
            @Flag var disable: Bool = false

            func run() throws {
                if enable && disable { throw ValidationError("Pass only one of --enable and --disable.") }
                var peerConfig = try TknetPeerCLI.loadConfig(config)
                let parsed: PoolStrategy?
                if let strategy {
                    do { parsed = try PeerAdmin.strategy(strategy) }
                    catch let error as PeerAdminError { throw ValidationError(error.message) }
                } else {
                    parsed = nil
                }
                let isEnabled: Bool? = enable ? true : (disable ? false : nil)
                do {
                    try PeerAdmin.updatePool(
                        &peerConfig, id: id, name: name, strategy: parsed, maxRetries: maxRetries,
                        addMembers: addMember, removeMembers: removeMember, isEnabled: isEnabled)
                } catch let error as PeerAdminError {
                    throw ValidationError(error.message)
                }
                try TknetPeerCLI.saveConfig(peerConfig, path: config)
                print("pool \(id)")
            }
        }
    }
}

// MARK: - migrate-from-novamlx

extension TknetPeerCLI {
    struct MigrateFromNova: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "migrate-from-novamlx",
            abstract: "Copy NovaMLX load balancers and Tokenhub providers into this peer config")

        @Option(help: "Config file path")
        var config: String = CLIConfig.defaultConfigPath

        @Option(name: .customLong("nova-db"), help: "Path to nova_config.db (default: ~/.config/novamlx/path)")
        var novaDB: String?

        @Option(name: .customLong("local-base"), help: "Base URL for local NovaMLX members")
        var localBase: String = "http://127.0.0.1:6590/v1"

        @Flag(name: .customLong("dry-run"), help: "Print the summary and write nothing")
        var dryRun: Bool = false

        @Flag(name: .customLong("force"), help: "Overwrite pools and members that already exist, and re-run after a previous migration")
        var force: Bool = false

        @Flag(name: .customLong("replace-pools"), help: "Drop existing pools before writing migrated ones")
        var replacePools: Bool = false

        func run() throws {
            guard let localURL = URL(string: localBase), localURL.host != nil else {
                throw ValidationError("Invalid --local-base \(localBase)")
            }
            let peerConfig = try TknetPeerCLI.loadConfig(config)
            if peerConfig.migratedFromNovaAt != nil && !force {
                let stamp = ISO8601DateFormatter().string(from: peerConfig.migratedFromNovaAt!)
                print("Already migrated at \(stamp). Pass --force to run again.")
                return
            }
            let database = try resolveDatabase()
            let input: NovaMigrationInput
            do {
                input = try NovaDBSnapshotReader.load(database: database)
            } catch let error as NovaDBReadError {
                throw ValidationError(error.description)
            }
            let result = NovaConfigMigrator.migrate(
                input, existing: peerConfig,
                options: NovaMigrationOptions(force: force, replacePools: replacePools, localNovaBaseURL: localURL))
            print("\(result.members.count) members, \(result.pools.count) pools, \(result.secrets.count) secrets, \(result.warnings.count) warnings")
            for warning in result.warnings {
                print("warning: \(warning)")
            }
            for pool in result.pools {
                print("pool \(pool.name) members=\(pool.memberIds.joined(separator: ","))")
            }
            if dryRun {
                print("dry-run: nothing written")
                return
            }
            var updated = result.config
            updated.migratedFromNovaAt = Date()
            let secrets = FileSecretStore(directory: TknetPeerCLI.secretsDirectory(for: config))
            for (ref, value) in result.secrets {
                secrets.save(value, for: ref)
            }
            try TknetPeerCLI.saveConfig(updated, path: config)
            print("wrote \(TknetPeerCLI.configURL(config).path)")
        }

        private func resolveDatabase() throws -> URL {
            if let novaDB {
                let url = URL(fileURLWithPath: CLIConfig.expandTilde(novaDB))
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw ValidationError("nova_config.db not found at \(url.path)")
                }
                return url
            }
            let home = FileManager.default.homeDirectoryForCurrentUser
            let pathFile = home.appendingPathComponent(".config/novamlx/path")
            guard let text = try? String(contentsOf: pathFile, encoding: .utf8) else {
                throw ValidationError(
                    "No NovaMLX data directory. Write the directory to ~/.config/novamlx/path or pass --nova-db (often ~/.nova/nova_config.db).")
            }
            let line = text.split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .first { !$0.isEmpty && !$0.hasPrefix("#") }
            guard let line else {
                throw ValidationError("~/.config/novamlx/path is empty. Pass --nova-db.")
            }
            let expanded = CLIConfig.expandTilde(line)
            let db = URL(fileURLWithPath: expanded, isDirectory: true).appendingPathComponent("nova_config.db")
            guard FileManager.default.fileExists(atPath: db.path) else {
                throw ValidationError("nova_config.db not found at \(db.path). Pass --nova-db if NovaMLX stores it elsewhere.")
            }
            return db
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
