import Foundation
import SwiftUI
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXAPI
import NovaMLXMenuBar
import NovaMLXDistributed
import NovaMLXDB

/// Early env-var setup: runs before any GPU work because the static
/// initializer is triggered when the module is loaded.
private struct MLXEnvSetup {
    static let configure: Void = {
        // Force MLX to commit Metal command buffers after every operation
        // (and 1 MB of data).  The default batching (up to 40 ops / 40–50 MB
        // per buffer) can cause a single command buffer to take tens of
        // seconds on SDXL-Turbo, which exceeds the macOS background-process
        // timeout (~5 s) and triggers a `MTLCommandBufferStatusError` /
        // `SIGABRT`.  MLX reads these variables once on first use, so we
        // must set them before any GPU evaluation.
        setenv("MLX_MAX_OPS_PER_BUFFER", "1", 1)
        setenv("MLX_MAX_MB_PER_BUFFER", "1", 1)
    }()
}

@main
struct NovaMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        let l10n = L10n.shared
        return MenuBarExtra("NovaMLX", systemImage: "brain.head.profile.fill") {
            Button { appDelegate.openMainWindow(to: .status) } label: {
                Label(l10n.tr("app.status"), systemImage: "gauge.with.dots.needle.bottom.50percent")
            }
            Button { appDelegate.openMainWindow(to: .models) } label: {
                Label(l10n.tr("app.models"), systemImage: "cube.box")
            }
            Button { appDelegate.openMainWindow(to: .tokenhub) } label: {
                Label(l10n.tr("app.tokenhub"), systemImage: "server.rack")
            }
            Button { appDelegate.openMainWindow(to: .chat) } label: {
                Label(l10n.tr("app.chat"), systemImage: "cpu")
            }
            if appDelegate.appState.clusterEnabled {
                Button { appDelegate.openMainWindow(to: .cluster) } label: {
                    Label(l10n.tr("app.cluster"), systemImage: "xserve")
                }
            }
            Button { appDelegate.openMainWindow(to: .settings) } label: {
                Label(l10n.tr("app.settings"), systemImage: "gearshape")
            }
            Divider()
            Button { NSApp.terminate(nil) } label: {
                Label(l10n.tr("menuBar.quit"), systemImage: "power")
            }
        }
        .menuBarExtraStyle(.menu)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let appState = MenuBarAppState()
    let engine = MLXEngine()
    let settingsManager: ModelSettingsManager
    let workerMode: Bool

    lazy var inferenceService: InferenceService = {
        // Read cluster settings (needed for both workerMode and non-workerMode paths)
        let (isCluster, clusterConfig) = Self.readClusterSettings()

        if workerMode {
            let workerPath = Bundle.main.executableURL!
                .deletingLastPathComponent()
                .appendingPathComponent("NovaMLXWorker")
                .path
            return InferenceService(
                engine: engine,
                settingsManager: settingsManager,
                workerMode: true,
                workerBinaryPath: workerPath,
                clusterMode: isCluster,
                clusterConfig: clusterConfig
            )
        }
        return InferenceService(
            engine: engine,
            settingsManager: settingsManager,
            clusterMode: isCluster,
            clusterConfig: clusterConfig
        )
    }()
    let modelManager: ModelManager
    var apiServer: NovaMLXAPIServer?
    var serverTask: Task<Void, Never>?
    var memoryPressureHandler: MemoryPressureHandler?
    let config = NovaMLXConfiguration.shared
    var mainWindow: NSWindow?

    /// Path errors captured in init() BEFORE ModelManager creates any directories
    private let pathValidationErrors: [String]

    override init() {
        // Trigger MLX environment setup before any GPU library initialisation.
        _ = MLXEnvSetup.configure

        // Validate BEFORE creating ModelManager (which auto-creates dirs)
        let errors = NovaMLXPaths.validateConfiguredPaths()
        self.pathValidationErrors = errors

        let baseDir = NovaMLXPaths.baseDir
        let modelsDir = NovaMLXPaths.modelsDir

        // Initialize SQLite databases (migrates legacy JSON if needed)
        do {
            try NovaDB.shared.setup(baseDir: baseDir)
            NovaMLXLog.info("SQLite databases initialized")
        } catch {
            NovaMLXLog.error("Failed to initialize SQLite: \(error)")
        }

        // Auto-migrate from old Application Support path if needed
        Self.migrateFromApplicationSupport(to: baseDir)

        self.modelManager = ModelManager(modelsDirectory: modelsDir)
        self.settingsManager = ModelSettingsManager(baseDirectory: baseDir)

        // Enable worker subprocess for crash isolation
        self.workerMode = true

        // Coordinator nodes start on the Cluster page so the Thunderbolt subnet
        // enforcement in scanNetwork() runs immediately on launch.
        if Self.readClusterSettings().0 {
            appState.requestedPage = .cluster
        }

        super.init()
    }

    /// One-time migration from ~/Library/Application Support/NovaMLX to ~/.nova
    private static func migrateFromApplicationSupport(to newBase: URL) {
        let fm = FileManager.default
        guard let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return }
        let oldBase = appSupport.appendingPathComponent("NovaMLX", isDirectory: true)

        guard fm.fileExists(atPath: oldBase.path) else { return }
        guard !fm.fileExists(atPath: newBase.path) else {
            // New dir exists — just fix registry paths if they still point to old location
            fixRegistryPaths(at: newBase, oldPrefix: oldBase.path)
            return
        }

        NovaMLXLog.info("Migrating from \(oldBase.path) to \(newBase.path)...")
        do {
            try fm.createDirectory(at: newBase.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.moveItem(at: oldBase, to: newBase)
            fixRegistryPaths(at: newBase, oldPrefix: oldBase.path)
            NovaMLXLog.info("Migration complete")
        } catch {
            NovaMLXLog.error("Migration failed: \(error)")
        }
    }

    /// Fix localURL paths in registry.json that still point to old location
    private static func fixRegistryPaths(at baseDir: URL, oldPrefix: String) {
        let registryPath = baseDir.appendingPathComponent("models/registry.json")
        guard let data = try? Data(contentsOf: registryPath),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        var changed = false
        let newModelsPrefix = "file://" + NovaMLXPaths.modelsDir.path

        for (key, value) in json {
            guard var record = value as? [String: Any],
                  let localURL = record["localURL"] as? String,
                  localURL.contains(oldPrefix) || localURL.contains("Application%20Support/NovaMLX")
            else { continue }
            record["localURL"] = localURL.replacingOccurrences(
                of: "file:///Users/lucas/Library/Application%20Support/NovaMLX/models/",
                with: newModelsPrefix
            )
            record["localURL"] = (record["localURL"] as? String ?? localURL).replacingOccurrences(
                of: "file:///Users/lucas/Library/Application Support/NovaMLX/models/",
                with: newModelsPrefix
            )
            json[key] = record
            changed = true
        }

        if changed {
            let newData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            try? newData?.write(to: registryPath, options: .atomic)
            NovaMLXLog.info("Fixed registry paths to point to \(NovaMLXPaths.modelsDir.path)")
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleOpenMainWindow),
            name: .openNovaAppWindow,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleRestartServer),
            name: .restartNovaMLXServer,
            object: nil
        )

        // Log path validation issues but don't block startup
        // (macOS may report external drives as unreadable even when accessible from shell)
        if !pathValidationErrors.isEmpty {
            for err in pathValidationErrors {
                NovaMLXLog.warning("[Paths] \(err)")
            }
        }

        Task {
            NovaMLXLog.rotateLogFile()

            try? await config.initializeDirectories()

            let configFile = NovaMLXPaths.configFile
            if FileManager.default.fileExists(atPath: configFile.path) {
                do {
                    try await config.loadFromFile(configFile)
                    let apiKeyCount = (try? NovaDB.shared.apiKeyStore.listAsAPIKey())?.count ?? 0
                    NovaMLXLog.info("Loaded config from \(configFile.path) (apiKeys: \(apiKeyCount))")
                } catch {
                    NovaMLXLog.error("Failed to load config: \(error)")
                }
            }

            // API keys now live in SQLite (NovaDB.apiKeyStore); no JSON load step.

            modelManager.registerPopularModels()
            modelManager.discoverModels()
            modelManager.cleanupEmptyDirectories()

            // Cleanup orphaned prefix cache dirs and old Application Support directory
            let downloadedIds = Set(modelManager.downloadedModels().map { $0.id })
            engine.cleanupOrphanedCacheDirs(downloadedModelIds: downloadedIds)
            Self.cleanupLegacyAppSupportDir()

            if workerMode {
                do {
                    try inferenceService.startWorker()
                    NovaMLXLog.info("Worker subprocess started")
                } catch {
                    NovaMLXLog.error("Failed to start worker: \(error)")
                }
            }

            let serverConfig = await config.serverConfig
            let hfEndpoint = await config.huggingfaceEndpoint
            apiServer = NovaMLXAPIServer(
                inferenceService: inferenceService,
                modelManager: modelManager,
                config: serverConfig,
                huggingfaceEndpoint: hfEndpoint
            )

            appState.isServerRunning = true
            appState.serverPort = serverConfig.port
            appState.adminPort = serverConfig.adminPort
            appState.apiKey = Self.firstRawAPIKey()

            NovaMLXLog.info("NovaMLX v\(NovaMLXCore.version) started")

            if let apiServer = apiServer {
                serverTask = Task {
                    try? await apiServer.start()
                }
            }

            // Restore models in background after API is live
            Task {
                await inferenceService.restoreModels(modelManager: modelManager)
                appState.detectIncompleteDownloads(modelsDirectory: modelManager.modelsDirectory)
                appState.resumeIncompleteDownloads()

                // Sync local model providers after models finish loading
                let loaded = inferenceService.listLoadedModels()
                TokenhubManager.shared.provisionLocalProviders(loadedModels: loaded)
                NovaMLXLog.info("Local managed providers synced: \(loaded.count) models")
            }

            appState.startStatsMonitoring(inferenceService: inferenceService)

            // Provision managed providers: cloud (if subscribed) + local loaded models
            Task {
                // Cloud providers — validate subscription with network check
                do {
                    _ = try await CloudAuth.validate()

                    // Verify tknet.ai API Key and provision nova providers
                    if let apiKey = TokenhubManager.shared.loadTknetApiKeyFromSettings(), !apiKey.isEmpty {
                        let isValid = await CloudBackend.shared.verifySettingsApiKey(apiKey: apiKey)
                        if isValid {
                            let models = await CloudBackend.shared.fetchTknetModels(apiKey: apiKey)
                            if !models.isEmpty {
                                try? TokenhubManager.shared.provisionTknetProviders(remoteModels: models)
                                NovaMLXLog.info("tknet.ai nova providers provisioned: \(models.count) models")
                            } else {
                                NovaMLXLog.info("tknet.ai API key valid but no nova models found")
                            }
                        } else {
                            NovaMLXLog.info("tknet.ai API key invalid, skipping nova provider provisioning")
                        }
                    } else {
                        NovaMLXLog.info("No tknet.ai API key configured")
                    }
                } catch {
                    NovaMLXLog.info("Not subscribed, skipping cloud provisioning")
                }
            }

            let memHandler = MemoryPressureHandler(engine: engine, settingsManager: settingsManager)
            memHandler.start()
            memoryPressureHandler = memHandler

            // Start ProcessMemoryEnforcer (1s polling, configurable limits)
            await engine.startMemoryEnforcer()
            await engine.configureEnforcerSettings { [settingsManager] modelId in
                settingsManager.getSettings(modelId)
            }

            // Distributed inference initialization
            NovaMLXLog.info("[Cluster] Distributed backend check: ring=\(MLXDistributedWrapper.isBackendAvailable("ring")), jaccl=\(MLXDistributedWrapper.isBackendAvailable("jaccl")), best=\(MLXDistributedWrapper.bestAvailableBackend())")
            let clusterSettings = serverConfig.cluster
            if let cluster = clusterSettings {
                switch cluster.role {
                case "coordinator":
                    let clusterConfig = ClusterConfig(
                        role: .coordinator,
                        coordinatorHost: cluster.coordinatorHost ?? "0.0.0.0",
                        coordinatorPort: cluster.coordinatorPort ?? 6591,
                        strategy: ClusterStrategy(rawValue: cluster.strategy ?? "minNodes") ?? .minNodes,
                        minLayersPerShard: cluster.minLayersPerShard ?? 32,
                        enableRingTransport: false   // TCP fallback — Ring broken with link-local Thunderbolt
                    )
                    try? ClusterManager.shared.startAsCoordinator(config: clusterConfig)

                    // Configure ClusterModelManager for distributed model lifecycle
                    ClusterModelManager.shared.configure(
                        engine: engine,
                        tokenizerProvider: { [engine] modelId in
                            guard let tokenizer = engine.getContainer(for: modelId)?.tokenizer else { return nil }
                            return DistributedTokenizer(
                                encode: { text in tokenizer.encode(text) },
                                decode: { tokens in tokenizer.decode(tokens) }
                            )
                        },
                        modelPathProvider: { modelId in
                            let path = NovaMLXPaths.modelsDir.appendingPathComponent(modelId).path
                            var isDir: ObjCBool = false
                            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
                                return nil
                            }
                            return path
                        }
                    )

                    NovaMLXLog.info("[Cluster] Started as coordinator on port \(clusterConfig.coordinatorPort)")

                    // Force the main window open on the Cluster page for autonomous
                    // validation of Thunderbolt subnet discovery (scanNetwork).
                    Task { @MainActor in
                        self.openMainWindow(to: .cluster)
                    }
                case "worker":
                    if let host = cluster.coordinatorHost {
                        let clusterConfig = ClusterConfig(
                            role: .worker,
                            coordinatorHost: host,
                            coordinatorPort: cluster.coordinatorPort ?? 6591,
                            strategy: ClusterStrategy(rawValue: cluster.strategy ?? "minNodes") ?? .minNodes,
                            minLayersPerShard: cluster.minLayersPerShard ?? 32,
                        enableRingTransport: false   // TCP fallback — Ring broken with link-local Thunderbolt
                        )
                        WorkerShardService.shared.setEngine(engine)
                        WorkerService.shared.start(config: clusterConfig)
                        NovaMLXLog.info("[Cluster] Started as worker, coordinator at \(host)")
                    }
                default:
                    break
                }
            }
        }
    }

    @objc func handleRestartServer(_ notification: Notification) {
        restartServer()
    }

    func restartServer() {
        NovaMLXLog.info("Restarting server with updated config...")
        serverTask?.cancel()
        serverTask = nil

        Task {
            let configFile = await config.configFileURL
            if FileManager.default.fileExists(atPath: configFile.path) {
                do {
                    try await config.loadFromFile(configFile)
                } catch {
                    NovaMLXLog.error("Failed to reload config: \(error)")
                }
            }

            let serverConfig = await config.serverConfig
            let hfEndpoint = await config.huggingfaceEndpoint
            apiServer = NovaMLXAPIServer(
                inferenceService: inferenceService,
                modelManager: modelManager,
                config: serverConfig,
                huggingfaceEndpoint: hfEndpoint
            )

            appState.serverPort = serverConfig.port
            appState.adminPort = serverConfig.adminPort
            appState.apiKey = Self.firstRawAPIKey()

            NovaMLXLog.info("Server restarted (apiKeys: \((try? NovaDB.shared.apiKeyStore.listAsAPIKey())?.count ?? 0))")

            // Re-initialize cluster on config change
            if let cluster = serverConfig.cluster {
                switch cluster.role {
                case "coordinator":
                    let clusterConfig = ClusterConfig(
                        role: .coordinator,
                        coordinatorHost: cluster.coordinatorHost ?? "0.0.0.0",
                        coordinatorPort: cluster.coordinatorPort ?? 6591,
                        strategy: ClusterStrategy(rawValue: cluster.strategy ?? "minNodes") ?? .minNodes,
                        minLayersPerShard: cluster.minLayersPerShard ?? 32,
                        enableRingTransport: false   // TCP fallback — Ring broken with link-local Thunderbolt
                    )
                    try? ClusterManager.shared.startAsCoordinator(config: clusterConfig)
                    NovaMLXLog.info("[Cluster] Restarted as coordinator on port \(clusterConfig.coordinatorPort)")
                case "worker":
                    if let host = cluster.coordinatorHost {
                        let clusterConfig = ClusterConfig(
                            role: .worker,
                            coordinatorHost: host,
                            coordinatorPort: cluster.coordinatorPort ?? 6591,
                            strategy: ClusterStrategy(rawValue: cluster.strategy ?? "minNodes") ?? .minNodes,
                            minLayersPerShard: cluster.minLayersPerShard ?? 32,
                        enableRingTransport: false   // TCP fallback — Ring broken with link-local Thunderbolt
                        )
                        WorkerShardService.shared.setEngine(engine)
                        WorkerService.shared.start(config: clusterConfig)
                        NovaMLXLog.info("[Cluster] Restarted as worker, coordinator at \(host)")
                    }
                default:
                    break
                }
            }

            if let apiServer = apiServer {
                serverTask = Task {
                    try? await apiServer.start()
                }
            }
        }
    }

    /// Fetch the raw plaintext of the most recently created API key from the
    /// SQLite store (`APIKeyStore.list()` orders by `created_at` DESC, so
    /// `.first` is the newest key). Used to seed `appState.apiKey` so the menu
    /// bar UI's internal Bearer-token calls to the local API server authenticate
    /// after startup or restart. Returns nil if there are no keys (open mode).
    private static func firstRawAPIKey() -> String? {
        guard let first = (try? NovaDB.shared.apiKeyStore.list())?.first else { return nil }
        return try? NovaDB.shared.apiKeyStore.getRawKey(id: first.id)
    }

    /// Remove the old ~/Library/Application Support/NovaMLX/ directory if it exists
    private static func cleanupLegacyAppSupportDir() {
        let fm = FileManager.default
        let legacyDir = NovaMLXPaths.legacyAppSupportDir
        guard fm.fileExists(atPath: legacyDir.path) else { return }

        // Move any prefix_cache content to new location first
        let legacyPrefixCache = legacyDir.appendingPathComponent("prefix_cache")
        let newPrefixCache = NovaMLXPaths.prefixCacheBaseDir
        if fm.fileExists(atPath: legacyPrefixCache.path) {
            try? fm.createDirectory(at: newPrefixCache, withIntermediateDirectories: true)
            if let contents = try? fm.contentsOfDirectory(at: legacyPrefixCache, includingPropertiesForKeys: nil) {
                for dir in contents {
                    let dest = newPrefixCache.appendingPathComponent(dir.lastPathComponent)
                    if !fm.fileExists(atPath: dest.path) {
                        try? fm.moveItem(at: dir, to: dest)
                    }
                }
            }
        }

        // Delete the old directory
        try? fm.removeItem(at: legacyDir)
        NovaMLXLog.info("Cleaned up legacy Application Support directory")
    }

    // MARK: - Config helpers

    /// Read cluster settings from config.json synchronously (avoids actor-isolated async access).
    private static func readClusterSettings() -> (Bool, ClusterConfig?) {
        let configFile = NovaMLXPaths.configFile
        guard let data = try? Data(contentsOf: configFile),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let server = json["server"] as? [String: Any],
              let cluster = server["cluster"] as? [String: Any],
              let role = cluster["role"] as? String,
              role == "coordinator"
        else {
            return (false, nil)
        }
        let config = ClusterConfig(
            role: .coordinator,
            coordinatorHost: cluster["coordinatorHost"] as? String ?? "0.0.0.0",
            coordinatorPort: cluster["coordinatorPort"] as? Int ?? 6591,
            strategy: ClusterStrategy(rawValue: cluster["strategy"] as? String ?? "minNodes") ?? .minNodes,
            minLayersPerShard: cluster["minLayersPerShard"] as? Int ?? 32
        )
        return (true, config)
    }

    func applicationWillTerminate(_ notification: Notification) {
        memoryPressureHandler?.stop()
        appState.stopStatsMonitoring()
        inferenceService.stopWorker()
    }

    nonisolated func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        MainActor.assumeIsolated {
            openMainWindow()
        }
        return true
    }

    @objc func handleOpenMainWindow(_ notification: Notification) {
        openMainWindow()
    }

    func openMainWindow(to page: AppPage = .status) {
        // When launched as coordinator, default to Cluster page so that
        // scanNetwork() (with Thunderbolt subnet enforcement) runs immediately.
        let effectivePage: AppPage = {
            if page != .status { return page }
            let (isCoordinator, _) = Self.readClusterSettings()
            return isCoordinator ? .cluster : .status
        }()

        appState.requestedPage = effectivePage

        if let window = mainWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let contentView = NovaAppView(
            appState: appState,
            inferenceService: inferenceService,
            modelManager: modelManager
        )

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1000, height: 650),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "NovaMLX"
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: contentView.environmentObject(L10n.shared))
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        mainWindow = window
        NovaMLXLog.info("Main window opened")
    }
}
