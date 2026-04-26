import SwiftUI
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXAPI
import NovaMLXMenuBar

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
            Button { appDelegate.openMainWindow(to: .chat) } label: {
                Label(l10n.tr("app.chat"), systemImage: "bubble.left.and.bubble.right")
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
        if workerMode {
            let workerPath = Bundle.main.executableURL!
                .deletingLastPathComponent()
                .appendingPathComponent("NovaMLXWorker")
                .path
            return InferenceService(
                engine: engine,
                settingsManager: settingsManager,
                workerMode: true,
                workerBinaryPath: workerPath
            )
        }
        return InferenceService(engine: engine, settingsManager: settingsManager)
    }()
    let modelManager: ModelManager
    var apiServer: NovaMLXAPIServer?
    var serverTask: Task<Void, Never>?
    var memoryPressureHandler: MemoryPressureHandler?
    let config = NovaMLXConfiguration.shared
    var mainWindow: NSWindow?

    override init() {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let baseDir = homeDir.appendingPathComponent(".nova", isDirectory: true)
        let modelsDir = baseDir.appendingPathComponent("models", isDirectory: true)

        // Auto-migrate from old Application Support path if needed
        Self.migrateFromApplicationSupport(to: baseDir)

        self.modelManager = ModelManager(modelsDirectory: modelsDir)
        self.settingsManager = ModelSettingsManager(baseDirectory: baseDir)

        // Enable worker subprocess for crash isolation
        self.workerMode = true

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

        guard let encoded = oldPrefix.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else { return }
        _ = "file://" + encoded
        var changed = false

        for (key, value) in json {
            guard var record = value as? [String: Any],
                  let localURL = record["localURL"] as? String,
                  localURL.contains(oldPrefix) || localURL.contains("Application%20Support/NovaMLX")
            else { continue }
            record["localURL"] = localURL.replacingOccurrences(
                of: "file:///Users/lucas/Library/Application%20Support/NovaMLX/models/",
                with: "file:///Users/lucas/.nova/models/"
            )
            // Also handle non-percent-encoded variant
            record["localURL"] = (record["localURL"] as? String ?? localURL).replacingOccurrences(
                of: "file:///Users/lucas/Library/Application Support/NovaMLX/models/",
                with: "file:///Users/lucas/.nova/models/"
            )
            json[key] = record
            changed = true
        }

        if changed {
            let newData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            try? newData?.write(to: registryPath, options: .atomic)
            NovaMLXLog.info("Fixed registry paths to point to ~/.nova")
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

        Task {
            NovaMLXLog.rotateLogFile()

            try? await config.initializeDirectories()

            let configFile = await config.modelsDirectory
                .deletingLastPathComponent()
                .appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configFile.path) {
                do {
                    try await config.loadFromFile(configFile)
                    let cfg = await config.serverConfig
                    NovaMLXLog.info("Loaded config from \(configFile.path) (apiKeys: \(cfg.apiKeys.count))")
                } catch {
                    NovaMLXLog.error("Failed to load config: \(error)")
                }
            }

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
            apiServer = NovaMLXAPIServer(
                inferenceService: inferenceService,
                modelManager: modelManager,
                config: serverConfig
            )

            appState.isServerRunning = true
            appState.serverPort = serverConfig.port
            appState.adminPort = serverConfig.adminPort
            appState.apiKey = serverConfig.apiKeys.first

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
            }

            appState.startStatsMonitoring(inferenceService: inferenceService)

            // Discover cloud models from remote endpoint
            Task {
                let _ = await CloudBackend.shared.fetchModels()
                appState.cloudModels = await inferenceService.listCloudModels()
                NovaMLXLog.info("Cloud models discovered: \(appState.cloudModels.count)")
            }

            let memHandler = MemoryPressureHandler(engine: engine, settingsManager: settingsManager)
            memHandler.start()
            memoryPressureHandler = memHandler

            // Start ProcessMemoryEnforcer (1s polling, configurable limits)
            await engine.startMemoryEnforcer()
            await engine.configureEnforcerSettings { [settingsManager] modelId in
                settingsManager.getSettings(modelId)
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
            apiServer = NovaMLXAPIServer(
                inferenceService: inferenceService,
                modelManager: modelManager,
                config: serverConfig
            )

            appState.serverPort = serverConfig.port
            appState.adminPort = serverConfig.adminPort
            appState.apiKey = serverConfig.apiKeys.first

            NovaMLXLog.info("Server restarted (apiKeys: \(serverConfig.apiKeys.count))")

            if let apiServer = apiServer {
                serverTask = Task {
                    try? await apiServer.start()
                }
            }
        }
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
        appState.requestedPage = page

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
