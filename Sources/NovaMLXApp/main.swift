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
        MenuBarExtra("NovaMLX", systemImage: "brain.head.profile.fill") {
            MenuBarContentView(
                appState: appDelegate.appState,
                inferenceService: appDelegate.inferenceService,
                modelManager: appDelegate.modelManager
            )
        }
        .menuBarExtraStyle(.window)

        Window("NovaMLX Dashboard", id: "dashboard") {
            Text("NovaMLX Dashboard")
                .frame(minWidth: 600, minHeight: 400)
        }
        .defaultSize(width: 800, height: 600)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let appState = MenuBarAppState()
    let engine = MLXEngine()
    let settingsManager: ModelSettingsManager
    lazy var inferenceService = InferenceService(engine: engine, settingsManager: settingsManager)
    let modelManager: ModelManager
    var apiServer: NovaMLXAPIServer?
    var memoryPressureHandler: MemoryPressureHandler?
    let config = NovaMLXConfiguration.shared

    override init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDir = appSupport.appendingPathComponent("NovaMLX/models", isDirectory: true)
        self.modelManager = ModelManager(modelsDirectory: modelsDir)
        self.settingsManager = ModelSettingsManager(baseDirectory: appSupport.appendingPathComponent("NovaMLX"))
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Task {
            try? await config.initializeDirectories()

            let configFile = await config.modelsDirectory
                .deletingLastPathComponent()
                .appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configFile.path) {
                try? await config.loadFromFile(configFile)
                NovaMLXLog.info("Loaded config from \(configFile.path)")
            }

            modelManager.registerPopularModels()
            modelManager.discoverModels()

            appState.startStatsMonitoring(inferenceService: inferenceService)

            let memHandler = MemoryPressureHandler(engine: engine, settingsManager: settingsManager)
            memHandler.start()
            memoryPressureHandler = memHandler

            let serverConfig = await config.serverConfig
            apiServer = NovaMLXAPIServer(
                inferenceService: inferenceService,
                modelManager: modelManager,
                config: serverConfig
            )

            appState.isServerRunning = true
            appState.serverPort = serverConfig.port

            NovaMLXLog.info("NovaMLX v\(NovaMLXCore.version) started")

            if let apiServer = apiServer {
                try? await apiServer.start()
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        memoryPressureHandler?.stop()
        appState.stopStatsMonitoring()
    }
}
