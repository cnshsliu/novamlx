import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

@MainActor
public final class MenuBarAppState: ObservableObject {
    @Published public var isServerRunning: Bool = false
    @Published public var serverPort: Int = 8080
    @Published public var loadedModels: [String] = []
    @Published public var systemStats: SystemStats = SystemStats()
    @Published public var inferenceStats: InferenceStats = InferenceStats()
    @Published public var totalTokensGenerated: UInt64 = 0
    @Published public var uptime: TimeInterval = 0

    private var statsTimer: Timer?

    public init() {}

    public func startStatsMonitoring(inferenceService: InferenceService) {
        statsTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            Task { @MainActor in
                self.systemStats = SystemMonitor.shared.currentStats(
                    activeRequests: inferenceService.stats.activeRequests,
                    tokensPerSecond: self.systemStats.tokensPerSecond
                )
                self.inferenceStats = inferenceService.stats
                self.loadedModels = inferenceService.listLoadedModels()
                self.uptime = SystemMonitor.shared.uptime
            }
        }
    }

    public func stopStatsMonitoring() {
        statsTimer?.invalidate()
        statsTimer = nil
    }
}
