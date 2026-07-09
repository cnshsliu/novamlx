import SwiftUI
import NovaMLXAPI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

public struct TPSHistoryPoint: Sendable, Equatable {
    public let tps: Double
    public let timestamp: Date

    public init(tps: Double, timestamp: Date = Date()) {
        self.tps = tps
        self.timestamp = timestamp
    }
}

public struct SpecBoostState: Sendable {
    public let status: String
    public let reason: String?
    public let draftModelId: String?
    public let draftDisplayName: String?
    public let draftDownloaded: Bool?
    public let draftLoaded: Bool?
}

@MainActor
public final class MenuBarAppState: ObservableObject {
    @Published public var isServerRunning: Bool = false
    @Published public var serverPort: Int = 6590
    @Published public var adminPort: Int = 6591
    @Published public var apiKey: String? = nil
    @Published public var loadedModels: [String] = []
    /// Models currently being restored on startup. Shown as spinner rows in the
    /// UI so the user sees progress while waiting (model loading can take minutes).
    @Published public var restoringModels: [String] = []
    @Published public var systemStats: SystemStats = SystemStats()
    @Published public var inferenceStats: InferenceStats = InferenceStats()
    @Published public var totalTokensGenerated: UInt64 = 0
    @Published public var uptime: TimeInterval = 0
    @Published public var downloadTasks: [String: DownloadTaskInfo] = [:]
    @Published public var requestedPage: AppPage? = nil
    /// When non-nil, ChatPageView should pre-select this model on its next
    /// appear / onReceive. Cleared after consumption. Drives the
    /// "Pick to Playground" buttons in Active Models, Tokenhub API models,
    /// and Load Balancers.
    @Published public var requestedPlaygroundModel: String? = nil
    @Published public var tpsHistory: [TPSHistoryPoint] = []
    @Published public var peakTokensPerSecond: Double = 0
    @Published public var currentInferenceModel: String? = nil
    /// Live, in-progress inference across ALL backends (LLM, VLM, ASR, TTS, image).
    /// Drives the "Realtime Inference Speed" panel. nil when idle.
    @Published public var liveActivity: LiveActivity? = nil

    // Cloud auth state — shared across all pages
    @Published public var cloudLoggedIn: Bool = false
    @Published public var cloudEmail: String = ""
    @Published public var cloudPlan: String = ""

    // Cluster state — read from config, drives sidebar visibility
    @Published public var clusterEnabled: Bool = false

    // Speed Boost state per model
    @Published public var specBoostStatus: [String: SpecBoostState] = [:]

    private var statsTimer: Timer?
    private var specBoostPollCounter = 0
    private let maxTpsHistory = 90

    /// Cap for realistic token generation speed on Apple Silicon (any value above is a measurement bug)
    private let maxRealisticTps: Double = 500

    public init() {}

    /// Jump to Playground and request that this model be pre-selected.
    /// Used by play.circle buttons in Models / Tokenhub / LoadBalancers pages.
    public func pickInPlayground(_ modelId: String) {
        requestedPlaygroundModel = modelId
        requestedPage = .chat
    }

    // MARK: - Stats Monitoring

    public func startStatsMonitoring(inferenceService: InferenceService) {
        statsTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            Task { @MainActor in
                let currentStats = inferenceService.stats
                let tps = currentStats.recentTokensPerSecond
                var systemStats = SystemMonitor.shared.currentStats(
                    activeRequests: currentStats.activeRequests,
                    tokensPerSecond: tps
                )
                // In worker mode, override CPU with worker-reported value
                // (SystemMonitor only measures the lightweight main process)
                if currentStats.workerCpuUsage > 0 {
                    systemStats = SystemStats(
                        cpuUsage: currentStats.workerCpuUsage,
                        memoryUsed: systemStats.memoryUsed,
                        memoryTotal: systemStats.memoryTotal,
                        gpuMemoryUsed: systemStats.gpuMemoryUsed,
                        activeRequests: systemStats.activeRequests,
                        tokensPerSecond: systemStats.tokensPerSecond,
                        uptime: systemStats.uptime
                    )
                }
                self.systemStats = systemStats
                self.inferenceStats = currentStats
                self.liveActivity = currentStats.liveActivity
                self.currentInferenceModel = CurrentInferenceModel.shared.modelID
                self.loadedModels = inferenceService.listLoadedModels()
                self.uptime = SystemMonitor.shared.uptime
                self.totalTokensGenerated = currentStats.totalTokensGenerated
                // Trim consecutive zeros: allow at most 1 zero data point (~2s) between
                // inference bursts. Long idle stretches produce a flat line that wastes
                // chart space — only keep the interesting TPS up/down curve.
                let point = TPSHistoryPoint(tps: tps, timestamp: Date())
                if tps > 0 {
                    self.tpsHistory.append(point)
                } else {
                    let trailingZeros = self.tpsHistory.reversed().prefix(while: { $0.tps == 0 }).count
                    if trailingZeros < 1 {
                        self.tpsHistory.append(point)
                    }
                }
                if self.tpsHistory.count > self.maxTpsHistory {
                    self.tpsHistory.removeFirst(self.tpsHistory.count - self.maxTpsHistory)
                }
                // Track peak, capping at realistic maximum to filter measurement bugs
                let displayTps = tps > 0 ? tps : 0
                if displayTps > self.peakTokensPerSecond && displayTps <= self.maxRealisticTps {
                    self.peakTokensPerSecond = displayTps
                }
                await self.pollDownloadStatus()
                if self.specBoostPollCounter % 5 == 0 {
                    await self.pollSpecBoostStatus()
                }
                self.specBoostPollCounter += 1
                self.refreshCloudAuthState()
                self.syncClusterEnabled()
            }
        }
    }

    private func syncClusterEnabled() {
        let configPath = NovaMLXPaths.configFile
        guard let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let server = json["server"] as? [String: Any] else { return }
        let enabled = server["cluster"] != nil
        if enabled != clusterEnabled { clusterEnabled = enabled }
    }

    public func stopStatsMonitoring() {
        statsTimer?.invalidate()
        statsTimer = nil
    }

    // MARK: - Cloud Auth

    public func refreshCloudAuthState() {
        if let cache = AuthCache.load(), !cache.isExpired, cache.valid {
            cloudLoggedIn = true
            cloudEmail = cache.userEmail
            cloudPlan = cache.plan
        } else {
            // No valid cache — show logged-out state
            cloudLoggedIn = false
            cloudEmail = ""
            cloudPlan = ""
        }
    }

    // MARK: - Download Management

    public var activeDownloadCount: Int {
        downloadTasks.values.filter { $0.isActive }.count
    }

    // MARK: - Hugging Face Mirror
    public var huggingfaceEndpoint: String? {
        get async { await NovaMLXConfiguration.shared.huggingfaceEndpoint }
    }

    public func setHuggingfaceEndpoint(_ endpoint: String?) async {
        await NovaMLXConfiguration.shared.setHuggingfaceEndpoint(endpoint)
        await NovaMLXConfiguration.shared.syncToStore()
    }

    public func startDownload(repoId: String) {
        // Allow click-through even when a download is "active" — the server's
        // cancelTasksForRepo guarantees single-flight per repo by killing any
        // in-flight task before starting the new one. Why allow this: a
        // stalled-but-not-failed download looks frozen to the user; letting
        // them click Resume (which kills + restarts server-side) is the
        // intended escape hatch. The local DownloadTaskInfo is rebuilt to
        // reset progress + status to .pending so the UI shows immediate feedback.
        downloadTasks[repoId] = DownloadTaskInfo(repoId: repoId)

        Task {
            // Read current mirror before entering the inner Task
            let currentEndpoint = await self.huggingfaceEndpoint

            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/download") else {
                downloadTasks[repoId]?.status = .failed
                downloadTasks[repoId]?.errorMessage = "Invalid URL"
                return
            }
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }

                // Send current mirror so the server uses the latest setting without restart
                let body: [String: Any] = [
                    "repo_id": repoId,
                    "endpoint": currentEndpoint as Any
                ]
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
                let (data, response) = try await URLSession.shared.data(for: request)
                if let httpResp = response as? HTTPURLResponse, httpResp.statusCode != 200 {
                    let msg = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
                        ?? "HTTP \(httpResp.statusCode)"
                    downloadTasks[repoId]?.status = .failed
                    downloadTasks[repoId]?.errorMessage = msg
                    return
                }
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let taskId = json["task_id"] as? String {
                    downloadTasks[repoId]?.taskId = taskId
                }
                downloadTasks[repoId]?.status = .downloading
            } catch {
                downloadTasks[repoId]?.status = .failed
                downloadTasks[repoId]?.errorMessage = error.localizedDescription
            }
        }
    }

    public func cancelDownload(repoId: String) {
        guard let task = downloadTasks[repoId] else { return }
        guard let taskId = task.taskId else {
            downloadTasks[repoId]?.status = .failed
            downloadTasks[repoId]?.errorMessage = "Cancelled"
            return
        }
        Task {
            guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/cancel") else { return }
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
                request.httpBody = try JSONSerialization.data(withJSONObject: ["task_id": taskId])
                _ = try await URLSession.shared.data(for: request)
                downloadTasks[repoId]?.status = .failed
                downloadTasks[repoId]?.errorMessage = "Cancelled"
            } catch {
                downloadTasks[repoId]?.status = .failed
                downloadTasks[repoId]?.errorMessage = "Cancel failed: \(error.localizedDescription)"
            }
        }
    }

    public func dismissDownload(repoId: String) {
        downloadTasks.removeValue(forKey: repoId)
    }

    /// Cancel a failed/incomplete download and delete all partial files from disk
    public func cancelAndDeleteDownload(repoId: String, modelsDirectory: URL) {
        let modelDir = modelsDirectory.appendingPathComponent(repoId, isDirectory: true)
        let fm = FileManager.default
        if fm.fileExists(atPath: modelDir.path) {
            try? fm.removeItem(at: modelDir)
            NovaMLXLog.info("Deleted partial download: \(modelDir.path)")
        }
        downloadTasks.removeValue(forKey: repoId)
    }

    /// Scan models directory for .download temp files left by interrupted downloads
    public func detectIncompleteDownloads(modelsDirectory: URL) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: modelsDirectory, includingPropertiesForKeys: [.isRegularFileKey]) else { return }

        var incompleteRepos = Set<String>()
        for case let url as URL in enumerator {
            guard url.pathExtension == "download" else { continue }
            // .download files sit in the model's directory
            // e.g., models/mlx-community/gemma-3-4b-it-4bit/model-00001.safetensors.download
            let modelDir = url.deletingLastPathComponent()
            let relativePath = modelDir.path.replacingOccurrences(of: modelsDirectory.path + "/", with: "")
            incompleteRepos.insert(relativePath)
        }

        for repoId in incompleteRepos {
            if downloadTasks[repoId] == nil {
                var task = DownloadTaskInfo(repoId: repoId)
                task.status = .failed
                task.errorMessage = "Interrupted — tap Retry to resume"
                downloadTasks[repoId] = task
                NovaMLXLog.info("Detected incomplete download: \(repoId)")
            }
        }
    }

    /// Auto-resume any downloads that were interrupted in a previous session
    public func resumeIncompleteDownloads() {
        for (repoId, task) in downloadTasks {
            if task.status == .failed {
                NovaMLXLog.info("Auto-resuming interrupted download: \(repoId)")
                startDownload(repoId: repoId)
            }
        }
    }

    private func pollDownloadStatus() async {
        let activeTasks = downloadTasks.values.filter { $0.isActive }
        guard !activeTasks.isEmpty else { return }

        guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/hf/tasks") else { return }
        do {
            var request = URLRequest(url: url)
            if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
            let (data, _) = try await URLSession.shared.data(for: request)
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let tasks = json["tasks"] as? [[String: Any]] else { return }

            for taskJson in tasks {
                guard let repoId = taskJson["repoId"] as? String,
                      downloadTasks[repoId] != nil else { continue }

                let status = taskJson["status"] as? String ?? ""
                let progress = taskJson["progress"] as? Double ?? 0
                let dlBytes = taskJson["downloadedBytes"] as? Int64 ?? 0
                let totalBytes = taskJson["totalBytes"] as? Int64 ?? 0

                // Parse per-file progress
                if let files = taskJson["fileProgresses"] as? [[String: Any]] {
                    let parsed: [FileDownloadInfo] = files.compactMap { f in
                        guard let name = f["filename"] as? String else { return nil }
                        let stallSeconds: Double? = (f["secondsSinceLastByte"] as? Double)
                        return FileDownloadInfo(
                            filename: name,
                            downloadedBytes: f["downloadedBytes"] as? Int64 ?? 0,
                            totalBytes: f["totalBytes"] as? Int64 ?? 0,
                            status: f["status"] as? String ?? "waiting",
                            currentURL: f["currentURL"] as? String,
                            retryCount: f["retryCount"] as? Int ?? 0,
                            isResuming: f["isResuming"] as? Bool ?? false,
                            speed: f["speed"] as? Double ?? 0,
                            secondsSinceLastByte: stallSeconds
                        )
                    }
                    downloadTasks[repoId]?.fileProgresses = parsed
                }

                switch status {
                case "completed":
                    downloadTasks[repoId]?.status = .completed
                    downloadTasks[repoId]?.progress = 100
                    downloadTasks[repoId]?.downloadedBytes = totalBytes
                    downloadTasks[repoId]?.totalBytes = totalBytes
                    #if DEBUG
                    NovaMLXLog.info("[Poll] Download completed: \(repoId), \(totalBytes) bytes")
                    #endif
                    // Trigger UI refresh so "My Models" picks up the newly downloaded model
                    NotificationCenter.default.post(name: .novaMLXModelsChanged, object: nil)
                case "failed":
                    downloadTasks[repoId]?.status = .failed
                    downloadTasks[repoId]?.errorMessage = taskJson["error"] as? String ?? "Download failed"
                case "cancelled":
                    downloadTasks[repoId]?.status = .failed
                    downloadTasks[repoId]?.errorMessage = "Cancelled"
                case "downloading":
                    downloadTasks[repoId]?.status = .downloading
                    downloadTasks[repoId]?.progress = progress
                    downloadTasks[repoId]?.downloadedBytes = dlBytes
                    downloadTasks[repoId]?.totalBytes = totalBytes
                default:
                    break
                }
            }
        } catch {
            // Silent — polling errors are non-critical
        }
    }

    private func pollSpecBoostStatus() async {
        guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/models") else { return }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 5
            if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
            let (data, _) = try await URLSession.shared.data(for: request)
            guard let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return }
            var updated: [String: SpecBoostState] = [:]
            for item in array {
                guard let id = item["id"] as? String,
                      let boost = item["specBoost"] as? [String: Any] else { continue }
                updated[id] = SpecBoostState(
                    status: boost["status"] as? String ?? "",
                    reason: boost["reason"] as? String,
                    draftModelId: boost["draftModelId"] as? String,
                    draftDisplayName: boost["draftDisplayName"] as? String,
                    draftDownloaded: boost["draftDownloaded"] as? Bool,
                    draftLoaded: boost["draftLoaded"] as? Bool
                )
            }
            self.specBoostStatus = updated
        } catch {
            // Silent
        }
    }

    func boostDownload(modelId: String) async {
        guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/models/boost/download") else { return }
        do {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
            request.httpBody = try JSONEncoder().encode(AdminLoadRequest(modelId: modelId))
            _ = try await URLSession.shared.data(for: request)
            // Immediately re-poll to get updated status
            await pollSpecBoostStatus()
        } catch {
            NovaMLXLog.error("[SpecBoost] Download failed: \(error)")
        }
    }

    func boostLoad(modelId: String) async {
        guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/models/boost/load") else { return }
        do {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
            request.httpBody = try JSONEncoder().encode(AdminLoadRequest(modelId: modelId))
            _ = try await URLSession.shared.data(for: request)
            await pollSpecBoostStatus()
        } catch {
            NovaMLXLog.error("[SpecBoost] Load failed: \(error)")
        }
    }
}
