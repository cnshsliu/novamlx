import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

@MainActor
public final class MenuBarAppState: ObservableObject {
    @Published public var isServerRunning: Bool = false
    @Published public var serverPort: Int = 6590
    @Published public var adminPort: Int = 6591
    @Published public var apiKey: String? = nil
    @Published public var loadedModels: [String] = []
    @Published public var cloudModels: [String] = []
    @Published public var systemStats: SystemStats = SystemStats()
    @Published public var inferenceStats: InferenceStats = InferenceStats()
    @Published public var totalTokensGenerated: UInt64 = 0
    @Published public var uptime: TimeInterval = 0
    @Published public var downloadTasks: [String: DownloadTaskInfo] = [:]
    @Published public var requestedPage: AppPage? = nil
    @Published public var tpsHistory: [Double] = []
    @Published public var peakTokensPerSecond: Double = 0

    // Cloud auth state — shared across all pages
    @Published public var cloudLoggedIn: Bool = false
    @Published public var cloudEmail: String = ""
    @Published public var cloudPlan: String = ""

    private var statsTimer: Timer?
    private let maxTpsHistory = 90

    /// Cap for realistic token generation speed on Apple Silicon (any value above is a measurement bug)
    private let maxRealisticTps: Double = 500

    public init() {}

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
                self.loadedModels = inferenceService.listLoadedModels()
                self.cloudModels = await inferenceService.listCloudModels()
                self.uptime = SystemMonitor.shared.uptime
                self.totalTokensGenerated = currentStats.totalTokensGenerated
                // Trim consecutive zeros: allow at most 1 zero data point (~2s) between
                // inference bursts. Long idle stretches produce a flat line that wastes
                // chart space — only keep the interesting TPS up/down curve.
                if tps > 0 {
                    self.tpsHistory.append(tps)
                } else {
                    let trailingZeros = self.tpsHistory.reversed().prefix(while: { $0 == 0 }).count
                    if trailingZeros < 1 {
                        self.tpsHistory.append(tps)
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
                self.refreshCloudAuthState()
            }
        }
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

    public func startDownload(repoId: String) {
        guard downloadTasks[repoId]?.isActive != true else { return }
        downloadTasks[repoId] = DownloadTaskInfo(repoId: repoId)

        Task {
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
                request.httpBody = try JSONSerialization.data(withJSONObject: ["repo_id": repoId])
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
                    downloadTasks[repoId]?.fileProgresses = files.compactMap { f in
                        guard let name = f["filename"] as? String else { return nil }
                        return FileDownloadInfo(
                            filename: name,
                            downloadedBytes: f["downloadedBytes"] as? Int64 ?? 0,
                            totalBytes: f["totalBytes"] as? Int64 ?? 0,
                            status: f["status"] as? String ?? "waiting"
                        )
                    }
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
}
