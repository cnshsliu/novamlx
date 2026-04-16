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
    @Published public var systemStats: SystemStats = SystemStats()
    @Published public var inferenceStats: InferenceStats = InferenceStats()
    @Published public var totalTokensGenerated: UInt64 = 0
    @Published public var uptime: TimeInterval = 0
    @Published public var downloadTasks: [String: DownloadTaskInfo] = [:]

    private var statsTimer: Timer?

    public init() {}

    // MARK: - Stats Monitoring

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
                await self.pollDownloadStatus()
            }
        }
    }

    public func stopStatsMonitoring() {
        statsTimer?.invalidate()
        statsTimer = nil
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
