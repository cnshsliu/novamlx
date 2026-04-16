import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

public struct DashboardView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @State private var availableModels: [ModelRecord] = []
    @State private var downloadStates: [String: DownloadState] = [:]
    @State private var deviceInfo: DeviceInfo?
    @State private var refreshTimer: Timer?

    public init(appState: MenuBarAppState, inferenceService: InferenceService, modelManager: ModelManager) {
        self.appState = appState
        self.inferenceService = inferenceService
        self.modelManager = modelManager
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                headerSection
                statsGrid
                deviceInfoSection
                loadedModelsSection
                availableModelsSection
            }
            .padding(20)
        }
        .frame(minWidth: 700, minHeight: 500)
        .background(NovaTheme.Colors.background)
        .onAppear { startRefresh() }
        .onDisappear { stopRefresh() }
    }

    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Nova").foregroundColor(.primary) +
                Text("MLX").foregroundColor(NovaTheme.Colors.accent)
                Text("v\(NovaMLXCore.version)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .font(.title2.bold())

            Spacer()

            HStack(spacing: 8) {
                Circle()
                    .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                    .frame(width: 10, height: 10)
                Text(appState.isServerRunning ? "Running" : "Stopped")
                    .font(.subheadline)
            }

            Text("127.0.0.1:\(String(appState.serverPort))")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var statsGrid: some View {
        let gridItems = [
            GridItem(.flexible(), spacing: 12),
            GridItem(.flexible(), spacing: 12),
            GridItem(.flexible(), spacing: 12),
            GridItem(.flexible(), spacing: 12),
        ]

        return LazyVGrid(columns: gridItems, spacing: 12) {
            MetricCard(title: "Models", value: "\(appState.loadedModels.count)", subtitle: "loaded")
            MetricCard(title: "Memory", value: appState.systemStats.memoryUsed.bytesFormatted, subtitle: "system")
            MetricCard(title: "GPU", value: appState.systemStats.gpuMemoryUsed.bytesFormatted, subtitle: "used")
            MetricCard(title: "Requests", value: "\(appState.inferenceStats.activeRequests)", subtitle: "active")
            MetricCard(title: "Speed", value: String(format: "%.1f", appState.systemStats.tokensPerSecond), subtitle: "tok/s")
            MetricCard(title: "CPU", value: String(format: "%.0f%%", appState.systemStats.cpuUsage * 100), subtitle: "usage")
            MetricCard(title: "Uptime", value: formatUptime(appState.uptime), subtitle: "session")
            MetricCard(title: "Disk", value: modelManager.totalDiskUsage().bytesFormatted, subtitle: "models")
        }
    }

    private var deviceInfoSection: some View {
        Group {
            if let info = deviceInfo {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Device")
                        .font(.headline)

                    HStack(spacing: 24) {
                        infoRow(label: "Chip", value: info.chipVariant.isEmpty ? info.chipName : "\(info.chipName) (\(info.chipVariant))")
                        infoRow(label: "Memory", value: "\(info.memoryGB) GB")
                        infoRow(label: "GPU", value: "\(info.gpuCores) cores")
                        infoRow(label: "CPU", value: "\(info.cpuCores) cores")
                    }
                }
                .padding(12)
                .background(NovaTheme.Colors.cardBackground)
                .cornerRadius(8)
            }
        }
    }

    private func infoRow(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption)
                .foregroundColor(.primary)
        }
    }

    private var loadedModelsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Loaded Models")
                .font(.headline)

            if appState.loadedModels.isEmpty {
                Text("No models loaded")
                    .foregroundColor(.secondary)
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(12)
            } else {
                ForEach(appState.loadedModels, id: \.self) { modelId in
                    HStack {
                        Image(systemName: "cpu")
                            .foregroundColor(NovaTheme.Colors.statusOK)
                            .font(.caption)
                        Text(modelId)
                            .font(.system(size: 12))
                            .lineLimit(1)
                        CopyIDButton(id: modelId)
                        Spacer()
                        Button {
                            Task {
                                if let record = modelManager.getRecord(modelId) {
                                    await inferenceService.unloadModel(
                                        ModelIdentifier(id: modelId, family: record.family)
                                    )
                                }
                            }
                        } label: {
                            Image(systemName: "eject.circle")
                                .font(.caption)
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                }
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .cornerRadius(8)
    }

    private var availableModelsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Available Models")
                .font(.headline)

            let downloaded = availableModels.filter { modelManager.isDownloaded($0.id) }
            if downloaded.isEmpty {
                Text("No downloaded models found")
                    .foregroundColor(.secondary)
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(12)
            } else {
                ForEach(downloaded, id: \.id) { record in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(record.id)
                                .font(.system(size: 12))
                                .lineLimit(1)
                            HStack(spacing: 8) {
                                Text(record.family.rawValue)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text(record.modelType.rawValue)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                if record.sizeBytes > 0 {
                                    Text(record.sizeBytes.bytesFormatted)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }

                        Spacer()

                        if inferenceService.isModelLoaded(record.id) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(NovaTheme.Colors.statusOK)
                                .font(.caption)
                        } else {
                            Button("Load") {
                                Task {
                                    let config = ModelConfig(
                                        identifier: ModelIdentifier(id: record.id, family: record.family),
                                        modelType: record.modelType
                                    )
                                    do {
                                        try await inferenceService.loadModel(at: record.localURL, config: config)
                                    } catch {
                                        NovaMLXLog.error("Failed to load model: \(error)")
                                    }
                                }
                            }
                            .font(.caption)
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                }
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .cornerRadius(8)
    }

    private func formatUptime(_ interval: TimeInterval) -> String {
        let hours = Int(interval) / 3600
        let minutes = Int(interval) % 3600 / 60
        if hours > 0 { return "\(hours)h \(minutes)m" }
        return "\(minutes)m"
    }

    private func startRefresh() {
        availableModels = modelManager.downloadedModels()
        deviceInfo = DeviceInfo.current()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { _ in
            Task { @MainActor in
                self.availableModels = modelManager.downloadedModels()
                self.deviceInfo = DeviceInfo.current()
            }
        }
    }

    private func stopRefresh() {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }
}
