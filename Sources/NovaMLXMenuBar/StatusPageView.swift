import SwiftUI
import Charts
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

struct StatusPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n
    @State private var deviceInfo: DeviceInfo?
    @State private var tqStats: [String: TQStatInfo] = [:]

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                serverStatusHero
                tpsChart
                metricsGrid
                deviceSection
                loadedModelsSection
            }
            .padding(24)
        }
        .onAppear { loadTQStats() }
    }

    private var serverStatusHero: some View {
        HStack(spacing: 20) {
            ZStack {
                Circle()
                    .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK.opacity(0.2) : NovaTheme.Colors.statusError.opacity(0.2))
                    .frame(width: 56, height: 56)
                Image(systemName: appState.isServerRunning ? "bolt.fill" : "bolt.slash")
                    .font(.title2)
                    .foregroundColor(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(appState.isServerRunning ? l10n.tr("status.serverOnline") : l10n.tr("status.serverOffline"))
                    .font(.title2.bold())
                Text("http://127.0.0.1:\(String(appState.serverPort))")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    + Text("  admin:\(String(appState.serverPort == 8080 ? 8081 : 8081))")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if appState.isServerRunning {
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f", appState.systemStats.tokensPerSecond))
                        .font(.title.bold())
                        .foregroundColor(NovaTheme.Colors.accent)
                    Text(l10n.tr("status.tokensPerSec"))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(20)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    private var tpsChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(l10n.tr("status.inferenceSpeed"))
                    .font(.headline)
                Spacer()
                if let last = appState.tpsHistory.last, last > 0 {
                    Text(String(format: "%.1f tok/s", last))
                        .font(.caption)
                        .foregroundColor(NovaTheme.Colors.accent)
                        .fontWeight(.medium)
                }
            }

            if appState.tpsHistory.allSatisfy({ $0 == 0 }) {
                Text(l10n.tr("status.noActivity"))
                    .foregroundColor(.secondary)
                    .font(.subheadline)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 24)
            } else {
                Chart {
                    ForEach(Array(appState.tpsHistory.enumerated()), id: \.offset) { index, tps in
                        LineMark(
                            x: .value("Time", index),
                            y: .value("tok/s", tps)
                        )
                        .foregroundStyle(NovaTheme.Colors.accent)
                        .interpolationMethod(.catmullRom)

                        AreaMark(
                            x: .value("Time", index),
                            y: .value("tok/s", tps)
                        )
                        .foregroundStyle(
                            .linearGradient(
                                colors: [NovaTheme.Colors.accent.opacity(0.3), NovaTheme.Colors.accent.opacity(0.05)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .interpolationMethod(.catmullRom)
                    }
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisValueLabel()
                            .font(.caption2)
                    }
                }
                .frame(height: 120)
            }
        }
        .padding(16)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    private var metricsGrid: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricCard(icon: "cpu", title: l10n.tr("status.cpu"), value: String(format: "%.0f%%", min(appState.systemStats.cpuUsage, 100.0)))
            MetricCard(icon: "memorychip", title: l10n.tr("status.memory"), value: appState.systemStats.memoryUsed.bytesFormatted)
            MetricCard(icon: "gpu", title: l10n.tr("status.gpu"), value: appState.systemStats.gpuMemoryUsed.bytesFormatted)
            MetricCard(icon: "arrow.triangle.2.circlepath", title: l10n.tr("status.uptime"), value: formatUptime(appState.uptime))
            MetricCard(icon: "text.bubble", title: l10n.tr("status.requests"), value: "\(appState.inferenceStats.activeRequests) \(l10n.tr("status.active"))")
            MetricCard(icon: "cube.box", title: l10n.tr("status.models"), value: "\(appState.loadedModels.count) \(l10n.tr("status.loaded"))")
            MetricCard(icon: "externaldrive", title: l10n.tr("status.disk"), value: modelManager.totalDiskUsage().bytesFormatted)
            MetricCard(icon: "chart.line.uptrend.xyaxis", title: l10n.tr("status.totalTokens"), value: formatNumber(appState.totalTokensGenerated))
        }
    }

    private var deviceSection: some View {
        let info = deviceInfo ?? DeviceInfo.current()
        return VStack(alignment: .leading, spacing: 12) {
            Text(l10n.tr("status.device"))
                .font(.headline)

            HStack(spacing: 32) {
                deviceItem(label: l10n.tr("status.chip"), value: info.chipVariant.isEmpty ? info.chipName : "\(info.chipName) (\(info.chipVariant))")
                deviceItem(label: l10n.tr("status.memory"), value: "\(info.memoryGB) GB")
                deviceItem(label: l10n.tr("status.gpuCores"), value: "\(info.gpuCores)")
                deviceItem(label: l10n.tr("status.cpuCores"), value: "\(info.cpuCores)")
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    private func deviceItem(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundColor(.secondary)
            Text(value).font(.caption).fontWeight(.medium)
        }
    }

    private var loadedModelsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(l10n.tr("status.activeModels"))
                .font(.headline)

            if appState.loadedModels.isEmpty {
                Text(l10n.tr("status.noModelsLoaded"))
                    .foregroundColor(.secondary)
                    .font(.subheadline)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(20)
            } else {
                ForEach(appState.loadedModels, id: \.self) { modelId in
                    HStack {
                        Image(systemName: "circle.fill")
                            .foregroundColor(NovaTheme.Colors.statusOK)
                            .font(.caption2)
                        Text(modelId)
                            .font(.system(size: 13))
                            .lineLimit(1)
                        CopyIDButton(id: modelId)
                        Spacer()
                        if modelManager.getRecord(modelId) != nil {
                            if let tq = tqStats[modelId] {
                                Text("KV \(tq.bits)-bit (\(String(format: "%.1f", tq.compressionRatio))x)")
                                    .font(.caption2)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(NovaTheme.Colors.accentDim)
                                    .foregroundColor(NovaTheme.Colors.accent)
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                        Text(l10n.tr("status.ready"))
                            .font(.caption)
                            .foregroundColor(NovaTheme.Colors.statusOK)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(NovaTheme.Colors.rowBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    private func formatUptime(_ interval: TimeInterval) -> String {
        let h = Int(interval) / 3600
        let m = Int(interval) % 3600 / 60
        return h > 0 ? "\(h)h \(m)m" : "\(m)m"
    }

    private func formatNumber(_ n: UInt64) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    private func loadTQStats() {
        let adminPort = appState.adminPort
        guard let url = URL(string: "http://127.0.0.1:\(String(adminPort))/admin/api/turboquant") else { return }
        Task {
            var request = URLRequest(url: url)
            if let apiKey = appState.apiKey {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            do {
                let (data, resp) = try await URLSession.shared.data(for: request)
                guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else { return }
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let models = json["models"] as? [[String: Any]] {
                    var stats: [String: TQStatInfo] = [:]
                    for m in models {
                        guard let id = m["modelId"] as? String,
                              let bits = m["bits"] as? Int else { continue }
                        let ratio = m["compressionRatio"] as? Double ?? 0
                        stats[id] = TQStatInfo(bits: bits, compressionRatio: ratio)
                    }
                    tqStats = stats
                }
            } catch {}
        }
    }
}

private struct TQStatInfo {
    let bits: Int
    let compressionRatio: Double
}
