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
    @State private var hoveredPoint: TPSHistoryPoint?
    @State private var hoveredIndex: Int = 0
    @State private var hoveredX: CGFloat?

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                serverStatusHero
                tpsChart
                metricsGrid
                deviceSection
            }
            .padding(24)
        }
    }

    private var serverStatusHero: some View {
        HStack(spacing: 20) {
            ZStack {
                Circle()
                    .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK.opacity(0.2) : NovaTheme.Colors.statusError.opacity(0.2))
                    .frame(width: 56, height: 56)
                Image(systemName: heroIcon)
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
                // Live inference indicator: show the active model + kind if a
                // backend (LLM, VLM, ASR, TTS, image) is currently running.
                if let activity = appState.liveActivity {
                    HStack(spacing: 4) {
                        ProgressView()
                            .scaleEffect(0.5)
                            .frame(width: 10, height: 10)
                        Text("\(activity.model) · \(activity.kind.label)")
                            .font(.caption)
                            .foregroundColor(NovaTheme.Colors.accent)
                            .lineLimit(1)
                    }
                } else if let model = appState.currentInferenceModel ?? appState.loadedModels.first {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(NovaTheme.Colors.accent)
                            .frame(width: 6, height: 6)
                        Text(model)
                            .font(.caption)
                            .foregroundColor(NovaTheme.Colors.accent)
                            .lineLimit(1)
                    }
                }
            }

            Spacer()

            if appState.isServerRunning {
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f", appState.peakTokensPerSecond))
                        .font(.title.bold())
                        .foregroundColor(NovaTheme.Colors.accent)
                    Text(l10n.tr("status.peakTokensPerSec"))
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
                Text(l10n.tr("status.realtimeInferenceSpeed"))
                    .font(.headline)
                if let activity = appState.liveActivity {
                    Text(activity.kind.label)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(NovaTheme.Colors.accent.opacity(0.15))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .clipShape(Capsule())
                }
                Spacer()
                if !liveSpeedText.isEmpty {
                    Text(liveSpeedText)
                        .font(.caption)
                        .foregroundColor(NovaTheme.Colors.accent)
                        .fontWeight(.medium)
                }
            }

            if appState.tpsHistory.allSatisfy({ $0.tps == 0 }) {
                Text(l10n.tr("status.noActivity"))
                    .foregroundColor(.secondary)
                    .font(.subheadline)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 24)
            } else {
                let history = appState.tpsHistory
                Chart {
                    ForEach(Array(history.enumerated()), id: \.offset) { index, point in
                        LineMark(
                            x: .value("Time", index),
                            y: .value("tok/s", point.tps)
                        )
                        .foregroundStyle(NovaTheme.Colors.accent)
                        .interpolationMethod(.catmullRom)

                        AreaMark(
                            x: .value("Time", index),
                            y: .value("tok/s", point.tps)
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

                    if let point = hoveredPoint {
                        RuleMark(y: .value("tok/s", point.tps))
                            .foregroundStyle(NovaTheme.Colors.accent.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))

                        PointMark(
                            x: .value("Time", hoveredIndex),
                            y: .value("tok/s", point.tps)
                        )
                        .foregroundStyle(NovaTheme.Colors.accent)
                        .symbolSize(50)
                    }
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisValueLabel()
                            .font(.caption2)
                    }
                }
                .chartOverlay { proxy in
                    GeometryReader { geometry in
                        ZStack(alignment: .topLeading) {
                            Rectangle()
                                .fill(.clear)
                                .contentShape(Rectangle())
                                .onContinuousHover { phase in
                                    switch phase {
                                    case .active(let location):
                                        guard let plotAnchor = proxy.plotFrame else {
                                            hoveredPoint = nil
                                            hoveredX = nil
                                            return
                                        }
                                        let plotFrame = geometry[plotAnchor]
                                        let relX = location.x - plotFrame.origin.x
                                        let xRatio = max(0, min(1, relX / max(plotFrame.width, 1)))
                                        let maxIndex = max(history.count - 1, 0)
                                        let idx = maxIndex > 0
                                            ? Int(round(xRatio * Double(maxIndex)))
                                            : 0
                                        let clamped = max(0, min(idx, maxIndex))
                                        hoveredPoint = history[clamped]
                                        hoveredIndex = clamped
                                        hoveredX = plotFrame.minX + xRatio * plotFrame.width
                                    case .ended:
                                        hoveredPoint = nil
                                        hoveredIndex = 0
                                        hoveredX = nil
                                    }
                                }

                            if let point = hoveredPoint, let x = hoveredX {
                                tpsHoverTooltip(for: point)
                                    .fixedSize()
                                    .position(
                                        x: min(max(x, 72), geometry.size.width - 72),
                                        y: 18
                                    )
                            }
                        }
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

    private func formatUptime(_ interval: TimeInterval) -> String {
        let h = Int(interval) / 3600
        let m = Int(interval) % 3600 / 60
        return h > 0 ? "\(h)h \(m)m" : "\(m)m"
    }

    private func tpsHoverTooltip(for point: TPSHistoryPoint) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "%.1f tok/s", point.tps))
                .font(.caption.bold())
                .foregroundColor(NovaTheme.Colors.accent)
            Text(formatTPSHoverTimestamp(point.timestamp))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(NovaTheme.Colors.cardBackground)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(NovaTheme.Colors.accent.opacity(0.4), lineWidth: 1))
        )
        .shadow(color: .black.opacity(0.12), radius: 4, y: 2)
    }

    private func formatTPSHoverTimestamp(_ date: Date) -> String {
        let interval = Date().timeIntervalSince(date)
        let relative: String
        if interval < 10 {
            relative = l10n.tr("status.tpsJustNow")
        } else if interval < 60 {
            relative = l10n.tr("status.tpsSecondsAgo", Int(interval))
        } else if interval < 3600 {
            relative = l10n.tr("status.tpsMinutesAgo", Int(interval / 60))
        } else {
            relative = l10n.tr("status.tpsHoursAgo", Int(interval / 3600))
        }
        let clock = DateFormatter.localizedString(from: date, dateStyle: .none, timeStyle: .medium)
        return "\(relative) · \(clock)"
    }

    private func formatNumber(_ n: UInt64) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    // MARK: - Live activity helpers

    /// SF Symbol for the hero badge, chosen by what is running (or idle bolt).
    private var heroIcon: String {
        guard appState.isServerRunning else { return "bolt.slash" }
        switch appState.liveActivity?.kind {
        case .asr: return "waveform"
        case .tts: return "speaker.wave.2"
        case .image: return "photo"
        case .vlm: return "eye"
        case .llm: return "bolt.fill"
        case nil: return "bolt.fill"
        }
    }

    /// Speed string for the active inference, e.g. "23.5 tok/s". Falls back to
    /// the recent LLM TPS when idle. Returns "" if nothing to show.
    private var liveSpeedText: String {
        if let activity = appState.liveActivity, activity.speed > 0 {
            return String(format: "%.1f %@", activity.speed, activity.unit)
        }
        if let last = appState.tpsHistory.last, last.tps > 0 {
            return String(format: "%.1f tok/s", last.tps)
        }
        return ""
    }
}
