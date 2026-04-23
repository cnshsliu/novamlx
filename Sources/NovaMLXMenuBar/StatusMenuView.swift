import SwiftUI
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

struct StatusMenuView: View {
    @ObservedObject var appState: MenuBarAppState

    var body: some View {
        let l10n = L10n.shared
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                    .frame(width: 8, height: 8)
                Text(appState.isServerRunning ? l10n.tr("app.running") : l10n.tr("app.stopped"))
                    .font(.headline)
            }

            Divider()

            LabeledContent(l10n.tr("status.serverOnline").components(separatedBy: " ").first ?? "Server", value: "127.0.0.1:\(String(appState.serverPort))")
            LabeledContent(l10n.tr("status.models"), value: "\(appState.loadedModels.count) \(l10n.tr("status.loaded"))")
            LabeledContent(l10n.tr("status.memory"), value: appState.systemStats.memoryUsed.bytesFormatted)
            LabeledContent(l10n.tr("status.requests"), value: "\(appState.inferenceStats.activeRequests) \(l10n.tr("status.active"))")

            if appState.isServerRunning {
                LabeledContent(l10n.tr("status.uptime"), value: formatUptime(appState.uptime))
                if appState.systemStats.tokensPerSecond > 0 {
                    LabeledContent(l10n.tr("status.inferenceSpeed").components(separatedBy: " ").first ?? "Speed", value: String(format: "%.1f tok/s", appState.systemStats.tokensPerSecond))
                }
            }
        }
        .padding(8)
        .frame(width: 260)
    }

    private func formatUptime(_ interval: TimeInterval) -> String {
        let l10n = L10n.shared
        let hours = Int(interval) / 3600
        let minutes = Int(interval) % 3600 / 60
        if hours > 0 {
            return "\(hours)\(l10n.tr("status.hour")) \(minutes)\(l10n.tr("status.minute"))"
        }
        return "\(minutes)\(l10n.tr("status.minute"))"
    }
}
