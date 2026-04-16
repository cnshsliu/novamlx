import SwiftUI
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

struct StatusMenuView: View {
    @ObservedObject var appState: MenuBarAppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                    .frame(width: 8, height: 8)
                Text(appState.isServerRunning ? "Running" : "Stopped")
                    .font(.headline)
            }

            Divider()

            LabeledContent("Server", value: "127.0.0.1:\(String(appState.serverPort))")
            LabeledContent("Models", value: "\(appState.loadedModels.count) loaded")
            LabeledContent("Memory", value: appState.systemStats.memoryUsed.bytesFormatted)
            LabeledContent("Requests", value: "\(appState.inferenceStats.activeRequests) active")

            if appState.isServerRunning {
                LabeledContent("Uptime", value: formatUptime(appState.uptime))
                if appState.systemStats.tokensPerSecond > 0 {
                    LabeledContent("Speed", value: String(format: "%.1f tok/s", appState.systemStats.tokensPerSecond))
                }
            }
        }
        .padding(8)
        .frame(width: 260)
    }

    private func formatUptime(_ interval: TimeInterval) -> String {
        let hours = Int(interval) / 3600
        let minutes = Int(interval) % 3600 / 60
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        }
        return "\(minutes)m"
    }
}
