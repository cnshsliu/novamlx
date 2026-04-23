import SwiftUI
import NovaMLXCore
import NovaMLXModelManager

struct ModelsMenuView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager

    var body: some View {
        let l10n = L10n.shared
        VStack(alignment: .leading, spacing: 8) {
            Text(l10n.tr("app.models"))
                .font(.headline)

            Divider()

            if appState.loadedModels.isEmpty {
                Text(l10n.tr("menu.noModels"))
                    .foregroundColor(.secondary)
                    .font(.caption)
            } else {
                ForEach(appState.loadedModels, id: \.self) { model in
                    HStack {
                        Image(systemName: "cpu")
                            .foregroundColor(NovaTheme.Colors.statusOK)
                        Text(model)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                    }
                    .font(.system(size: 11))
                }
            }

            Divider()

            let downloaded = modelManager.downloadedModels()
            Text(l10n.tr("menu.downloadedCount", downloaded.count))
                .font(.caption)
                .foregroundColor(.secondary)

            let totalSize = modelManager.totalDiskUsage()
            Text(l10n.tr("menu.diskUsage", totalSize.bytesFormatted))
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .frame(width: 260)
    }
}
