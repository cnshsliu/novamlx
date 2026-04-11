import SwiftUI
import NovaMLXCore
import NovaMLXModelManager

struct ModelsMenuView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models")
                .font(.headline)

            Divider()

            if appState.loadedModels.isEmpty {
                Text("No models loaded")
                    .foregroundColor(.secondary)
                    .font(.caption)
            } else {
                ForEach(appState.loadedModels, id: \.self) { model in
                    HStack {
                        Image(systemName: "cpu")
                            .foregroundColor(.green)
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
            Text("Downloaded: \(downloaded.count) models")
                .font(.caption)
                .foregroundColor(.secondary)

            let totalSize = modelManager.totalDiskUsage()
            Text("Disk usage: \(totalSize.bytesFormatted)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .frame(width: 260)
    }
}
