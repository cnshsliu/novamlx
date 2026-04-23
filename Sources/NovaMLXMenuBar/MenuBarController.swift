import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXAPI

@MainActor
struct NovaMLXMenuBarController {
    let appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    func createMenuBar() -> some Scene {
        MenuBarExtra("NovaMLX", systemImage: "brain.head.profile.fill") {
            MenuBarContentView(
                appState: appState,
                inferenceService: inferenceService,
                modelManager: modelManager
            )
        }
        .menuBarExtraStyle(.window)
    }
}

public struct MenuBarContentView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @State private var selectedTab = 0

    public init(appState: MenuBarAppState, inferenceService: InferenceService, modelManager: ModelManager) {
        self.appState = appState
        self.inferenceService = inferenceService
        self.modelManager = modelManager
    }

    public var body: some View {
        let l10n = L10n.shared
        VStack(spacing: 0) {
            TabView(selection: $selectedTab) {
                StatusMenuView(appState: appState)
                    .tabItem { Label(l10n.tr("app.status"), systemImage: "circle.fill") }
                    .tag(0)

                ModelsMenuView(appState: appState, modelManager: modelManager)
                    .tabItem { Label(l10n.tr("app.models"), systemImage: "cube.box") }
                    .tag(1)
            }
            .frame(width: 280, height: 200)

            Divider()

            HStack(spacing: 8) {
                Button {
                    NotificationCenter.default.post(name: .openNovaAppWindow, object: nil)
                } label: {
                    Image(systemName: "macwindow")
                    Text(l10n.tr("menuBar.openWindow"))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Spacer()

                Button {
                    NSApp.terminate(nil)
                } label: {
                    Image(systemName: "power")
                    Text(l10n.tr("menuBar.quit"))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .foregroundColor(NovaTheme.Colors.statusError)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }
}

extension Notification.Name {
    public static let openNovaAppWindow = Notification.Name("openNovaAppWindow")
    public static let restartNovaMLXServer = Notification.Name("restartNovaMLXServer")
    public static let novaMLXModelsChanged = Notification.Name("novaMLXModelsChanged")
}
