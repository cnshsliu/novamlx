import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

public enum AppPage: String, CaseIterable, Identifiable, Sendable {
    case status = "Status"
    case models = "Models"
    case chat = "Chat"
    case agents = "Agents"
    case settings = "Settings"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .status: return "gauge.with.dots.needle.bottom.50percent"
        case .models: return "cube.box"
        case .chat: return "bubble.left.and.bubble.right"
        case .agents: return "app.badge.checkmark"
        case .settings: return "gearshape"
        }
    }
}

public struct NovaAppView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @EnvironmentObject var l10n: L10n
    @State private var selectedPage: AppPage
    @State private var columnVisibility: NavigationSplitViewVisibility = .doubleColumn

    public init(appState: MenuBarAppState, inferenceService: InferenceService, modelManager: ModelManager) {
        self.appState = appState
        self.inferenceService = inferenceService
        self.modelManager = modelManager
        _selectedPage = State(initialValue: appState.requestedPage ?? .status)
    }

    public var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebar
        } detail: {
            detailView
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipped()
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 900, minHeight: 600)
        .onChange(of: appState.requestedPage) { _, newValue in
            if let page = newValue {
                selectedPage = page
                appState.requestedPage = nil
            }
        }
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            sidebarHeader
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .padding(.bottom, 12)

            Divider()

            VStack(spacing: 2) {
                ForEach(AppPage.allCases) { page in
                    sidebarItem(page)
                }
            }
            .padding(.horizontal, 8)
            .padding(.top, 8)

            Spacer()

            sidebarFooter
                .padding(.horizontal, 16)
                .padding(.bottom, 12)
        }
        .background(Color(nsColor: .controlBackgroundColor))
        .navigationSplitViewColumnWidth(200)
    }

    private func sidebarItem(_ page: AppPage) -> some View {
        Button {
            selectedPage = page
        } label: {
            HStack(spacing: 8) {
                Image(systemName: page.icon)
                    .font(.system(size: 14))
                    .frame(width: 20)
                    .foregroundColor(selectedPage == page ? .primary : .secondary)

                Text(localizedName(page))
                    .font(.system(size: 13, weight: selectedPage == page ? .semibold : .regular))

                Spacer()

                if page == .models && appState.activeDownloadCount > 0 {
                    Text("\(appState.activeDownloadCount)")
                        .font(.caption2)
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(NovaTheme.Colors.accent)
                        .clipShape(Capsule())
                }
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 8)
            .background(selectedPage == page ? NovaTheme.Colors.accentDim : Color.clear)
            .overlay(alignment: .leading) {
                if selectedPage == page {
                    RoundedRectangle(cornerRadius: 1)
                        .fill(NovaTheme.Colors.accent)
                        .frame(width: 2)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var sidebarHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if let logo = NSImage(named: "AppIcon") ?? Bundle.module.image(forResource: "AppIcon") {
                    Image(nsImage: logo)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 36, height: 36)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 0) {
                        Text("Nova").font(.title3.bold())
                        Text("MLX").font(.title3.bold()).foregroundColor(NovaTheme.Colors.accent)
                    }
                    HStack(spacing: 4) {
                        Circle()
                            .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
                            .frame(width: 6, height: 6)
                        Text(appState.isServerRunning ? l10n.tr("app.running") : l10n.tr("app.stopped"))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
    }

    private var sidebarFooter: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !appState.loadedModels.isEmpty {
                Text(l10n.tr("app.modelsLoaded", appState.loadedModels.count))
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text("v\(NovaMLXCore.version)")
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var detailView: some View {
        ZStack {
            StatusPageView(appState: appState, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .status ? 1 : 0)
            ModelsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .models ? 1 : 0)
            ChatPageView(appState: appState, inferenceService: inferenceService)
                .environmentObject(l10n)
                .opacity(selectedPage == .chat ? 1 : 0)
            AgentsPageView(appState: appState)
                .environmentObject(l10n)
                .opacity(selectedPage == .agents ? 1 : 0)
            SettingsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .settings ? 1 : 0)
        }
    }

    private func localizedName(_ page: AppPage) -> String {
        switch page {
        case .status: return l10n.tr("app.status")
        case .models: return l10n.tr("app.models")
        case .chat: return l10n.tr("app.chat")
        case .agents: return l10n.tr("app.agents")
        case .settings: return l10n.tr("app.settings")
        }
    }
}
