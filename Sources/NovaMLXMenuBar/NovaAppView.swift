import SwiftUI
import Charts
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

public enum AppPage: String, CaseIterable, Identifiable, Sendable {
    case status = "Status"
    case localInference = "Local Inference"   // was: models = "Models"
    case tokenhub = "Tokenhub"
    case loadBalancers = "Load Balancers"     // NEW — placeholder until Task 10
    case chat = "Playground"
    case cluster = "Cluster"
    case apiKeys = "API Keys"
    case settings = "Settings"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .status: return "gauge.with.dots.needle.bottom.50percent"
        case .localInference: return "cube.box"
        case .tokenhub: return "server.rack"
        case .loadBalancers: return "scalemass"
        case .chat: return "cpu"
        case .cluster: return "xserve"
        case .apiKeys: return "key.fill"
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

    public init(appState: MenuBarAppState, inferenceService: InferenceService, modelManager: ModelManager) {
        self.appState = appState
        self.inferenceService = inferenceService
        self.modelManager = modelManager
        _selectedPage = State(initialValue: appState.requestedPage ?? .status)
    }

    public var body: some View {
        HSplitView {
            sidebar
                .frame(width: 200)
            detailView
                .frame(minWidth: 700, minHeight: 600)
        }
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
                    if page != .cluster || appState.clusterEnabled {
                        sidebarItem(page)
                    }
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

                if page == .localInference && appState.activeDownloadCount > 0 {
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
        .accessibilityIdentifier("sidebar-\(page.rawValue.lowercased())")
    }

    private var sidebarHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if let logo = NSImage(named: "AppIcon") ?? (ResourceBundleLocator.find(bundleName: "NovaMLX_NovaMLXMenuBar")?.image(forResource: "AppIcon")) {
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
        VStack(alignment: .leading, spacing: 6) {
            miniTpsChart
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

    private var miniTpsChart: some View {
        let history = appState.tpsHistory
        let hasData = !history.allSatisfy({ $0 == 0 })
        let currentTps = history.last ?? 0

        return Group {
            if hasData {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(String(format: "%.0f tok/s", currentTps))
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundColor(currentTps > 0 ? NovaTheme.Colors.accent : .secondary)
                        Spacer()
                        if appState.peakTokensPerSecond > 0 {
                            Text(String(format: "peak %.0f", appState.peakTokensPerSecond))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    Chart {
                        ForEach(Array(history.enumerated()), id: \.offset) { index, tps in
                            LineMark(
                                x: .value("T", index),
                                y: .value("tps", tps)
                            )
                            .foregroundStyle(NovaTheme.Colors.accent)
                            .interpolationMethod(.catmullRom)
                            .lineStyle(StrokeStyle(lineWidth: 1.2))

                            AreaMark(
                                x: .value("T", index),
                                y: .value("tps", tps)
                            )
                            .foregroundStyle(
                                .linearGradient(
                                    colors: [NovaTheme.Colors.accent.opacity(0.25), NovaTheme.Colors.accent.opacity(0.02)],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                            .interpolationMethod(.catmullRom)
                        }
                    }
                    .chartXAxis(.hidden)
                    .chartYAxis(.hidden)
                    .frame(height: 32)
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(NovaTheme.Colors.cardBackground)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
                )
            }
        }
    }

    private var detailView: some View {
        ZStack {
            StatusPageView(appState: appState, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .status ? 1 : 0)
            ClusterPageView(appState: appState)
                .environmentObject(l10n)
                .opacity(selectedPage == .cluster ? 1 : 0)
            ModelsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .localInference ? 1 : 0)
            TokenhubPageView(appState: appState)
                .environmentObject(l10n)
                .opacity(selectedPage == .tokenhub ? 1 : 0)
            LoadBalancersPageView()
                .opacity(selectedPage == .loadBalancers ? 1 : 0)
            ChatPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .chat ? 1 : 0)
            APIKeysPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .apiKeys ? 1 : 0)
            SettingsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager)
                .environmentObject(l10n)
                .opacity(selectedPage == .settings ? 1 : 0)
        }
    }

    private func localizedName(_ page: AppPage) -> String {
        switch page {
        case .status: return l10n.tr("app.status")
        case .cluster: return l10n.tr("app.cluster")
        case .localInference: return l10n.tr("app.models")  // i18n key rename deferred to Task 12
        case .tokenhub: return l10n.tr("app.tokenhub")
        case .loadBalancers: return "Load Balancers"  // TODO(Task 12): l10n.tr("app.load_balancers")
        case .chat: return l10n.tr("app.chat")
        case .apiKeys: return "API Keys"
        case .settings: return l10n.tr("app.settings")
        }
    }
}
