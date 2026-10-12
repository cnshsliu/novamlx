import SwiftUI
import Charts
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

public enum AppPage: String, CaseIterable, Identifiable, Sendable {
    case status = "Status"
    case localInference = "Local Inference"   // was: models = "Models"
    case downloads = "Downloads"
    case voiceClone = "Voice Clone"
    case videoSlice = "Video Slice"
    case talkingHead = "Talking Head"
    case cluster = "Cluster"
    case requests = "Requests"
    case apiKeys = "API Keys"
    case catalogAdmin = "Catalog Admin"
    case settings = "Settings"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .status: return "gauge.with.dots.needle.bottom.50percent"
        case .localInference: return "cube.box"
        case .downloads: return "arrow.down.circle"
        case .voiceClone: return "waveform.badge.microphone"
        case .videoSlice: return "film"
        case .talkingHead: return "person.crop.rectangle"
        case .cluster: return "xserve"
        case .requests: return "list.bullet.rectangle"
        case .apiKeys: return "key.fill"
        case .catalogAdmin: return "checkmark.seal.fill"
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
    /// Page drafts. Not observed here, so a keystroke does not rebuild the sidebar.
    @State private var pageMemory = PageMemoryBox()

    public init(
        appState: MenuBarAppState,
        inferenceService: InferenceService,
        modelManager: ModelManager
    ) {
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
        .task { await modelManager.refreshModelCaches() }
        .onReceive(NotificationCenter.default.publisher(for: .novaMLXModelsChanged)) { _ in
            Task { await modelManager.refreshModelCaches() }
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
                    if page == .cluster && !appState.clusterEnabled {
                        EmptyView()
                    } else if page == .catalogAdmin && !appState.isCatalogAdmin {
                        EmptyView()
                    } else {
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

                if page == .downloads && appState.activeDownloadCount > 0 {
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
        let hasData = !history.allSatisfy({ $0.tps == 0 })
        let currentTps = history.last?.tps ?? 0
        let imageActivity = appState.liveActivity?.kind == .image ? appState.liveActivity : nil

        return Group {
            if let imageActivity, !hasData {
                Text(imageActivity.displayLine)
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.accent)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 4)
            } else if hasData {
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
                        ForEach(Array(history.enumerated()), id: \.offset) { index, point in
                            LineMark(
                                x: .value("T", index),
                                y: .value("tps", point.tps)
                            )
                            .foregroundStyle(NovaTheme.Colors.accent)
                            .interpolationMethod(.catmullRom)
                            .lineStyle(StrokeStyle(lineWidth: 1.2))

                            AreaMark(
                                x: .value("T", index),
                                y: .value("tps", point.tps)
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

    @ViewBuilder
    private var detailView: some View {
        switch selectedPage {
        case .status:
            StatusPageView(appState: appState, modelManager: modelManager)
                .environmentObject(l10n)
        case .cluster:
            ClusterPageView(appState: appState, memory: pageMemory.cluster)
                .environmentObject(l10n)
        case .localInference:
            ModelsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager, memory: pageMemory.models)
                .environmentObject(l10n)
        case .downloads:
            DownloadsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager, memory: pageMemory.downloads)
                .environmentObject(l10n)
        case .voiceClone:
            VoiceClonePageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager, memory: pageMemory.voice)
                .environmentObject(l10n)
        case .videoSlice:
            VideoSlicePageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager, memory: pageMemory.video)
                .environmentObject(l10n)
        case .talkingHead:
            TalkingHeadPageView(appState: appState, inferenceService: inferenceService, memory: pageMemory.talkingHead)
                .environmentObject(l10n)
        case .requests:
            RequestLogPageView(appState: appState, inferenceService: inferenceService, memory: pageMemory.requests)
                .environmentObject(l10n)
        case .apiKeys:
            APIKeysPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager, memory: pageMemory.apiKeys)
                .environmentObject(l10n)
        case .settings:
            SettingsPageView(appState: appState, inferenceService: inferenceService, modelManager: modelManager, memory: pageMemory.settings)
                .environmentObject(l10n)
        case .catalogAdmin:
            if appState.isCatalogAdmin {
                CatalogAdminPageView(appState: appState, modelManager: modelManager, memory: pageMemory.catalog)
                    .environmentObject(l10n)
            }
        }
    }

    private func localizedName(_ page: AppPage) -> String {
        switch page {
        case .status: return l10n.tr("app.status")
        case .cluster: return l10n.tr("app.cluster")
        case .localInference: return l10n.tr("app.local_inference")
        case .downloads: return l10n.tr("app.downloads")
        case .voiceClone: return l10n.tr("app.voiceClone")
        case .videoSlice: return l10n.tr("app.videoSlice")
        case .talkingHead: return l10n.tr("app.talkingHead")
        case .requests: return l10n.tr("app.requests")
        case .apiKeys: return "API Keys"
        case .catalogAdmin: return l10n.tr("app.catalogAdmin")
        case .settings: return l10n.tr("app.settings")
        }
    }
}
