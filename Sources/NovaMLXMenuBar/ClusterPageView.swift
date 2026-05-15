import SwiftUI
import NovaMLXCore
import NovaMLXDistributed
import NovaMLXUtils

// MARK: - Data Models

struct WorkerSnapshot: Identifiable {
    let id: String
    let nodeId: String
    let hostname: String
    let port: Int
    let status: String
    let totalMemoryBytes: UInt64
    let computeCapability: Double
    let cpuModel: String
    let registeredAt: Date?
    let lastHeartbeat: Date?

    init?(json: [String: Any]) {
        guard let nodeId = json["nodeId"] as? String else { return nil }
        self.id = nodeId
        self.nodeId = nodeId
        self.hostname = json["hostname"] as? String ?? nodeId
        self.port = json["port"] as? Int ?? 6591
        self.status = json["status"] as? String ?? "unknown"
        self.totalMemoryBytes = json["totalMemoryBytes"] as? UInt64 ?? json["memory"] as? UInt64 ?? 0
        self.computeCapability = json["computeCapability"] as? Double ?? 1.0
        self.cpuModel = json["cpuModel"] as? String ?? ""

        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let reg = json["registeredAt"] as? String {
            self.registeredAt = df.date(from: reg)
        } else { self.registeredAt = nil }
        if let hb = json["lastHeartbeat"] as? String {
            self.lastHeartbeat = df.date(from: hb)
        } else { self.lastHeartbeat = nil }
    }
}

/// A host discovered on the local network via ARP scanning.
struct DiscoveredHost: Identifiable {
    let id: String        // IP address
    let ipAddress: String
    let hostname: String  // resolved or "?"
    let interface: String // e.g. "bridge0"
    let macAddress: String

    /// Whether this host matches a registered worker (matched by hostname).
    var matchedWorker: WorkerSnapshot?
}

enum ClusterHealth: String {
    case healthy, degraded, waiting, down

    var color: Color {
        switch self {
        case .healthy: return NovaTheme.Colors.statusOK
        case .degraded: return NovaTheme.Colors.statusWarn
        case .waiting: return NovaTheme.Colors.accent
        case .down: return NovaTheme.Colors.statusError
        }
    }

    var label: String {
        switch self {
        case .healthy: return "Healthy"
        case .degraded: return "Degraded"
        case .waiting: return "Waiting for Workers"
        case .down: return "Down"
        }
    }
}

enum ClusterEventType {
    case workerJoined, workerLeft, statusChange
}

struct ModelNodeReadiness: Identifiable {
    let id: String
    let nodeId: String
    let hostname: String
    let layerCount: Int
    let startLayer: Int
    let endLayer: Int
    let status: String
    let memoryUsedBytes: UInt64
    let errorMessage: String?

    var statusColor: Color {
        switch status {
        case "ready": return NovaTheme.Colors.statusOK
        case "loading": return NovaTheme.Colors.statusWarn
        case "failed": return NovaTheme.Colors.statusError
        default: return NovaTheme.Colors.textSecondary
        }
    }
}

struct ClusterEvent: Identifiable {
    let id = UUID()
    let timestamp = Date()
    let type: ClusterEventType
    let nodeId: String
    let detail: String
}

enum WorkerSortOrder: String, CaseIterable {
    case joinTime
    case onlineDuration
    case cpuScore
    case memory

    var label: String {
        switch self {
        case .joinTime: return "Join Time"
        case .onlineDuration: return "Online Duration"
        case .cpuScore: return "CPU"
        case .memory: return "Memory"
        }
    }

    var icon: String {
        switch self {
        case .joinTime: return "clock.badge"
        case .onlineDuration: return "hourglass"
        case .cpuScore: return "cpu"
        case .memory: return "memorychip"
        }
    }
}

// MARK: - ClusterPageView

struct ClusterPageView: View {
    @ObservedObject var appState: MenuBarAppState
    @EnvironmentObject var l10n: L10n

    @State private var isRunning: Bool = false
    @State private var clusterRole: String = "none"
    @State private var clusterStrategy: String = "minNodes"
    @State private var coordinatorHost: String = ""
    @State private var workers: [WorkerSnapshot] = []
    @State private var events: [ClusterEvent] = []
    @State private var expandedNodeId: String? = nil
    @State private var workerSort: WorkerSortOrder = .joinTime
    @State private var pollTimer: Timer?
    @State private var lastPollTime: Date?

    // Network discovery
    @State private var discoveredHosts: [DiscoveredHost] = []
    @State private var scanTimer: Timer?

    // Local hardware info (read once)
    private let localCPUModel: String = {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return CString( buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }()

    // Worker deployment
    @State private var deployPhases: [String: DeployPhase] = [:]
    @State private var deployErrors: [String: String] = [:]
    @State private var showCredentialDialog = false
    @State private var credentialHost = ""
    @State private var credentialUsername = ""
    @State private var credentialPassword = ""

    // Model activation
    @State private var activeModel: String?
    @State private var clusterModelState: String = "idle"
    @State private var modelReadiness: [ModelNodeReadiness] = []
    @State private var isActivating: Bool = false
    @State private var activationError: String?
    @State private var distributedTPS: Double?
    @State private var distributedSpecAccuracy: Double?
    @State private var distributedLastAgo: String?

    private let maxEvents = 50

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if isRunning || appState.clusterEnabled {
                    clusterHealthHero
                    modelActivationSection
                    nodeGrid
                    networkDiscoverySection
                    if !events.isEmpty { recentEventsSection }
                } else {
                    clusterDisabledView
                }
            }
            .padding(24)
        }
        .onAppear { startPolling() }
        .onDisappear { stopPolling() }
        .sheet(isPresented: $showCredentialDialog) {
            credentialDialogSheet
        }
    }

    // MARK: - Computed

    private var health: ClusterHealth {
        if workers.isEmpty { return .waiting }
        let bad = workers.filter { $0.status == "disconnected" || $0.status == "failed" }
        let good = workers.filter { $0.status == "active" || $0.status == "ready" }
        if !bad.isEmpty && !good.isEmpty { return .degraded }
        if good.isEmpty { return .degraded }
        return .healthy
    }

    private var activeCount: Int {
        workers.filter { $0.status == "active" || $0.status == "ready" }.count
    }

    private var totalMemory: UInt64 {
        // Include Coordinator's own memory — it participates in inference too
        let localMem = ProcessInfo.processInfo.physicalMemory
        return workers.reduce(localMem) { $0 + $1.totalMemoryBytes }
    }

    /// Total number of nodes: Coordinator + Workers
    private var totalNodeCount: Int {
        1 + workers.count
    }

    private var sortedWorkers: [WorkerSnapshot] {
        workers.sorted { a, b in
            switch workerSort {
            case .joinTime:
                let ta = a.registeredAt ?? .distantPast
                let tb = b.registeredAt ?? .distantPast
                return ta < tb
            case .onlineDuration:
                let ta = a.registeredAt ?? .distantFuture
                let tb = b.registeredAt ?? .distantFuture
                return ta < tb
            case .cpuScore:
                return cpuScore(a.cpuModel) > cpuScore(b.cpuModel)
            case .memory:
                return a.totalMemoryBytes > b.totalMemoryBytes
            }
        }
    }

    /// Rough chip generation score for sorting: M1=1, M2=2, M3=3, M4=4; Pro +0.5, Max +1, Ultra +1.5
    private func cpuScore(_ model: String) -> Double {
        guard !model.isEmpty else { return 0 }
        let m = model.lowercased()
        var score = 0.0
        if m.contains("m1") { score = 1 }
        else if m.contains("m2") { score = 2 }
        else if m.contains("m3") { score = 3 }
        else if m.contains("m4") { score = 4 }
        else if m.contains("m5") { score = 5 }
        if m.contains("ultra") { score += 1.5 }
        else if m.contains("max") { score += 1.0 }
        else if m.contains("pro") { score += 0.5 }
        return score
    }

    private var unregisteredHosts: [DiscoveredHost] {
        let workerHostnames = Set(workers.map { $0.hostname.lowercased() })
        return discoveredHosts.filter { host in
            !workerHostnames.contains(host.hostname.lowercased())
        }
    }

    // MARK: - Health Hero

    private var clusterHealthHero: some View {
        HStack(spacing: 20) {
            ZStack {
                Circle()
                    .fill(health.color.opacity(0.2))
                    .frame(width: 56, height: 56)
                Image(systemName: "xserve")
                    .font(.title2)
                    .foregroundColor(health.color)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(health.label)
                    .font(.title2.bold())
                HStack(spacing: 8) {
                    StatusBadge(
                        text: clusterRole.capitalized,
                        color: NovaTheme.Colors.accent
                    )
                    if clusterStrategy != "minNodes" {
                        Text(clusterStrategy)
                            .font(.caption)
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                    }
                }
                if clusterRole == "worker" && !coordinatorHost.isEmpty {
                    Text("Coordinator: \(coordinatorHost)")
                        .font(.caption)
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                }
            }

            Spacer()

            HStack(spacing: 12) {
                MetricCard(
                    icon: "server.rack",
                    title: l10n.tr("cluster.workers"),
                    value: "\(workers.count)",
                    subtitle: "\(activeCount) " + l10n.tr("cluster.active")
                )
                MetricCard(
                    icon: "memorychip",
                    title: l10n.tr("cluster.totalMemory"),
                    value: bytesFormatted(totalMemory),
                    subtitle: totalNodeCount > 0 ? "\(bytesFormatted(totalMemory / UInt64(totalNodeCount))) / node" : nil
                )
            }
        }
        .padding(20)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    // MARK: - Model Activation

    private var modelActivationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Section header
            HStack {
                Image(systemName: "cube.transparent")
                    .foregroundColor(NovaTheme.Colors.accent)
                Text("Model Activation")
                    .font(.headline)
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                Spacer()
                if activeModel != nil {
                    StatusBadge(text: clusterModelState.uppercased(), color: modelStateColor)
                }
            }

            if let model = activeModel {
                // Active model info
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(model)
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(NovaTheme.Colors.textPrimary)
                            .lineLimit(1)
                        Spacer()
                        Button(action: { deactivateModel() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "stop.circle")
                                Text("Deactivate")
                            }
                            .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(isActivating)
                    }

                    // Per-node readiness
                    if !modelReadiness.isEmpty {
                        VStack(spacing: 6) {
                            ForEach(modelReadiness) { node in
                                HStack(spacing: 8) {
                                    Circle()
                                        .fill(node.statusColor)
                                        .frame(width: 8, height: 8)
                                    Text(node.hostname)
                                        .font(.caption)
                                        .foregroundColor(NovaTheme.Colors.textPrimary)
                                    Text("Layers \(node.startLayer)–\(node.endLayer)")
                                        .font(.caption)
                                        .foregroundColor(NovaTheme.Colors.textSecondary)
                                    Spacer()
                                    Text(node.status.capitalized)
                                        .font(.caption)
                                        .foregroundColor(node.statusColor)
                                    if node.memoryUsedBytes > 0 {
                                        Text(ByteCountFormatter.string(fromByteCount: Int64(node.memoryUsedBytes), countStyle: .memory))
                                            .font(.caption2)
                                            .foregroundColor(NovaTheme.Colors.textSecondary)
                                    }
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(NovaTheme.Colors.rowBackground)
                                .cornerRadius(4)
                            }
                        }
                    }

                    if let error = activationError {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(NovaTheme.Colors.statusError)
                    }

                    // Distributed inference metrics
                    if let tps = distributedTPS, tps > 0 {
                        HStack(spacing: 12) {
                            MetricCard(
                                icon: "gauge.with.dots.needle.bottom.50percent",
                                title: "Distributed TPS",
                                value: String(format: "%.1f", tps),
                                subtitle: distributedLastAgo.map { "\($0) ago" }
                            )
                            if let acc = distributedSpecAccuracy, acc > 0 {
                                MetricCard(
                                    icon: "bolt.horizontal",
                                    title: "Spec Accuracy",
                                    value: String(format: "%.0f%%", acc * 100),
                                    subtitle: nil
                                )
                            }
                        }
                    }
                }
                .padding(12)
                .background(NovaTheme.Colors.cardBackground)
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
                )
            } else {
                // No model active — activate prompt
                HStack {
                    Text("No model activated")
                        .font(.body)
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                    Spacer()
                    Button(action: { activateModel() }) {
                        HStack(spacing: 4) {
                            Image(systemName: "bolt.circle")
                            Text("Activate Model")
                        }
                        .font(.caption)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(isActivating || workers.isEmpty)
                }
                .padding(12)
                .background(NovaTheme.Colors.cardBackground)
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
                )
            }
        }
    }

    private var modelStateColor: Color {
        switch clusterModelState {
        case "ready": return NovaTheme.Colors.statusOK
        case "activating": return NovaTheme.Colors.statusWarn
        case "failed": return NovaTheme.Colors.statusError
        default: return NovaTheme.Colors.textSecondary
        }
    }

    // MARK: - Node Grid

    private var nodeGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            let totalNodes = 1 + workers.count
            HStack {
                sectionHeader(l10n.tr("cluster.nodes"), icon: "server.rack", count: totalNodes)
                Spacer()
                if workers.count > 1 {
                    sortPicker
                }
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                localNodeCard
                ForEach(sortedWorkers) { worker in
                    nodeCard(worker)
                }
            }
        }
        .padding(NovaTheme.Spacing.lg)
        .sectionCard()
    }

    private var sortPicker: some View {
        Menu {
            ForEach(WorkerSortOrder.allCases, id: \.self) { order in
                Button {
                    workerSort = order
                } label: {
                    HStack {
                        Label(order.label, systemImage: order.icon)
                        if workerSort == order {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.arrow.down")
                    .font(.system(size: 10))
                Text(workerSort.label)
                    .font(.system(size: 10))
            }
            .foregroundColor(NovaTheme.Colors.textSecondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(NovaTheme.Colors.rowBackground)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.sm))
        }
    }

    private var localNodeCard: some View {
        let mem = ProcessInfo.processInfo.physicalMemory
        let cpu = localCPUModel
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 4) {
                        Text(Host.current().localizedName ?? "Local")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(NovaTheme.Colors.textPrimary)
                            .lineLimit(1)
                        Text("(\(clusterRole == "none" ? "Coordinator" : clusterRole.capitalized))")
                            .font(.system(size: 10))
                            .foregroundColor(NovaTheme.Colors.accent)
                    }
                    Text(":\(appState.adminPort)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
                Spacer()
                StatusBadge(text: "Active", color: NovaTheme.Colors.statusOK)
            }
            HStack(spacing: 16) {
                if !cpu.isEmpty {
                    Label {
                        Text(cpu)
                            .font(.system(size: 11))
                    } icon: {
                        Image(systemName: "cpu")
                            .font(.system(size: 9))
                    }
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                }

                Label {
                    Text(bytesFormatted(mem))
                        .font(.system(size: 11))
                } icon: {
                    Image(systemName: "memorychip")
                        .font(.system(size: 9))
                }
                .foregroundColor(NovaTheme.Colors.textSecondary)
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.rowBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.md).stroke(NovaTheme.Colors.accent.opacity(0.3), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }

    private func nodeCard(_ w: WorkerSnapshot) -> some View {
        let statusColor = workerStatusColor(w.status)
        let isExpanded = expandedNodeId == w.id

        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(w.hostname)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(NovaTheme.Colors.textPrimary)
                        .lineLimit(1)
                    Text(":\(w.port)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
                Spacer()
                StatusBadge(text: w.status.capitalized, color: statusColor)
            }

            HStack(spacing: 16) {
                if !w.cpuModel.isEmpty {
                    Label {
                        Text(w.cpuModel)
                            .font(.system(size: 11))
                    } icon: {
                        Image(systemName: "cpu")
                            .font(.system(size: 9))
                    }
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                }

                Label {
                    Text(bytesFormatted(w.totalMemoryBytes))
                        .font(.system(size: 11))
                } icon: {
                    Image(systemName: "memorychip")
                        .font(.system(size: 9))
                }
                .foregroundColor(NovaTheme.Colors.textSecondary)

                if let hb = w.lastHeartbeat {
                    Label {
                        Text(relativeTime(hb))
                            .font(.system(size: 11))
                    } icon: {
                        Image(systemName: "heart.fill")
                            .font(.system(size: 9))
                    }
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                }
            }

            if isExpanded {
                Divider()
                VStack(alignment: .leading, spacing: 4) {
                    detailRow("Node ID", value: w.nodeId, monospaced: true)
                    if let reg = w.registeredAt {
                        detailRow("Registered", value: formattedDate(reg))
                    }
                    if let hb = w.lastHeartbeat {
                        detailRow("Last Heartbeat", value: formattedDate(hb))
                    }
                    detailRow("Compute", value: String(format: "%.1f", w.computeCapability))
                }
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.rowBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.md).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.15)) {
                expandedNodeId = expandedNodeId == w.id ? nil : w.id
            }
        }
    }

    private func detailRow(_ label: String, value: String, monospaced: Bool = false) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 10))
                .foregroundColor(NovaTheme.Colors.textTertiary)
            Spacer()
            Text(value)
                .font(.system(size: 10, design: monospaced ? .monospaced : .default))
                .foregroundColor(NovaTheme.Colors.textSecondary)
        }
    }

    // MARK: - Network Discovery

    /// Dynamically query macOS for Thunderbolt interface names via `networksetup`.
    /// Lazily cached — interface names don't change during a session.
    /// Returns names like ["en1", "en2", "en3", "bridge0"] depending on hardware.
    nonisolated(unsafe) private static let thunderboltInterfaces: Set<String> = {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/networksetup")
        p.arguments = ["-listallhardwareports"]
        let pipe = Pipe()
        p.standardOutput = pipe
        do {
            try p.run()
            p.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8) else { return [] }
            var result = Set<String>()
            var isTB = false
            for line in output.components(separatedBy: "\n") {
                let t = line.trimmingCharacters(in: .whitespaces)
                if t.hasPrefix("Hardware Port:") {
                    isTB = t.localizedCaseInsensitiveContains("Thunderbolt")
                } else if t.hasPrefix("Device:") && isTB {
                    let dev = t.dropFirst("Device:".count).trimmingCharacters(in: .whitespaces)
                    result.insert(String(dev))
                }
            }
            return result
        } catch {
            return []
        }
    }()

    /// Check if a network interface is Thunderbolt.
    /// Uses the dynamically discovered set from `networksetup -listallhardwareports`.
    nonisolated private func isThunderboltInterface(_ iface: String) -> Bool {
        Self.thunderboltInterfaces.contains(iface)
    }

    private var networkDiscoverySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(
                l10n.tr("cluster.networkDiscovery"),
                icon: "bolt.horizontal",
                count: discoveredHosts.count
            )

            if discoveredHosts.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "bolt.horizontal")
                        .font(.system(size: 11))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                    Text(l10n.tr("cluster.noHostsFound"))
                        .font(.system(size: 12))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ForEach(discoveredHosts) { host in
                    discoveredHostRow(host)
                }
            }
        }
        .padding(NovaTheme.Spacing.lg)
        .sectionCard()
    }

    private func discoveredHostRow(_ host: DiscoveredHost) -> some View {
        let isWorker = host.matchedWorker != nil
        let phase = deployPhases[host.ipAddress]
        let isDeploying = phase != nil && phase != .idle && phase != .running && phase != .stopped && phase != .failed
        let errorMsg = deployErrors[host.ipAddress]

        return HStack(spacing: 10) {
            Image(systemName: isWorker ? "checkmark.circle.fill" : (phase == .running ? "server.rack" : "desktopcomputer"))
                .font(.system(size: 12))
                .foregroundColor(isWorker ? NovaTheme.Colors.statusOK : (phase == .running ? NovaTheme.Colors.statusOK : NovaTheme.Colors.accent))
                .frame(width: 16)

            VStack(alignment: .leading, spacing: 1) {
                Text(host.hostname == "?" ? host.ipAddress : host.hostname)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                HStack(spacing: 6) {
                    Text(host.ipAddress)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                    if host.interface != "?" {
                        Text(host.interface)
                            .font(.system(size: 9))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                    }
                    if isDeploying {
                        Text(phaseLabel(phase))
                            .font(.system(size: 9, weight: .medium))
                            .foregroundColor(NovaTheme.Colors.accent)
                    }
                    if let err = errorMsg {
                        Text(err)
                            .font(.system(size: 9))
                            .foregroundColor(NovaTheme.Colors.statusError)
                            .lineLimit(1)
                    }
                }
            }

            Spacer()

            if isDeploying {
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(width: 16, height: 16)
            } else if isWorker {
                StatusBadge(text: "Worker", color: NovaTheme.Colors.statusOK)
            } else if phase == .failed {
                Button("Retry") { deployToHost(host) }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
            } else if phase == .stopped {
                Button("Start") { startRemoteWorker(host) }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
            } else {
                Button("Deploy") { deployToHost(host) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(NovaTheme.Colors.rowBackground.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.sm))
    }

    // MARK: - Events

    private var recentEventsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(l10n.tr("cluster.events"), icon: "clock.arrow.circlepath", count: events.count)

            VStack(spacing: 4) {
                ForEach(events.prefix(20)) { event in
                    eventRow(event)
                }
            }
        }
        .padding(NovaTheme.Spacing.lg)
        .sectionCard()
    }

    private func eventRow(_ event: ClusterEvent) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(eventColor(event.type))
                .frame(width: 6, height: 6)
            Text(relativeTime(event.timestamp))
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textTertiary)
                .frame(width: 48, alignment: .leading)
            Text(event.nodeId)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textSecondary)
                .lineLimit(1)
            Text(event.detail)
                .font(.system(size: 11))
                .foregroundColor(NovaTheme.Colors.textPrimary)
            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    // MARK: - Disabled View

    private var clusterDisabledView: some View {
        VStack(spacing: 16) {
            Spacer().frame(height: 60)
            Image(systemName: "xserve")
                .font(.system(size: 48))
                .foregroundColor(NovaTheme.Colors.textTertiary)
            Text(l10n.tr("cluster.disabled"))
                .font(.title3.bold())
                .foregroundColor(NovaTheme.Colors.textPrimary)
            Text(l10n.tr("cluster.enableHint"))
                .font(.subheadline)
                .foregroundColor(NovaTheme.Colors.textSecondary)
                .multilineTextAlignment(.center)

            Button {
                appState.requestedPage = .settings
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "gearshape")
                    Text(l10n.tr("cluster.goToSettings"))
                }
                .font(.system(size: 13, weight: .medium))
            }
            .buttonStyle(.plain)
            .foregroundColor(NovaTheme.Colors.accent)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(NovaTheme.Colors.accentDim)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Polling

    private func startPolling() {
        poll()
        scanNetwork()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            DispatchQueue.main.async { self.poll() }
        }
        scanTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { _ in
            DispatchQueue.main.async { self.scanNetwork() }
        }
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
        scanTimer?.invalidate()
        scanTimer = nil
    }

    private func poll() {
        let port = appState.adminPort
        let key = appState.apiKey ?? "abcd1234"
        let prevWorkers = workers

        fetchClusterStatus(port: port, apiKey: key) { isRunning, role, strategy, coordinatorHost, newWorkers in
            NovaMLXLog.info("[ClusterPage] poll: isRunning=\(isRunning), role=\(role), workers=\(newWorkers.count)")
            self.isRunning = isRunning
            self.clusterRole = role
            self.clusterStrategy = strategy
            self.coordinatorHost = coordinatorHost
            self.diffEvents(from: prevWorkers, to: newWorkers)
            self.workers = newWorkers
            self.lastPollTime = Date()
        }

        fetchDiscoveryDebug(port: port, apiKey: key) { enriched in
            if !enriched.isEmpty {
                self.diffEvents(from: self.workers, to: enriched)
                self.workers = enriched
            }
        }

        fetchModelStatus(port: port, apiKey: key)
    }

    // MARK: - Network Scanning — Thunderbolt + direct cable

    /// Is this a link-local IP (169.254.x.x) indicating a direct cable connection?
    nonisolated private func isLinkLocal(_ ip: String) -> Bool {
        ip.hasPrefix("169.254.")
    }

    /// Is this interface likely WiFi? (en0 on most Macs, plus common WiFi names)
    nonisolated private func isWiFiInterface(_ iface: String) -> Bool {
        iface == "en0" || iface == "awdl0" || iface == "utun0" || iface.hasPrefix("llw")
    }

    private func scanNetwork() {
        let allLocalAddrs = collectAllWiredIPAddresses()

        // Run arp parsing off the main thread to avoid blocking UI.
        // Detect: (1) Thunderbolt interfaces, (2) link-local IPs on wired (non-WiFi) interfaces.
        Task.detached(priority: .utility) { [workers = self.workers] in
            let arpEntries = await self.parseArpTableAsync()
            let myIPs = Set(allLocalAddrs.map(\.ip))
            var hosts: [DiscoveredHost] = []

            for entry in arpEntries {
                guard !myIPs.contains(entry.ip) else { continue }
                guard !entry.ip.hasSuffix(".255") && entry.ip != "0.0.0.0" else { continue }

                // Accept: Thunderbolt interface OR link-local on non-WiFi wired interface
                let isTB = isThunderboltInterface(entry.interface)
                let isDirectCable = isLinkLocal(entry.ip) && !isWiFiInterface(entry.interface)
                guard isTB || isDirectCable else { continue }

                let matchedWorker = workers.first { w in
                    w.hostname.lowercased() == entry.hostname.lowercased()
                }

                hosts.append(DiscoveredHost(
                    id: entry.ip,
                    ipAddress: entry.ip,
                    hostname: entry.hostname,
                    interface: entry.interface,
                    macAddress: entry.mac,
                    matchedWorker: matchedWorker
                ))
            }

            let sorted = hosts.sorted { $0.matchedWorker != nil && $1.matchedWorker == nil }
            NovaMLXLog.info("[ClusterPage] Network discovery: \(sorted.count) hosts found (from \(arpEntries.count) ARP entries)")
            DispatchQueue.main.async {
                self.discoveredHosts = sorted
            }
        }
    }

    /// Async version of parseArpTable that doesn't block the calling thread.
    /// nonisolated so Task.detached can run it without hopping back to MainActor.
    nonisolated private func parseArpTableAsync() async -> [ArpEntry] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/arp")
        process.arguments = ["-a"]
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8) else {
                NovaMLXLog.warning("[ClusterPage] ARP scan: no UTF8 output")
                return []
            }
            let entries = parseArpOutput(output)
            NovaMLXLog.info("[ClusterPage] ARP scan: \(entries.count) entries parsed")
            return entries
        } catch {
            NovaMLXLog.warning("[ClusterPage] ARP scan Process() failed: \(error)")
            return []
        }
    }

    // Pure value types — no actor isolation needed
    private struct TbAddr: Sendable {
        let ip: String
        let prefix: String
    }

    private struct ArpEntry: Sendable {
        let hostname: String
        let ip: String
        let mac: String
        let interface: String
    }

    /// Collect IP addresses from Thunderbolt interfaces.
    nonisolated private func collectThunderboltIPAddresses() -> [TbAddr] {
        var addrs: [TbAddr] = []
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return addrs }

        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let name = CString( ptr.pointee.ifa_name)
            guard isThunderboltInterface(name) else { continue }
            guard let sa = ptr.pointee.ifa_addr else { continue }
            guard sa.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST)
            let ip = CString( host)
            guard !ip.isEmpty && ip != "0.0.0.0" && ip != "127.0.0.1" else { continue }
            let parts = ip.split(separator: ".")
            guard parts.count >= 3 else { continue }
            let prefix = "\(parts[0]).\(parts[1]).\(parts[2])"
            addrs.append(TbAddr(ip: ip, prefix: prefix))
        }
        freeifaddrs(ifaddr)
        return addrs
    }

    /// Collect IP addresses from ALL wired (non-WiFi) interfaces.
    /// Used to find the coordinator's IP on the same subnet as the target worker.
    nonisolated private func collectAllWiredIPAddresses() -> [TbAddr] {
        var addrs: [TbAddr] = []
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return addrs }

        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let name = CString( ptr.pointee.ifa_name)
            guard !isWiFiInterface(name) && name != "lo0" else { continue }
            guard let sa = ptr.pointee.ifa_addr else { continue }
            guard sa.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST)
            let ip = CString( host)
            guard !ip.isEmpty && ip != "0.0.0.0" && ip != "127.0.0.1" else { continue }
            // For link-local (169.254.x.x), match on /16 prefix; otherwise /24
            let prefix: String
            if ip.hasPrefix("169.254.") {
                prefix = "169.254"
            } else {
                let parts = ip.split(separator: ".")
                guard parts.count >= 3 else { continue }
                prefix = "\(parts[0]).\(parts[1]).\(parts[2])"
            }
            addrs.append(TbAddr(ip: ip, prefix: prefix))
        }
        freeifaddrs(ifaddr)
        return addrs
    }

    // Pure function — no UI state access, safe to call from any isolation context
    nonisolated private func isOnThunderboltSubnet(_ ip: String, tbAddrs: [TbAddr]) -> Bool {
        for addr in tbAddrs {
            if ip.hasPrefix(addr.prefix) { return true }
        }
        return false
    }

    private func parseArpTable() -> [ArpEntry] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/arp")
        process.arguments = ["-a"]
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8) else { return [] }
            return parseArpOutput(output)
        } catch {
            return []
        }
    }

    // Parse: ? (192.168.1.5) at ab:cd:ef:12:34:56 on en0 ifscope [ethernet]
    nonisolated private func parseArpOutput(_ output: String) -> [ArpEntry] {
        var entries: [ArpEntry] = []
        for line in output.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.contains("(") && trimmed.contains(")") && trimmed.contains(" at ") else { continue }

            // Extract hostname (before parenthesis)
            guard let parenOpen = trimmed.firstIndex(of: "("),
                  let parenClose = trimmed.firstIndex(of: ")") else { continue }
            let hostname = String(trimmed[..<parenOpen]).trimmingCharacters(in: .whitespaces)

            // Extract IP (between parentheses)
            let ip = String(trimmed[trimmed.index(after: parenOpen)..<parenClose])

            // Extract MAC (after " at ")
            guard let atRange = trimmed.range(of: " at ") else { continue }
            let afterAt = trimmed[atRange.upperBound...]
            let macPart = afterAt.split(separator: " ").first ?? ""
            let mac = String(macPart)
            guard mac.contains(":") && mac != "(incomplete)" else { continue }

            // Extract interface (after " on ")
            var iface = "?"
            if let onRange = trimmed.range(of: " on ") {
                let afterOn = trimmed[onRange.upperBound...]
                let ifacePart = afterOn.split(separator: " ").first ?? "?"
                iface = String(ifacePart)
            }

            entries.append(ArpEntry(hostname: hostname, ip: ip, mac: mac, interface: iface))
        }
        return entries
    }

    // MARK: - Model Activation Actions

    private func activateModel() {
        guard let port = appState.apiKey != nil ? appState.adminPort : nil,
              let key = appState.apiKey else { return }

        // Pick the first loaded model that's not a draft/ASR/embedding model
        let excludedSuffixes = ["0.6B", "ASR", "bge-", "whisper"]
        let candidate = appState.loadedModels.first { id in
            !excludedSuffixes.contains(where: { id.contains($0) })
        }
        guard let modelId = candidate ?? appState.loadedModels.first else { return }

        isActivating = true
        activationError = nil
        let url = URL(string: "http://127.0.0.1:\(port)/admin/api/cluster/activate-model")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["modelId": modelId])

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isActivating = false
                if let error = error {
                    self.activationError = error.localizedDescription
                    return
                }
                self.fetchModelStatus(port: port, apiKey: key)
            }
        }.resume()
    }

    private func deactivateModel() {
        guard let port = appState.apiKey != nil ? appState.adminPort : nil,
              let key = appState.apiKey else { return }

        isActivating = true
        let url = URL(string: "http://127.0.0.1:\(port)/admin/api/cluster/deactivate-model")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isActivating = false
                self.fetchModelStatus(port: port, apiKey: key)
            }
        }.resume()
    }

    private func fetchModelStatus(port: Int, apiKey: String) {
        let url = URL(string: "http://127.0.0.1:\(port)/admin/api/cluster/model-status")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        URLSession.shared.dataTask(with: request) { data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            DispatchQueue.main.async {
                self.activeModel = json["activeModel"] as? String
                self.clusterModelState = json["state"] as? String ?? "idle"

                if let nodes = json["nodes"] as? [[String: Any]] {
                    self.modelReadiness = nodes.compactMap { node in
                        guard let nodeId = node["nodeId"] as? String else { return nil }
                        return ModelNodeReadiness(
                            id: nodeId,
                            nodeId: nodeId,
                            hostname: node["hostname"] as? String ?? nodeId,
                            layerCount: node["layerCount"] as? Int ?? 0,
                            startLayer: node["startLayer"] as? Int ?? 0,
                            endLayer: node["endLayer"] as? Int ?? 0,
                            status: node["status"] as? String ?? "pending",
                            memoryUsedBytes: node["memoryUsedBytes"] as? UInt64 ?? 0,
                            errorMessage: node["errorMessage"] as? String
                        )
                    }
                }

                if let stats = json["inferenceStats"] as? [String: Any] {
                    self.distributedTPS = stats["tokensPerSecond"] as? Double
                    self.distributedSpecAccuracy = stats["speculationAccuracy"] as? Double
                    self.distributedLastAgo = stats["timestampAgo"] as? String
                }
            }
        }.resume()
    }

    // MARK: - API Fetching

    private func fetchClusterStatus(port: Int, apiKey: String, handler: @escaping @MainActor (Bool, String, String, String, [WorkerSnapshot]) -> Void) {
        guard let url = URL(string: "http://127.0.0.1:\(port)/admin/api/cluster/status") else { return }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 3
        URLSession.shared.dataTask(with: request) { data, _, _ in
            guard let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                NovaMLXLog.warning("[ClusterPage] fetchClusterStatus: no valid JSON response")
                return
            }
            let isRunning = json["isRunning"] as? Bool ?? false
            let configDict = json["config"] as? [String: Any]
            let role = configDict?["role"] as? String ?? json["role"] as? String ?? "none"
            let strategy = configDict?["strategy"] as? String ?? "minNodes"
            let host = configDict?["coordinatorHost"] as? String ?? ""
            let workers = (json["workers"] as? [[String: Any]])?
                .compactMap { WorkerSnapshot(json: $0) }
                .sorted { $0.hostname < $1.hostname } ?? []
            Task { @MainActor in handler(isRunning, role, strategy, host, workers) }
        }.resume()
    }

    private func fetchDiscoveryDebug(port: Int, apiKey: String, handler: @escaping @MainActor ([WorkerSnapshot]) -> Void) {
        guard let url = URL(string: "http://127.0.0.1:\(port)/admin/api/cluster/discovery-debug") else { return }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 3
        URLSession.shared.dataTask(with: request) { data, _, _ in
            guard let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let rawWorkers = json["workers"] as? [[String: Any]] else { return }
            let workers = rawWorkers.compactMap { WorkerSnapshot(json: $0) }
                .sorted { $0.hostname < $1.hostname }
            Task { @MainActor in handler(workers) }
        }.resume()
    }

    // MARK: - Worker Deployment

    private var credentialDialogSheet: some View {
        VStack(spacing: 16) {
            Text("Deploy NovaMLX to \(credentialHost)")
                .font(.headline)
            Text("Enter SSH credentials for the remote Mac.\nMake sure \"Remote Login\" is enabled in System Settings > General > Sharing.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            TextField("Username", text: $credentialUsername)
                .textFieldStyle(.roundedBorder)
            SecureField("Password", text: $credentialPassword)
                .textFieldStyle(.roundedBorder)

            HStack(spacing: 12) {
                Button("Cancel") {
                    showCredentialDialog = false
                    credentialPassword = ""
                }
                .keyboardShortcut(.cancelAction)

                Button("Deploy") {
                    showCredentialDialog = false
                    performDeploy()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(credentialUsername.isEmpty || credentialPassword.isEmpty)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 360)
    }

    private func phaseLabel(_ phase: DeployPhase?) -> String {
        switch phase {
        case .generatingKey: return "Generating SSH key..."
        case .installingKey: return "Installing key..."
        case .transferring: return "Transferring app..."
        case .configuring: return "Configuring..."
        case .launching: return "Launching..."
        case .running: return "Running"
        case .stopped: return "Stopped"
        case .failed: return "Failed"
        default: return ""
        }
    }

    private func deployToHost(_ host: DiscoveredHost) {
        // Always show credential dialog — let user confirm or re-enter
        credentialHost = host.ipAddress
        // Pre-fill username from Keychain if available
        if let creds = try? KeychainHelper.loadSSHCredential(host: host.ipAddress) {
            credentialUsername = creds.username
        } else {
            credentialUsername = NSUserName()
        }
        credentialPassword = ""
        showCredentialDialog = true
    }

    private func performDeploy() {
        let host = credentialHost
        let username = credentialUsername
        let password = credentialPassword
        credentialPassword = "" // clear from memory ASAP

        // Save credentials to Keychain
        try? KeychainHelper.saveSSHCredential(host: host, username: username, password: password)

        deployPhases[host] = .generatingKey
        deployErrors.removeValue(forKey: host)

        // Find coordinator's IP on the same subnet as the target worker
        let allAddrs = collectAllWiredIPAddresses()
        let coordinatorIP: String
        if let match = allAddrs.first(where: { $0.prefix != "169.254" && host.hasPrefix($0.prefix) })
            ?? allAddrs.first(where: { $0.prefix == "169.254" && host.hasPrefix("169.254.") }) {
            coordinatorIP = match.ip
        } else {
            coordinatorIP = allAddrs.first?.ip ?? "127.0.0.1"
        }

        Task {
            do {
                let deployer = WorkerDeployer.shared

                // Install public key (first time)
                try await deployer.installPublicKey(host: host, username: username, password: password)

                // Run full deploy
                for try await phase in deployer.deploy(
                    host: host,
                    username: username,
                    coordinatorHost: coordinatorIP,
                    coordinatorPort: appState.adminPort,
                    appBundlePath: nil
                ) {
                    await MainActor.run { deployPhases[host] = phase }
                }
                deployErrors.removeValue(forKey: host)

                // Ensure cluster mode is enabled in config (idempotent, no restart)
                await MainActor.run {
                    ensureClusterConfig()
                }
            } catch {
                // Clear stored credentials on auth failure so retry prompts for new ones
                KeychainHelper.deleteSSHCredential(host: host)
                await MainActor.run {
                    deployPhases[host] = .failed
                    deployErrors[host] = error.localizedDescription
                }
            }
        }
    }

    private func ensureClusterConfig() {
        let configPath = NovaMLXPaths.configFile

        guard let data = try? Data(contentsOf: configPath),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var server = json["server"] as? [String: Any] else { return }

        // Already has cluster config — nothing to do
        if server["cluster"] != nil { return }

        server["cluster"] = [
            "role": "coordinator",
            "coordinatorPort": appState.adminPort
        ]
        json["server"] = server

        do {
            let newData = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            try newData.write(to: configPath, options: .atomic)
            NovaMLXLog.info("[ClusterPage] Enabled cluster mode in config after deploy")
        } catch {
            NovaMLXLog.error("[ClusterPage] Failed to update cluster config: \(error)")
        }
    }

    private func startRemoteWorker(_ host: DiscoveredHost) {
        let ip = host.ipAddress
        guard let creds = try? KeychainHelper.loadSSHCredential(host: ip) else {
            deployToHost(host)
            return
        }
        Task {
            do {
                try await WorkerDeployer.shared.startWorker(host: ip, username: creds.username)
                deployPhases[ip] = .running
            } catch {
                deployPhases[ip] = .failed
                deployErrors[ip] = error.localizedDescription
            }
        }
    }

    private func stopRemoteWorker(_ host: DiscoveredHost) {
        let ip = host.ipAddress
        guard let creds = try? KeychainHelper.loadSSHCredential(host: ip) else { return }
        Task {
            do {
                try await WorkerDeployer.shared.stopWorker(host: ip, username: creds.username)
                deployPhases[ip] = .stopped
            } catch {
                deployErrors[ip] = error.localizedDescription
            }
        }
    }

    // MARK: - Event Diffing

    private func diffEvents(from prev: [WorkerSnapshot], to curr: [WorkerSnapshot]) {
        let prevMap = Dictionary(uniqueKeysWithValues: prev.map { ($0.nodeId, $0) })
        let currMap = Dictionary(uniqueKeysWithValues: curr.map { ($0.nodeId, $0) })
        let prevIds = Set(prevMap.keys)
        let currIds = Set(currMap.keys)

        for id in currIds.subtracting(prevIds) {
            appendEvent(.workerJoined, nodeId: id, detail: l10n.tr("cluster.workerJoined"))
        }
        for id in prevIds.subtracting(currIds) {
            appendEvent(.workerLeft, nodeId: id, detail: l10n.tr("cluster.workerLeft"))
        }
        for id in currIds.intersection(prevIds) {
            if let old = prevMap[id], let new = currMap[id], old.status != new.status {
                appendEvent(.statusChange, nodeId: id, detail: "\(old.status) → \(new.status)")
            }
        }
    }

    private func appendEvent(_ type: ClusterEventType, nodeId: String, detail: String) {
        events.insert(ClusterEvent(type: type, nodeId: nodeId, detail: detail), at: 0)
        if events.count > maxEvents { events.removeLast() }
    }

    // MARK: - Helpers

    private func workerStatusColor(_ status: String) -> Color {
        switch status {
        case "active", "ready": return NovaTheme.Colors.statusOK
        case "loading", "syncing", "registering": return NovaTheme.Colors.statusWarn
        case "disconnected", "failed": return NovaTheme.Colors.statusError
        default: return NovaTheme.Colors.textTertiary
        }
    }

    private func eventColor(_ type: ClusterEventType) -> Color {
        switch type {
        case .workerJoined: return NovaTheme.Colors.statusOK
        case .workerLeft: return NovaTheme.Colors.statusError
        case .statusChange: return NovaTheme.Colors.statusWarn
        }
    }

    private func bytesFormatted(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        if gb >= 1024 {
            return String(format: "%.1f TB", gb / 1024.0)
        }
        return String(format: "%.0f GB", gb)
    }

    private func relativeTime(_ date: Date) -> String {
        let interval = Date().timeIntervalSince(date)
        if interval < 5 { return "now" }
        if interval < 60 { return "\(Int(interval))s" }
        if interval < 3600 { return "\(Int(interval / 60))m" }
        return "\(Int(interval / 3600))h"
    }

    private func formattedDate(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .medium
        return f.string(from: date)
    }
}
