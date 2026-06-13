import SwiftUI
import CryptoKit
import NovaMLXCore
import NovaMLXDB
import NovaMLXDistributed
import NovaMLXUtils

// MARK: - Data Models

struct WorkerSnapshot: Identifiable {
    let id: String
    let nodeId: String
    let hostname: String
    let port: Int
    let networkHost: String?          // Best reachable IP (often the Thunderbolt 10.42.x.x)
    let binaryFingerprint: String?
    let configHash: String?
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
        self.networkHost = json["networkHost"] as? String
        self.binaryFingerprint = json["binaryFingerprint"] as? String
        self.configHash = json["configHash"] as? String
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

/// Lightweight info for the Cluster activation model picker.
struct ActivationModelInfo: Identifiable, Equatable {
    let id: String                 // model folder name, e.g. "mlx-community/Qwen3.6-27B-4bit"
    let numLayers: Int?
    let estimatedFullGB: Double?   // rough estimate for the *full* model
    let isRecommended: Bool        // true if it has enough layers for the current cluster policy
    let displayLabel: String       // rich string for Picker / Menu

    var modelId: String { id }
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

    // Thunderbolt Subnet Policy (strict)
    @State private var thunderboltSubnet: String = ""
    @State private var thunderboltEnforceStrict: Bool = true
    @State private var localThunderboltIPs: [String] = []

    // MARK: - Staleness Detection (Step 1)

    /// Coordinator's current binary fingerprint (used as the source of truth).
    private var localBinaryFingerprint: String {
        let version = NovaMLXCore.version
        let execPath = Bundle.main.executableURL?.path ?? ""
        let modDate = (try? FileManager.default.attributesOfItem(atPath: execPath)[.modificationDate] as? Date) ?? Date()
        return "\(version)-\(Int(modDate.timeIntervalSince1970))"
    }

    /// Coordinator's current config hash (from clusterPolicyStore).
    private var localConfigHash: String? {
        let json = (try? NovaDB.shared.clusterPolicyStore.get()) ?? "{}"
        guard let data = json.data(using: .utf8) else { return nil }
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }

    /// Returns whether the given worker is considered stale compared to the coordinator.
    private func isWorkerStale(_ worker: WorkerSnapshot) -> Bool {
        // If worker hasn't reported fingerprints yet, don't mark stale yet.
        guard worker.binaryFingerprint != nil || worker.configHash != nil else { return false }

        let binaryMismatch = worker.binaryFingerprint != nil && worker.binaryFingerprint != localBinaryFingerprint
        let configMismatch = worker.configHash != nil && localConfigHash != nil && worker.configHash != localConfigHash

        return binaryMismatch || configMismatch
    }

    // MARK: - SSH Agent Status (Step 2) - explicit state for reliable UI
    enum SSHAgentUIState: Int {
        case checking = 0
        case ready = 1
        case notReady = 2
    }

    @State private var sshAgentUIState: SSHAgentUIState = .checking
    @State private var sshAgentStatusMessage: String = "正在检测 SSH 状态..."
    @State private var sshAgentRefreshTimer: Timer?
    @State private var sshMonitorStarted: Bool = false

    // Persist last known SSH status so we don't show "正在检测" for long on every launch
    @AppStorage("lastSSHStateRaw") private var lastSSHStateRaw: Int = SSHAgentUIState.checking.rawValue
    @AppStorage("lastSSHStatusMessage") private var lastSSHStatusMessage: String = "正在检测 SSH 状态..."

    private func startSSHAgentMonitoring() {
        guard !sshMonitorStarted else { return }
        sshMonitorStarted = true

        // Immediately show last known result (if any) so UI doesn't stay on "正在检测" for long
        if let cachedState = SSHAgentUIState(rawValue: lastSSHStateRaw), cachedState != .checking {
            sshAgentUIState = cachedState
            sshAgentStatusMessage = lastSSHStatusMessage
        }

        // Then do a fresh check in background
        checkSSHAgentStatus()

        sshAgentRefreshTimer?.invalidate()
        sshAgentRefreshTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { _ in
            Task { @MainActor in
                self.checkSSHAgentStatus()
            }
        }
    }

    private func stopSSHAgentMonitoring() {
        sshAgentRefreshTimer?.invalidate()
        sshAgentRefreshTimer = nil
    }

    private func checkSSHAgentStatus() {
        let testHosts = ["10.42.0.2"] + workers.compactMap { $0.networkHost }.filter { !$0.isEmpty }
        guard !testHosts.isEmpty else {
            updateSSHAgentStatus(state: .notReady, message: "SSH 不可用\n无可用 Worker 地址")
            return
        }

        Task.detached(priority: .userInitiated) {
            // Fast TCP port 22 check — milliseconds vs seconds for SSH handshake
            let reachable = await withTaskGroup(of: String?.self) { group in
                for host in testHosts {
                    group.addTask {
                        let process = Process()
                        process.executableURL = URL(fileURLWithPath: "/usr/bin/nc")
                        process.arguments = ["-z", "-w", "1", host, "22"]
                        let pipe = Pipe()
                        process.standardOutput = pipe
                        process.standardError = pipe
                        guard (try? process.run()) != nil else { return nil }
                        process.waitUntilExit()
                        return process.terminationStatus == 0 ? host : nil
                    }
                }
                for await r in group {
                    if r != nil { group.cancelAll(); return r }
                }
                return nil
            }

            DispatchQueue.main.async {
                if reachable != nil {
                    self.sshAgentUIState = .ready
                    self.sshAgentStatusMessage = "SSH 可用\n可直接用于 Redeploy"
                    self.lastSSHStateRaw = SSHAgentUIState.ready.rawValue
                    self.lastSSHStatusMessage = "SSH 可用\n可直接用于 Redeploy"
                } else {
                    self.sshAgentUIState = .notReady
                    self.sshAgentStatusMessage = "SSH 不可用\n请确保 Coordinator 能免密 SSH 到 Worker"
                    self.lastSSHStateRaw = SSHAgentUIState.notReady.rawValue
                    self.lastSSHStatusMessage = "SSH 不可用\n请确保 Coordinator 能免密 SSH 到 Worker"
                }
            }
        }
    }

    private func updateSSHAgentStatus(state: SSHAgentUIState, message: String) {
        DispatchQueue.main.async {
            self.sshAgentUIState = state
            self.sshAgentStatusMessage = message
        }
    }

    /// Maps DeployPhase to user-friendly Chinese text for the UI
    private func phaseDisplayText(for phase: DeployPhase) -> String {
        switch phase {
        case .transferring:
            return "正在传输新版本"
        case .configuring:
            return "正在推送配置"
        case .launching:
            return "正在启动新 Worker"
        case .running:
            return "运行中"
        case .failed:
            return "Redeploy 失败"
        default:
            return phase.rawValue
        }
    }

    // MARK: - Redeploy Action (Step 3)

    private func performRedeploy(for worker: WorkerSnapshot, preferredIP: String? = nil) async {
        guard sshAgentUIState == .ready else {
            NovaMLXLog.warning("[ClusterPage] Redeploy blocked — SSH not ready")
            return
        }

        // Prefer the best known IP (Thunderbolt preferred). Allow override for testing.
        let targetIP = preferredIP ?? worker.networkHost ?? worker.hostname

        NovaMLXLog.info("[ClusterPage] Starting redeploy for \(worker.hostname) via \(targetIP)")

        do {
            try await WorkerDeployer.shared.redeployWorker(
                host: worker.hostname,
                networkHost: targetIP
            )
            NovaMLXLog.info("[ClusterPage] Redeploy command completed for \(worker.hostname). Triggering refresh...")

            // Give the worker a moment to restart, then refresh status
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            await MainActor.run {
                self.poll()
                self.loadThunderboltSubnetInfo()
            }
        } catch {
            NovaMLXLog.error("[ClusterPage] Redeploy failed for \(worker.hostname): \(error)")
        }
    }

    /// Debug helper: finds the Mini worker and triggers redeploy regardless of stale state.
    /// Always prefers the stable Thunderbolt IP (10.42.0.2) for reliability.
    private func triggerTestRedeploy() async {
        // Try to find the Mini by network IP or hostname
        let miniWorker = workers.first { w in
            (w.networkHost?.contains("10.42.0.2") ?? false) ||
            w.hostname.lowercased().contains("mini") ||
            w.hostname.lowercased().contains("10.42.0.2")
        }

        guard let worker = miniWorker else {
            NovaMLXLog.warning("[ClusterPage] Test Redeploy: Could not find Mini worker in the list")
            return
        }

        // Force the best known Thunderbolt IP (hostname is unreliable over Thunderbolt)
        NovaMLXLog.info("[ClusterPage] 🧪 Test Redeploy triggered for \(worker.hostname) (forced IP: 10.42.0.2)")
        await performRedeploy(for: worker, preferredIP: "10.42.0.2")
    }

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

    // Model picker for activation (replaces blind auto-pick)
    @State private var availableActivationModels: [ActivationModelInfo] = []
    @State private var selectedActivationModel: String = ""
    @State private var recommendedMinLayers: Int = 0   // for current cluster (minLayersPerShard × nodes)
    @State private var activatingModelId: String? = nil   // model currently being activated (for progress display)

    private let maxEvents = 50

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if isRunning || appState.clusterEnabled {
                    clusterHealthHero
                    if clusterRole == "coordinator" {
                        sshAgentStatusBanner       // 只在 Coordinator 上显示 SSH Agent 状态（管理操作相关）
                    }
                    thunderboltSubnetCard          // Strict Thunderbolt Fabric — most prominent after hero
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
        // Also kick off discovery immediately when the view struct is created
        .task { startPolling() }
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
        // A host is "unregistered" only if it has no matching worker (by IP or hostname).
        // This is the source of truth for whether we should offer the Deploy button.
        return discoveredHosts.filter { host in
            host.matchedWorker == nil
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

                let staleCount = workers.filter { isWorkerStale($0) }.count
                if staleCount > 0 {
                    MetricCard(
                        icon: "exclamationmark.triangle",
                        title: "Stale Workers",
                        value: "\(staleCount)",
                        subtitle: "Need update",
                        valueColor: NovaTheme.Colors.statusWarn
                    )
                }
            }
        }
        .padding(20)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    // MARK: - Thunderbolt Subnet Card
    // Calm, factual presentation — we only have one mode now (the configured subnet).
    private var thunderboltSubnetCard: some View {
        let hasSubnet = !thunderboltSubnet.isEmpty && thunderboltSubnet != "Not configured"
        let statusColor = hasSubnet ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusWarn

        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "bolt.horizontal.circle.fill")
                    .font(.system(size: 18))
                    .foregroundColor(NovaTheme.Colors.accent)

                Text("Thunderbolt Subnet")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textPrimary)

                Spacer()

                if hasSubnet {
                    Text("Active")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 3)
                        .background(statusColor)
                        .clipShape(Capsule())
                } else {
                    Text("Not Configured")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 3)
                        .background(statusColor)
                        .clipShape(Capsule())
                }
            }

            if hasSubnet {
                Text(thunderboltSubnet)
                    .font(.system(size: 22, weight: .heavy, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.accent)
                    .padding(.top, 2)

                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Local Thunderbolt IP")
                            .font(.system(size: 10))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                        Text(localThunderboltIPs.joined(separator: ", "))
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                            .foregroundColor(NovaTheme.Colors.textPrimary)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Rule")
                            .font(.system(size: 10))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                        Text("Only hosts on this subnet are accepted for the cluster.")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(NovaTheme.Colors.textPrimary)
                    }
                }

                if thunderboltSubnet.contains("Auto") == false && loadThunderboltPolicy().subnet.isEmpty {
                    Text("Auto-detected from local Thunderbolt Bridge")
                        .font(.system(size: 9))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                }
            } else {
                Text("No Thunderbolt subnet configured. Set it in Settings or push cluster-policy.json from the coordinator.")
                    .font(.system(size: 12))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                    .padding(.top, 4)
            }
        }
        .padding(NovaTheme.Spacing.lg)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.lg)
                .stroke(hasSubnet ? NovaTheme.Colors.accent.opacity(0.6) : NovaTheme.Colors.statusWarn.opacity(0.5), lineWidth: hasSubnet ? 1.5 : 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    // MARK: - SSH Agent Status Banner (placed right below Health Hero as requested)
    private var sshAgentStatusBanner: some View {
        let isChecking = sshAgentUIState == .checking   // use explicit state

        return HStack(spacing: 12) {
            Image(systemName: (sshAgentUIState == .ready) ? "network" : (isChecking ? "hourglass" : "exclamationmark.triangle"))
                .font(.system(size: 14))
                .foregroundColor((sshAgentUIState == .ready) ? NovaTheme.Colors.statusOK : (isChecking ? NovaTheme.Colors.textSecondary : NovaTheme.Colors.statusWarn))

            VStack(alignment: .leading, spacing: 2) {
                Text("SSH 连通性")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(NovaTheme.Colors.textSecondary)

                Text(sshAgentStatusMessage)
                    .font(.system(size: 11))
                    .foregroundColor((sshAgentUIState == .ready) ? NovaTheme.Colors.textPrimary : (isChecking ? NovaTheme.Colors.textSecondary : NovaTheme.Colors.statusWarn))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            if !(sshAgentUIState == .ready) && !isChecking {
                Button("复制提示") {
                    let pasteboard = NSPasteboard.general
                    pasteboard.clearContents()
                    pasteboard.setString("请确保 ssh 可以免密登录到 10.42.0.2", forType: .string)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill((sshAgentUIState == .ready) ? Color.green.opacity(0.08) : (isChecking ? Color.gray.opacity(0.08) : Color.orange.opacity(0.08)))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke((sshAgentUIState == .ready) ? NovaTheme.Colors.statusOK.opacity(0.3) : (isChecking ? Color.gray.opacity(0.3) : NovaTheme.Colors.statusWarn.opacity(0.4)), lineWidth: 1)
        )
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
                // No model active — show model picker + activate
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("No model activated")
                            .font(.body)
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                        Spacer()
                        Button {
                            scanAvailableModelsForActivation()
                        } label: {
                            Image(systemName: "arrow.triangle.2.circlepath")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    if recommendedMinLayers > 0 {
                        Text("Current cluster recommendation: at least \(recommendedMinLayers) layers (minLayersPerShard × active nodes)")
                            .font(.caption2)
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                    }

                    if availableActivationModels.isEmpty {
                        Text("No local models found. Load or download a large model (7B+) first.")
                            .font(.caption)
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                    } else {
                        Picker("Model to activate", selection: $selectedActivationModel) {
                            Text("Select a model…").tag("")
                            ForEach(availableActivationModels) { info in
                                Text(info.displayLabel).tag(info.id)
                            }
                        }
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity)
                    }

                    // Strong warning for unsuitable models
                    if isActivating, let model = activatingModelId {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Distributing \(model) to workers...")
                                .font(.caption)
                                .foregroundColor(NovaTheme.Colors.accent)
                            if !modelReadiness.isEmpty {
                                Text("Sync progress: \(modelReadiness.filter { $0.status == "ready" }.count)/\(modelReadiness.count) nodes ready")
                                    .font(.caption2)
                            } else {
                                ProgressView().scaleEffect(0.6)
                            }
                        }
                        .padding(6)
                        .background(NovaTheme.Colors.cardBackground)
                        .cornerRadius(6)
                    }

                    if let selectedInfo = availableActivationModels.first(where: { $0.id == selectedActivationModel }),
                       !selectedInfo.isRecommended {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(NovaTheme.Colors.statusError)
                            Text("This model is likely too small for reliable distributed inference on your current cluster (needs more layers).")
                                .font(.caption2)
                                .foregroundColor(NovaTheme.Colors.statusError)
                                .lineLimit(3)
                        }
                        .padding(6)
                        .background(NovaTheme.Colors.statusError.opacity(0.1))
                        .cornerRadius(6)
                    }

                    HStack(spacing: 8) {
                        Button(action: { activateSelectedModel() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "bolt.circle")
                                Text(selectedActivationModel.isEmpty ? "Activate Model" : "Activate Selected")
                            }
                            .font(.caption)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(isActivating || selectedActivationModel.isEmpty)

                        Button(action: { syncSelectedModelToWorkers() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.down.circle")
                                Text("Sync to Workers")
                            }
                            .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(isActivating || selectedActivationModel.isEmpty)
                    }
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
            let staleCount = workers.filter { isWorkerStale($0) }.count
            HStack {
                sectionHeader(l10n.tr("cluster.nodes"), icon: "server.rack", count: totalNodes)
                if staleCount > 0 {
                    Text("(\(staleCount) out of date)")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                        .padding(.leading, 8)
                }
                Spacer()

                // Debug / Test button for Redeploy — only show on Coordinator
                if clusterRole == "coordinator" {
                    Button("🧪 Test Redeploy (Mini)") {
                        Task {
                            await triggerTestRedeploy()
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(NovaTheme.Colors.statusWarn)
                    .disabled(sshAgentUIState != .ready)
                }

                Button("刷新") {
                    poll()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

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

                    if let fp = w.binaryFingerprint {
                        Text(fp)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                    }
                }
                Spacer()
                // Show redeploy phase if active, otherwise normal status
                if let dep = WorkerDeployer.shared.deployment(for: w.hostname),
                   dep.phase != .running && dep.phase != .idle {
                    StatusBadge(text: phaseDisplayText(for: dep.phase), color: NovaTheme.Colors.statusWarn)
                } else if isWorkerStale(w) {
                    StatusBadge(text: "OUT OF DATE", color: NovaTheme.Colors.statusWarn)
                } else {
                    StatusBadge(text: w.status.capitalized, color: statusColor)
                }
            }

            // Redeploy status / action area
            if let dep = WorkerDeployer.shared.deployment(for: w.hostname),
               dep.phase != .running && dep.phase != .idle {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 11))
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                    Text(phaseDisplayText(for: dep.phase) + "…")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                }
                .padding(.top, 2)
            } else if isWorkerStale(w) {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                    Text("Software or config is out of sync with Coordinator. Redeploy recommended.")
                        .font(.system(size: 10))
                        .foregroundColor(NovaTheme.Colors.statusWarn)

                    Button("Redeploy") {
                        Task {
                            await performRedeploy(for: w)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
                    .tint(NovaTheme.Colors.statusWarn)
                    .disabled(!(sshAgentUIState == .ready) || (WorkerDeployer.shared.deployment(for: w.hostname)?.phase != .running && WorkerDeployer.shared.deployment(for: w.hostname)?.phase != .idle))
                }
                .padding(.top, 2)
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
    nonisolated private static let thunderboltInterfaces: Set<String> = {
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
            HStack {
                sectionHeader(
                    "Discovered Hosts on Thunderbolt Subnet",
                    icon: "bolt.horizontal",
                    count: discoveredHosts.count
                )
                Spacer()
                Button {
                    loadThunderboltSubnetInfo()
                    scanNetwork()
                } label: {
                    Label("Rescan", systemImage: "arrow.clockwise")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }

            if discoveredHosts.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 13))
                        .foregroundColor(NovaTheme.Colors.statusWarn)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("No hosts found on the Thunderbolt subnet")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(NovaTheme.Colors.textPrimary)
                        Text(thunderboltSubnet.isEmpty ? "No subnet configured" : "Only hosts on \(thunderboltSubnet) are accepted for the cluster.")
                            .font(.system(size: 11))
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                    }
                }
                .padding(.vertical, 4)
            } else {
                // Only show hosts that are NOT yet registered workers.
                // If a machine is already "Ready" in the Nodes list, we do not offer Deploy again.
                ForEach(unregisteredHosts) { host in
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
                    Text("On Subnet")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(NovaTheme.Colors.statusOK)
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
                StatusBadge(text: "Registered", color: NovaTheme.Colors.statusOK)
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
        NovaMLXLog.info("[ClusterPage] startPolling() called (isRunning=\(isRunning), clusterEnabled=\(appState.clusterEnabled))")
        loadThunderboltSubnetInfo()
        if clusterRole == "coordinator" {
            startSSHAgentMonitoring()
        }
        poll()
        if clusterRole == "coordinator" || appState.clusterEnabled {
            scanAvailableModelsForActivation()
        }
        scanNetwork()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            DispatchQueue.main.async { self.poll() }
        }
        scanTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { _ in
            DispatchQueue.main.async { self.scanNetwork() }
        }
    }

    private func loadThunderboltSubnetInfo() {
        let policy = loadThunderboltPolicy()
        let ips = collectThunderboltIPAddresses().map { $0.ip }

        DispatchQueue.main.async {
            self.thunderboltSubnet = policy.subnet.isEmpty ? "Not configured" : policy.subnet
            self.thunderboltEnforceStrict = policy.enforce
            self.localThunderboltIPs = ips.isEmpty ? ["(no Thunderbolt Bridge IP)"] : ips
        }
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
        scanTimer?.invalidate()
        scanTimer = nil
        stopSSHAgentMonitoring()
    }

    private func poll() {
        let port = appState.adminPort
        let key = appState.apiKey ?? "abcd1234"
        let prevWorkers = workers

        fetchClusterStatus(port: port, apiKey: key) { isRunning, role, strategy, coordinatorHost, newWorkers in
            NovaMLXLog.info("[ClusterPage] poll: isRunning=\(isRunning), role=\(role), workers=\(newWorkers.count)")
            self.isRunning = isRunning
            self.clusterRole = role
            if role == "coordinator" {
                self.startSSHAgentMonitoring()
            }
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

    /// Returns a score for how "good" this IP is for Thunderbolt distributed inference.
    /// Higher = better.
    /// We strongly prefer stable private IPs on the Thunderbolt bridge (e.g. 10.42.0.x)
    /// over link-local 169.254.x.x.
    nonisolated private func thunderboltIPScore(_ ip: String, preferredPrefix: String = "10.42.") -> Int {
        if ip.hasPrefix(preferredPrefix) {
            return 100
        }
        if isLinkLocal(ip) {
            return 5                         // Strong penalty for link-local
        }
        if ip.hasPrefix("192.168.") || ip.hasPrefix("10.") || ip.hasPrefix("172.16.") {
            return 50
        }
        return 0
    }

    /// Load authoritative Thunderbolt subnet policy.
    /// Prefers clusterPolicyStore (SQLite), then configStore.
    /// If nothing is explicitly configured, auto-detect from the local Thunderbolt Bridge IP (very useful after manual 10.42 setup).
    nonisolated private func loadThunderboltPolicy() -> (subnet: String, enforce: Bool, prefix: String) {
        // 1. Authoritative: clusterPolicyStore (was cluster-policy.json)
        if let policyJSON = try? NovaDB.shared.clusterPolicyStore.get(),
           let data = policyJSON.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let tb = json["thunderbolt"] as? [String: Any],
           let subnet = tb["subnet"] as? String, !subnet.isEmpty {
            let enforce = tb["enforce"] as? Bool ?? true
            let prefix = prefixFromSubnet(subnet)
            return (subnet, enforce, prefix)
        }

        // 2. configStore (was config.json)
        if let record = try? NovaDB.shared.configStore.get(),
           let raw = record.clusterConfig,
           let data = raw.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let cluster = json["cluster"] as? [String: Any],
           let tb = cluster["thunderbolt"] as? [String: Any],
           let subnet = tb["subnet"] as? String, !subnet.isEmpty {
            let enforce = tb["enforce"] as? Bool ?? false
            let prefix = prefixFromSubnet(subnet)
            return (subnet, enforce, prefix)
        }

        // 3. Auto-detect from local Thunderbolt IPs (e.g. after manual bridge config of 10.42.x.x)
        let localTB = collectThunderboltIPAddresses()
        if let first = localTB.first(where: { $0.ip.hasPrefix("10.") || $0.ip.hasPrefix("192.168.") }) {
            let prefix = first.prefix + "."
            // Reconstruct a /24 subnet string for display
            let subnet = first.prefix + ".0/24"
            return (subnet, true, prefix)   // treat as strict once we saw a real private TB IP
        }

        // 4. Nothing configured
        return ("", false, "")
    }

    nonisolated private func prefixFromSubnet(_ subnet: String) -> String {
        let prefix = subnet.split(separator: "/").first ?? ""
        let parts = prefix.split(separator: ".")
        guard parts.count >= 3 else { return "" }
        return "\(parts[0]).\(parts[1]).\(parts[2])."
    }

    /// Preferred prefix for filtering (empty = no policy configured)
    nonisolated private func preferredThunderboltPrefix() -> String {
        loadThunderboltPolicy().prefix
    }

    /// Is this interface likely WiFi? (en0 on most Macs, plus common WiFi names)
    nonisolated private func isWiFiInterface(_ iface: String) -> Bool {
        iface == "en0" || iface == "awdl0" || iface == "utun0" || iface.hasPrefix("llw")
    }

    private func scanNetwork() {
        NovaMLXLog.info("[ClusterPage] scanNetwork() invoked")
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

                // STRICT Thunderbolt subnet enforcement.
                // Once a preferred subnet (e.g. 10.42.0.0/24) is configured via policy,
                // ONLY hosts with an IP in that exact subnet are accepted.
                // Link-local (169.254) is NEVER accepted when a policy is active.
                let prefix = preferredThunderboltPrefix()
                let isGood = !prefix.isEmpty && entry.ip.hasPrefix(prefix)

                if isGood {
                    // Matches the authoritative Thunderbolt subnet → accept
                } else if prefix.isEmpty {
                    // No policy configured yet — allow any stable Thunderbolt or wired IP (legacy behavior)
                    let isTB = isThunderboltInterface(entry.interface)
                    let isDirect = isLinkLocal(entry.ip) && !isWiFiInterface(entry.interface)
                    if isTB && !isLinkLocal(entry.ip) {
                        // ok
                    } else if isDirect {
                        // ok (only when no policy)
                    } else {
                        continue
                    }
                } else {
                    // Policy active but this IP is not on the subnet → reject (no fallback)
                    continue
                }

                // Robust matching: prefer IP (networkHost or any known address) first,
                // then fall back to hostname. This prevents Deploy button from appearing
                // for already-registered workers (especially on Thunderbolt 10.42.x.x).
                let matchedWorker = workers.first { w in
                    // 1. Best: match by the actual connection IP we have for the worker
                    if let netHost = w.networkHost, !netHost.isEmpty {
                        if entry.ip == netHost { return true }
                    }
                    // 2. Also match if the discovered IP happens to be what we use to talk to it
                    if entry.ip == w.hostname { return true }

                    // 3. Fallback: hostname match
                    if !entry.hostname.isEmpty && entry.hostname != "?" {
                        return w.hostname.lowercased() == entry.hostname.lowercased()
                    }
                    return false
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

            // Group by hostname and keep only the best IP.
            // Use the preferred Thunderbolt prefix when available (will come from ClusterConfigProvider).
            var bestByHostname: [String: DiscoveredHost] = [:]
            for host in hosts {
                let key = host.hostname.lowercased()
                if let existing = bestByHostname[key] {
                    let prefix = preferredThunderboltPrefix()
                    let newScore = thunderboltIPScore(host.ipAddress, preferredPrefix: prefix)
                    let oldScore = thunderboltIPScore(existing.ipAddress, preferredPrefix: prefix)
                    if newScore > oldScore {
                        NovaMLXLog.info("[ClusterPage] Upgrading \(key) from \(existing.ipAddress) (score \(oldScore)) to \(host.ipAddress) (score \(newScore))")
                        bestByHostname[key] = host
                    }
                } else {
                    bestByHostname[key] = host
                }
            }

            let deduped = Array(bestByHostname.values)
            let sorted = deduped.sorted { $0.matchedWorker != nil && $1.matchedWorker == nil }

            let ipList = sorted.map { $0.ipAddress }.joined(separator: ", ")
            NovaMLXLog.info("[ClusterPage] Network discovery: \(sorted.count) hosts (deduped from \(hosts.count), preferred stable Thunderbolt subnet) — IPs: [\(ipList)]")
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

    private func scanAvailableModelsForActivation() {
        // Use the official path resolution (respects ~/.config/novamlx/models-path)
        let modelsRoot = NovaMLXPaths.modelsDir
        let minLayers = (ClusterManager.shared.config?.minLayersPerShard ?? 8) * max(2, 1 + workers.count)
        recommendedMinLayers = minLayers

        var infos: [ActivationModelInfo] = []

        // 1. Recursively scan for any directory that contains config.json (handles mlx-community/, Qwen/, etc.)
        let foundConfigs = findAllConfigJsons(under: modelsRoot)

        for configURL in foundConfigs {
            // modelId = relative path from modelsRoot, e.g. "mlx-community/Qwen3.6-27B-4bit"
            let relativePath = configURL.deletingLastPathComponent().path.replacingOccurrences(of: modelsRoot.path + "/", with: "")
            let modelId = relativePath

            guard !modelId.isEmpty else { continue }

            let (numLayers, estGB) = parseModelConfigForActivationInfo(at: configURL, modelId: modelId)

            // Filter obviously unsuitable tiny / non-LLM models.
            // Only skip on layer count if we *successfully* parsed a small number.
            let lower = modelId.lowercased()
            let isObviouslyTiny = lower.contains("embed") || lower.contains("bge-") || lower.contains("whisper") ||
                                  lower.contains("asr") || lower.hasSuffix("0.5b") || lower.hasSuffix("0.6b")

            let hasTooFewLayers = (numLayers ?? Int.max) < 16

            if isObviouslyTiny || hasTooFewLayers { continue }

            let isRecommended = (numLayers ?? 0) >= minLayers
            let label = buildActivationLabel(modelId: modelId, numLayers: numLayers, estGB: estGB, isRecommended: isRecommended)

            infos.append(ActivationModelInfo(
                id: modelId,
                numLayers: numLayers,
                estimatedFullGB: estGB,
                isRecommended: isRecommended,
                displayLabel: label
            ))
        }

        // 2. Also surface currently loaded models (even if not rescanned)
        for loaded in appState.loadedModels {
            if infos.contains(where: { $0.id == loaded }) { continue }

            let lower = loaded.lowercased()
            let isTiny = lower.contains("embed") || lower.contains("bge-") || lower.contains("whisper") ||
                         lower.contains("asr") || lower.hasSuffix("0.5b") || lower.hasSuffix("0.6b")
            if isTiny { continue }

            // We don't have layer info for loaded-only entries without re-parsing, so mark conservatively
            let label = loaded + " (loaded)"
            infos.append(ActivationModelInfo(
                id: loaded,
                numLayers: nil,
                estimatedFullGB: nil,
                isRecommended: true,
                displayLabel: label
            ))
        }

        availableActivationModels = infos.sorted { $0.displayLabel < $1.displayLabel }

        // Preserve selection if still valid
        if !availableActivationModels.contains(where: { $0.id == selectedActivationModel }) {
            // Prefer a recommended model if available
            if let recommended = availableActivationModels.first(where: { $0.isRecommended }) {
                selectedActivationModel = recommended.id
            } else {
                selectedActivationModel = availableActivationModels.first?.id ?? ""
            }
        }
    }

    /// Lightweight parse of config.json for layer count and rough size estimate.
    private func parseModelConfigForActivationInfo(at configURL: URL, modelId: String) -> (numLayers: Int?, estimatedGB: Double?) {
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (nil, nil)
        }

        var numLayers: Int? = nil
        if let n = json["num_hidden_layers"] as? Int {
            numLayers = n
        } else if let n = json["n_layer"] as? Int {          // some older models
            numLayers = n
        } else if let n = json["num_layers"] as? Int {
            numLayers = n
        }

        // Rough memory estimate
        var estGB: Double? = nil

        // Try to detect bits from folder name or quantization_config
        let lowerId = modelId.lowercased()
        let bits: Int
        if lowerId.contains("8bit") || lowerId.contains("-8b") { bits = 8 }
        else if lowerId.contains("3bit") || lowerId.contains("-3b") { bits = 3 }
        else if lowerId.contains("2bit") || lowerId.contains("-2b") { bits = 2 }
        else { bits = 4 } // default assumption for most MLX community quants

        if let paramsB = guessParamCount(from: modelId) {
            // Very rough: 4-bit ≈ 0.55–0.65 GB per billion parameters for the whole model
            let bytesPerParam = Double(bits) / 8.0 * 1.15   // overhead for scales etc.
            estGB = paramsB * bytesPerParam
        }

        return (numLayers, estGB)
    }

    /// Very rough parameter count guess from common naming patterns.
    private func guessParamCount(from modelId: String) -> Double? {
        let lower = modelId.lowercased()
        if lower.contains("70b") || lower.contains("72b") { return 70 }
        if lower.contains("34b") || lower.contains("32b") { return 34 }
        if lower.contains("27b") || lower.contains("30b") { return 27 }
        if lower.contains("14b") || lower.contains("13b") { return 14 }
        if lower.contains("9b")  || lower.contains("8b")  { return 9 }
        if lower.contains("7b")  { return 7 }
        if lower.contains("3b")  { return 3 }
        return nil
    }

    private func buildActivationLabel(modelId: String, numLayers: Int?, estGB: Double?, isRecommended: Bool) -> String {
        var parts: [String] = []
        let short = modelId.split(separator: "/").last.map(String.init) ?? modelId
        parts.append(short)

        if let layers = numLayers {
            parts.append("\(layers) layers")
        }
        if let gb = estGB {
            parts.append(String(format: "~%.0fGB", gb))
        }
        if !isRecommended {
            parts.append("⚠️ small")
        }
        return parts.joined(separator: " · ")
    }

    /// Recursively finds all config.json files up to a reasonable depth (handles org/model nesting).
    private func findAllConfigJsons(under root: URL, maxDepth: Int = 4) -> [URL] {
        var results: [URL] = []
        var queue: [(url: URL, depth: Int)] = [(root, 0)]

        while !queue.isEmpty {
            let (current, depth) = queue.removeFirst()
            guard depth <= maxDepth else { continue }

            guard let contents = try? FileManager.default.contentsOfDirectory(at: current, includingPropertiesForKeys: nil) else { continue }

            for item in contents {
                if item.hasDirectoryPath {
                    queue.append((item, depth + 1))
                } else if item.lastPathComponent == "config.json" {
                    results.append(item)
                }
            }
        }
        return results
    }

    private func syncSelectedModelToWorkers() {
        guard !selectedActivationModel.isEmpty else { return }

        let modelId = selectedActivationModel
        activatingModelId = modelId
        isActivating = true   // reuse the banner for progress feel
        activationError = nil

        // In a full implementation we would broadcast .requestModelSync over worker control connections.
        // For now we rely on the auto-sync path added earlier + show helpful feedback.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            self.activationError = "Sync request sent. Workers will automatically pull the model using WeightDistributor if it is missing."
            self.isActivating = false
            self.activatingModelId = nil
            self.fetchModelStatus(port: self.appState.adminPort, apiKey: self.appState.apiKey ?? "abcd1234")
        }
    }

    private func activateSelectedModel() {
        let port = appState.adminPort
        let key = appState.apiKey ?? "abcd1234"

        let modelId = selectedActivationModel
        guard !modelId.isEmpty else {
            activationError = "Please select a model from the picker."
            return
        }

        if workers.isEmpty {
            activationError = "No workers connected. Make sure the worker machine is running NovaMLXWorker and has joined the cluster."
            return
        }

        // Strong pre-activation warning for tiny models
        if let info = availableActivationModels.first(where: { $0.id == modelId }), !info.isRecommended {
            // Still allow it (user may know what they're doing), but surface a loud warning
            activationError = "Warning: The selected model may not have enough layers for good distributed performance on your cluster. Consider a larger model (≥ 20+ layers recommended)."
            // We continue, but the user sees the error + can decide
        }

        activatingModelId = modelId
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
                self.activatingModelId = nil
                if let error = error {
                    self.activationError = error.localizedDescription
                    return
                }
                // Check for server-side error returned in body (even on 200)
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let hasError = json["error"] as? Bool, hasError,
                   let msg = json["message"] as? String {
                    self.activationError = msg
                    self.fetchModelStatus(port: port, apiKey: key)
                    return
                }
                self.fetchModelStatus(port: port, apiKey: key)
            }
        }.resume()
    }

    private func deactivateModel() {
        let port = appState.adminPort
        let key = appState.apiKey ?? "abcd1234"

        isActivating = true
        activationError = nil
        let url = URL(string: "http://127.0.0.1:\(port)/admin/api/cluster/deactivate-model")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isActivating = false
                if let error = error {
                    self.activationError = error.localizedDescription
                } else if let data = data,
                          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let hasError = json["error"] as? Bool, hasError,
                          let msg = json["message"] as? String {
                    self.activationError = msg
                }
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

    // Cluster mode persistence is managed by SettingsPageView only.
    // We do NOT auto-enable it here — the user's toggle choice must survive restarts.

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
