import AppKit
import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXModelManager
import NovaMLXTknetPeer

// MARK: - TknetPeerPageView (Task 11)
//
// In-page strings are literal English, matching LoadBalancersPageView's
// convention; only the sidebar page title is localized (app.tknetPeer).

struct TknetPeerPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager
    @ObservedObject var peer: TknetPeerState
    let memory: TknetPageMemory
    @State private var selectedServer = TknetServers.official
    @State private var peerNameText = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                if peer.status?.connection == .upgradeRequired {
                    upgradeBanner
                }
                statusRow
                serverSection
                if peer.config.peerId == nil {
                    unregisteredNote
                } else {
                    credentialsSection
                    demandSection
                    if !peer.config.capabilities.isEmpty {
                        capabilitiesSection
                    }
                }
            }
            .padding(24)
        }
        .navigationTitle("Tknet Peer")
        .task { if peer.config.peerId != nil { await peer.fetchDemand() } }
        .onAppear {
            alignServer()
            if memory.keep {
                selectedServer = memory.selectedServer
                if memory.nameEdited { peerNameText = memory.peerNameText }
                revealKey = memory.revealKey
                loginEmail = memory.loginEmail
                loginPassword = memory.loginPassword
            } else if loginEmail.isEmpty {
                loginEmail = peer.savedLoginEmail
            }
            Task {
                await peer.refreshAccount()
                await peer.fillMissingAPI(
                    modelType: { modelRecord($0)?.modelType.rawValue },
                    family: { modelRecord($0)?.family.rawValue })
            }
        }
        .onDisappear {
            memory.keep = true
            memory.selectedServer = selectedServer
            memory.peerNameText = peerNameText
            memory.revealKey = revealKey
            memory.loginEmail = loginEmail
            memory.loginPassword = loginPassword
        }
        .onChange(of: peerNameText) { _, value in
            memory.peerNameText = value
            memory.nameEdited = value != peer.peerName(for: selectedServer)
        }
        .onChange(of: peer.activeREST) { _, _ in
            alignServer()
            loginEmail = peer.savedLoginEmail
            loginPassword = ""
            Task { await peer.refreshAccount() }
        }
    }

    // MARK: Server
    //
    // One product server: tknet.ai. A checkout build can also use localhost.
    // Identity stays the peer id plus the connection key for that server.

    private var allowsLocalServer: Bool {
        TknetServers.allowsLocalServer(bundlePath: Bundle.main.bundlePath)
    }

    private var knownServers: [String] {
        TknetServers.choices(allowsLocal: allowsLocalServer)
    }

    private var savedName: String { peer.peerName(for: selectedServer) }

    private var selectedRegistered: Bool { peer.isRegistered(selectedServer) }

    private var serverSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Server").font(.system(size: 13, weight: .semibold))
            if allowsLocalServer {
                Picker("Server", selection: $selectedServer) {
                    Text("tknet.ai").tag(TknetServers.official)
                    Text("localhost").tag(TknetServers.localDev)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text("localhost is only for development. A released app uses tknet.ai.")
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("tknet.ai")
                    .font(.system(size: 13, weight: .medium))
            }

            if selectedRegistered, let peerId = peer.config.registrations[tunnelKey(selectedServer)] {
                Text("Registered").font(.system(size: 11, weight: .semibold)).foregroundColor(.green)
                Text(peerId)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
            } else {
                Text("Not registered on this server").font(.system(size: 11)).foregroundColor(.secondary)
            }

            Text("Peer name").font(.caption)
            TextField("Peer name", text: $peerNameText, prompt: Text(defaultPeerName))
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))
            Text(nameCaption)
                .font(.system(size: 10.5))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                if !selectedRegistered {
                    Button("Register") { registerSelected() }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                        .disabled(peerNameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                } else if peerNameText.trimmingCharacters(in: .whitespacesAndNewlines) != savedName,
                          !peerNameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button("Save name") {
                        let server = selectedServer
                        let name = peerNameText
                        Task { await peer.rememberName(server, name: name) }
                    }
                    .buttonStyle(.bordered).controlSize(.small)
                }
            }
            if !selectedRegistered {
                registerErrorText
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .cornerRadius(10)
        .onChange(of: selectedServer) { _, newValue in
            peerNameText = peer.peerName(for: newValue)
            guard newValue != peer.activeREST, let url = URL(string: newValue) else { return }
            Task { await peer.selectServer(url) }
        }
    }

    @ViewBuilder
    private var registerErrorText: some View {
        if let error = peer.lastError {
            Text(error)
                .font(.system(size: 11))
                .foregroundColor(.red)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var nameCaption: String {
        if selectedRegistered && savedName.isEmpty {
            return "This server is registered. No peer name is saved on this Mac yet. Save name stores the label here. It does not call the server."
        }
        if selectedRegistered {
            return "Saved peer name for this server. The name is a label. The peer ID above is the identity."
        }
        return "Register asks this server for a new peer ID. This Mac then remembers that ID and this name."
    }

    private var defaultPeerName: String {
        "tknet-peer-\(ProcessInfo.processInfo.hostName.prefix(20))"
    }

    private func tunnelKey(_ rest: String) -> String {
        guard let url = URL(string: rest) else { return "" }
        return PeerConfig.tunnelKey(PeerConfig.tunnelURL(fromREST: url))
    }

    /// Show tknet.ai or, in a checkout, the active known server.
    /// A released app that was left on localhost moves back to tknet.ai.
    private func alignServer() {
        let rest = peer.activeREST
        let shown = knownServers.contains(rest) ? rest : TknetServers.official
        peerNameText = peer.peerName(for: shown)
        if selectedServer != shown {
            selectedServer = shown
        } else if shown != rest, let url = URL(string: shown) {
            Task { await peer.selectServer(url) }
        }
    }

    private func registerSelected() {
        guard let url = URL(string: selectedServer) else { return }
        let name = peerNameText.trimmingCharacters(in: .whitespacesAndNewlines)
        Task { await peer.register(server: url, peerName: name) }
    }

    private var unregisteredNote: some View {
        Text("Start stays off until this Mac has a peer ID.")
            .font(.system(size: 11))
            .foregroundColor(.secondary)
    }

    // MARK: Claim credentials (绑机器要用)

    @State private var revealKey = false
    @State private var loginEmail = ""
    @State private var loginPassword = ""

    private var credentialsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("认领这台机器（Claim this machine）")
                .font(.system(size: 13, weight: .semibold))
            Text("登录你的 tknet 账号，然后点认领。密码不会留在这台 Mac 上。登录会话保存 24 小时。")
                .font(.system(size: 11)).foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if peer.accountEmail != nil {
                Text(signedInLine)
                    .font(.system(size: 11))
                if peer.claimedPeerId == peer.config.peerId {
                    Text("已认领（Claimed）")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.green)
                }
                HStack(spacing: 8) {
                    Button("认领（Claim）") { Task { await peer.claimThisMac() } }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                    Button("退出（Sign out）") { Task { await peer.signOut() } }
                        .buttonStyle(.bordered).controlSize(.small)
                }
            } else {
                TextField("Email", text: $loginEmail)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))
                SecureField("Password", text: $loginPassword)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))
                Button("登录（Sign in）") {
                    let email = loginEmail
                    let password = loginPassword
                    Task {
                        if await peer.signIn(email: email, password: password) {
                            loginPassword = ""
                        }
                    }
                }
                .buttonStyle(.borderedProminent).controlSize(.small)
                .disabled(loginEmail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || loginPassword.isEmpty)
            }
            if let error = peer.accountError {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundColor(.red)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("也可以把这两个值贴到网页。")
                .font(.system(size: 10.5)).foregroundColor(.secondary)

            credentialRow(label: "Peer ID", value: peer.config.peerId ?? "")

            HStack(spacing: 8) {
                Text("连接密钥").font(.system(size: 11)).foregroundColor(.secondary).frame(width: 64, alignment: .leading)
                if revealKey, let key = peer.machineKey {
                    Text(key)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .lineLimit(1).truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text("••••••••••••••••")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                Button(revealKey ? "Hide" : "Show") { revealKey.toggle() }
                    .buttonStyle(.bordered).controlSize(.mini)
                if let key = peer.machineKey {
                    Button("Copy") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(key, forType: .string) }
                        .buttonStyle(.bordered).controlSize(.mini)
                }
            }
            .padding(8)
            .background(NovaTheme.Colors.rowBackground)
            .cornerRadius(6)
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .cornerRadius(10)
    }

    private var signedInLine: String {
        let email = peer.accountEmail ?? ""
        let name = peer.accountName ?? ""
        if name.isEmpty { return "已登录 \(email)" }
        return "已登录 \(name)（\(email)）"
    }

    private func credentialRow(label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.system(size: 11)).foregroundColor(.secondary).frame(width: 64, alignment: .leading)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .textSelection(.enabled)
                .lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Copy") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(value, forType: .string) }
                .buttonStyle(.bordered).controlSize(.mini)
        }
        .padding(8)
        .background(NovaTheme.Colors.rowBackground)
        .cornerRadius(6)
    }

    // MARK: Header + status

    private var connectionText: String {
        switch peer.status?.connection {
        case .connecting: return "connecting"
        case .connected: return "connected"
        case .backingOff(let seconds): return "reconnecting in \(Int(seconds))s"
        case .stopping: return "stopping"
        case .upgradeRequired: return "upgrade required"
        case .idle, nil: return "idle"
        }
    }

    private var connectionColor: Color {
        switch peer.status?.connection {
        case .connected: return .green
        case .connecting, .backingOff: return .orange
        case .upgradeRequired: return .red
        default: return .secondary
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Tknet Peer").font(.title2.bold())
                Text("Serve tknet.ai demand from this Mac")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Button(peer.running ? "Stop" : "Start") {
                Task { peer.running ? await peer.stop() : await peer.start() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(peer.config.peerId == nil)
            .help(peer.config.peerId == nil ? "Register this peer first" : "")
        }
    }

    /// Persistent: the tunnel is terminally refused until the app is
    /// updated. The URL is baked in — the server's error frame is
    /// best-effort and can be lost before close.
    private var upgradeBanner: some View {
        HStack(spacing: 10) {
            Label("Upgrade required", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundColor(.red)
            Text("This version of the peer protocol is no longer accepted by tknet.ai. Download the latest NovaMLX to keep serving.")
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            if let url = URL(string: NovaMLXTknetPeer.TknetPeer.downloadURL) {
                Link("Download", destination: url)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(12)
        .background(Color.red.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.red.opacity(0.3)))
    }

    private var statusRow: some View {
        HStack(spacing: 12) {
            Label(connectionText, systemImage: "antenna.radiowaves.left.and.right")
                .font(.caption)
                .foregroundColor(connectionColor)
            Text("\(peer.status?.activeRequests ?? 0)/\(peer.status?.totalRequests ?? 0) requests")
                .font(.caption.monospaced()).foregroundColor(.secondary)
            Text("\(peer.status?.totalCompletionTokens ?? 0) tokens")
                .font(.caption.monospaced()).foregroundColor(.secondary)
            Spacer()
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: Earnings

    private var earningsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Earnings").font(.headline)
                Spacer()
                Button("Refresh") { Task { await peer.fetchEarnings() } }
            }
            if let e = peer.earnings {
                let a = e.availability
                HStack(spacing: 16) {
                    VStack(alignment: .leading) {
                        Text("Total earned").font(.caption2).foregroundColor(.secondary)
                        Text("$\(a.totalEarned)").font(.title3.bold().monospacedDigit())
                    }
                    VStack(alignment: .leading) {
                        Text("Payout available").font(.caption2).foregroundColor(.secondary)
                        Text("$\(a.available)").font(.title3.bold().monospacedDigit()).foregroundColor(.green)
                    }
                    VStack(alignment: .leading) {
                        Text("On hold (7d)").font(.caption2).foregroundColor(.secondary)
                        Text("$\(a.onHold)").font(.caption.monospacedDigit()).foregroundColor(.secondary)
                    }
                    Spacer()
                }
                ForEach(e.byModel, id: \.model) { m in
                    HStack {
                        Text(m.model).font(.caption.monospaced())
                        Spacer()
                        Text("\(m.requests) req · \(m.tokens) tok").font(.caption).foregroundColor(.secondary)
                        Text("$\(m.earned)").font(.caption.monospacedDigit())
                    }
                }
            } else {
                Text("No earnings data yet — refresh after serving traffic.")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: Demand list

    private var downloadedIds: [String] {
        modelManager.downloadedModels().map(\.id)
    }

    private var matchedDemand: [DemandEntry] {
        let ids = downloadedIds
        let loaded = appState.loadedModels
        return peer.demand.filter {
            DemandModelMatch.localId(for: $0.model, downloadedIds: ids, loadedIds: loaded) != nil
        }
    }

    private func localMatch(for entry: DemandEntry) -> (localId: String?, loaded: Bool) {
        guard let id = DemandModelMatch.localId(
            for: entry.model, downloadedIds: downloadedIds, loadedIds: appState.loadedModels)
        else { return (nil, false) }
        return (id, DemandModelMatch.isLoaded(id, loadedIds: appState.loadedModels))
    }

    private var demandSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Demand").font(.headline)
                Spacer()
                Button("Refresh") { Task { await peer.fetchDemand() } }
            }
            Text("Matched (\(matchedDemand.count))").font(.subheadline.weight(.semibold))
            if peer.demand.isEmpty {
                Text("No open demand right now.")
                    .font(.caption).foregroundColor(.secondary)
            } else if matchedDemand.isEmpty {
                Text("No open demand matches a model on this Mac.")
                    .font(.caption).foregroundColor(.secondary)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(matchedDemand, id: \.demandId) { entry in
                        demandRow(entry)
                    }
                }
            }

            Text("All demands (\(peer.demand.count))").font(.subheadline.weight(.semibold))
                .padding(.top, 6)
            if peer.demand.isEmpty {
                Text("No open demand right now.")
                    .font(.caption).foregroundColor(.secondary)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(peer.demand, id: \.demandId) { entry in
                        demandRow(entry)
                    }
                }
            }
        }
    }

    private func modelRecord(_ id: String) -> ModelRecord? {
        modelManager.downloadedModels().first {
            $0.id.caseInsensitiveCompare(id) == .orderedSame
        }
    }

    private func demandRow(_ entry: DemandEntry) -> some View {
        let claimed = peer.config.capabilities.first { $0.demandId == entry.demandId }
        let active = claimed.map { peer.config.activeCapabilities.contains($0) } ?? false
        let match = localMatch(for: entry)
        return HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(entry.model).font(.headline)
                    Text(entry.modality)
                        .font(.caption2.bold())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .foregroundColor(.accentColor)
                        .clipShape(Capsule())
                }
                if let note = entry.note {
                    Text(note).font(.caption).foregroundColor(.secondary)
                }
                if let localId = match.localId, localId.caseInsensitiveCompare(entry.model) != .orderedSame {
                    Text(localId).font(.caption2.monospaced()).foregroundColor(.secondary)
                }
            }
            Spacer()
            if active {
                VStack(alignment: .trailing, spacing: 2) {
                    Label("Serving", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundColor(.green)
                    if let api = claimed?.api, let op = api.operations.first {
                        let extra = api.operations.count > 1 ? " +\(api.operations.count - 1)" : ""
                        Text("\(op.method) \(op.path)\(extra)")
                            .font(.caption2.monospaced())
                            .foregroundColor(.secondary)
                    }
                }
            } else if claimed != nil {
                Label("Retired — waiting for demand to return", systemImage: "clock")
                    .font(.caption).foregroundColor(.secondary)
            } else if let localId = match.localId {
                VStack(alignment: .trailing, spacing: 4) {
                    Button("Serve via NovaMLX (local)") {
                        guard DemandModelMatch.isLoaded(localId, loadedIds: appState.loadedModels) else { return }
                        let record = modelRecord(localId)
                        let api = SupplyAPI.contract(
                            modelId: localId,
                            modelType: record?.modelType.rawValue ?? "unknown",
                            family: record?.family.rawValue ?? "")
                        Task {
                            await peer.addLocalSource(
                                for: entry, upstreamModel: localId,
                                apiKeyStore: NovaDB.shared.apiKeyStore, api: api)
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(!match.loaded)
                    if !match.loaded {
                        Text("Downloaded. Load it first.")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                }
            } else {
                Text("Not downloaded")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: Declared capabilities

    private var capabilitiesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Declared Capabilities (\(peer.config.capabilities.count))").font(.headline)
            LazyVStack(spacing: 8) {
                ForEach(peer.config.capabilities, id: \.demandId) { cap in
                    CapabilityRow(capability: cap, peer: peer)
                }
            }
        }
    }
}

// MARK: - CapabilityRow (editable upstream model)

private struct CapabilityRow: View {
    let capability: Capability
    @ObservedObject var peer: TknetPeerState
    @State private var upstream: String
    /// Loaded chat-capable models, recomputed on every render so load/unload
    /// in the Models page is reflected the next time this row redraws.
    @State private var loadedChatModels: [String] = []

    init(capability: Capability, peer: TknetPeerState) {
        self.capability = capability
        self.peer = peer
        _upstream = State(initialValue: peer.upstreamModel(for: capability))
    }

    private var retired: Bool {
        !peer.config.activeCapabilities.contains(capability)
    }

    /// Picker options: loaded llm/vlm models, plus the current value (kept
    /// even when its model is not loaded — CLI-set values and temporarily
    /// unloaded models must display unchanged, never be silently swapped).
    private var options: [String] {
        if loadedChatModels.contains(upstream) { return loadedChatModels }
        return loadedChatModels + [upstream]
    }

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(capability.model).font(.headline)
                    Text(capability.demandId)
                        .font(.caption.monospaced())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.gray.opacity(0.15))
                        .foregroundColor(.secondary)
                        .clipShape(Capsule())
                }
                Text("in \(capability.priceIn) / out \(capability.priceOut) per 1k tokens")
                    .font(.caption).foregroundColor(.secondary)
                if let api = capability.api {
                    Text(api.summary).font(.caption2).foregroundColor(.secondary)
                    ForEach(api.operations, id: \.path) { op in
                        Text("\(op.method) \(op.path)")
                            .font(.caption2.monospaced())
                            .foregroundColor(.secondary)
                    }
                }
            }
            Spacer()
            Picker("Upstream model", selection: $upstream) {
                ForEach(options, id: \.self) { model in
                    Text(loadedChatModels.contains(model)
                         ? model
                         : "\(model) (not loaded)")
                        .tag(model)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 240)
            .help("Model name sent to the local NovaMLX server. Load models in the Models page to add entries.")
            if retired {
                Text("retired").font(.caption2).foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .opacity(retired ? 0.5 : 1.0)
        .onAppear {
            loadedChatModels = Self.fetchLoadedChatModels()
            upstream = peer.upstreamModel(for: capability)
        }
        .onChange(of: upstream) { _, newValue in
            Task { await peer.updateUpstreamModel(for: capability.demandId, to: newValue) }
        }
    }

    /// Loaded models that can serve /v1/chat/completions (llm + vlm),
    /// from the same loaded_models store the LoadBalancers page reads.
    private static func fetchLoadedChatModels() -> [String] {
        let loaded = (try? NovaDB.shared.loadedModelsStore.list()) ?? []
        let chatTypes: Set<String> = ["llm", "vlm"]
        return loaded.filter { id in
            guard let record = try? NovaDB.shared.modelRegistryStore.get(modelId: id),
                  let type = record.modelType else { return true }
            return chatTypes.contains(type)
        }.sorted()
    }
}
