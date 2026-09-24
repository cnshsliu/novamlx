import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXTknetPeer

// MARK: - TknetPeerPageView (Task 11)
//
// In-page strings are literal English, matching LoadBalancersPageView's
// convention; only the sidebar page title is localized (app.tknetPeer).

struct TknetPeerPageView: View {
    @StateObject private var peer = TknetPeerState()
    @State private var serverText = "https://tknet.ai"
    @State private var peerNameText = "tknet-peer-\(ProcessInfo.processInfo.hostName.prefix(20))"

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                statusRow
                if peer.config.peerId == nil {
                    registrationSection
                } else {
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
    }

    // MARK: Header + status

    private var connectionText: String {
        switch peer.status?.connection {
        case .connecting: return "connecting"
        case .connected: return "connected"
        case .backingOff(let seconds): return "reconnecting in \(Int(seconds))s"
        case .stopping: return "stopping"
        case .idle, nil: return "idle"
        }
    }

    private var connectionColor: Color {
        switch peer.status?.connection {
        case .connected: return .green
        case .connecting, .backingOff: return .orange
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

    private var statusRow: some View {
        HStack(spacing: 12) {
            Label(connectionText, systemImage: "antenna.radiowaves.left.and.right")
                .font(.caption)
                .foregroundColor(connectionColor)
            Text("\(peer.status?.activeRequests ?? 0)/\(peer.status?.totalRequests ?? 0) requests")
                .font(.caption.monospaced()).foregroundColor(.secondary)
            Text("\(peer.status?.totalCompletionTokens ?? 0) tokens")
                .font(.caption.monospaced()).foregroundColor(.secondary)
            if let error = peer.lastError {
                Text(error).font(.caption).foregroundColor(.red).lineLimit(2)
            }
            Spacer()
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: Registration

    private var registrationSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Register this Mac").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 10) {
                GridRow {
                    Text("Server").font(.caption)
                    TextField("https://tknet.ai", text: $serverText)
                }
                GridRow {
                    Text("Peer name").font(.caption)
                    TextField("Peer name", text: $peerNameText)
                }
            }
            Button("Register") {
                guard let url = URL(string: serverText.trimmingCharacters(in: .whitespaces)) else { return }
                Task { await peer.register(server: url, peerName: peerNameText) }
            }
            .buttonStyle(.bordered)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: Demand list

    private var demandSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Demand (\(peer.demand.count))").font(.headline)
                Spacer()
                Button("Refresh") { Task { await peer.fetchDemand() } }
            }
            if peer.demand.isEmpty {
                Text("No open demand right now.")
                    .font(.caption).foregroundColor(.secondary)
                    .padding(.top, 4)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(peer.demand, id: \.demandId) { entry in
                        demandRow(entry)
                    }
                }
            }
        }
    }

    private func demandRow(_ entry: DemandEntry) -> some View {
        let claimed = peer.config.capabilities.first { $0.demandId == entry.demandId }
        let active = claimed.map { peer.config.activeCapabilities.contains($0) } ?? false
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
            }
            Spacer()
            if active {
                Label("Serving", systemImage: "checkmark.circle.fill")
                    .font(.caption).foregroundColor(.green)
            } else if claimed != nil {
                Label("Retired — waiting for demand to return", systemImage: "clock")
                    .font(.caption).foregroundColor(.secondary)
            } else {
                Button("Serve via NovaMLX (local)") {
                    Task { await peer.addLocalSource(for: entry, apiKeyStore: NovaDB.shared.apiKeyStore) }
                }
                .buttonStyle(.bordered)
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

    init(capability: Capability, peer: TknetPeerState) {
        self.capability = capability
        self.peer = peer
        _upstream = State(initialValue: peer.upstreamModel(for: capability))
    }

    private var retired: Bool {
        !peer.config.activeCapabilities.contains(capability)
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
            }
            Spacer()
            TextField("Upstream model", text: $upstream)
                .textFieldStyle(.roundedBorder)
                .frame(width: 220)
                .onSubmit { Task { await peer.updateUpstreamModel(for: capability.demandId, to: upstream) } }
            if retired {
                Text("retired").font(.caption2).foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .opacity(retired ? 0.5 : 1.0)
    }
}
