import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXTknetNode

// MARK: - TknetNodePageView (Task 11)
//
// In-page strings are literal English, matching LoadBalancersPageView's
// convention; only the sidebar page title is localized (app.tknetNode).

struct TknetNodePageView: View {
    @StateObject private var node = TknetNodeState()
    @State private var serverText = "https://tknet.ai"
    @State private var nodeNameText = "tknet-node-\(ProcessInfo.processInfo.hostName.prefix(20))"

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                statusRow
                if node.config.nodeId == nil {
                    registrationSection
                } else {
                    demandSection
                    if !node.config.capabilities.isEmpty {
                        capabilitiesSection
                    }
                }
            }
            .padding(24)
        }
        .navigationTitle("Tknet Node")
        .task { if node.config.nodeId != nil { await node.fetchDemand() } }
    }

    // MARK: Header + status

    private var connectionText: String {
        switch node.status?.connection {
        case .connecting: return "connecting"
        case .connected: return "connected"
        case .backingOff(let seconds): return "reconnecting in \(Int(seconds))s"
        case .stopping: return "stopping"
        case .idle, nil: return "idle"
        }
    }

    private var connectionColor: Color {
        switch node.status?.connection {
        case .connected: return .green
        case .connecting, .backingOff: return .orange
        default: return .secondary
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Tknet Node").font(.title2.bold())
                Text("Serve tknet.ai demand from this Mac")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Button(node.running ? "Stop" : "Start") {
                Task { node.running ? await node.stop() : await node.start() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(node.config.nodeId == nil)
            .help(node.config.nodeId == nil ? "Register this node first" : "")
        }
    }

    private var statusRow: some View {
        HStack(spacing: 12) {
            Label(connectionText, systemImage: "antenna.radiowaves.left.and.right")
                .font(.caption)
                .foregroundColor(connectionColor)
            Text("\(node.status?.activeRequests ?? 0)/\(node.status?.totalRequests ?? 0) requests")
                .font(.caption.monospaced()).foregroundColor(.secondary)
            Text("\(node.status?.totalCompletionTokens ?? 0) tokens")
                .font(.caption.monospaced()).foregroundColor(.secondary)
            if let error = node.lastError {
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
                    Text("Node name").font(.caption)
                    TextField("Node name", text: $nodeNameText)
                }
            }
            Button("Register") {
                guard let url = URL(string: serverText.trimmingCharacters(in: .whitespaces)) else { return }
                Task { await node.register(server: url, nodeName: nodeNameText) }
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
                Text("Demand (\(node.demand.count))").font(.headline)
                Spacer()
                Button("Refresh") { Task { await node.fetchDemand() } }
            }
            if node.demand.isEmpty {
                Text("No open demand right now.")
                    .font(.caption).foregroundColor(.secondary)
                    .padding(.top, 4)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(node.demand, id: \.demandId) { entry in
                        demandRow(entry)
                    }
                }
            }
        }
    }

    private func demandRow(_ entry: DemandEntry) -> some View {
        let claimed = node.config.capabilities.first { $0.demandId == entry.demandId }
        let active = claimed.map { node.config.activeCapabilities.contains($0) } ?? false
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
                    Task { await node.addLocalSource(for: entry, apiKeyStore: NovaDB.shared.apiKeyStore) }
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
            Text("Declared Capabilities (\(node.config.capabilities.count))").font(.headline)
            LazyVStack(spacing: 8) {
                ForEach(node.config.capabilities, id: \.demandId) { cap in
                    CapabilityRow(capability: cap, node: node)
                }
            }
        }
    }
}

// MARK: - CapabilityRow (editable upstream model)

private struct CapabilityRow: View {
    let capability: Capability
    @ObservedObject var node: TknetNodeState
    @State private var upstream: String

    init(capability: Capability, node: TknetNodeState) {
        self.capability = capability
        self.node = node
        _upstream = State(initialValue: node.upstreamModel(for: capability))
    }

    private var retired: Bool {
        !node.config.activeCapabilities.contains(capability)
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
                .onSubmit { Task { await node.updateUpstreamModel(for: capability.demandId, to: upstream) } }
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
