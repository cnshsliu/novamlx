import AppKit
import SwiftUI

struct TunnelStatusView: View {
    @EnvironmentObject private var state: PeerAppState
    @State private var serverText = ""
    @State private var peerName = ""
    @State private var revealKey = false
    @State private var didLoadDrafts = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                statusCard
                tunnelControls
                identityCard
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear {
            guard !didLoadDrafts else { return }
            didLoadDrafts = true
            serverText = state.activeREST
        }
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(state.connectionLabel, systemImage: state.connectionSymbol)
                .font(.headline)
                .foregroundStyle(connectionColor)
            HStack(spacing: 8) {
                Circle()
                    .fill(state.isGatewayUp ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(state.isGatewayUp ? "Gateway listening \(state.gatewayLabel)" : "Gateway \(state.gatewayLabel)")
            }
            .font(.callout)
            if let peerId = state.config.peerId, !peerId.isEmpty {
                Text("Peer id \(peerId)")
                    .font(.callout)
                    .textSelection(.enabled)
            }
            Text("Active requests \(state.status?.activeRequests ?? 0)")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(state.tunnelWanted
                 ? "Tunnel auto-dials on launch."
                 : "Tunnel stays down on the next launch. The gateway still listens.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var tunnelControls: some View {
        HStack(spacing: 12) {
            if state.isTunnelActive {
                // Stop must stay available while Connecting/Busy — isBusy only
                // gates Start/Register so a hung dial cannot lock the operator out.
                Button("Stop") { Task { await state.stopTunnelFromUser() } }
                    .disabled(state.lockError != nil)
            } else {
                Button("Start") { Task { await state.startTunnelFromUser() } }
                    .disabled(state.lockError != nil || state.isBusy || state.config.peerId == nil)
                if state.tunnelWanted {
                    Button("Stop auto-dial") { Task { await state.stopTunnelFromUser() } }
                        .disabled(state.lockError != nil)
                }
            }
            Text("Stop leaves the local port open. Quit shuts the gateway down.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var identityCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Registration")
                .font(.headline)
            if let peerId = state.config.peerId, !peerId.isEmpty, let key = state.connectionKey {
                Text("Peer id")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(peerId)
                    .font(.body.monospaced())
                    .textSelection(.enabled)
                Text("连接密钥")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(revealKey ? key : String(repeating: "•", count: min(key.count, 24)))
                    .font(.body.monospaced())
                    .textSelection(.enabled)
                HStack {
                    Button(revealKey ? "Hide connection key" : "Show connection key") {
                        revealKey.toggle()
                    }
                    Button("Copy connection key") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(key, forType: .string)
                    }
                }
            } else {
                Text("Register once. The connection key is kept in the Keychain and mirrored for `tknet-peer`.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                TextField("Server", text: $serverText)
                    .textFieldStyle(.roundedBorder)
                TextField("Peer name", text: $peerName)
                    .textFieldStyle(.roundedBorder)
                Button("Register") {
                    Task { await state.register(serverText: serverText, peerName: peerName) }
                }
                .disabled(state.lockError != nil || state.isBusy)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var connectionColor: Color {
        switch state.status?.connection {
        case .connected: return .green
        case .connecting, .backingOff: return .orange
        case .upgradeRequired: return .red
        default: return .secondary
        }
    }
}
