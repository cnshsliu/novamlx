import SwiftUI

struct PeerRootView: View {
    @EnvironmentObject private var state: PeerAppState
    // UI-test hook: `--start-tab=members` lands directly on a section so
    // read-only GUI verification needs no synthetic clicks. Case-insensitive
    // against the tab raw values ("Members", "Pools", ...).
    @State private var tab: PeerTab = {
        guard let arg = ProcessInfo.processInfo.arguments
            .first(where: { $0.hasPrefix("--start-tab=") }) else { return .tunnel }
        let raw = String(arg.dropFirst("--start-tab=".count))
        return PeerTab.allCases.first { $0.rawValue.lowercased() == raw.lowercased() } ?? .tunnel
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Tknet Peer")
                        .font(.title2.bold())
                    Text("连接 token 工厂与市场")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Picker("Section", selection: $tab) {
                    ForEach(PeerTab.allCases) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 460)
            }
            if let lockError = state.lockError {
                Text(lockError)
                    .font(.callout)
                    .foregroundStyle(.red)
            } else if let lastError = state.lastError {
                Text(lastError)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            Group {
                switch tab {
                case .tunnel:
                    TunnelStatusView()
                case .members:
                    MembersView()
                case .pools:
                    PoolsView()
                case .account:
                    AccountView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .padding(16)
        .frame(minWidth: 680, minHeight: 480)
    }
}

private enum PeerTab: String, CaseIterable, Identifiable {
    case tunnel = "Tunnel"
    case members = "Members"
    case pools = "Pools"
    case account = "Account"

    var id: String { rawValue }
}
