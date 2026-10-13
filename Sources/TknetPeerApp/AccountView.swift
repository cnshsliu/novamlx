import TknetPeer
import AppKit
import SwiftUI

/// Account claim and supplier earnings. Pools are what tknet sells.
struct AccountView: View {
    @EnvironmentObject private var state: PeerAppState
    @State private var loginEmail = ""
    @State private var loginPassword = ""
    @State private var didLoadEmail = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                claimCard
                earningsCard
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear {
            guard !didLoadEmail else { return }
            didLoadEmail = true
            if loginEmail.isEmpty {
                loginEmail = state.savedLoginEmail
            }
            Task {
                await state.refreshAccount()
                await state.fetchEarnings()
            }
        }
    }

    private var claimCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Account")
                .font(.headline)
            Text("认领 token 工厂")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text("Sign in to your tknet account, then claim this token factory. The password is not stored. tknet sells pools.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if state.config.peerId == nil {
                Text("Register this token factory on the Tunnel tab before claiming it.")
                    .font(.callout)
            } else if state.accountEmail != nil {
                Text(signedInLine)
                    .font(.callout)
                if state.claimedPeerId == state.config.peerId {
                    Text("Claimed")
                        .font(.callout.bold())
                        .foregroundStyle(.green)
                }
                HStack(spacing: 8) {
                    Button("Claim token factory") {
                        Task { await state.claimTokenFactory() }
                    }
                    .disabled(state.isAccountBusy || state.connectionKey == nil)
                    Button("Sign out") {
                        Task { await state.signOut() }
                    }
                    .disabled(state.isAccountBusy)
                }
            } else {
                TextField("Email", text: $loginEmail)
                    .textFieldStyle(.roundedBorder)
                SecureField("Password", text: $loginPassword)
                    .textFieldStyle(.roundedBorder)
                Button("Sign in") {
                    let email = loginEmail
                    let password = loginPassword
                    Task {
                        if await state.signIn(email: email, password: password) {
                            loginPassword = ""
                        }
                    }
                }
                .disabled(state.isAccountBusy
                          || loginEmail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || loginPassword.isEmpty)
            }
            if let error = state.accountError {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var earningsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Earnings")
                    .font(.headline)
                Spacer()
                Button(state.isEarningsBusy ? "Refreshing…" : "Refresh") {
                    Task { await state.fetchEarnings() }
                }
                .disabled(state.isEarningsBusy || state.connectionKey == nil)
            }
            if let summary = state.earnings {
                let availability = summary.availability
                HStack(alignment: .top, spacing: 16) {
                    moneyColumn("Total earned", availability.totalEarned, emphasize: false)
                    moneyColumn("Payout available", availability.available, emphasize: true)
                    moneyColumn("On hold (7d)", availability.onHold, emphasize: false)
                    Spacer(minLength: 0)
                }
                Text("Paid out $\(availability.paidOut)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !summary.byModel.isEmpty {
                    Text("By pool")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)
                    ForEach(Array(summary.byModel.enumerated()), id: \.offset) { _, row in
                        HStack {
                            Text(row.model)
                                .font(.caption.monospaced())
                            Spacer()
                            Text("\(row.requests) req · \(row.tokens) tok")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("$\(row.earned)")
                                .font(.caption.monospacedDigit())
                        }
                    }
                }
                if !summary.recent.isEmpty {
                    Text("Recent")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)
                    ForEach(Array(summary.recent.prefix(8).enumerated()), id: \.offset) { _, entry in
                        HStack {
                            Text(entry.createdAt)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            Text(entry.model)
                                .font(.caption.monospaced())
                            Spacer()
                            Text("$\(entry.grossAmount)\(entry.payoutId != nil ? " paid" : "")")
                                .font(.caption.monospacedDigit())
                        }
                    }
                }
            } else if state.connectionKey == nil {
                Text("Register this token factory on the Tunnel tab, then refresh after pools have served traffic.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                Text("No earnings yet. Refresh after pools have served traffic.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if let error = state.earningsError {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var signedInLine: String {
        let email = state.accountEmail ?? ""
        let name = state.accountName ?? ""
        if name.isEmpty { return "Signed in as \(email)" }
        return "Signed in as \(name) (\(email))"
    }

    private func moneyColumn(_ title: String, _ amount: String, emphasize: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text("$\(amount)")
                .font(emphasize ? .title3.bold().monospacedDigit() : .title3.monospacedDigit())
                .foregroundStyle(emphasize ? Color.green : Color.primary)
        }
    }
}
