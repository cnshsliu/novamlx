import TknetPeer
import SwiftUI

struct PoolsView: View {
    @EnvironmentObject private var state: PeerAppState
    @State private var draft: PoolDraft?
    @State private var pendingDelete: PoolConfig?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Pools")
                    .font(.headline)
                Spacer()
                Button("Add pool") { draft = PoolDraft() }
                    .disabled(state.lockError != nil || state.config.members.isEmpty)
            }
            if state.config.pools.isEmpty {
                Text("No pools yet. A pool name is the model id clients send to the gateway.")
                    .foregroundStyle(.secondary)
            } else {
                List {
                    ForEach(state.config.pools) { pool in
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(pool.name).font(.body.bold())
                                Text(strategyLabel(pool.strategy))
                                    .font(.caption)
                                Text(pool.memberIds.joined(separator: ", "))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text(pool.isEnabled ? "Enabled · retries \(pool.maxRetries)" : "Disabled")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Edit") { draft = PoolDraft(pool) }
                            Button("Delete") { pendingDelete = pool }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .sheet(item: $draft) { draft in
            PoolForm(draft: draft, members: state.config.members)
        }
        .confirmationDialog(
            "Delete \(pendingDelete?.name ?? "pool")?",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let id = pendingDelete?.id {
                    Task { await state.deletePool(id: id) }
                }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }
}

private struct PoolDraft: Identifiable {
    let token = UUID()
    var id: UUID { token }
    var poolId: String
    var name: String
    var strategy: PoolStrategy
    var memberIds: [String]
    var maxRetries: Int
    var isEnabled: Bool
    var isNew: Bool

    init() {
        poolId = ""
        name = ""
        strategy = .tiered
        memberIds = []
        maxRetries = 3
        isEnabled = true
        isNew = true
    }

    init(_ pool: PoolConfig) {
        poolId = pool.id
        name = pool.name
        strategy = pool.strategy
        memberIds = pool.memberIds
        maxRetries = pool.maxRetries
        isEnabled = pool.isEnabled
        isNew = false
    }
}

private struct PoolForm: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var state: PeerAppState
    let draft: PoolDraft
    let members: [MemberConfig]

    @State private var poolId: String
    @State private var name: String
    @State private var strategy: PoolStrategy
    @State private var selected: Set<String>
    @State private var maxRetries: Int
    @State private var isEnabled: Bool
    @State private var formError: String?

    init(draft: PoolDraft, members: [MemberConfig]) {
        self.draft = draft
        self.members = members
        _poolId = State(initialValue: draft.poolId)
        _name = State(initialValue: draft.name)
        _strategy = State(initialValue: draft.strategy)
        _selected = State(initialValue: Set(draft.memberIds))
        _maxRetries = State(initialValue: draft.maxRetries)
        _isEnabled = State(initialValue: draft.isEnabled)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(draft.isNew ? "Add pool" : "Edit pool")
                .font(.title3.bold())
            TextField("Id", text: $poolId)
                .disabled(!draft.isNew)
            TextField("Name (model id clients use)", text: $name)
            Picker("Strategy", selection: $strategy) {
                ForEach(PoolStrategy.allCases, id: \.self) { strategy in
                    Text(strategyLabel(strategy)).tag(strategy)
                }
            }
            Stepper("Max retries \(maxRetries)", value: $maxRetries, in: 1...20)
            Toggle("Enabled", isOn: $isEnabled)
            Text("Members")
                .font(.headline)
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(members) { member in
                        Toggle(isOn: Binding(
                            get: { selected.contains(member.id) },
                            set: { isOn in
                                if isOn { selected.insert(member.id) } else { selected.remove(member.id) }
                            }
                        )) {
                            Text("\(member.name) (\(member.id))")
                        }
                    }
                }
            }
            .frame(maxHeight: 180)
            if let formError {
                Text(formError).foregroundStyle(.red).font(.callout)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { Task { await save() } }
            }
        }
        .padding(20)
        .frame(width: 480)
    }

    private func save() async {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            formError = "Pool name is required."
            return
        }
        let idSource = poolId.trimmingCharacters(in: .whitespacesAndNewlines)
        let id = idSource.isEmpty ? trimmedName : idSource
        let ordered = members.map(\.id).filter { selected.contains($0) }
        guard !ordered.isEmpty else {
            formError = "A pool needs at least one member."
            return
        }
        let saved = await state.savePool(
            id: id, name: trimmedName, strategy: strategy, memberIds: ordered,
            maxRetries: maxRetries, isEnabled: isEnabled, isNew: draft.isNew)
        if saved {
            dismiss()
        } else {
            formError = state.lastError ?? "Could not save the pool."
        }
    }
}

private func strategyLabel(_ strategy: PoolStrategy) -> String {
    switch strategy {
    case .tiered: return "Tiered"
    case .roundRobin: return "Round robin"
    case .weighted: return "Weighted"
    case .lowestLatency: return "Lowest latency"
    case .random: return "Random"
    }
}
