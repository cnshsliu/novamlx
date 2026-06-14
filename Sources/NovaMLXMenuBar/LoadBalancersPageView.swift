import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils

// MARK: - LoadBalancersPageView (Task 10)

struct LoadBalancersPageView: View {
    @State private var lbs: [LoadBalancer] = []
    @State private var editing: LoadBalancer?
    @State private var creating = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                if lbs.isEmpty {
                    emptyState
                } else {
                    LazyVStack(spacing: 8) {
                        ForEach(lbs) { lb in
                            LBRow(lb: lb) { editing = lb }
                        }
                    }
                }
            }
            .padding(24)
        }
        .navigationTitle("Load Balancers")
        .sheet(item: $editing) { lb in
            LBEditView(lbId: lb.id)
        }
        .sheet(isPresented: $creating) {
            LBEditView(lbId: nil)
        }
        .task { await reload() }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Load Balancers").font(.title2.bold())
                Text("Route requests across pools via `lb:<slug>`")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Button("+ New LB") { creating = true }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "scalemass")
                .font(.system(size: 32))
                .foregroundColor(.secondary)
            Text("No load balancers yet").font(.headline)
            Text("Create one to route requests across local and remote models.")
                .font(.caption).foregroundColor(.secondary)
        }
        .padding(.top, 60)
    }

    private func reload() async {
        do {
            lbs = try NovaDB.shared.loadBalancerStore.listLBs()
        } catch {
            NovaMLXLog.error("[LB] list failed: \(error)")
        }
    }
}

// MARK: - LBRow (list card)

struct LBRow: View {
    let lb: LoadBalancer
    let onEdit: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(lb.name).font(.headline)
                    Text("lb:\(lb.slug)")
                        .font(.caption.monospaced())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .foregroundColor(.accentColor)
                        .clipShape(Capsule())
                }
                Text("\(lb.strategy.rawValue) · \(lb.requestCount) requests")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Circle()
                .fill(lb.isEnabled ? Color.green : Color.gray.opacity(0.4))
                .frame(width: 10, height: 10)
            Button("Edit", action: onEdit)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .opacity(lb.isEnabled ? 1.0 : 0.5)
    }
}

// MARK: - LBEditView (form + members)

struct LBEditView: View {
    let lbId: UUID?

    @Environment(\.dismiss) private var dismiss
    @State private var lb: LoadBalancer?
    @State private var members: [LBMember] = []
    @State private var stats: [UUID: LBMemberStats] = [:]
    @State private var showAddMember = false
    @State private var errorMsg: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            if let lb {
                formFields(lb: lb)
                membersSection(lb: lb)
                if let errorMsg { Text(errorMsg).foregroundColor(.red).font(.caption) }
            } else {
                ProgressView()
            }
            Spacer()
        }
        .padding(24)
        .frame(minWidth: 600, minHeight: 500)
        .sheet(isPresented: $showAddMember) {
            if let liveId = self.lb?.id {
                LBMemberPickerSheet(lbId: liveId) { _ in
                    Task { await reload() }
                }
            } else {
                // Defensive: the "+ Add member" button is disabled until self.lb
                // is set, so this branch should be unreachable. Show a hint anyway
                // rather than silently doing nothing if the invariant ever slips.
                Text("Save the LB first.").padding()
            }
        }
        .task {
            if lbId != nil {
                await reload()
            } else {
                // Create new LB with default values
                let new = LoadBalancer(name: "New LB", slug: "new-lb")
                try? NovaDB.shared.loadBalancerStore.upsertLB(new)
                self.lb = new
            }
        }
    }

    private var header: some View {
        HStack {
            Text(lbId == nil ? "New Load Balancer" : "Edit Load Balancer")
                .font(.title2.bold())
            Spacer()
            Button("Done") { dismiss() }
        }
    }

    @ViewBuilder
    private func formFields(lb: LoadBalancer) -> some View {
        let strategyBinding = Binding<LBStrategy>(
            get: { self.lb?.strategy ?? .tiered },
            set: { newStrategy in
                self.lb?.strategy = newStrategy
                save()
            }
        )
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 10) {
            GridRow {
                Text("Name").font(.caption)
                TextField("Name", text: Binding(
                    get: { self.lb?.name ?? "" },
                    set: { self.lb?.name = $0; save() }
                ))
            }
            GridRow {
                Text("Slug").font(.caption)
                TextField("Slug", text: Binding(
                    get: { self.lb?.slug ?? "" },
                    set: { newSlug in
                        guard isValidLBSlug(newSlug) else {
                            self.errorMsg = "slug must match ^[a-z0-9-]+$"
                            return
                        }
                        self.errorMsg = nil
                        self.lb?.slug = newSlug
                        save()
                    }
                ))
            }
            GridRow {
                Text("Strategy").font(.caption)
                Picker("Strategy", selection: strategyBinding) {
                    ForEach(LBStrategy.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
            }
            GridRow {
                Text("Max retries").font(.caption)
                Stepper(value: Binding(
                    get: { self.lb?.maxRetries ?? 3 },
                    set: { self.lb?.maxRetries = $0; save() }
                ), in: 1...10) {
                    Text("\(self.lb?.maxRetries ?? 3)")
                }
            }
            GridRow {
                Text("Enabled").font(.caption)
                Toggle("", isOn: Binding(
                    get: { self.lb?.isEnabled ?? true },
                    set: { self.lb?.isEnabled = $0; save() }
                ))
            }
        }
    }

    @ViewBuilder
    private func membersSection(lb: LoadBalancer) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Members (\(members.count))").font(.headline)
                Spacer()
                Button("+ Add member") { showAddMember = true }
                    .disabled(self.lb?.id == nil)  // disabled until the LB is saved
            }
            ForEach(members) { m in
                LBMemberRow(
                    member: m,
                    stats: stats[m.id],
                    lb: lb,
                    onChange: { Task { await reload() } }
                )
            }
        }
    }

    private func save() {
        guard var updated = lb else { return }
        updated.updatedAt = Date()
        do {
            try NovaDB.shared.loadBalancerStore.upsertLB(updated)
            self.lb = updated
        } catch {
            self.errorMsg = String(describing: error)
        }
    }

    private func reload() async {
        guard let lbId else { return }
        do {
            lb = try NovaDB.shared.loadBalancerStore.getLB(lbId)
            members = try NovaDB.shared.lbMemberStore.listMembers(lbId: lbId)
            var statsMap: [UUID: LBMemberStats] = [:]
            for m in members {
                if let s = try NovaDB.shared.lbMemberStatsStore.getStats(m.id) {
                    statsMap[m.id] = s
                }
            }
            self.stats = statsMap
        } catch {
            NovaMLXLog.error("[LBEdit] reload failed: \(error)")
        }
    }
}

// MARK: - LBMemberRow

struct LBMemberRow: View {
    let member: LBMember
    let stats: LBMemberStats?
    let lb: LoadBalancer
    let onChange: () -> Void

    var body: some View {
        HStack {
            // Kind badge
            Text(member.kind == .local ? "LOCAL" : "REMOTE")
                .font(.caption2.bold())
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(member.kind == .local
                    ? Color.green.opacity(0.15) : Color.yellow.opacity(0.15))
                .foregroundColor(member.kind == .local ? .green : .orange)
                .clipShape(Capsule())

            // Reference
            Text(member.ref).font(.caption.monospaced())
                .foregroundColor(.primary)

            // Status
            if member.kind == .local {
                Text("see Local Inference page")  // MLXEngine.shared.isModelLoaded not accessible here
                    .font(.caption2).foregroundColor(.secondary)
            } else if let stats {
                Text("\(stats.avgLatencyMs)ms avg")
                    .font(.caption2).foregroundColor(.secondary)
            }

            Spacer()

            // Weight (only if weighted strategy)
            if lb.strategy == .weighted {
                Text("w:").font(.caption2).foregroundColor(.secondary)
                TextField("", value: Binding(
                    get: { member.weight ?? 1 },
                    set: { newWeight in
                        var updated = member
                        updated.weight = max(1, newWeight)
                        try? NovaDB.shared.lbMemberStore.upsertMember(updated)
                        onChange()
                    }
                ), format: .number)
                .frame(width: 40)
                .textFieldStyle(.roundedBorder)
            }

            // Enable toggle
            Toggle("", isOn: Binding(
                get: { member.isEnabled },
                set: { v in
                    var updated = member
                    updated.isEnabled = v
                    try? NovaDB.shared.lbMemberStore.upsertMember(updated)
                    onChange()
                }
            )).labelsHidden()

            // Remove
            Button(role: .destructive) {
                try? NovaDB.shared.lbMemberStore.deleteMember(member.id)
                onChange()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red.opacity(0.6))
            }.buttonStyle(.plain)
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// Local slug validation (mirrors isValidLBSlug in admin API)
private func isValidLBSlug(_ s: String) -> Bool {
    guard !s.isEmpty, s.count <= 64 else { return false }
    return s.allSatisfy { c in
        (c >= "a" && c <= "z") || (c >= "0" && c <= "9") || c == "-"
    }
}
