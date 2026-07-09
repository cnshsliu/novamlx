import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils

// MARK: - LoadBalancersPageView (Task 10)

struct LoadBalancersPageView: View {
    @ObservedObject var appState: MenuBarAppState
    @State private var lbs: [LoadBalancer] = []
    @State private var editing: LoadBalancer?
    @State private var creating = false
    /// Accordion state: UUID of the currently-expanded LB, or nil if all collapsed.
    /// At most one LB's members panel is open at a time. Clicking the expanded
    /// row again collapses it (toggle); clicking a different row collapses the
    /// current one and expands the new one (mutual exclusion).
    @State private var expandedId: UUID?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                if lbs.isEmpty {
                    emptyState
                } else {
                    LazyVStack(spacing: 8) {
                        ForEach(lbs) { lb in
                            lbCard(lb)
                        }
                    }
                }
            }
            .padding(24)
        }
        .navigationTitle("Load Balancers")
        .sheet(item: $editing, onDismiss: { Task { await reload() } }) { lb in
            LBEditView(lbId: lb.id)
        }
        .sheet(isPresented: $creating, onDismiss: { Task { await reload() } }) {
            LBEditView(lbId: nil)
        }
        .task { await reload() }
    }

    /// One accordion card: header row + collapsible read-only members panel.
    @ViewBuilder
    private func lbCard(_ lb: LoadBalancer) -> some View {
        let isExpanded = expandedId == lb.id
        VStack(spacing: 0) {
            LBRow(
                lb: lb,
                isExpanded: isExpanded,
                onEdit: { editing = lb },
                onToggle: {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        expandedId = isExpanded ? nil : lb.id
                    }
                },
                onPlay: { appState.pickInPlayground("lb:" + lb.slug) }
            )
            if isExpanded {
                LBMembersPreviewPanel(lbId: lb.id)
                    .background(Color(nsColor: .controlBackgroundColor).opacity(0.4))
                    .transition(.opacity)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .opacity(lb.isEnabled ? 1.0 : 0.5)
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

// MARK: - LBRow (list card header — tap toggles member panel, Edit opens sheet)

struct LBRow: View {
    let lb: LoadBalancer
    let isExpanded: Bool
    let onEdit: () -> Void
    let onToggle: () -> Void
    let onPlay: () -> Void

    var body: some View {
        HStack {
            // Chevron that flips when expanded — visual cue that the row is tappable.
            Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                .font(.caption.bold())
                .foregroundColor(.secondary)
                .frame(width: 12)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(lb.name).font(.headline)
                    Text("lb:\(lb.slug)")
                        .font(.caption.monospaced())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .foregroundColor(.accentColor)
                        .clipShape(Capsule())
                    // Pick-to-Playground: copies lb:<slug> into the Playground
                    // model picker and jumps there. Kept outside the row's
                    // contentShape tap so it doesn't toggle the accordion.
                    Button(action: onPlay) {
                        Image(systemName: "play.circle")
                            .font(.system(size: 11))
                            .foregroundColor(.accentColor)
                    }
                    .buttonStyle(.plain)
                    .help("Open in Playground")
                }
                Text("\(lb.strategy.rawValue) · \(lb.requestCount) requests")
                    .font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Circle()
                .fill(lb.isEnabled ? Color.green : Color.gray.opacity(0.4))
                .frame(width: 10, height: 10)
            // Edit button stays independent of the row's tap gesture so the
            // outer contentShape doesn't swallow its click.
            Button("Edit", action: onEdit)
                .buttonStyle(.bordered)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .contentShape(Rectangle())
        .onTapGesture { onToggle() }
    }
}

// MARK: - LBMembersPreviewPanel (read-only member list shown when row is expanded)

/// Read-only member list. Loads members + stats on appear and renders one row
/// per member with its kind badge, ref, and live status (loaded for locals,
/// avg latency for remotes). No controls here — editing happens in LBEditView.
struct LBMembersPreviewPanel: View {
    let lbId: UUID
    @State private var members: [LBMember] = []
    @State private var stats: [UUID: LBMemberStats] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if members.isEmpty {
                Text("No members yet — click Edit to add some.")
                    .font(.caption).foregroundColor(.secondary)
                    .padding(.vertical, 4)
            } else {
                ForEach(members) { m in
                    memberRow(m)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .task { await reload() }
    }

    @ViewBuilder
    private func memberRow(_ m: LBMember) -> some View {
        HStack(spacing: 8) {
            Text(m.kind == .local ? "LOCAL" : "REMOTE")
                .font(.caption2.bold())
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(m.kind == .local
                    ? Color.green.opacity(0.15) : Color.yellow.opacity(0.15))
                .foregroundColor(m.kind == .local ? .green : .orange)
                .clipShape(Capsule())

            Text(m.ref).font(.caption.monospaced())
                .foregroundColor(.primary)
                .lineLimit(1)
                .truncationMode(.middle)

            Spacer()

            if !m.isEnabled {
                Text("disabled")
                    .font(.caption2).foregroundColor(.secondary)
            } else if m.kind == .local {
                let loaded = isLocalModelLoaded(m.ref)
                Text(loaded ? "✓ loaded" : "⚠ not loaded")
                    .font(.caption2)
                    .foregroundColor(loaded ? .green : .orange)
            } else if let s = stats[m.id] {
                Text("\(s.avgLatencyMs)ms avg")
                    .font(.caption2).foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    private func reload() async {
        do {
            members = try NovaDB.shared.lbMemberStore.listMembers(lbId: lbId)
            var map: [UUID: LBMemberStats] = [:]
            for m in members {
                if let s = try NovaDB.shared.lbMemberStatsStore.getStats(m.id) {
                    map[m.id] = s
                }
            }
            stats = map
        } catch {
            NovaMLXLog.error("[LBPreview] reload failed: \(error)")
        }
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
    @State private var showEmptyMembersWarning = false
    /// Snapshot of the LB as it was when the edit sheet opened. Used to roll
    /// back changes on "Cancel edit" (Edit path: restore; New path: delete).
    @State private var originalLB: LoadBalancer?

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
        .alert("Add at least one member", isPresented: $showEmptyMembersWarning) {
            // OK: just dismiss the alert, stay in the form for further editing.
            Button("OK", role: .cancel) {}
            // Cancel edit: rollback + close.
            Button("Cancel edit", role: .destructive) { cancelEdit() }
        } message: {
            Text("This load balancer has no members. Requests to it will fail with 503 until you add at least one. Click \"OK\" to keep editing, or \"Cancel edit\" to \(lbId == nil ? "discard this LB" : "revert your changes").")
        }
        .task {
            if let lbId {
                await reload()
                // Edit path: snapshot the original so Cancel can revert.
                originalLB = lb
            } else {
                // New path: mint a unique default slug so the second consecutive
                // "New LB" doesn't trip UNIQUE(slug). originalLB stays nil so
                // Cancel knows to DELETE the row we're about to create.
                let slug = makeUniqueDefaultSlug()
                let new = LoadBalancer(name: "New LB", slug: slug)
                do {
                    try NovaDB.shared.loadBalancerStore.upsertLB(new)
                    self.lb = new
                } catch {
                    self.errorMsg = friendlyLBError(error)
                }
            }
        }
    }

    private var header: some View {
        HStack {
            Text(lbId == nil ? "New Load Balancer" : "Edit Load Balancer")
                .font(.title2.bold())
            Spacer()
            Button("Done") {
                if members.isEmpty {
                    showEmptyMembersWarning = true
                } else {
                    dismiss()
                }
            }
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
                            self.errorMsg = "Slug must match ^[a-z0-9-]+$ and be 1-64 chars."
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
            self.errorMsg = nil
        } catch {
            self.errorMsg = friendlyLBError(error)
        }
    }

    /// Roll back any unsaved-as-final edits and close the sheet.
    /// - Edit path (`originalLB != nil`): restore the LB to its pre-edit state.
    /// - New path (`originalLB == nil`): delete the row created by `.task`.
    private func cancelEdit() {
        if let original = originalLB {
            // Edit: revert to the snapshot taken on `.task`.
            do {
                try NovaDB.shared.loadBalancerStore.upsertLB(original)
            } catch {
                NovaMLXLog.error("[LBEdit] revert failed: \(error)")
            }
        } else if let created = lb {
            // New: delete the placeholder row we created in `.task`. Cascades
            // to lb_members + lb_member_stats so nothing leaks.
            do {
                try NovaDB.shared.loadBalancerStore.deleteLB(created.id)
            } catch {
                NovaMLXLog.error("[LBEdit] delete failed: \(error)")
            }
        }
        dismiss()
    }

    private func reload() async {
        // Resolve live ID: for the edit path, `lbId` is non-nil; for the create
        // path, `lbId` is nil but `self.lb?.id` is set after `.task` creates
        // the new row. Use whichever is available so both paths can re-fetch
        // members after the picker sheet adds rows.
        guard let liveId = lbId ?? self.lb?.id else { return }
        do {
            lb = try NovaDB.shared.loadBalancerStore.getLB(liveId)
            members = try NovaDB.shared.lbMemberStore.listMembers(lbId: liveId)
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

/// Mint a slug that doesn't collide with any existing LB's slug.
/// Tries `new-lb`, then `new-lb-2`, `new-lb-3`, …
private func makeUniqueDefaultSlug() -> String {
    let existing = (try? NovaDB.shared.loadBalancerStore.listLBs()) ?? []
    let taken = Set(existing.map(\.slug))
    if !taken.contains("new-lb") { return "new-lb" }
    var i = 2
    while taken.contains("new-lb-\(i)") { i += 1 }
    return "new-lb-\(i)"
}

/// Translate a GRDB/SQLite error into a human-readable message. Falls back
/// to the raw error description for anything we don't recognize.
private func friendlyLBError(_ error: Error) -> String {
    let raw = String(describing: error)
    // GRDB surfaces UNIQUE violations as "SQLite error 19: UNIQUE constraint failed: <table>.<col>"
    if raw.contains("UNIQUE constraint failed: load_balancers.slug") {
        return "A load balancer with this slug already exists. Choose a different slug."
    }
    if raw.contains("UNIQUE constraint failed") && raw.contains("slug") {
        return "A load balancer with this slug already exists. Choose a different slug."
    }
    return raw
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
                // Query the loaded_models table (the same store InferenceService
                // writes to on every load/unload) to reflect live MLXEngine state
                // without needing the InferenceService instance threaded in.
                let loaded = isLocalModelLoaded(member.ref)
                if loaded {
                    Text("✓ loaded")
                        .font(.caption2).foregroundColor(.green)
                } else {
                    Text("⚠ not loaded")
                        .font(.caption2).foregroundColor(.orange)
                }
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

/// True if the model is currently loaded in MLXEngine. Reads the same
/// `loaded_models` SQLite table that `InferenceService.saveLoadedModelsList`
/// writes to on every load/unload, so it reflects live state without
/// needing the InferenceService instance threaded through the view hierarchy.
private func isLocalModelLoaded(_ modelId: String) -> Bool {
    let loaded = (try? NovaDB.shared.loadedModelsStore.list()) ?? []
    return loaded.contains(modelId)
}
