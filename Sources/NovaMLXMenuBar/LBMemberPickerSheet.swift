import SwiftUI
import NovaMLXCore
import NovaMLXDB
import NovaMLXModelManager
import NovaMLXUtils

// MARK: - LBMemberPickerSheet (Task 11)

/// Multi-select sheet for adding members to an LB.
///
/// Two tabs:
/// - **Local** — models whose weights are on disk (read from `ModelRegistryStore`,
///   the same source `ModelManager.loadRegistry()` reads at startup and writes back
///   to on every change, so the picker sees fresh state without depending on the
///   `ModelManager` instance owned by `NovaAppView`).
/// - **Remote** — enabled `TokenhubProvider`s. Stores `provider.name` as the ref
///   because `LBProxy` dispatches remote members as `"tknet:" + ref`, and
///   `TokenhubManager.resolve(modelName:)` looks providers up by name.
///
/// Members already attached to this LB are shown greyed-out and unselectable.
struct LBMemberPickerSheet: View {
    let lbId: UUID
    let onAdded: ([LBMember]) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab: Tab = .local
    @State private var localModels: [String] = []                       // downloaded model IDs
    @State private var remoteProviders: [(name: String, ref: String)] = []
    @State private var existingMemberRefs: Set<String> = []
    /// Per-tab selection. Critical: a single shared `Set<String>` would
    /// conflate local model_ids with remote provider names and stamp them
    /// all with the active tab's `kind` on Add (bug: switching Local→Remote
    /// and adding would mark all prior local picks as `.remote`).
    @State private var selectedByTab: [Tab: Set<String>] = [.local: [], .remote: []]
    @State private var searchText: String = ""

    private var selectedCount: Int {
        (selectedByTab[.local]?.count ?? 0) + (selectedByTab[.remote]?.count ?? 0)
    }

    enum Tab: String, CaseIterable, Identifiable {
        case local = "Local", remote = "Remote"
        var id: String { rawValue }
    }

    /// Case-insensitive substring match against the search field. Empty query
    /// matches everything. Applies to whichever tab is active so the single
    /// search bar above the segmented control filters Local and Remote alike.
    private func matchesSearch(_ candidate: String) -> Bool {
        let q = searchText.trimmingCharacters(in: .whitespaces).lowercased()
        if q.isEmpty { return true }
        return candidate.lowercased().contains(q)
    }

    private var filteredLocalModels: [String] {
        localModels.filter { matchesSearch($0) }
    }

    private var filteredRemoteProviders: [(name: String, ref: String)] {
        remoteProviders.filter { matchesSearch($0.name) || matchesSearch($0.ref) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Add members").font(.title3.bold())

            // Single shared search bar — filters whichever tab is active.
            TextField("Search local models and remote providers…", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()

            Picker("", selection: $selectedTab) {
                ForEach(Tab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            switch selectedTab {
            case .local:  localList
            case .remote: remoteList
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Add \(selectedCount)") { addSelected() }
                    .buttonStyle(.borderedProminent)
                    .disabled(selectedCount == 0)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 500, minHeight: 400)
        .task { await reload() }
    }

    // MARK: - Local tab

    private var localList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                if localModels.isEmpty {
                    emptyHint("No downloaded local models. Use Download Models to fetch one first.")
                } else if filteredLocalModels.isEmpty {
                    emptyHint("No local models match \"\(searchText)\".")
                } else {
                    ForEach(filteredLocalModels, id: \.self) { modelId in
                        memberRow(
                            title: modelId,
                            subtitle: nil,
                            key: modelId,
                            badgeColor: .green,
                            badgeText: "LOCAL"
                        )
                    }
                }
            }
        }
    }

    // MARK: - Remote tab

    private var remoteList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                if remoteProviders.isEmpty {
                    emptyHint("No enabled remote providers. Add one on the TokenHub page first.")
                } else if filteredRemoteProviders.isEmpty {
                    emptyHint("No remote providers match \"\(searchText)\".")
                } else {
                    ForEach(filteredRemoteProviders, id: \.ref) { p in
                        memberRow(
                            title: p.name,
                            subtitle: p.ref,
                            key: p.ref,
                            badgeColor: .orange,
                            badgeText: "REMOTE"
                        )
                    }
                }
            }
        }
    }

    // MARK: - Row view

    @ViewBuilder
    private func memberRow(
        title: String, subtitle: String?, key: String,
        badgeColor: Color, badgeText: String
    ) -> some View {
        let isExisting = existingMemberRefs.contains(key)
        let isSelected = selectedByTab[selectedTab]?.contains(key) ?? false

        HStack(spacing: 10) {
            Image(systemName: isSelected ? "checkmark.square" : "square")
                .foregroundColor(isExisting ? .secondary : .accentColor)

            Text(badgeText)
                .font(.caption2.bold())
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(badgeColor.opacity(0.15))
                .foregroundColor(badgeColor)
                .clipShape(Capsule())

            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.caption)
                if let subtitle, !subtitle.isEmpty, subtitle != title {
                    Text(subtitle).font(.caption2.monospaced()).foregroundColor(.secondary)
                }
            }

            Spacer()

            if isExisting {
                Text("already added")
                    .font(.caption2).foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(isExisting ? Color.gray.opacity(0.15) : Color.clear)
        .contentShape(Rectangle())
        .opacity(isExisting ? 0.6 : 1.0)
        .onTapGesture {
            guard !isExisting else { return }
            var current = selectedByTab[selectedTab] ?? []
            if current.contains(key) {
                current.remove(key)
            } else {
                current.insert(key)
            }
            selectedByTab[selectedTab] = current
        }
    }

    private func emptyHint(_ text: String) -> some View {
        Text(text)
            .font(.caption).foregroundColor(.secondary)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.top, 40)
    }

    // MARK: - Actions

    private func addSelected() {
        // Each tab's picks get that tab's kind. Picks made in the *other*
        // tab (still in selectedByTab) are also added with their own kind,
        // so cross-tab selection is preserved without the kind-stamping bug.
        var added: [LBMember] = []
        let locals = selectedByTab[.local] ?? []
        let remotes = selectedByTab[.remote] ?? []
        for ref in locals {
            let m = LBMember(lbId: lbId, kind: .local, ref: ref)
            do {
                try NovaDB.shared.lbMemberStore.upsertMember(m)
                added.append(m)
            } catch {
                NovaMLXLog.error("[LBMemberPicker] add local failed: \(error)")
            }
        }
        for ref in remotes {
            let m = LBMember(lbId: lbId, kind: .remote, ref: ref)
            do {
                try NovaDB.shared.lbMemberStore.upsertMember(m)
                added.append(m)
            } catch {
                NovaMLXLog.error("[LBMemberPicker] add remote failed: \(error)")
            }
        }
        onAdded(added)
        dismiss()
    }

    private func reload() async {
        // Local: ModelRegistryStore is SQLite — the same source ModelManager reads
        // at startup and writes to on every registry mutation. Filter to records
        // that have completed downloads (downloadedAt != nil), matching
        // ModelManager.downloadedModels() but without needing the ModelManager
        // instance threaded through the view hierarchy.
        if let registry = try? NovaDB.shared.modelRegistryStore.listAsRegistry() {
            localModels = registry.values
                .filter { $0.downloadedAt != nil }
                .map { $0.id }
                .sorted()
        }

        // Remote: enabled providers. ref = provider.name because LBProxy dispatches
        // remote members as "tknet:<ref>" and TokenhubManager.resolve() looks
        // providers up by name (get(name:) → record keyed on the name column).
        let allProviders = (try? NovaDB.shared.tokenhubStore.listAsProviders()) ?? []
        remoteProviders = allProviders
            .filter { $0.isEnabled }
            .map { (name: $0.name, ref: $0.name) }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }

        // Existing members (to grey out in the picker). Match on ref across both
        // kinds — a local model id and a remote provider name could in principle
        // collide, but in practice they don't (model ids are repo paths like
        // "org/model"; provider names are short display names like "OpenAI").
        let existing = (try? NovaDB.shared.lbMemberStore.listMembers(lbId: lbId)) ?? []
        existingMemberRefs = Set(existing.map(\.ref))
    }
}
