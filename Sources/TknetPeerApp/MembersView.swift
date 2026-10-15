import AppKit
import TknetPeer
import SwiftUI

/// Left pane selection: `all` or one concrete factory id.
private enum FactorySelection: Hashable {
    case all
    case factory(String)

    var factoryId: String? {
        if case .factory(let id) = self { return id }
        return nil
    }
}

struct MembersView: View {
    @EnvironmentObject private var state: PeerAppState
    @State private var selectedFactory: FactorySelection = .all
    @State private var selectedMemberId: String?
    @State private var factoryDraft: FactoryDraft?
    @State private var memberDraft: MemberDraft?
    @State private var pendingDeleteFactory: FactoryConfig?
    @State private var pendingDeleteMember: MemberConfig?
    @State private var importNotice: String?

    private var factories: [FactoryConfig] { state.config.factories }

    private var visibleMembers: [MemberConfig] {
        switch selectedFactory {
        case .all:
            return state.config.members
        case .factory(let id):
            guard let factory = factories.first(where: { $0.id == id }) else { return [] }
            return state.config.members(in: factory)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Members")
                    .font(.headline)
                Spacer()
                Button("Add Factory") { factoryDraft = FactoryDraft() }
                    .disabled(state.lockError != nil)
                Button("Add member") { openMemberDraft() }
                    .disabled(state.lockError != nil || selectedFactory.factoryId == nil)
            }
            HStack(spacing: 0) {
                factoryPane
                Divider()
                memberPane
            }
            .frame(maxHeight: .infinity)
        }
        .sheet(item: $factoryDraft) { draft in
            FactoryForm(draft: draft)
        }
        .sheet(item: $memberDraft) { draft in
            MemberForm(draft: draft)
        }
        .confirmationDialog(
            "Delete member \(pendingDeleteMember.map { MemberLabel.title($0, among: state.config.members, config: state.config) } ?? "member")?",
            isPresented: Binding(
                get: { pendingDeleteMember != nil },
                set: { if !$0 { pendingDeleteMember = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let id = pendingDeleteMember?.id {
                    Task { await state.deleteMember(id: id) }
                    if selectedMemberId == id { selectedMemberId = nil }
                }
                pendingDeleteMember = nil
            }
            Button("Cancel", role: .cancel) { pendingDeleteMember = nil }
        }
        .confirmationDialog(
            "Delete factory \(pendingDeleteFactory?.name ?? "")? Its members go with it.",
            isPresented: Binding(
                get: { pendingDeleteFactory != nil },
                set: { if !$0 { pendingDeleteFactory = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                deleteFactory()
            }
            Button("Cancel", role: .cancel) { pendingDeleteFactory = nil }
        }
    }

    // MARK: - Left pane (factories)

    private var factoryPane: some View {
        VStack(spacing: 0) {
            List(selection: Binding(
                get: { selectedFactory },
                set: {
                    selectedFactory = $0 ?? .all
                    selectedMemberId = nil
                })) {
                HStack {
                    Text("All")
                        .font(.body.weight(.medium))
                    Spacer()
                    Button("Test") { testAllMembers() }
                        .buttonStyle(.link)
                        .controlSize(.small)
                        .disabled(state.lockError != nil || factoryTestingIds.contains(allFactoryId))
                        .accessibilityIdentifier("all-test")
                }
                .tag(FactorySelection.all)
                if factories.isEmpty {
                    // No factories yet: visible but never selectable.
                    Text("(none)")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(factories) { factory in
                        HStack {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(factory.name).font(.body)
                                Text(factory.displayURL)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer()
                            Text("\(state.config.members(in: factory).count)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Button("Test") {
                                selectFactory(factory.id)
                                testFactoryMembers(factory)
                            }
                            .buttonStyle(.link)
                            .controlSize(.small)
                            .disabled(state.lockError != nil || factoryTestingIds.contains(factory.id))
                            .accessibilityIdentifier("factory-test")
                            Button("Edit") {
                                selectFactory(factory.id)
                                factoryDraft = FactoryDraft(factory)
                            }
                            .buttonStyle(.link)
                            .controlSize(.small)
                            .disabled(state.lockError != nil)
                            .accessibilityIdentifier("factory-edit")
                        }
                        .tag(FactorySelection.factory(factory.id))
                    }
                }
            }
            .frame(width: 220)
            paneControls(
                minusAction: {
                    if let id = selectedFactory.factoryId,
                       let factory = factories.first(where: { $0.id == id }) {
                        pendingDeleteFactory = factory
                    }
                },
                minusEnabled: selectedFactory.factoryId != nil && state.lockError == nil,
                plusAction: { factoryDraft = FactoryDraft() },
                plusEnabled: state.lockError == nil,
                minusID: "factory-remove", plusID: "factory-add")
        }
    }

    // MARK: - Right pane (members of the selection)

    private var memberPane: some View {
        VStack(spacing: 0) {
            if let importNotice {
                Text(importNotice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 8)
                    .padding(.top, 4)
            }
            if visibleMembers.isEmpty {
                Spacer()
                Text(emptyMembersText)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                Spacer()
            } else {
                List(selection: $selectedMemberId) {
                    ForEach(visibleMembers) { member in
                        memberRow(member)
                            .tag(member.id)
                    }
                }
            }
            paneControls(
                minusAction: {
                    if let id = selectedMemberId,
                       let member = state.config.members.first(where: { $0.id == id }) {
                        pendingDeleteMember = member
                    }
                },
                minusEnabled: selectedMemberId != nil && state.lockError == nil,
                plusAction: { openMemberDraft() },
                // "+" only when a concrete factory (not All) is selected.
                plusEnabled: selectedFactory.factoryId != nil && state.lockError == nil,
                minusID: "member-remove", plusID: "member-add",
                importAction: selectedFactory.factoryId == nil ? nil : { importModels() },
                importNotice: importNotice)
        }
    }

    // MARK: - Member testing

    /// Probe results keyed by member id, shared by row buttons and the
    /// factory-wide "Test" so a failed probe leaves a warning icon behind.
    @State private var memberTestResults: [String: MemberProbeResult] = [:]
    @State private var testingMemberIds: Set<String> = []
    @State private var factoryTestingIds: Set<String> = []

    private func testMember(_ member: MemberConfig) {
        guard !testingMemberIds.contains(member.id) else { return }
        testingMemberIds.insert(member.id)
        Task {
            memberTestResults[member.id] = await state.probeMember(member)
            testingMemberIds.remove(member.id)
        }
    }

    /// Factory test: probe this factory's members one by one.
    private func testFactoryMembers(_ factory: FactoryConfig) {
        guard !factoryTestingIds.contains(factory.id) else { return }
        factoryTestingIds.insert(factory.id)
        Task {
            for member in state.config.members(in: factory) {
                memberTestResults[member.id] = await state.probeMember(member)
            }
            factoryTestingIds.remove(factory.id)
        }
    }

    @ViewBuilder
    private func memberRow(_ member: MemberConfig) -> some View {
        MemberRowView(
            member: member,
            result: memberTestResults[member.id],
            isTesting: testingMemberIds.contains(member.id),
            onEdit: { memberDraft = MemberDraft(member, factories: factories) },
            onTest: { testMember(member) })
    }

    // MARK: - Actions

    /// Bulk-add one member per model the selected factory advertises.
    private func importModels() {
        guard let id = selectedFactory.factoryId,
              let factory = factories.first(where: { $0.id == id }) else { return }
        importNotice = "Importing…"
        Task {
            if let outcome = await state.importMembersFromFactory(factory) {
                var text = outcome.added == 0
                    ? "No new models."
                    : "Added \(outcome.added) models."
                if outcome.skipped > 0 {
                    text += " Skipped \(outcome.skipped) draft models."
                }
                importNotice = text
            } else {
                importNotice = state.lastError ?? "Import failed."
            }
        }
    }

    /// Sentinel id so "All" can ride the same testing-disable set.
    private let allFactoryId = "__all__"

    /// Test every member across every factory, one by one.
    private func testAllMembers() {
        guard !factoryTestingIds.contains(allFactoryId) else { return }
        factoryTestingIds.insert(allFactoryId)
        Task {
            for member in state.config.members {
                memberTestResults[member.id] = await state.probeMember(member)
            }
            factoryTestingIds.remove(allFactoryId)
        }
    }

    /// Row buttons (Test/Edit) select the factory first, like a row click.
    private func selectFactory(_ id: String) {
        selectedFactory = .factory(id)
        selectedMemberId = nil
    }

    private func openMemberDraft() {
        guard let id = selectedFactory.factoryId,
              let factory = factories.first(where: { $0.id == id }) else { return }
        memberDraft = MemberDraft(factory: factory)
    }

    private func deleteFactory() {
        guard let factory = pendingDeleteFactory else { return }
        if selectedFactory == .factory(factory.id) {
            selectedFactory = .all
            selectedMemberId = nil
        }
        pendingDeleteFactory = nil
        Task { await state.deleteFactory(id: factory.id) }
    }

    private var emptyMembersText: String {
        switch selectedFactory {
        case .all:
            return factories.isEmpty
                ? "No members yet. Add a Factory, then add members to it."
                : "No members yet."
        case .factory(let id):
            guard let factory = factories.first(where: { $0.id == id }) else { return "" }
            return "No members in \(factory.name). Use + or Import models."
        }
    }

    private func paneControls(
        minusAction: @escaping () -> Void, minusEnabled: Bool,
        plusAction: @escaping () -> Void, plusEnabled: Bool,
        minusID: String, plusID: String,
        importAction: (() -> Void)? = nil,
        importNotice: String? = nil
    ) -> some View {
        HStack(spacing: 4) {
            Button(action: minusAction) {
                Image(systemName: "minus")
            }
            .disabled(!minusEnabled)
            .accessibilityIdentifier(minusID)
            Button(action: plusAction) {
                Image(systemName: "plus")
            }
            .disabled(!plusEnabled)
            .accessibilityIdentifier(plusID)
            if let importAction {
                Button("Import models", action: importAction)
                    .buttonStyle(.link)
                    .controlSize(.small)
                    .disabled(state.lockError != nil || importNotice == "Importing…")
                    .accessibilityIdentifier("member-import")
            }
            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.bar)
    }
}

/// Title for a member. The host suffix only appears when the same model name
/// is reachable through different factory hosts — otherwise it is noise.
enum MemberLabel {
    static func title(_ member: MemberConfig, among members: [MemberConfig], config: PeerConfig) -> String {
        let model = PeerAdmin.publicModel(member)
        let sharers = members.filter { PeerAdmin.publicModel($0) == model }
        guard sharers.count > 1 else { return model }
        let hosts = Set(sharers.compactMap { config.memberFactoryBase($0)?.host })
        guard hosts.count > 1,
              let host = config.memberFactoryBase(member)?.host, !host.isEmpty else { return model }
        return "\(model) · \(host)"
    }
}

/// One member row: model + interface lines on the left, Test/Edit links on
/// the right. The probe result comes from the parent so the factory-wide
/// test can fill it in too; a failed probe leaves a warning icon.
private struct MemberRowView: View {
    @EnvironmentObject private var state: PeerAppState
    let member: MemberConfig
    let result: MemberProbeResult?
    let isTesting: Bool
    let onEdit: () -> Void
    let onTest: () -> Void

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    // Click copies the model name (without any "· host" suffix).
                    Button {
                        copyToPasteboard(PeerAdmin.publicModel(member))
                    } label: {
                        Text(MemberLabel.title(member, among: state.config.members, config: state.config))
                            .font(.body.bold())
                    }
                    .buttonStyle(.plain)
                    .help("Click to copy the model name")
                    if let result, !result.ok {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help(result.message)
                    }
                }
                Button {
                    copyToPasteboard(fullEndpointURL)
                } label: {
                    Text(fullEndpointURL)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Click to copy the endpoint URL")
                Text(detailLine)
                    .font(.caption)
                Text(member.isEnabled ? "Enabled" : "Disabled")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if isTesting {
                    Text("Testing…")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else if let result {
                    Text(result.ok ? "Test ✓ \(result.message)" : "Test ✗ \(result.message)")
                        .font(.caption2)
                        .foregroundStyle(result.ok ? Color.green : Color.red)
                        .textSelection(.enabled)
                }
            }
            Spacer()
            HStack(spacing: 10) {
                Button("Test", action: onTest)
                    .buttonStyle(.link)
                    .disabled(isTesting)
                    .accessibilityIdentifier("member-test")
                Button("Edit", action: onEdit)
                    .buttonStyle(.link)
            }
        }
        .padding(.vertical, 4)
    }

    /// Full forward URL: the factory base plus the interface's canonical path.
    private var fullEndpointURL: String {
        state.config.memberEndpointURL(member)?.absoluteString ?? "(no interface)"
    }

    private var detailLine: String {
        let model = PeerAdmin.publicModel(member)
        let kind = PeerAdmin.interfaceKind(of: member, in: state.config)?.displayName ?? "?"
        if member.upstreamModel != model {
            return "\(kind) · upstream \(member.upstreamModel)"
        }
        return kind
    }

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

// MARK: - Factory form

private struct FactoryDraft: Identifiable {
    let token = UUID()
    var id: UUID { token }
    var factoryId: String?
    var name: String
    var url: String
    var isNew: Bool

    init() {
        factoryId = nil
        name = ""
        url = "https://"
        isNew = true
    }

    init(_ factory: FactoryConfig) {
        factoryId = factory.id
        name = factory.name
        url = factory.displayURL
        isNew = false
    }
}

private struct FactoryForm: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var state: PeerAppState
    let draft: FactoryDraft

    @State private var name: String
    @State private var url: String
    @State private var apiKey: String
    @State private var formError: String?
    @State private var revealKey = false
    /// The stored key as loaded, so Save can skip a no-op rewrite.
    @State private var originalKey = ""

    init(draft: FactoryDraft) {
        self.draft = draft
        _name = State(initialValue: draft.name)
        _url = State(initialValue: draft.url)
        _apiKey = State(initialValue: "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(draft.isNew ? "Add Factory" : "Edit Factory")
                .font(.title3.bold())
            TextField("Name", text: $name)
            TextField("URL (host:port)", text: $url)
            HStack(spacing: 6) {
                Group {
                    if revealKey {
                        TextField("API key", text: $apiKey)
                    } else {
                        SecureField("API key (optional)", text: $apiKey)
                    }
                }
                Button {
                    revealKey.toggle()
                } label: {
                    Image(systemName: revealKey ? "eye.slash" : "eye")
                }
                .buttonStyle(.link)
                .help(revealKey ? "Hide the key" : "Reveal the key")
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(apiKey, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.link)
                .disabled(apiKey.isEmpty)
                .help("Copy the key")
            }
            Text("ALL-UPPERCASE values are looked up in \(state.envFileURL.path).")
                .font(.caption2)
                .foregroundStyle(.secondary)
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
        .frame(width: 420)
        .onAppear {
            guard !draft.isNew, let id = draft.factoryId,
                  let factory = state.config.factory(id: id) else { return }
            if let stored = state.factoryStoredKey(factory) {
                apiKey = stored
                originalKey = stored
            }
        }
    }

    private func save() async {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            formError = "Factory name is required."
            return
        }
        guard let base = URL(string: url.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            formError = "Factory URL must be an http(s) host[:port]."
            return
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if let problem = state.envKeyProblem(trimmedKey) {
            formError = problem
            return
        }
        // Unchanged (or untouched new) key: keep the stored one.
        let key: String? = (apiKey == originalKey) ? nil : trimmedKey
        let saved = await state.saveFactory(
            id: draft.factoryId, name: trimmedName, baseUrl: base, apiKey: key)
        if saved {
            dismiss()
        } else {
            formError = state.lastError ?? "Could not save the factory."
        }
    }
}

// MARK: - Member form

/// A member is (interface, model). The interface carries the full endpoint
/// URL and calling template; the model names what to invoke on it.
private struct MemberDraft: Identifiable {
    let token = UUID()
    var id: UUID { token }
    var memberId: String?
    var model: String
    var upstreamOverride: String
    /// The (factory, kind) pair this member is invoked through.
    var factoryId: String
    var kind: InterfaceKind
    var weight: String
    var isEnabled: Bool
    var isNew: Bool
    /// Factories available for reassignment while editing.
    var factories: [FactoryConfig]
    /// Member's stored key (loaded on appear); nil until then.
    var memberAPIKey: String?

    init(factory: FactoryConfig) {
        memberId = nil
        model = ""
        upstreamOverride = ""
        factoryId = factory.id
        kind = .openaiChat
        weight = ""
        isEnabled = true
        isNew = true
        factories = []
        memberAPIKey = nil
    }

    init(_ member: MemberConfig, factories: [FactoryConfig]) {
        let model = PeerAdmin.publicModel(member)
        memberId = member.id
        self.model = model
        // Member key is loaded in the form's onAppear (needs state).
        memberAPIKey = nil
        upstreamOverride = member.upstreamModel == model ? "" : member.upstreamModel
        let interface = factories.flatMap(\.interfaces).first { $0.id == member.interfaceId }
        factoryId = factories.first {
            $0.interfaces.contains { $0.id == member.interfaceId }
        }?.id ?? ""
        kind = interface?.kind ?? .openaiChat
        weight = member.weight.map(String.init) ?? ""
        isEnabled = member.isEnabled
        isNew = false
        self.factories = factories
    }
}

private struct MemberForm: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var state: PeerAppState
    let draft: MemberDraft

    @State private var model: String
    @State private var upstreamOverride: String
    @State private var factoryId: String
    @State private var kind: InterfaceKind
    @State private var weight: String
    @State private var isEnabled: Bool
    @State private var showMore: Bool
    @State private var formError: String?
    /// Test outcome for the (interface, model) as currently entered.
    @State private var testResult: MemberProbeResult?
    @State private var isTesting = false
    /// Blank = use the factory key; a value overrides it.
    @State private var apiKey = ""
    @State private var revealKey = false
    @State private var originalKey: String?

    init(draft: MemberDraft) {
        self.draft = draft
        _model = State(initialValue: draft.model)
        _upstreamOverride = State(initialValue: draft.upstreamOverride)
        _factoryId = State(initialValue: draft.factoryId)
        _kind = State(initialValue: draft.kind)
        _weight = State(initialValue: draft.weight)
        _isEnabled = State(initialValue: draft.isEnabled)
        let hasMore = !draft.upstreamOverride.isEmpty || !draft.weight.isEmpty
        _showMore = State(initialValue: hasMore)
    }

    private var selectedFactory: FactoryConfig? {
        draft.factories.first { $0.id == factoryId }
            ?? state.config.factory(id: factoryId)
    }

    /// New members may only be saved after a passing bare-call test; edits
    /// stay open so an offline factory never blocks unrelated changes.
    private var saveDisabled: Bool {
        draft.isNew && testResult?.ok != true
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(draft.isNew ? "Add member" : "Edit member")
                .font(.title3.bold())
            TextField("Model", text: $model)
                .onChange(of: model) { _, _ in testResult = nil }
            if draft.isNew {
                // Adding from a factory pane: the factory is fixed.
                if let factory = selectedFactory {
                    LabeledContent("Factory") {
                        Text(factory.name).foregroundStyle(.secondary)
                    }
                }
            } else {
                // Editing: allow moving between factories.
                Picker("Factory", selection: $factoryId) {
                    ForEach(draft.factories) { factory in
                        Text(factory.name).tag(factory.id)
                    }
                }
                .onChange(of: factoryId) { _, _ in testResult = nil }
            }
            Picker("Interface", selection: $kind) {
                ForEach(InterfaceKind.memberSelectable) { kind in
                    Text(kind.displayName).tag(kind)
                }
            }
            .onChange(of: kind) { _, _ in testResult = nil }
            if let factory = selectedFactory, let url = factory.endpointURL(of: kind) {
                Text(url.absoluteString)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            HStack(spacing: 6) {
                Group {
                    if revealKey {
                        TextField("API key (blank uses the factory key)", text: $apiKey)
                    } else {
                        SecureField("API key (blank uses the factory key)", text: $apiKey)
                    }
                }
                .onChange(of: apiKey) { _, _ in testResult = nil }
                Button {
                    revealKey.toggle()
                } label: {
                    Image(systemName: revealKey ? "eye.slash" : "eye")
                }
                .buttonStyle(.link)
                .help(revealKey ? "Hide the key" : "Reveal the key")
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(apiKey, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.link)
                .disabled(apiKey.isEmpty)
                .help("Copy the key")
            }
            Text("Blank uses the factory's key; ALL-UPPERCASE values resolve from \(state.envFileURL.path).")
                .font(.caption2)
                .foregroundStyle(.secondary)
            DisclosureGroup("More", isExpanded: $showMore) {
                VStack(alignment: .leading, spacing: 8) {
                    TextField("Upstream model (blank uses Model)", text: $upstreamOverride)
                    TextField("Weight (optional)", text: $weight)
                }
                .padding(.top, 4)
            }
            if let formError {
                Text(formError).foregroundStyle(.red).font(.callout)
            }
            if let testResult {
                Text(testResult.ok
                     ? "Test ✓ \(testResult.message)"
                     : "Test ✗ \(testResult.message)")
                    .font(.callout)
                    .foregroundStyle(testResult.ok ? Color.green : Color.red)
            }
            HStack {
                Button("Test") { Task { await runTest() } }
                    .disabled(isTesting || !canTest)
                    .accessibilityIdentifier("member-form-test")
                if draft.isNew && saveDisabled {
                    Text("Test the member to enable Save.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { Task { await save() } }
                    .disabled(saveDisabled)
            }
        }
        .padding(20)
        .frame(width: 460)
        .onAppear {
            guard !draft.isNew, let id = draft.memberId,
                  let member = state.config.member(id: id) else { return }
            let stored = state.memberStoredKey(member) ?? ""
            apiKey = stored
            originalKey = stored
        }
    }

    private var canTest: Bool {
        selectedFactory != nil
            && !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func runTest() async {
        // Save the dialog as-is first, then test the persisted member: the
        // test must exercise exactly what would be saved — including key
        // inheritance, which a form-local probe cannot see.
        guard let memberId = await persist() else { return }
        originalKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        isTesting = true
        defer { isTesting = false }
        guard let member = state.config.member(id: memberId) else {
            formError = "The member vanished before it could be tested."
            return
        }
        testResult = await state.probeMember(member)
        formError = nil
    }

    /// Model on the wire: the override when present, else the typed name.
    private var testModelName: String {
        let override = upstreamOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        if !override.isEmpty { return override }
        return model.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func save() async {
        guard await persist() != nil else { return }
        dismiss()
    }

    /// Validates and writes the dialog; returns the member id, nil on error
    /// with `formError` set.
    private func persist() async -> String? {
        let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModel.isEmpty else {
            formError = "Model name is required."
            return nil
        }
        guard let factoryId = selectedFactory?.id else {
            formError = "No factory selected."
            return nil
        }
        let weightValue: Int?
        let trimmedWeight = weight.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedWeight.isEmpty {
            weightValue = nil
        } else if let parsed = Int(trimmedWeight), parsed >= 1 {
            weightValue = parsed
        } else {
            formError = "Weight must be a whole number of at least 1."
            return nil
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if let problem = state.envKeyProblem(trimmedKey) {
            formError = problem
            return nil
        }
        // nil = unchanged on edit; new members: "" (empty) = inherit factory.
        let keyValue: String? = draft.isNew
            ? trimmedKey
            : (trimmedKey == originalKey ? nil : trimmedKey)
        return await state.saveMember(
            existingId: draft.memberId,
            model: trimmedModel,
            upstreamOverride: upstreamOverride,
            factoryId: factoryId,
            kind: kind,
            apiKey: keyValue,
            weight: weightValue,
            isEnabled: isEnabled)
    }
}
