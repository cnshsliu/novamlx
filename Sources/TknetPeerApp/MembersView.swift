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
            "Delete member \(pendingDeleteMember.map { MemberLabel.title($0, among: state.config.members) } ?? "member")?",
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
            "Delete factory \(pendingDeleteFactory?.name ?? "")?",
            isPresented: Binding(
                get: { pendingDeleteFactory != nil },
                set: { if !$0 { pendingDeleteFactory = nil } }),
            titleVisibility: .visible
        ) {
            Button("Keep members (ungrouped)", role: .destructive) {
                deleteFactory(deleteMembers: false)
            }
            Button("Delete members too", role: .destructive) {
                deleteFactory(deleteMembers: true)
            }
            Button("Cancel", role: .cancel) { pendingDeleteFactory = nil }
        }
    }

    // MARK: - Left pane (factories)

    private var factoryPane: some View {
        VStack(spacing: 0) {
            List(selection: Binding(
                get: { selectedFactory },
                set: { selectedFactory = $0 ?? .all; selectedMemberId = nil })) {
                Text("All")
                    .tag(FactorySelection.all)
                    .font(.body.weight(.medium))
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

    /// Bulk-add one member per model the selected factory advertises.
    private func importModels() {
        guard let id = selectedFactory.factoryId,
              let factory = factories.first(where: { $0.id == id }) else { return }
        importNotice = "Importing…"
        Task {
            if let added = await state.importMembersFromFactory(factory) {
                importNotice = added == 0
                    ? "No new models."
                    : "Added \(added) models."
            } else {
                importNotice = state.lastError ?? "Import failed."
            }
        }
    }

    @ViewBuilder
    private func memberRow(_ member: MemberConfig) -> some View {
        MemberRowView(member: member, factories: factories) {
            memberDraft = MemberDraft(member, factories: factories)
        }
    }

    private var emptyMembersText: String {
        switch selectedFactory {
        case .all:
            return factories.isEmpty
                ? "No members yet. Add a Factory, then add members to it."
                : "No members yet."
        case .factory(let id):
            guard let factory = factories.first(where: { $0.id == id }) else { return "" }
            return "No members in \(factory.name). Use + to add one."
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
            if let importNotice {
                Text(importNotice)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.bar)
    }

    private func openMemberDraft() {
        guard let id = selectedFactory.factoryId,
              let factory = factories.first(where: { $0.id == id }) else { return }
        memberDraft = MemberDraft(factory: factory, defaultKey: state.factoryDefaultKey(factory))
    }

    private func deleteFactory(deleteMembers: Bool) {
        guard let factory = pendingDeleteFactory else { return }
        if selectedFactory == .factory(factory.id) {
            selectedFactory = .all
            selectedMemberId = nil
        }
        pendingDeleteFactory = nil
        Task { await state.deleteFactory(id: factory.id, deleteMembers: deleteMembers) }
    }
}

enum MemberLabel {
    static func title(_ member: MemberConfig, among members: [MemberConfig]) -> String {
        let model = PeerAdmin.publicModel(member)
        let sharers = members.filter { PeerAdmin.publicModel($0) == model }
        guard sharers.count > 1 else { return model }
        // The host suffix exists to tell the same model apart across different
        // hosts. When every sharer sits on one host (duplicate entries of one
        // endpoint) it disambiguates nothing — every row would grow a
        // "· 127.0.0.1" that just wraps and reads like a broken URL.
        let hosts = Set(sharers.compactMap(\.endpoint.host))
        guard hosts.count > 1, let host = member.endpoint.host, !host.isEmpty else { return model }
        return "\(model) · \(host)"
    }
}

/// One member row: identity lines on the left, Test/Edit links on the right.
/// Test state is per-row so several members can be checked independently.
private struct MemberRowView: View {
    @EnvironmentObject private var state: PeerAppState
    let member: MemberConfig
    let factories: [FactoryConfig]
    let onEdit: () -> Void

    private enum TestState {
        case idle
        case running
        case done(MemberProbeResult)
    }

    @State private var testState: TestState = .idle

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(MemberLabel.title(member, among: state.config.members))
                    .font(.body.bold())
                Text(member.endpoint.absoluteString)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Text(detailLine)
                    .font(.caption)
                Text(member.isEnabled ? "Enabled" : "Disabled")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                switch testState {
                case .idle:
                    EmptyView()
                case .running:
                    Text("Testing…")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                case .done(let result):
                    Text(result.ok ? "Test ✓ \(result.message)" : "Test ✗ \(result.message)")
                        .font(.caption2)
                        .foregroundStyle(result.ok ? Color.green : Color.red)
                        .textSelection(.enabled)
                }
            }
            Spacer()
            HStack(spacing: 10) {
                Button("Test") { Task { await runTest() } }
                    .buttonStyle(.link)
                    .disabled(isRunning)
                    .accessibilityIdentifier("member-test")
                Button("Edit", action: onEdit)
                    .buttonStyle(.link)
            }
        }
        .padding(.vertical, 4)
    }

    private var isRunning: Bool {
        if case .running = testState { return true }
        return false
    }

    private var detailLine: String {
        let model = PeerAdmin.publicModel(member)
        let type = typeLabel(member.type)
        if member.upstreamModel != model {
            return "\(type) · upstream \(member.upstreamModel)"
        }
        return type
    }

    private func runTest() async {
        testState = .running
        testState = .done(await state.probeMember(member))
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
            SecureField(draft.isNew ? "API key (optional)" : "API key (blank keeps the current key)", text: $apiKey)
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
        let key = apiKey.isEmpty ? nil : apiKey
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

private struct MemberDraft: Identifiable {
    let token = UUID()
    var id: UUID { token }
    var memberId: String?
    var model: String
    var upstreamOverride: String
    var type: SourceType
    var factoryId: String?
    /// Path on the factory base (leading "/"), or the full URL when ungrouped.
    var endpoint: String
    var apiKey: String
    var weight: String
    var isEnabled: Bool
    var isNew: Bool
    /// Factories available for reassignment while editing.
    var factories: [FactoryConfig]

    init(factory: FactoryConfig, defaultKey: String?) {
        memberId = nil
        model = ""
        upstreamOverride = ""
        type = .openaiCompatible
        factoryId = factory.id
        endpoint = "/v1"
        apiKey = defaultKey ?? ""
        weight = ""
        isEnabled = true
        isNew = true
        factories = []
    }

    init(_ member: MemberConfig, factories: [FactoryConfig]) {
        let model = PeerAdmin.publicModel(member)
        memberId = member.id
        self.model = model
        upstreamOverride = member.upstreamModel == model ? "" : member.upstreamModel
        type = member.type
        factoryId = member.factoryId
        if let factoryId = member.factoryId,
           let factory = factories.first(where: { $0.id == factoryId }) {
            let base = factory.baseUrl.absoluteString
            var path = member.endpoint.absoluteString
            if path.hasPrefix(base) { path.removeFirst(base.count) }
            if path.isEmpty { path = "/" }
            endpoint = path
        } else {
            endpoint = member.endpoint.absoluteString
        }
        apiKey = ""
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
    @State private var type: SourceType
    @State private var typeEdited: Bool
    /// nil = ungrouped (full URL field); otherwise the chosen factory id.
    @State private var factoryId: String?
    @State private var endpoint: String
    @State private var apiKey: String
    @State private var weight: String
    @State private var isEnabled: Bool
    @State private var showMore: Bool
    @State private var formError: String?
    /// Test outcome for the endpoint as currently entered. Nil = not tested.
    @State private var testResult: MemberProbeResult?
    @State private var isTesting = false

    init(draft: MemberDraft) {
        self.draft = draft
        _model = State(initialValue: draft.model)
        _upstreamOverride = State(initialValue: draft.upstreamOverride)
        _type = State(initialValue: draft.type)
        _typeEdited = State(initialValue: !draft.isNew)
        _factoryId = State(initialValue: draft.factoryId)
        _endpoint = State(initialValue: draft.endpoint)
        _apiKey = State(initialValue: draft.apiKey)
        _weight = State(initialValue: draft.weight)
        _isEnabled = State(initialValue: draft.isEnabled)
        let hasMore = !draft.upstreamOverride.isEmpty || !draft.weight.isEmpty
        _showMore = State(initialValue: hasMore)
    }

    private var selectedFactory: FactoryConfig? {
        draft.factories.first { $0.id == factoryId }
    }

    /// New members may only be saved after a passing endpoint test; edits
    /// stay open so an offline factory never blocks unrelated changes.
    private var saveDisabled: Bool {
        draft.isNew && testResult?.ok != true
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(draft.isNew ? "Add member" : "Edit member")
                .font(.title3.bold())
            TextField("Model", text: $model)
            if let models = testResult?.models, !models.isEmpty {
                Menu("Choose advertised model…") {
                    ForEach(models, id: \.self) { id in
                        Button(id) { model = id }
                    }
                }
                .controlSize(.small)
            }
            if draft.isNew {
                // Adding from a factory pane: the factory is fixed.
                if let factory = selectedFactory {
                    LabeledContent("Factory") {
                        Text(factory.name).foregroundStyle(.secondary)
                    }
                }
            } else {
                // Editing: allow reassignment between factories / ungrouped.
                Picker("Factory", selection: Binding(
                    get: { factoryId },
                    set: { reassign($0) })) {
                    Text("(none)").tag(String?.none)
                    ForEach(draft.factories) { factory in
                        Text(factory.name).tag(String?.some(factory.id))
                    }
                }
            }
            if selectedFactory != nil {
                TextField("Path (e.g. /v1)", text: $endpoint)
                    .onChange(of: endpoint) { _, _ in testResult = nil }
            } else {
                TextField("Endpoint URL", text: $endpoint)
                    .onChange(of: endpoint) { _, newValue in
                        testResult = nil
                        guard !typeEdited, factoryId == nil else { return }
                        type = inferredType(newValue)
                    }
            }
            SecureField(draft.isNew ? "API key" : "API key (blank keeps the current key)", text: $apiKey)
                .onChange(of: apiKey) { _, _ in testResult = nil }
            HStack(spacing: 16) {
                HStack(spacing: 6) {
                    Text("Type")
                    Picker("Type", selection: Binding(
                        get: { type },
                        set: { type = $0; typeEdited = true }
                    )) {
                        Text("OpenAI-compatible").tag(SourceType.openaiCompatible)
                        Text("Local NovaMLX").tag(SourceType.localNovaMLX)
                        Text("Anthropic").tag(SourceType.anthropic)
                    }
                    .labelsHidden()
                    .fixedSize()
                }
                Toggle("Enabled", isOn: $isEnabled)
                Spacer(minLength: 0)
            }
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
                    .disabled(isTesting || resolvedEndpoint() == nil)
                    .accessibilityIdentifier("member-form-test")
                if draft.isNew && saveDisabled {
                    Text("Test the endpoint to enable Save.")
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
    }

    /// Switching factories converts the field between path and full-URL form.
    private func reassign(_ newId: String?) {
        let oldFactory = selectedFactory
        factoryId = newId
        testResult = nil
        let newFactory = draft.factories.first { $0.id == newId }
        let text = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        if oldFactory != nil, newFactory == nil {
            // path -> full URL
            if let old = oldFactory {
                var path = text
                if !path.hasPrefix("/") { path = "/" + path }
                endpoint = old.baseUrl.absoluteString + path
            }
        } else if oldFactory == nil, let newBase = newFactory?.baseUrl.absoluteString {
            // full URL -> path against the new base
            if text.hasPrefix(newBase) {
                var path = String(text.dropFirst(newBase.count))
                if path.isEmpty { path = "/" }
                endpoint = path
            } else if let url = URL(string: text),
                      PeerConfig.factoryBase(of: url) == PeerConfig.factoryBase(of: URL(string: newBase)!) {
                var path = url.path
                if path.isEmpty { path = "/" }
                if let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.query {
                    path += "?\(query)"
                }
                endpoint = path
            } else if text.hasPrefix("/") {
                endpoint = text
            } else if !text.isEmpty {
                // Different host entirely: keep only the path portion.
                var path = URL(string: text)?.path ?? "/"
                if path.isEmpty { path = "/" }
                endpoint = path
            }
        }
    }

    /// Endpoint URL exactly as `save()` would build it, or nil when invalid.
    private func resolvedEndpoint() -> URL? {
        if let factory = selectedFactory {
            var path = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
            if path.isEmpty { path = "/" }
            if !path.hasPrefix("/") { path = "/" + path }
            return URL(string: factory.baseUrl.absoluteString + path)
        }
        guard let url = URL(string: endpoint.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              url.host != nil else { return nil }
        return url
    }

    /// Key used for the test: the field, else the member's stored key when
    /// editing with a blank field ("blank keeps the current key").
    private func effectiveAPIKey() -> String? {
        if !apiKey.isEmpty { return apiKey }
        if !draft.isNew, let id = draft.memberId,
           let member = state.config.member(id: id) {
            return state.memberAPIKey(member)
        }
        return nil
    }

    private func runTest() async {
        guard let url = resolvedEndpoint() else {
            testResult = MemberProbeResult(
                ok: false, httpStatus: 0, models: [],
                message: "Endpoint must be an http(s) URL.")
            return
        }
        isTesting = true
        defer { isTesting = false }
        testResult = await state.probeEndpoint(url, apiKey: effectiveAPIKey())
        formError = nil
    }

    private func save() async {
        let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModel.isEmpty else {
            formError = "Model name is required."
            return
        }
        guard let endpointURL = resolvedEndpoint() else {
            formError = selectedFactory != nil
                ? "Path is not valid."
                : "Endpoint must be an http(s) URL."
            return
        }
        let weightValue: Int?
        let trimmedWeight = weight.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedWeight.isEmpty {
            weightValue = nil
        } else if let parsed = Int(trimmedWeight), parsed >= 1 {
            weightValue = parsed
        } else {
            formError = "Weight must be a whole number of at least 1."
            return
        }
        let key: String? = apiKey.isEmpty ? (draft.isNew ? "" : nil) : apiKey
        let saved = await state.saveMember(
            existingId: draft.memberId,
            model: trimmedModel,
            upstreamOverride: upstreamOverride,
            type: type,
            endpoint: endpointURL,
            apiKey: key,
            weight: weightValue,
            isEnabled: isEnabled,
            factoryId: factoryId)
        if saved {
            dismiss()
        } else {
            formError = state.lastError ?? "Could not save the member."
        }
    }
}

private func inferredType(_ endpoint: String) -> SourceType {
    guard let url = URL(string: endpoint.trimmingCharacters(in: .whitespacesAndNewlines)) else {
        return .openaiCompatible
    }
    if MemberAddress.isLoopback(url) { return .localNovaMLX }
    let host = url.host?.lowercased() ?? ""
    if host.contains("anthropic.com") { return .anthropic }
    return .openaiCompatible
}

private func typeLabel(_ type: SourceType) -> String {
    switch type {
    case .openaiCompatible: return "OpenAI-compatible"
    case .localNovaMLX: return "Local NovaMLX"
    case .anthropic: return "Anthropic"
    }
}
