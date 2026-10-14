import TknetPeer
import SwiftUI

struct MembersView: View {
    @EnvironmentObject private var state: PeerAppState
    @State private var draft: MemberDraft?
    @State private var pendingDelete: MemberConfig?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Members")
                    .font(.headline)
                Spacer()
                Button("Add member") { draft = MemberDraft() }
                    .disabled(state.lockError != nil)
            }
            if state.config.members.isEmpty {
                Text("No members yet. Add an endpoint and the model name clients should send.")
                    .foregroundStyle(.secondary)
            } else {
                List {
                    ForEach(state.config.members) { member in
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(MemberLabel.title(member, among: state.config.members))
                                    .font(.body.bold())
                                Text(member.endpoint.absoluteString)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                                Text(detailLine(member))
                                    .font(.caption)
                                Text(member.isEnabled ? "Enabled" : "Disabled")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Edit") { draft = MemberDraft(member) }
                            Button("Delete") { pendingDelete = member }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .sheet(item: $draft) { draft in
            MemberForm(draft: draft)
        }
        .confirmationDialog(
            "Delete \(pendingDelete.map { MemberLabel.title($0, among: state.config.members) } ?? "member")?",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let id = pendingDelete?.id {
                    Task { await state.deleteMember(id: id) }
                }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }

    private func detailLine(_ member: MemberConfig) -> String {
        let model = PeerAdmin.publicModel(member)
        let type = typeLabel(member.type)
        if member.upstreamModel != model {
            return "\(type) · upstream \(member.upstreamModel)"
        }
        return type
    }
}

enum MemberLabel {
    static func title(_ member: MemberConfig, among members: [MemberConfig]) -> String {
        let model = PeerAdmin.publicModel(member)
        let shared = members.filter { PeerAdmin.publicModel($0) == model }.count > 1
        if shared, let host = member.endpoint.host, !host.isEmpty {
            return "\(model) · \(host)"
        }
        return model
    }
}

private struct MemberDraft: Identifiable {
    let token = UUID()
    var id: UUID { token }
    var memberId: String?
    var model: String
    var upstreamOverride: String
    var type: SourceType
    var endpoint: String
    var apiKey: String
    var weight: String
    var isEnabled: Bool
    var isNew: Bool

    init() {
        memberId = nil
        model = ""
        upstreamOverride = ""
        type = .localNovaMLX
        endpoint = "http://127.0.0.1:6590/v1"
        apiKey = ""
        weight = ""
        isEnabled = true
        isNew = true
    }

    init(_ member: MemberConfig) {
        let model = PeerAdmin.publicModel(member)
        memberId = member.id
        self.model = model
        upstreamOverride = member.upstreamModel == model ? "" : member.upstreamModel
        type = member.type
        endpoint = member.endpoint.absoluteString
        apiKey = ""
        weight = member.weight.map(String.init) ?? ""
        isEnabled = member.isEnabled
        isNew = false
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
    @State private var endpoint: String
    @State private var apiKey: String
    @State private var weight: String
    @State private var isEnabled: Bool
    @State private var showMore: Bool
    @State private var formError: String?

    init(draft: MemberDraft) {
        self.draft = draft
        _model = State(initialValue: draft.model)
        _upstreamOverride = State(initialValue: draft.upstreamOverride)
        _type = State(initialValue: draft.type)
        _typeEdited = State(initialValue: !draft.isNew)
        _endpoint = State(initialValue: draft.endpoint)
        _apiKey = State(initialValue: draft.apiKey)
        _weight = State(initialValue: draft.weight)
        _isEnabled = State(initialValue: draft.isEnabled)
        let hasMore = !draft.upstreamOverride.isEmpty || !draft.weight.isEmpty
        _showMore = State(initialValue: hasMore)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(draft.isNew ? "Add member" : "Edit member")
                .font(.title3.bold())
            TextField("Model", text: $model)
            TextField("Endpoint URL", text: $endpoint)
                .onChange(of: endpoint) { _, newValue in
                    guard !typeEdited else { return }
                    type = inferredType(newValue)
                }
            SecureField(draft.isNew ? "API key" : "API key (blank keeps the current key)", text: $apiKey)
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
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { Task { await save() } }
            }
        }
        .padding(20)
        .frame(width: 460)
    }

    private func save() async {
        let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModel.isEmpty else {
            formError = "Model name is required."
            return
        }
        guard let url = URL(string: endpoint.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              url.host != nil else {
            formError = "Endpoint must be an http(s) URL."
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
            endpoint: url,
            apiKey: key,
            weight: weightValue,
            isEnabled: isEnabled)
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
