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
                Text("No members yet. Add an OpenAI-compatible endpoint or the local NovaMLX port.")
                    .foregroundStyle(.secondary)
            } else {
                List {
                    ForEach(state.config.members) { member in
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(member.name).font(.body.bold())
                                Text(member.endpoint.absoluteString)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                                Text("\(typeLabel(member.type)) · \(member.upstreamModel)")
                                    .font(.caption)
                                Text(member.isEnabled ? (member.isFree ? "Enabled · free" : "Enabled · paid") : "Disabled")
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
            "Delete \(pendingDelete?.name ?? "member")?",
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
}

private struct MemberDraft: Identifiable {
    let token = UUID()
    var id: UUID { token }
    var memberId: String
    var name: String
    var type: SourceType
    var endpoint: String
    var upstreamModel: String
    var apiKey: String
    var weight: String
    var isEnabled: Bool
    var isFree: Bool
    var isNew: Bool

    init() {
        memberId = ""
        name = ""
        type = .openaiCompatible
        endpoint = "http://127.0.0.1:6590/v1"
        upstreamModel = ""
        apiKey = ""
        weight = ""
        isEnabled = true
        isFree = false
        isNew = true
    }

    init(_ member: MemberConfig) {
        memberId = member.id
        name = member.name
        type = member.type
        endpoint = member.endpoint.absoluteString
        upstreamModel = member.upstreamModel
        apiKey = ""
        weight = member.weight.map(String.init) ?? ""
        isEnabled = member.isEnabled
        isFree = member.isFree
        isNew = false
    }
}

private struct MemberForm: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var state: PeerAppState
    let draft: MemberDraft

    @State private var memberId: String
    @State private var name: String
    @State private var type: SourceType
    @State private var endpoint: String
    @State private var upstreamModel: String
    @State private var apiKey: String
    @State private var weight: String
    @State private var isEnabled: Bool
    @State private var isFree: Bool
    @State private var formError: String?

    init(draft: MemberDraft) {
        self.draft = draft
        _memberId = State(initialValue: draft.memberId)
        _name = State(initialValue: draft.name)
        _type = State(initialValue: draft.type)
        _endpoint = State(initialValue: draft.endpoint)
        _upstreamModel = State(initialValue: draft.upstreamModel)
        _apiKey = State(initialValue: draft.apiKey)
        _weight = State(initialValue: draft.weight)
        _isEnabled = State(initialValue: draft.isEnabled)
        _isFree = State(initialValue: draft.isFree)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(draft.isNew ? "Add member" : "Edit member")
                .font(.title3.bold())
            TextField("Id", text: $memberId)
                .disabled(!draft.isNew)
            TextField("Name", text: $name)
            Picker("Type", selection: $type) {
                Text("OpenAI-compatible").tag(SourceType.openaiCompatible)
                Text("Local NovaMLX").tag(SourceType.localNovaMLX)
                Text("Anthropic").tag(SourceType.anthropic)
            }
            TextField("Endpoint URL", text: $endpoint)
            TextField("Upstream model", text: $upstreamModel)
            SecureField(draft.isNew ? "API key" : "API key (blank keeps the current key)", text: $apiKey)
            TextField("Weight (optional)", text: $weight)
            Toggle("Enabled", isOn: $isEnabled)
            Toggle("Free", isOn: $isFree)
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
        let id = memberId.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty else {
            formError = "Member id is required."
            return
        }
        guard let url = URL(string: endpoint.trimmingCharacters(in: .whitespacesAndNewlines)) else {
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
            id: id,
            name: trimmedName.isEmpty ? id : trimmedName,
            type: type,
            endpoint: url,
            upstreamModel: upstreamModel,
            apiKey: key,
            weight: weightValue,
            isEnabled: isEnabled,
            isFree: isFree)
        if saved {
            dismiss()
        } else {
            formError = state.lastError ?? "Could not save the member."
        }
    }
}

private func typeLabel(_ type: SourceType) -> String {
    switch type {
    case .openaiCompatible: return "OpenAI-compatible"
    case .localNovaMLX: return "Local NovaMLX"
    case .anthropic: return "Anthropic"
    }
}
