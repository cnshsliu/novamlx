import SwiftUI
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

struct CatalogAdminPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n

    @State private var store: CatalogAdminStore?
    @State private var models: [CatalogEntry] = []
    @State private var selectedId: String?
    @State private var query = ""
    @State private var categoryFilter: String = ""
    @State private var updatedAt: String?
    @State private var statusMessage: String?
    @State private var isError = false
    @State private var isSaving = false
    @State private var draft: CatalogDraft = .blank

    private var filtered: [CatalogEntry] {
        models.filter { entry in
            if !categoryFilter.isEmpty, entry.category.rawValue != categoryFilter {
                return false
            }
            let q = query.trimmingCharacters(in: .whitespaces).lowercased()
            if q.isEmpty { return true }
            let blob = ([entry.id, entry.name, entry.description ?? ""] + entry.tags)
                .joined(separator: " ")
                .lowercased()
            return blob.contains(q)
        }
        .sorted { ($0.addedAt ?? "") > ($1.addedAt ?? "") }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            HSplitView {
                listPane
                    .frame(minWidth: 260, idealWidth: 320)
                editorPane
                    .frame(minWidth: 420)
            }
        }
        .onAppear { loadCatalog() }
        .onChange(of: appState.pendingCatalogEntry) { _, pending in
            guard let pending else { return }
            ingestPending(pending)
        }
    }

    private var toolbar: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .foregroundColor(NovaTheme.Colors.accent)
            Text(l10n.tr("app.catalogAdmin"))
                .font(.headline)
            Text(metaLine)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button("Add") { addBlank() }
                .controlSize(.small)
            Button("Duplicate") { duplicateSelected() }
                .controlSize(.small)
                .disabled(selectedId == nil)
            Button("Delete", role: .destructive) { deleteSelected() }
                .controlSize(.small)
                .disabled(selectedId == nil)
            Button(isSaving ? "Saving…" : "Save & push to GitHub") {
                Task { await saveAndPush() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isSaving || store == nil)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var metaLine: String {
        let path = store?.catalogURL.path ?? "repo not found"
        let stamp = updatedAt ?? ""
        return "\(path) · \(models.count) models · \(stamp)"
    }

    private var listPane: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                TextField("Search id, name, tags", text: $query)
                    .textFieldStyle(.roundedBorder)
                Picker("", selection: $categoryFilter) {
                    Text("All").tag("")
                    ForEach(ModelType.allCases, id: \.rawValue) { type in
                        Text(type.rawValue).tag(type.rawValue)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 110)
            }
            .padding(10)
            Divider()
            List(filtered, selection: $selectedId) { entry in
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(entry.name)
                            .font(.system(size: 13, weight: .medium))
                            .lineLimit(1)
                        if ModelCatalogPolicy.isIdPattern(entry.id) {
                            badge("family", color: NovaTheme.Colors.accent)
                        }
                        badge(entry.status.rawValue, color: entry.status == .verified ? NovaTheme.Colors.statusOK : .orange)
                    }
                    Text("\(entry.id) · \(entry.category.rawValue) · \(entry.size ?? "")")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                .tag(entry.id)
                .contentShape(Rectangle())
            }
            .listStyle(.sidebar)
            .onChange(of: selectedId) { oldId, newId in
                if let oldId { commitDraft(for: oldId) }
                if let newId, let entry = models.first(where: { $0.id == newId }) {
                    draft = CatalogDraft(entry)
                }
            }
        }
    }

    private var editorPane: some View {
        ScrollView {
            if selectedId == nil {
                Text("Select a model or add one.")
                    .foregroundColor(.secondary)
                    .padding(24)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                formGrid
                    .padding(20)
            }
            if let statusMessage {
                Text(statusMessage)
                    .font(.caption)
                    .foregroundColor(isError ? .red : NovaTheme.Colors.statusOK)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 16)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var formGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            labeled("id (Hub id, or owner/prefix* family)") {
                TextField("mlx-community/Qwen3.8-*", text: $draft.id)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: draft.id) { _, newId in
                        if draft.url.isEmpty {
                            draft.url = CatalogEntry.defaultURL(forId: newId)
                        }
                        renameSelected(to: newId)
                    }
            }
            labeled("url") {
                TextField("https://huggingface.co/…", text: $draft.url)
                    .textFieldStyle(.roundedBorder)
            }
            HStack(alignment: .top, spacing: 12) {
                labeled("name") {
                    TextField("Display name", text: $draft.name)
                        .textFieldStyle(.roundedBorder)
                }
                labeled("status") {
                    Picker("", selection: $draft.status) {
                        ForEach(CatalogStatus.allCases, id: \.self) { s in
                            Text(s.rawValue).tag(s)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }
            HStack(alignment: .top, spacing: 12) {
                labeled("category") {
                    Picker("", selection: $draft.category) {
                        ForEach(ModelType.allCases, id: \.self) { t in
                            Text(t.rawValue).tag(t)
                        }
                    }
                    .pickerStyle(.menu)
                }
                labeled("family") {
                    Picker("", selection: $draft.family) {
                        ForEach(ModelFamily.allCases, id: \.self) { f in
                            Text(f.rawValue).tag(f)
                        }
                    }
                    .pickerStyle(.menu)
                }
                labeled("format") {
                    Picker("", selection: $draft.format) {
                        ForEach(CatalogFormat.allCases, id: \.self) { f in
                            Text(f.rawValue).tag(f)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }
            labeled("quant") {
                TextField("4bit / 8bit / fp16", text: $draft.quant)
                    .textFieldStyle(.roundedBorder)
            }
            labeled("description") {
                TextEditor(text: $draft.description)
                    .font(.system(size: 12))
                    .frame(minHeight: 72)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
                    )
            }
            HStack(alignment: .top, spacing: 12) {
                labeled("size (display)") {
                    TextField("~15 GB", text: $draft.size)
                        .textFieldStyle(.roundedBorder)
                }
                labeled("sizeBytes") {
                    TextField("0", text: $draft.sizeBytes)
                        .textFieldStyle(.roundedBorder)
                }
                labeled("minRamGB") {
                    TextField("24", text: $draft.minRamGB)
                        .textFieldStyle(.roundedBorder)
                }
            }
            HStack(alignment: .top, spacing: 12) {
                labeled("revision") {
                    TextField("optional SHA", text: $draft.revision)
                        .textFieldStyle(.roundedBorder)
                }
                labeled("testedOn") {
                    TextField("NovaMLX version", text: $draft.testedOn)
                        .textFieldStyle(.roundedBorder)
                }
                labeled("addedAt") {
                    TextField("ISO-8601", text: $draft.addedAt)
                        .textFieldStyle(.roundedBorder)
                }
            }
            labeled("tags (comma-separated)") {
                TextField("MLX, 4-bit", text: $draft.tags)
                    .textFieldStyle(.roundedBorder)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("capabilities")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                HStack(spacing: 12) {
                    ForEach(CatalogFile.catalogCapabilities, id: \.self) { cap in
                        Toggle(cap, isOn: capabilityBinding(cap))
                            .toggleStyle(.checkbox)
                            .font(.system(size: 12))
                    }
                }
            }
            Button("Apply to list") {
                if let id = selectedId { commitDraft(for: id) }
            }
            .controlSize(.small)
        }
        .onChange(of: draft) { _, _ in
            if let id = selectedId { commitDraft(for: id) }
        }
    }

    private func labeled(_ title: String, @ViewBuilder field: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            field()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold))
            .foregroundColor(color)
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
    }

    private func capabilityBinding(_ cap: String) -> Binding<Bool> {
        Binding(
            get: { draft.capabilities.contains(cap) },
            set: { on in
                if on { draft.capabilities.insert(cap) }
                else { draft.capabilities.remove(cap) }
            }
        )
    }

    private func loadCatalog() {
        store = CatalogAdminStore.discover()
        guard let store else {
            statusMessage = CatalogAdminError.repoNotFound.localizedDescription
            isError = true
            ingestPendingIfNeeded()
            return
        }
        do {
            let file = try store.load()
            models = file.models
            updatedAt = file.updatedAt
            if selectedId == nil { selectedId = models.first?.id }
            if let id = selectedId, let entry = models.first(where: { $0.id == id }) {
                draft = CatalogDraft(entry)
            }
            statusMessage = nil
            isError = false
        } catch {
            statusMessage = error.localizedDescription
            isError = true
        }
        ingestPendingIfNeeded()
    }

    private func ingestPendingIfNeeded() {
        if let pending = appState.pendingCatalogEntry {
            ingestPending(pending)
        }
    }

    private func ingestPending(_ pending: CatalogEntry) {
        appState.pendingCatalogEntry = nil
        if let existing = models.first(where: { $0.id == pending.id }) {
            selectedId = existing.id
            draft = CatalogDraft(existing)
            statusMessage = "Already in catalog — edit and save if you want to change it."
            isError = false
            return
        }
        models.insert(pending, at: 0)
        selectedId = pending.id
        draft = CatalogDraft(pending)
        statusMessage = "Added as verified. Review, then Save & push to GitHub."
        isError = false
    }

    private func addBlank() {
        if let id = selectedId { commitDraft(for: id) }
        let blank = CatalogEntry(
            id: "new-model-\(Int(Date().timeIntervalSince1970))",
            url: "",
            name: "",
            category: .llm,
            family: .qwen,
            format: .mlx,
            tags: ["MLX"],
            status: .preview,
            addedAt: CatalogFile.utcNow()
        )
        models.insert(blank, at: 0)
        selectedId = blank.id
        draft = CatalogDraft(blank)
    }

    private func duplicateSelected() {
        guard let id = selectedId, let entry = models.first(where: { $0.id == id }) else { return }
        commitDraft(for: id)
        let copy = CatalogEntry(
            id: entry.id + "-copy",
            url: entry.url,
            name: entry.name,
            category: entry.category,
            family: entry.family,
            format: entry.format,
            description: entry.description,
            revision: entry.revision,
            quant: entry.quant,
            size: entry.size,
            sizeBytes: entry.sizeBytes,
            minRamGB: entry.minRamGB,
            tags: entry.tags,
            capabilities: entry.capabilities,
            testedOn: entry.testedOn,
            status: entry.status,
            addedAt: CatalogFile.utcNow()
        )
        if let idx = models.firstIndex(where: { $0.id == id }) {
            models.insert(copy, at: idx + 1)
        } else {
            models.insert(copy, at: 0)
        }
        selectedId = copy.id
        draft = CatalogDraft(copy)
    }

    private func deleteSelected() {
        guard let id = selectedId else { return }
        models.removeAll { $0.id == id }
        selectedId = models.first?.id
        if let next = selectedId, let entry = models.first(where: { $0.id == next }) {
            draft = CatalogDraft(entry)
        } else {
            draft = .blank
        }
    }

    private func renameSelected(to newId: String) {
        let trimmed = newId.trimmingCharacters(in: .whitespaces)
        guard let old = selectedId, old != trimmed else { return }
        guard !trimmed.isEmpty else { return }
        guard !models.contains(where: { $0.id == trimmed && $0.id != old }) else { return }
        selectedId = trimmed
    }

    private func commitDraft(for id: String) {
        guard let idx = models.firstIndex(where: { $0.id == id || $0.id == selectedId }) else { return }
        models[idx] = draft.entry()
        if models[idx].id != id {
            selectedId = models[idx].id
        }
    }

    private func saveAndPush() async {
        if let id = selectedId { commitDraft(for: id) }
        guard let store else {
            statusMessage = CatalogAdminError.repoNotFound.localizedDescription
            isError = true
            return
        }
        isSaving = true
        defer { isSaving = false }
        do {
            let file = CatalogFile(schemaVersion: 1, models: models)
            let saved = try store.save(file)
            models = saved.models
            updatedAt = saved.updatedAt
            modelManager.catalogStore.applyLocal(saved)
            do {
                let pushNote = try store.pushToGitHub()
                statusMessage = "Saved \(saved.models.count) models. \(pushNote)"
                isError = false
            } catch {
                statusMessage = "Saved locally. GitHub push failed: \(error.localizedDescription)"
                isError = true
            }
        } catch {
            statusMessage = error.localizedDescription
            isError = true
        }
    }
}

struct CatalogDraft: Equatable {
    var id: String
    var url: String
    var name: String
    var description: String
    var category: ModelType
    var family: ModelFamily
    var format: CatalogFormat
    var status: CatalogStatus
    var quant: String
    var size: String
    var sizeBytes: String
    var minRamGB: String
    var revision: String
    var testedOn: String
    var addedAt: String
    var tags: String
    var capabilities: Set<String>

    static let blank = CatalogDraft(CatalogEntry(
        id: "", url: "", name: "", category: .llm, family: .other, format: .mlx
    ))

    init(_ entry: CatalogEntry) {
        id = entry.id
        url = entry.url
        name = entry.name
        description = entry.description ?? ""
        category = entry.category
        family = entry.family
        format = entry.format
        status = entry.status
        quant = entry.quant ?? ""
        size = entry.size ?? ""
        sizeBytes = entry.sizeBytes.map(String.init) ?? ""
        minRamGB = entry.minRamGB.map(String.init) ?? ""
        revision = entry.revision ?? ""
        testedOn = entry.testedOn ?? ""
        addedAt = entry.addedAt ?? ""
        tags = entry.tags.joined(separator: ", ")
        capabilities = Set(entry.capabilities)
    }

    func entry() -> CatalogEntry {
        let emptyToNil: (String) -> String? = {
            let t = $0.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : t
        }
        return CatalogEntry(
            id: id.trimmingCharacters(in: .whitespaces),
            url: url.trimmingCharacters(in: .whitespaces),
            name: name.trimmingCharacters(in: .whitespaces),
            category: category,
            family: family,
            format: format,
            description: emptyToNil(description),
            revision: emptyToNil(revision),
            quant: emptyToNil(quant),
            size: emptyToNil(size),
            sizeBytes: UInt64(sizeBytes),
            minRamGB: Int(minRamGB),
            tags: tags.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty },
            capabilities: CatalogFile.catalogCapabilities.filter { capabilities.contains($0) },
            testedOn: emptyToNil(testedOn),
            status: status,
            addedAt: emptyToNil(addedAt)
        )
    }
}
