import SwiftUI
import AppKit
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

/// Inline two-pane model manager shown under Models Path in Settings: each
/// side picks a storage vault (root), the middle column moves the selected
/// model (or all of them) left/right, or deletes the selection.
struct MoveModelsPanel: View {
    let modelManager: ModelManager
    /// Blocks moves of models currently loaded in the engine.
    let isModelLoaded: (String) -> Bool

    @State private var leftVault: URL?
    @State private var rightVault: URL?
    @State private var leftModels: [ModelManager.VaultModelEntry] = []
    @State private var rightModels: [ModelManager.VaultModelEntry] = []
    @State private var leftSelection = Set<String>()
    @State private var rightSelection = Set<String>()
    @State private var leftSort: VaultSort = .name
    @State private var rightSort: VaultSort = .name
    @State private var message: String?
    /// Live operation progress; nil when idle.
    @State private var progress: MoveProgress?
    /// Outcome of the last completed batch.
    @State private var completion: Completion?
    /// Pending delete confirmation.
    @State private var confirmDelete = false
    /// Vault listing load state; discovery walks whole model trees and must
    /// stay off the main thread or the panel blocks for seconds on appear.
    @State private var loading = false
    @State private var reloadGeneration = 0
    /// Text most recently copied via click; shown briefly in the status bar.
    @State private var copiedHint: String?

    /// One queued operation. Source/dest/path are captured at enqueue time,
    /// so later picker changes never corrupt a pending task.
    struct QueueTask: Identifiable {
        let id = UUID()
        let modelId: String
        let path: URL
        let source: URL
        let dest: URL?
        var isDelete: Bool { dest == nil }
    }

    /// Pending tasks, drained one at a time in the background. Users can
    /// keep adding while it runs.
    @State private var queue: [QueueTask] = []
    @State private var queueRunning = false
    @State private var drainDone = 0

    private var busy: Bool { queueRunning || !queue.isEmpty }

    struct MoveProgress: Equatable {
        let done: Int
        let total: Int
        let current: String
        let verb: String
    }

    struct Completion: Equatable {
        let deleted: Bool
        let names: [String]
    }

    private var vaults: [URL] { NovaMLXPaths.configuredModelsDirs }

    /// Model ids present on BOTH panes' vaults — moving them across would
    /// hit "already exists at destination". Derived from the listings, so it
    /// refreshes automatically after every move/delete/reload. Meaningless
    /// when both pickers show the same vault.
    private var bothSidesIds: Set<String> {
        guard !sameVault else { return [] }
        return Set(leftModels.map(\.modelId))
            .intersection(Set(rightModels.map(\.modelId)))
    }

    var body: some View {
        VStack(spacing: 0) {
            if vaults.count < 2 {
                Text("Add at least two storage directories in Settings to move models between them.")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 20)
            } else {
                HStack(spacing: 0) {
                    vaultPane(selection: $leftVault, models: leftSort.apply(leftModels),
                              paneSelection: $leftSelection, sortOrder: $leftSort)
                    moveButtons
                    vaultPane(selection: $rightVault, models: rightSort.apply(rightModels),
                              paneSelection: $rightSelection, sortOrder: $rightSort)
                }
                .frame(height: 300)
            }
            statusBar
            if message != nil { footer }
        }
        .padding(.vertical, 8)
        .background(Color.primary.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 8)
        .onAppear {
            leftVault = vaults.first
            rightVault = vaults.count > 1 ? vaults[1] : nil
            reload()
        }
        .onChange(of: leftVault) { _ in reload() }
        .onChange(of: rightVault) { _ in reload() }
        // A stale unload-first error or an acknowledged 移动/删除完成
        // readout goes away once the user interacts with the lists again.
        .onChange(of: leftSelection) { _ in clearTransientStatus() }
        .onChange(of: rightSelection) { _ in clearTransientStatus() }
        .confirmationDialog(
            "Delete \(pendingDeleteEntries.count) model\(pendingDeleteEntries.count == 1 ? "" : "s")?",
            isPresented: $confirmDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { deleteSelected() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(pendingDeleteEntries.map(\.modelId).joined(separator: ", "))
        }
    }

    // MARK: - Panes

    private enum VaultSort: String, CaseIterable, Identifiable {
        case name, size
        var id: String { rawValue }
        var label: String {
            switch self {
            case .name: return "Name"
            case .size: return "Size"
            }
        }
        func apply(_ models: [ModelManager.VaultModelEntry]) -> [ModelManager.VaultModelEntry] {
            switch self {
            case .name: return models.sorted { $0.modelId.localizedStandardCompare($1.modelId) == .orderedAscending }
            case .size: return models.sorted { $0.sizeBytes != $1.sizeBytes ? $0.sizeBytes > $1.sizeBytes : $0.modelId < $1.modelId }
            }
        }
    }

    private func vaultPane(
        selection: Binding<URL?>,
        models: [ModelManager.VaultModelEntry],
        paneSelection: Binding<Set<String>>,
        sortOrder: Binding<VaultSort>
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Picker("", selection: selection) {
                ForEach(vaults, id: \.path) { vault in
                    Text(vault.path).tag(vault as URL?)
                }
            }
            .labelsHidden()
            .padding(.horizontal, 8)
            .padding(.top, 4)

            Picker("", selection: sortOrder) {
                ForEach(VaultSort.allCases) { sort in
                    Text(sort.label).tag(sort)
                }
            }
            .pickerStyle(.segmented)
            .controlSize(.small)
            .padding(.horizontal, 8)
            .padding(.bottom, 4)

            List(models, selection: paneSelection) { entry in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(entry.modelId)
                            .font(.system(size: 12, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .onTapGesture {
                                // Click the name to copy it; toggle the row
                                // so repeated name clicks build a multi-select.
                                copyToClipboard(entry.modelId)
                                if paneSelection.wrappedValue.contains(entry.id) {
                                    paneSelection.wrappedValue.remove(entry.id)
                                } else {
                                    paneSelection.wrappedValue.insert(entry.id)
                                }
                            }
                        HStack(spacing: 6) {
                            Text(ByteCountFormatter.string(
                                fromByteCount: Int64(entry.sizeBytes), countStyle: .file))
                            if !entry.isComplete { Text("incomplete").foregroundColor(.orange) }
                            if isModelLoaded(entry.modelId) {
                                Text("loaded").foregroundColor(NovaTheme.Colors.accent)
                            }
                            if bothSidesIds.contains(entry.modelId) {
                                Text("both sides").foregroundColor(.orange)
                            }
                        }
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    }
                    Spacer()
                }
                .tag(entry.id)
            }
            .listStyle(.inset)
            .overlay {
                if models.isEmpty {
                    if loading {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("No models")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Move Buttons

    private var leftSelected: Bool { !leftSelection.isEmpty }
    private var rightSelected: Bool { !rightSelection.isEmpty }

    private var moveButtons: some View {
        VStack(spacing: 10) {
            Spacer()
            // Arrows act on their own pane's selection and ENQUEUE tasks —
            // the queue keeps draining while the user lines up the next ones.
            moveButton("→", help: "Queue: move selected to the right vault",
                       enabled: leftSelected, action: { enqueue(direction: .right, all: false) })
            moveButton("→→", help: "Queue: move all to the right vault",
                       enabled: !leftModels.isEmpty, action: { enqueue(direction: .right, all: true) })
            Divider().frame(width: 36)
            moveButton("←", help: "Queue: move selected to the left vault",
                       enabled: rightSelected, action: { enqueue(direction: .left, all: false) })
            moveButton("←←", help: "Queue: move all to the left vault",
                       enabled: !rightModels.isEmpty, action: { enqueue(direction: .left, all: true) })
            Divider().frame(width: 36)
            Button {
                confirmDelete = true
            } label: {
                Image(systemName: "trash")
                    .frame(width: 44, height: 28)
            }
            .buttonStyle(.bordered)
            .help("Queue: delete selected models")
            .disabled(loading || !(leftSelected || rightSelected))
            .tint(.red)
            Spacer()
        }
        .padding(.horizontal, 10)
    }

    private func moveButton(_ title: String, help: String, enabled: Bool,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, design: .monospaced))
                .frame(width: 44, height: 28)
        }
        .buttonStyle(.bordered)
        .help(help)
        .disabled(loading || sameVault || !enabled)
    }

    private var sameVault: Bool {
        guard let l = leftVault, let r = rightVault else { return true }
        return l.standardizedFileURL.path == r.standardizedFileURL.path
    }

    // MARK: - Status Bar / Footer

    /// Full-name readout under the lists: current selection while idle, live
    /// progress during an operation, and the completed batch afterwards.
    private var statusBar: some View {
        HStack(alignment: .center, spacing: 8) {
            if let copiedHint {
                Image(systemName: "doc.on.doc")
                    .foregroundColor(.secondary)
                Text("已复制：\(copiedHint)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
            } else if let progress {
                ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                    .frame(width: 100)
                Text("\(progress.verb) \(progress.done)/\(progress.total): \(progress.current)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .truncationMode(.tail)
                if queue.count > 1 {
                    Text("队列 \(queue.count - 1) 待处理")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Button("清空队列") { queue.removeSubrange(1...) }
                        .controlSize(.mini)
                        .buttonStyle(.borderless)
                }
            } else if let completion, !completion.names.isEmpty {
                Image(systemName: completion.deleted ? "trash.fill" : "checkmark.circle.fill")
                    .foregroundColor(completion.deleted ? .orange : NovaTheme.Colors.statusOK)
                Text("\(completion.deleted ? "删除" : "移动")完成：\(completion.names.joined(separator: ", "))")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(completion.deleted ? .orange : NovaTheme.Colors.statusOK)
                    .lineLimit(3)
            } else {
                selectionSummary
            }
            Spacer(minLength: 12)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(minHeight: 44, alignment: .center)
    }

    /// Both panes' selected model names, untruncated (wraps up to 3 lines).
    private var selectionSummary: some View {
        let leftNames = leftModels.filter { leftSelection.contains($0.id) }.map(\.modelId)
        let rightNames = rightModels.filter { rightSelection.contains($0.id) }.map(\.modelId)
        return HStack(spacing: 8) {
            if leftNames.isEmpty && rightNames.isEmpty {
                Text("Select models to move or delete")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            } else {
                if !leftNames.isEmpty {
                    Text("← \(leftNames.joined(separator: ", "))")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                if !rightNames.isEmpty {
                    Text("\(rightNames.joined(separator: ", ")) →")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.accent)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            if let message {
                Text(message)
                    .font(.system(size: 11))
                    .foregroundColor(message.hasPrefix("Error") ? .red : NovaTheme.Colors.statusOK)
                    .lineLimit(3)
                    .truncationMode(.middle)
                    .onTapGesture { copyToClipboard(message) }
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(minHeight: 30, alignment: .center)
    }

    // MARK: - Clipboard

    private func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedHint = text
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            if copiedHint == text { copiedHint = nil }
        }
    }

    /// Drop the status-bar readout once the user moves on (selects another
    /// model) — the completion notice is only interesting right after the
    /// batch finished.
    private func clearTransientStatus() {
        completion = nil
        if let m = message, m.hasPrefix("Error: unload first") { message = nil }
    }

    // MARK: - Actions

    private func reload() {
        let left = leftVault
        let right = rightVault
        reloadGeneration += 1
        let generation = reloadGeneration
        loading = true
        let manager = modelManager
        Task.detached(priority: .userInitiated) {
            let l = left.map { manager.modelsInVault($0) } ?? []
            let r = right.map { manager.modelsInVault($0) } ?? []
            await MainActor.run {
                // A newer reload (vault switch) superseded this one.
                guard generation == reloadGeneration else { return }
                leftModels = l
                rightModels = r
                leftSelection.formIntersection(Set(l.map(\.id)))
                rightSelection.formIntersection(Set(r.map(\.id)))
                loading = false
            }
        }
    }

    private func finishBatch(deleted: Bool, names: [String], errors: [String]) {
        progress = nil
        reload()
        completion = Completion(deleted: deleted, names: names)
        message = errors.isEmpty ? nil : "Error: \(names.count) succeeded, \(errors.count) failed — "
            + errors.prefix(3).joined(separator: "; ")
    }

    /// Selected entries on whichever pane holds the selection.
    private var pendingDeleteEntries: [ModelManager.VaultModelEntry] {
        leftModels.filter { leftSelection.contains($0.id) }
            + rightModels.filter { rightSelection.contains($0.id) }
    }

    private func deleteSelected() {
        enqueue(entries: pendingDeleteEntries, source: nil, dest: nil)
    }

    private enum Direction { case left, right }

    /// Turn the current selection into queue tasks. `source`/`dest` are the
    /// move endpoints (nil = delete). Called from the arrow and trash
    /// buttons — works any time, even while the queue is draining.
    private func enqueue(direction: Direction, all: Bool) {
        guard !sameVault,
              let source = direction == .right ? leftVault : rightVault,
              let dest = direction == .right ? rightVault : leftVault else { return }
        let models = direction == .right ? leftModels : rightModels
        let selectedIds = direction == .right ? leftSelection : rightSelection
        let entries = all ? models : models.filter { selectedIds.contains($0.id) }
        enqueue(entries: entries, source: source, dest: dest)
        // Free the selection so the user can line up the next batch.
        if direction == .right { leftSelection = [] } else { rightSelection = [] }
    }

    private func enqueue(entries: [ModelManager.VaultModelEntry], source: URL?, dest: URL?) {
        guard !entries.isEmpty else { return }
        let loaded = entries.filter { isModelLoaded($0.modelId) }.map(\.modelId)
        if !loaded.isEmpty {
            message = "Error: unload first — \(loaded.joined(separator: ", "))"
            return
        }
        // Both panes may hold selections (delete button): resolve per entry.
        var tasks: [QueueTask] = []
        if let source, let dest {
            tasks = entries.map {
                QueueTask(modelId: $0.modelId, path: $0.path, source: source, dest: dest)
            }
        } else {
            let leftIds = Set(leftModels.map(\.id))
            tasks = entries.map { entry in
                let src = leftIds.contains(entry.id) ? leftVault : rightVault
                return QueueTask(modelId: entry.modelId, path: entry.path,
                                 source: src ?? entry.path.deletingLastPathComponent(),
                                 dest: nil)
            }
        }
        message = nil
        completion = nil
        queue.append(contentsOf: tasks)
        if source == nil { leftSelection = []; rightSelection = [] }
        drainQueue()
    }

    /// Execute queued tasks one at a time in the background. New tasks may
    /// be appended while draining; the loop picks them up. Failures are
    /// collected and reported when the queue empties.
    private func drainQueue() {
        guard !queueRunning else { return }
        queueRunning = true
        drainDone = 0
        let manager = modelManager
        Task.detached(priority: .userInitiated) {
            var movedNames: [String] = []
            var deletedNames: [String] = []
            var errors: [String] = []
            while true {
                let task: QueueTask? = await MainActor.run { queue.first }
                guard let task = task else { break }
                let done = await MainActor.run { drainDone }
                let pending = await MainActor.run { queue.count }
                await MainActor.run {
                    progress = MoveProgress(done: done, total: done + pending,
                                            current: task.modelId,
                                            verb: task.isDelete ? "Deleting" : "Moving")
                }
                do {
                    if let dest = task.dest {
                        try manager.moveModel(task.modelId, from: task.source, to: dest)
                        movedNames.append(task.modelId)
                    } else {
                        try manager.deleteModel(at: task.path)
                        deletedNames.append(task.modelId)
                    }
                } catch {
                    errors.append("\(task.modelId): \(error.localizedDescription)")
                }
                // Pop the finished task; refresh listings so both-sides
                // badges and the moved rows update as the queue drains.
                await MainActor.run {
                    if !queue.isEmpty, queue[0].id == task.id { queue.remove(at: 0) }
                    drainDone += 1
                    reload()
                }
            }
            _ = manager.discoverModels()
            await manager.refreshModelCaches()
            await MainActor.run {
                queueRunning = false
                let deleted = deletedNames.count > movedNames.count
                finishBatch(deleted: deleted,
                            names: movedNames + deletedNames,
                            errors: errors)
            }
        }
    }
}
