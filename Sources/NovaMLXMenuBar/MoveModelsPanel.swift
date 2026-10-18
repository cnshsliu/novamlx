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
    @State private var busy = false
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
        // Selection is exclusive: picking rows on one side clears the other,
        // so the arrow buttons always reflect a single direction. A stale
        // unload-first error goes away once the user picks something else.
        .onChange(of: leftSelection) { sel in
            if !sel.isEmpty, !rightSelection.isEmpty { rightSelection = [] }
            clearStaleError()
        }
        .onChange(of: rightSelection) { sel in
            if !sel.isEmpty, !leftSelection.isEmpty { leftSelection = [] }
            clearStaleError()
        }
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
                                // Click the name to copy it; also keep the
                                // row selectable like a plain click.
                                copyToClipboard(entry.modelId)
                                paneSelection.wrappedValue = [entry.id]
                            }
                        HStack(spacing: 6) {
                            Text(ByteCountFormatter.string(
                                fromByteCount: Int64(entry.sizeBytes), countStyle: .file))
                            if !entry.isComplete { Text("incomplete").foregroundColor(.orange) }
                            if isModelLoaded(entry.modelId) {
                                Text("loaded").foregroundColor(NovaTheme.Colors.accent)
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
            // Right-movers act on the left list; they are only available when
            // the selection sits on the left side.
            moveButton("→", help: "Move selected to the right vault",
                       enabled: leftSelected, action: { move(direction: .right, all: false) })
            moveButton("→→", help: "Move all to the right vault",
                       enabled: !leftModels.isEmpty && !rightSelected, action: { move(direction: .right, all: true) })
            Divider().frame(width: 36)
            moveButton("←", help: "Move selected to the left vault",
                       enabled: rightSelected, action: { move(direction: .left, all: false) })
            moveButton("←←", help: "Move all to the left vault",
                       enabled: !rightModels.isEmpty && !leftSelected, action: { move(direction: .left, all: true) })
            Divider().frame(width: 36)
            Button {
                confirmDelete = true
            } label: {
                Image(systemName: "trash")
                    .frame(width: 44, height: 28)
            }
            .buttonStyle(.bordered)
            .help("Delete selected models")
            .disabled(busy || loading || !(leftSelected || rightSelected))
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
        .disabled(busy || loading || sameVault || !enabled)
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

    private func clearStaleError() {
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
        busy = false
        progress = nil
        reload()
        leftSelection = []
        rightSelection = []
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
        let entries = pendingDeleteEntries
        guard !entries.isEmpty, !busy else { return }
        let loaded = entries.filter { isModelLoaded($0.modelId) }.map(\.modelId)
        if !loaded.isEmpty {
            message = "Error: unload first — \(loaded.joined(separator: ", "))"
            return
        }
        busy = true
        message = nil
        completion = nil
        progress = MoveProgress(done: 0, total: entries.count,
                                current: entries.first?.modelId ?? "", verb: "Deleting")
        let manager = modelManager
        Task.detached(priority: .userInitiated) {
            var deletedNames: [String] = []
            var errors: [String] = []
            for (index, entry) in entries.enumerated() {
                await MainActor.run {
                    progress = MoveProgress(done: index, total: entries.count,
                                            current: entry.modelId, verb: "Deleting")
                }
                do {
                    try manager.deleteModel(at: entry.path)
                    deletedNames.append(entry.modelId)
                } catch {
                    errors.append("\(entry.modelId): \(error.localizedDescription)")
                }
                await MainActor.run {
                    progress = MoveProgress(done: index + 1, total: entries.count,
                                            current: entry.modelId, verb: "Deleting")
                }
            }
            _ = manager.discoverModels()
            await manager.refreshModelCaches()
            await MainActor.run {
                finishBatch(deleted: true, names: deletedNames, errors: errors)
            }
        }
    }

    private enum Direction { case left, right }

    private func move(direction: Direction, all: Bool) {
        guard !busy, !sameVault,
              let source = direction == .right ? leftVault : rightVault,
              let dest = direction == .right ? rightVault : leftVault else { return }
        let models = direction == .right ? leftModels : rightModels
        let selectedIds = direction == .right ? leftSelection : rightSelection
        let entries = all ? models : models.filter { selectedIds.contains($0.id) }
        guard !entries.isEmpty else { return }

        let loaded = entries.filter { isModelLoaded($0.modelId) }.map(\.modelId)
        if !loaded.isEmpty {
            message = "Error: unload first — \(loaded.joined(separator: ", "))"
            return
        }

        busy = true
        message = nil
        completion = nil
        progress = MoveProgress(done: 0, total: entries.count,
                                current: entries.first?.modelId ?? "", verb: "Moving")
        let manager = modelManager
        Task.detached(priority: .userInitiated) {
            var movedNames: [String] = []
            var errors: [String] = []
            for (index, entry) in entries.enumerated() {
                await MainActor.run {
                    progress = MoveProgress(done: index, total: entries.count,
                                            current: entry.modelId, verb: "Moving")
                }
                do {
                    try manager.moveModel(entry.modelId, from: source, to: dest)
                    movedNames.append(entry.modelId)
                } catch {
                    errors.append("\(entry.modelId): \(error.localizedDescription)")
                }
                await MainActor.run {
                    progress = MoveProgress(done: index + 1, total: entries.count,
                                            current: entry.modelId, verb: "Moving")
                }
            }
            _ = manager.discoverModels()
            await manager.refreshModelCaches()
            await MainActor.run {
                finishBatch(deleted: false, names: movedNames, errors: errors)
            }
        }
    }
}
