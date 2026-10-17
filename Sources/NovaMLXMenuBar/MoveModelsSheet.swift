import SwiftUI
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils

/// Two-pane model mover: each side picks a storage vault (root), the middle
/// column moves the selected model (or all of them) left/right.
struct MoveModelsSheet: View {
    let modelManager: ModelManager
    /// Blocks moves of models currently loaded in the engine.
    let isModelLoaded: (String) -> Bool

    @Environment(\.dismiss) private var dismiss
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
    /// Live move progress; nil when idle.
    @State private var progress: MoveProgress?
    /// Names of the models moved by the last completed batch.
    @State private var lastMovedNames: [String]?

    struct MoveProgress: Equatable {
        let done: Int
        let total: Int
        let current: String
    }

    private var vaults: [URL] { NovaMLXPaths.configuredModelsDirs }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if vaults.count < 2 {
                Text("Add at least two storage directories in Settings to move models between them.")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HStack(spacing: 0) {
                    vaultPane(selection: $leftVault, models: leftSort.apply(leftModels),
                              paneSelection: $leftSelection, sortOrder: $leftSort)
                    moveButtons
                    vaultPane(selection: $rightVault, models: rightSort.apply(rightModels),
                              paneSelection: $rightSelection, sortOrder: $rightSort)
                }
                .frame(maxHeight: .infinity)
            }
            Divider()
            statusBar
            if message != nil { footer }
        }
        .frame(minWidth: 720, minHeight: 460)
        .onAppear {
            leftVault = vaults.first
            rightVault = vaults.count > 1 ? vaults[1] : nil
            reload()
        }
        .onChange(of: leftVault) { _ in reload() }
        .onChange(of: rightVault) { _ in reload() }
        // Selection is exclusive: picking rows on one side clears the other,
        // so the arrow buttons always reflect a single direction.
        .onChange(of: leftSelection) { sel in
            if !sel.isEmpty, !rightSelection.isEmpty { rightSelection = [] }
        }
        .onChange(of: rightSelection) { sel in
            if !sel.isEmpty, !leftSelection.isEmpty { leftSelection = [] }
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
            .padding(.top, 8)

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
                    Text("No models")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Move Buttons

    private var leftSelected: Bool { !leftSelection.isEmpty }
    private var rightSelected: Bool { !rightSelection.isEmpty }

    private var moveButtons: some View {
        VStack(spacing: 12) {
            Spacer()
            // Right-movers act on the left list; they are only available when
            // the selection sits on the left side.
            moveButton("→", help: "Move selected to the right vault",
                       enabled: leftSelected, direction: .right, all: false)
            moveButton("→→", help: "Move all to the right vault",
                       enabled: !leftModels.isEmpty && !rightSelected, direction: .right, all: true)
            Divider().frame(width: 36)
            moveButton("←", help: "Move selected to the left vault",
                       enabled: rightSelected, direction: .left, all: false)
            moveButton("←←", help: "Move all to the left vault",
                       enabled: !rightModels.isEmpty && !leftSelected, direction: .left, all: true)
            Spacer()
        }
        .padding(.horizontal, 10)
    }

    private enum Direction { case left, right }

    private func moveButton(_ title: String, help: String, enabled: Bool,
                            direction: Direction, all: Bool) -> some View {
        Button {
            move(direction: direction, all: all)
        } label: {
            Text(title)
                .font(.system(size: 14, design: .monospaced))
                .frame(width: 44, height: 28)
        }
        .buttonStyle(.bordered)
        .help(help)
        .disabled(busy || sameVault || !enabled)
    }

    private var sameVault: Bool {
        guard let l = leftVault, let r = rightVault else { return true }
        return l.standardizedFileURL.path == r.standardizedFileURL.path
    }

    // MARK: - Header / Status Bar / Footer

    private var header: some View {
        HStack {
            Text("Move Models")
                .font(.system(size: 13, weight: .semibold))
            Spacer()
            if busy {
                ProgressView()
                    .controlSize(.small)
            }
            Button("Done") { dismiss() }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
        .padding(12)
    }

    /// Full-name readout under the lists: current selection while idle, live
    /// progress during a move, and the completed batch afterwards.
    private var statusBar: some View {
        HStack(alignment: .center, spacing: 8) {
            if let progress {
                ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                    .frame(width: 120)
                Text("Moving \(progress.done)/\(progress.total): \(progress.current)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .truncationMode(.tail)
            } else if let moved = lastMovedNames, !moved.isEmpty {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(NovaTheme.Colors.statusOK)
                Text("移动完成：\(moved.joined(separator: ", "))")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.statusOK)
                    .lineLimit(3)
            } else {
                selectionSummary
            }
            Spacer(minLength: 12)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(minHeight: 44, alignment: .center)
        .background(Color.primary.opacity(0.03))
    }

    /// Both panes' selected model names, untruncated (wraps up to 3 lines).
    private var selectionSummary: some View {
        let leftNames = leftModels.filter { leftSelection.contains($0.id) }.map(\.modelId)
        let rightNames = rightModels.filter { rightSelection.contains($0.id) }.map(\.modelId)
        return HStack(spacing: 8) {
            if leftNames.isEmpty && rightNames.isEmpty {
                Text("Select models to move")
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
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(minHeight: 30, alignment: .center)
    }

    // MARK: - Actions

    private func reload() {
        if let left = leftVault {
            leftModels = modelManager.modelsInVault(left)
        } else { leftModels = [] }
        if let right = rightVault {
            rightModels = modelManager.modelsInVault(right)
        } else { rightModels = [] }
        let validLeft = Set(leftModels.map(\.id))
        let validRight = Set(rightModels.map(\.id))
        leftSelection.formIntersection(validLeft)
        rightSelection.formIntersection(validRight)
    }

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
        lastMovedNames = nil
        progress = MoveProgress(done: 0, total: entries.count, current: entries.first?.modelId ?? "")
        let manager = modelManager
        Task.detached(priority: .userInitiated) {
            var movedNames: [String] = []
            var errors: [String] = []
            for (index, entry) in entries.enumerated() {
                await MainActor.run {
                    progress = MoveProgress(done: index, total: entries.count, current: entry.modelId)
                }
                do {
                    try manager.moveModel(entry.modelId, from: source, to: dest)
                    movedNames.append(entry.modelId)
                } catch {
                    errors.append("\(entry.modelId): \(error.localizedDescription)")
                }
                await MainActor.run {
                    progress = MoveProgress(done: index + 1, total: entries.count, current: entry.modelId)
                }
            }
            _ = manager.discoverModels()
            await manager.refreshModelCaches()
            await MainActor.run {
                busy = false
                progress = nil
                reload()
                leftSelection = []
                rightSelection = []
                lastMovedNames = movedNames
                if errors.isEmpty {
                    message = nil
                } else {
                    message = "Error: \(movedNames.count) moved, \(errors.count) failed — "
                        + errors.prefix(3).joined(separator: "; ")
                }
            }
        }
    }
}
