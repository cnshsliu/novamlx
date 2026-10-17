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
    @State private var busy = false
    @State private var message: String?

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
                    vaultPane(selection: $leftVault, models: leftModels,
                              paneSelection: $leftSelection)
                    moveButtons
                    vaultPane(selection: $rightVault, models: rightModels,
                              paneSelection: $rightSelection)
                }
                .frame(maxHeight: .infinity)
            }
            Divider()
            footer
        }
        .frame(minWidth: 720, minHeight: 420)
        .onAppear {
            leftVault = vaults.first
            rightVault = vaults.count > 1 ? vaults[1] : nil
            reload()
        }
        .onChange(of: leftVault) { _ in reload() }
        .onChange(of: rightVault) { _ in reload() }
    }

    // MARK: - Panes

    private func vaultPane(
        selection: Binding<URL?>,
        models: [ModelManager.VaultModelEntry],
        paneSelection: Binding<Set<String>>
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Picker("", selection: selection) {
                ForEach(vaults, id: \.path) { vault in
                    Text(vault.path).tag(vault as URL?)
                }
            }
            .labelsHidden()
            .padding(8)

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

    private var moveButtons: some View {
        VStack(spacing: 12) {
            Spacer()
            moveButton("→", help: "Move selected to the right vault", direction: .right, all: false)
                .disabled(rightSelection.isEmpty)
            moveButton("→→", help: "Move all to the right vault", direction: .right, all: true)
                .disabled(leftModels.isEmpty)
            Divider().frame(width: 36)
            moveButton("←", help: "Move selected to the left vault", direction: .left, all: false)
                .disabled(leftSelection.isEmpty)
            moveButton("←←", help: "Move all to the left vault", direction: .left, all: true)
                .disabled(rightModels.isEmpty)
            Spacer()
        }
        .padding(.horizontal, 10)
        .opacity(sameVault ? 0.3 : 1)
    }

    private enum Direction { case left, right }

    private func moveButton(_ title: String, help: String,
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
        .disabled(busy || sameVault)
    }

    private var sameVault: Bool {
        guard let l = leftVault, let r = rightVault else { return true }
        return l.standardizedFileURL.path == r.standardizedFileURL.path
    }

    // MARK: - Header / Footer

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
                .keyboardShortcut(.defaultAction)
        }
        .padding(12)
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
        message = "Moving \(entries.count) model\(entries.count == 1 ? "" : "s")…"
        let manager = modelManager
        Task.detached(priority: .userInitiated) {
            var moved = 0
            var errors: [String] = []
            for entry in entries {
                do {
                    try manager.moveModel(entry.modelId, from: source, to: dest)
                    moved += 1
                } catch {
                    errors.append("\(entry.modelId): \(error.localizedDescription)")
                }
            }
            _ = manager.discoverModels()
            await manager.refreshModelCaches()
            await MainActor.run {
                busy = false
                reload()
                if errors.isEmpty {
                    message = "Moved \(moved) model\(moved == 1 ? "" : "s")."
                } else {
                    message = "Error: \(moved) moved, \(errors.count) failed — "
                        + errors.prefix(3).joined(separator: "; ")
                }
            }
        }
    }
}
