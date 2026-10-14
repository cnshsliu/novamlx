import SwiftUI
import NovaMLXCore
import NovaMLXEngine

/// Load-time yes/no for companions that have no measured tokens-per-second win.
struct AcceleratorPromptSheet: View {
    let prompt: AcceleratorPrompt
    let onDecide: (Set<String>) -> Void

    @EnvironmentObject var l10n: L10n
    @State private var enabled: [String: Bool] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(l10n.tr("models.acceleratorTitle"))
                .font(.title3.weight(.semibold))
            Text(l10n.tr("models.acceleratorBody"))
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if let attached = prompt.autoAttachedId {
                Text(l10n.tr("models.acceleratorAuto") + " " + attached)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ForEach(prompt.options, id: \.acceleratorId) { option in
                Toggle(isOn: binding(option.acceleratorId)) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(option.displayName)
                            .font(.body.weight(.medium))
                        Text(option.reason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text("\(option.kind) · ~\(option.estimatedSizeMB) MB")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                .toggleStyle(.switch)
            }

            HStack {
                Button(l10n.tr("models.acceleratorSkip")) {
                    onDecide([])
                }
                .keyboardShortcut(.cancelAction)
                Spacer()
                Button(l10n.tr("models.acceleratorUse")) {
                    let chosen = Set(enabled.filter(\.value).map(\.key))
                    onDecide(chosen)
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(minWidth: 460)
        .onAppear {
            if enabled.isEmpty {
                for option in prompt.options {
                    enabled[option.acceleratorId] = false
                }
            }
        }
    }

    private func binding(_ id: String) -> Binding<Bool> {
        Binding(
            get: { enabled[id] ?? false },
            set: { enabled[id] = $0 }
        )
    }
}
