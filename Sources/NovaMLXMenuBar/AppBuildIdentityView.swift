import SwiftUI
import NovaMLXCore

/// Debug/Release, build clock, and the bundle path of the running app.
struct AppBuildIdentityView: View {
    enum Style {
        /// Badge, clock, and path stacked. Used in the sidebar and settings.
        case stacked
        /// Two lines of small type pinned under a page.
        case footer
    }

    var style: Style = .stacked

    private let identity = AppBuildIdentity.current

    var body: some View {
        Group {
            switch style {
            case .stacked:
                stacked
            case .footer:
                footer
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("app-build-identity")
        .help(identity.summaryLine + "\n" + identity.bundlePath)
    }

    private var footer: some View {
        Text("\(identity.configurationLabel) · v\(NovaMLXCore.version) · \(identity.stamp) · \(identity.bundlePath)")
            .font(.system(size: 10, design: .monospaced))
            .foregroundColor(.secondary)
            .lineLimit(1)
            .truncationMode(.middle)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var stacked: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(identity.configurationLabel)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(identity.configuration == .debug ? .orange : NovaTheme.Colors.statusOK)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(
                        (identity.configuration == .debug ? Color.orange : NovaTheme.Colors.statusOK)
                            .opacity(0.16)
                    )
                    .clipShape(Capsule())
                Text("v\(NovaMLXCore.version)")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            Text(identity.stamp)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundColor(.primary)
            Text(identity.bundlePath)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)
                .lineLimit(4)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
