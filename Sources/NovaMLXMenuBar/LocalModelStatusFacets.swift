import SwiftUI
import NovaMLXCore

/// Downloaded and Verified chips for the Local Inference model list.
struct LocalModelFacetBar: View {
    @Binding var downloaded: LocalModelFacet
    @Binding var verified: LocalModelFacet
    @EnvironmentObject var l10n: L10n

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 18) {
                facetGroup(
                    title: l10n.tr("models.facetDownloaded"),
                    selection: $downloaded,
                    yes: l10n.tr("models.downloaded"),
                    no: l10n.tr("models.notDownloadedBadge"),
                    identifier: "models-facet-downloaded"
                )
                facetGroup(
                    title: l10n.tr("models.facetVerified"),
                    selection: $verified,
                    yes: l10n.tr("models.verifiedBadge"),
                    no: l10n.tr("models.unverifiedBadge"),
                    identifier: "models-facet-verified"
                )
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 4)
        }
        .accessibilityIdentifier("models-status-facets")
    }

    private func facetGroup(
        title: String,
        selection: Binding<LocalModelFacet>,
        yes: String,
        no: String,
        identifier: String
    ) -> some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)
            facetChip(l10n.tr("models.diskAll"), value: .any, selection: selection, identifier: "\(identifier)-any")
            facetChip(yes, value: .yes, selection: selection, identifier: "\(identifier)-yes")
            facetChip(no, value: .no, selection: selection, identifier: "\(identifier)-no")
        }
    }

    private func facetChip(
        _ title: String,
        value: LocalModelFacet,
        selection: Binding<LocalModelFacet>,
        identifier: String
    ) -> some View {
        let selected = selection.wrappedValue == value
        return Button {
            selection.wrappedValue = value
        } label: {
            Text(title)
                .font(.system(size: 11, weight: selected ? .semibold : .regular))
                .foregroundColor(selected ? .white : NovaTheme.Colors.accent)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(selected ? NovaTheme.Colors.accent : NovaTheme.Colors.accent.opacity(0.15))
                .overlay(Capsule().stroke(NovaTheme.Colors.accent.opacity(selected ? 0 : 0.3), lineWidth: 0.5))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(identifier)
        .accessibilityAddTraits(selected ? AccessibilityTraits.isSelected : AccessibilityTraits())
    }
}

/// Verified / downloaded badges on every row, plus loaded only while resident.
struct LocalModelStatusBadges: View {
    let modelId: String
    let isVerified: Bool
    let isDownloaded: Bool
    let isLoaded: Bool
    @EnvironmentObject var l10n: L10n

    var body: some View {
        let row = LocalModelRowState(
            id: modelId,
            isDownloaded: isDownloaded,
            isVerified: isVerified,
            isLoaded: isLoaded
        )
        FlowLayout(spacing: 4) {
            badge(
                text: l10n.tr(row.isVerified ? "models.verifiedBadge" : "models.unverifiedBadge"),
                color: row.isVerified ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusWarn,
                identifier: "models-badge-verified-\(modelId)"
            )
            badge(
                text: l10n.tr(row.isDownloaded ? "models.downloaded" : "models.notDownloadedBadge"),
                color: row.isDownloaded ? NovaTheme.Colors.statusOK : .secondary,
                identifier: "models-badge-downloaded-\(modelId)"
            )
            if row.showsLoadedBadge {
                badge(
                    text: l10n.tr("models.loadedBadge"),
                    color: NovaTheme.Colors.accent,
                    identifier: "models-badge-loaded-\(modelId)"
                )
            }
        }
        .accessibilityIdentifier("models-status-\(modelId)")
    }

    private func badge(text: String, color: Color, identifier: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .foregroundColor(color)
            .background(color.opacity(0.14))
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .accessibilityIdentifier(identifier)
    }
}
