import SwiftUI

// MARK: - View Modifiers

extension View {
    /// Section-level card with material background, luminous border, and radius.
    func sectionCard() -> some View {
        self
            .padding(NovaTheme.Spacing.lg)
            .background(.ultraThinMaterial)
            .background(NovaTheme.Colors.cardBackground.opacity(0.6))
            .overlay(
                RoundedRectangle(cornerRadius: NovaTheme.Radius.lg)
                    .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
    }

    /// Inner row within a section card.
    func rowCard() -> some View {
        self
            .padding(.horizontal, NovaTheme.Spacing.md)
            .padding(.vertical, 10)
            .background(NovaTheme.Colors.rowBackground)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.sm))
    }
}

// MARK: - Section Header

/// Unified section header with themed icon, title, and optional count badge.
func sectionHeader(_ title: String, icon: String, count: Int? = nil) -> some View {
    HStack(spacing: NovaTheme.Spacing.sm) {
        Image(systemName: icon)
            .foregroundColor(NovaTheme.Colors.accent)
            .font(.system(size: 13))
        Text(title)
            .font(.headline)
            .foregroundColor(NovaTheme.Colors.textPrimary)
        if let count {
            Text("\(count)")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(NovaTheme.Colors.accent)
                .clipShape(Capsule())
        }
    }
}

// MARK: - Status Badge

/// Luminous status badge for Running/Stopped/Error states.
struct StatusBadge: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(text)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundColor(color)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(color.opacity(0.12))
        .overlay(
            Capsule().stroke(color.opacity(0.25), lineWidth: 1)
        )
        .clipShape(Capsule())
    }
}

// MARK: - Metric Card

/// Stat card with icon, title, value, and optional subtitle.
struct MetricCard: View {
    var icon: String? = nil
    let title: String
    let value: String
    var subtitle: String? = nil
    var valueColor: Color = NovaTheme.Colors.textPrimary

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 9))
                        .foregroundColor(NovaTheme.Colors.accent)
                }
                Text(title)
                    .font(.system(size: 9))
                    .foregroundColor(NovaTheme.Colors.textTertiary)
                    .textCase(.uppercase)
            }
            Text(value)
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(valueColor)
            if let subtitle {
                Text(subtitle)
                    .font(.system(size: 10))
                    .foregroundColor(NovaTheme.Colors.accent)
            }
        }
        .padding(NovaTheme.Spacing.md)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }
}

// MARK: - Copy ID Button

/// Button that copies a string to the clipboard with visual feedback.
struct CopyIDButton: View {
    let id: String
    @State private var copied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(id, forType: .string)
            copied = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                copied = false
            }
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 10))
                .foregroundColor(copied ? NovaTheme.Colors.statusOK : NovaTheme.Colors.textTertiary)
                .frame(width: 16, height: 16)
        }
        .buttonStyle(.plain)
        .help("Copy: \(id)")
    }
}

// MARK: - Flow Layout

struct FlowLayout: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrange(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
        }

        return (CGSize(width: maxWidth, height: y + rowHeight), positions)
    }
}
