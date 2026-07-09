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

// MARK: - ItemInput (Tag/Chip Multi-Select)

/// Multi-select tag/chip input with dropdown suggestions.
/// Ported from Svelte ItemInput.svelte.
struct ItemInput: View {
    @Binding var items: [String]
    let suggestions: [String]
    let placeholder: String

    @State private var query = ""
    @State private var isFocused = false
    @State private var highlightIndex = -1
    @FocusState private var fieldFocused: Bool

    init(items: Binding<[String]>, suggestions: [String] = [], placeholder: String = "Type or select...") {
        self._items = items
        self.suggestions = suggestions
        self.placeholder = placeholder
    }

    private var filtered: [String] {
        guard !suggestions.isEmpty else { return [] }
        let q = query.lowercased()
        return suggestions
            .filter { !items.contains($0) }
            .filter { q.isEmpty || $0.lowercased().contains(q) }
    }

    @State private var dropdownDismissed = false

    private var showDropdown: Bool {
        isFocused && !filtered.isEmpty && !dropdownDismissed
    }

    var body: some View {
        VStack(spacing: 0) {
            inputField
            if showDropdown {
                dropdown
            }
        }
    }

    private var inputField: some View {
        FlowLayout(spacing: 4) {
            // Tag chips
            ForEach(items, id: \.self) { item in
                tagChip(item)
            }

            // Text input — always flows right after the last chip
            TextField(placeholder, text: $query)
                .font(.system(size: 12))
                .textFieldStyle(.plain)
                .focused($fieldFocused)
                .frame(minWidth: 80, idealWidth: 160, maxWidth: .infinity)
                .onSubmit { addCurrentOrHighlighted() }
                .onChange(of: query) { _, _ in
                    dropdownDismissed = false
                }
                .onChange(of: fieldFocused) { _, focused in
                    isFocused = focused
                    if focused { dropdownDismissed = false }
                    if !focused && !query.trimmingCharacters(in: .whitespaces).isEmpty {
                        add(query)
                    }
                }

            // Clear all button
            if !items.isEmpty {
                Button {
                    items = []
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                        .frame(width: 14, height: 14)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(fieldFocused ? NovaTheme.Colors.accent : NovaTheme.Colors.cardBorder, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .onTapGesture {
            if showDropdown {
                dropdownDismissed = true
            } else if fieldFocused {
                dropdownDismissed = false
            }
        }
        .onKeyPress(.escape) {
            if showDropdown {
                dropdownDismissed = true
                return .handled
            }
            return .ignored
        }
    }

    private func tagChip(_ item: String) -> some View {
        HStack(spacing: 3) {
            Text(item)
                .font(.system(size: 10))
                .lineLimit(1)
            Button {
                remove(item)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundColor(.secondary)
                    .frame(width: 12, height: 12)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(NovaTheme.Colors.accent.opacity(0.15))
        .overlay(Capsule().stroke(NovaTheme.Colors.accent.opacity(0.3), lineWidth: 0.5))
        .clipShape(Capsule())
    }

    private var dropdown: some View {
        VStack(spacing: 0) {
            ForEach(Array(filtered.enumerated()), id: \.offset) { index, item in
                Button {
                    add(item)
                } label: {
                    Text(item)
                        .font(.system(size: 11))
                        .foregroundColor(index == highlightIndex ? .white : NovaTheme.Colors.textPrimary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(index == highlightIndex ? NovaTheme.Colors.accent : Color.clear)
                }
                .buttonStyle(.plain)
                .onHover { hovering in
                    if hovering { highlightIndex = index }
                }
            }
        }
        .padding(.vertical, 4)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func add(_ item: String) {
        let trimmed = item.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, !items.contains(trimmed) else { return }
        items.append(trimmed)
        query = ""
        highlightIndex = -1
    }

    private func remove(_ item: String) {
        items.removeAll { $0 == item }
    }

    private func addCurrentOrHighlighted() {
        if highlightIndex >= 0 && highlightIndex < filtered.count {
            add(filtered[highlightIndex])
        } else if !query.trimmingCharacters(in: .whitespaces).isEmpty {
            add(query)
        }
    }
}

// MARK: - PlainSlider

/// Custom slider drawn from SwiftUI primitives to dodge the macOS native `Slider`
/// bottom-line rendering glitch inside grouped forms. Pointer-driven; no native chrome.
struct PlainSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    init(value: Binding<Double>, range: ClosedRange<Double>, step: Double = 0.01) {
        self._value = value
        self.range = range
        self.step = max(step, .leastNormalMagnitude)
    }

    var body: some View {
        GeometryReader { geo in
            let thumbDiameter: CGFloat = 14
            let trackHeight: CGFloat = 5
            let effectiveWidth = max(geo.size.width - thumbDiameter, 0)
            let span = max(range.upperBound - range.lowerBound, .leastNormalMagnitude)
            let progress = (value - range.lowerBound) / span
            let thumbOffset = CGFloat(max(0, min(1, progress))) * effectiveWidth

            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2.5)
                    .fill(Color.gray.opacity(0.35))
                    .frame(height: trackHeight)

                RoundedRectangle(cornerRadius: 2.5)
                    .fill(NovaTheme.Colors.accent)
                    .frame(width: thumbOffset + thumbDiameter / 2, height: trackHeight)

                Circle()
                    .fill(Color.white)
                    .frame(width: thumbDiameter, height: thumbDiameter)
                    .shadow(color: Color.black.opacity(0.2), radius: 2, x: 0, y: 1)
                    .offset(x: thumbOffset)
            }
            .frame(height: max(thumbDiameter, trackHeight))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        let denom = max(effectiveWidth, 1)
                        let rawProgress = Double((g.location.x - thumbDiameter / 2) / denom)
                        let clamped = max(0, min(1, rawProgress))
                        let raw = range.lowerBound + clamped * span
                        let stepped = (round(raw / step) * step)
                        value = max(range.lowerBound, min(range.upperBound, stepped))
                    }
            )
        }
        .frame(height: 14)
    }
}
