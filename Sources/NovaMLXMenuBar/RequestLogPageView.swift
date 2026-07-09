import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils

/// Request Log page — shows live in-flight requests on top and a scrollable
/// history of completed requests below. Driven by `RequestLogStore`, which is
/// fed by the HTTP middleware (start) and the inference layer (finish).
struct RequestLogPageView: View {
    @ObservedObject var appState: MenuBarAppState

    /// Refresh timer so the active-requests section updates live (the store
    /// mutates on the inference queue; we poll on the main thread).
    @State private var refreshTick: Date = Date()
    @State private var onlyErrors: Bool = false

    private let store = RequestLogStore.shared

    private var active: [RequestLogEntry] { store.activeRequests }
    private var completed: [RequestLogEntry] {
        let all = store.completedRequests
        return onlyErrors ? all.filter { $0.status == .error || $0.status == .cancelled } : all
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: NovaTheme.Spacing.xl) {
                header
                activeSection
                historySection
            }
            .padding(NovaTheme.Spacing.xxl)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: NovaTheme.Spacing.xs) {
                Text("Request Log")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                Text("\(active.count) active  ·  \(store.completedRequests.count) recent")
                    .font(.subheadline)
                    .foregroundColor(NovaTheme.Colors.textSecondary)
            }
            Spacer()
            Button(action: { store.clearCompleted() }) {
                Text("Clear")
                    .font(.system(size: 12))
            }
            .buttonStyle(.bordered)
            .help("Clear completed request history")
        }
    }

    // MARK: - Active requests

    private var activeSection: some View {
        VStack(alignment: .leading, spacing: NovaTheme.Spacing.md) {
            sectionHeader("Active Requests", count: active.count, systemName: "bolt.horizontal.fill")

            if active.isEmpty {
                emptyState("No requests in flight")
            } else {
                VStack(spacing: 6) {
                    ForEach(active) { entry in
                        ActiveRequestRow(entry: entry, now: refreshTick)
                    }
                }
            }
        }
        .onReceive(Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()) { _ in
            refreshTick = Date()
            // Safety net: prune any in-flight entries whose finalization hook
            // never fired (crashed worker, broken stream, etc.) so they don't
            // spin forever. The 120s default only catches genuinely orphaned rows.
            store.cancelStale(olderThan: 120)
        }
    }

    // MARK: - History

    private var historySection: some View {
        VStack(alignment: .leading, spacing: NovaTheme.Spacing.md) {
            HStack(alignment: .center) {
                sectionHeader("Recent Requests", count: store.completedRequests.count, systemName: "clock.arrow.circlepath")
                Spacer()
                Toggle("Errors only", isOn: $onlyErrors)
                    .toggleStyle(.switch)
                    .font(.system(size: 11))
            }

            if completed.isEmpty {
                emptyState(onlyErrors ? "No errors in recent history" : "No requests recorded yet")
            } else {
                VStack(spacing: 6) {
                    ForEach(completed) { entry in
                        CompletedRequestRow(entry: entry)
                    }
                }
            }
        }
    }

    // MARK: - Shared subviews

    private func sectionHeader(_ title: String, count: Int, systemName: String) -> some View {
        HStack(spacing: NovaTheme.Spacing.sm) {
            Image(systemName: systemName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(NovaTheme.Colors.accent)
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(NovaTheme.Colors.textPrimary)
            Text("(\(count))")
                .font(.system(size: 12))
                .foregroundColor(NovaTheme.Colors.textTertiary)
        }
    }

    private func emptyState(_ message: String) -> some View {
        Text(message)
            .font(.system(size: 13))
            .foregroundColor(NovaTheme.Colors.textTertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, NovaTheme.Spacing.lg)
    }
}

/// A single active (in-flight) request row — shows model, kind, key, and a
/// live duration/elapsed counter that re-renders via the parent timer tick.
private struct ActiveRequestRow: View {
    let entry: RequestLogEntry
    let now: Date

    private var elapsed: TimeInterval { now.timeIntervalSince(entry.startedAt) }

    var body: some View {
        HStack(spacing: NovaTheme.Spacing.md) {
            ProgressView()
                .scaleEffect(0.55)
                .frame(width: 14, height: 14)

            kindBadge(entry.kind)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(entry.model ?? "resolving…")
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundColor(NovaTheme.Colors.textPrimary)
                        .lineLimit(1)
                    Text(entry.endpoint)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundColor(NovaTheme.Colors.textSecondary)
                        .lineLimit(1)
                }
                HStack(spacing: 6) {
                    Text(entry.apiKeyName ?? "no-key")
                        .font(.system(size: 10.5))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                    if entry.kind == nil {
                        Text("pending")
                            .font(.system(size: 10.5))
                            .foregroundColor(NovaTheme.Colors.textTertiary)
                    }
                }
            }

            Spacer()

            if entry.tps ?? 0 > 0 {
                Text(String(format: "%.1f tok/s", entry.tps ?? 0))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.accent)
            }
            Text(String(format: "%.1fs", elapsed))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textSecondary)
        }
        .padding(.horizontal, NovaTheme.Spacing.md)
        .padding(.vertical, NovaTheme.Spacing.sm + 1)
        .background(NovaTheme.Colors.rowBackground)
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }
}

/// A single completed request row.
private struct CompletedRequestRow: View {
    let entry: RequestLogEntry

    var body: some View {
        HStack(spacing: NovaTheme.Spacing.md) {
            statusIcon(entry.status)
                .frame(width: 14)

            Text(relativeTime(entry.startedAt))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textTertiary)
                .frame(width: 38, alignment: .leading)

            Text(entry.method)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(methodColor(entry.method))
                .frame(width: 42, alignment: .leading)

            Text(entry.endpoint)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textSecondary)
                .frame(minWidth: 130, alignment: .leading)
                .lineLimit(1)

            if let kind = entry.kind {
                kindBadge(kind)
            } else {
                Text("—").font(.system(size: 11)).foregroundColor(NovaTheme.Colors.textTertiary)
                    .frame(width: 36, alignment: .center)
            }

            Text(entry.model ?? "—")
                .font(.system(size: 11.5, weight: .medium))
                .foregroundColor(NovaTheme.Colors.textPrimary)
                .frame(minWidth: 90, alignment: .leading)
                .lineLimit(1)

            Text(entry.apiKeyName ?? "no-key")
                .font(.system(size: 10.5))
                .foregroundColor(NovaTheme.Colors.textTertiary)
                .frame(maxWidth: 110, alignment: .leading)
                .lineLimit(1)
                .help(entry.apiKeyName ?? "no-key")

            Spacer()

            if entry.tps ?? 0 > 0 {
                Text(String(format: "%.1f tok/s", entry.tps ?? 0))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                    .frame(width: 70, alignment: .trailing)
            }

            if let dur = entry.durationMs {
                Text(durationLabel(dur))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.textSecondary)
                    .frame(width: 56, alignment: .trailing)
            }

            if let err = entry.error, !err.isEmpty {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 10))
                    .foregroundColor(NovaTheme.Colors.statusWarn)
                    .help(err)
            }
        }
        .padding(.horizontal, NovaTheme.Spacing.md)
        .padding(.vertical, NovaTheme.Spacing.sm)
        .background(NovaTheme.Colors.rowBackground)
        .overlay(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }
}

// MARK: - Shared small components

private func kindBadge(_ kind: InferenceKind?) -> some View {
    let label = kind?.label ?? "—"
    let color: Color = {
        switch kind {
        case .asr: return NovaTheme.Colors.statusWarn
        case .tts: return .purple
        case .image: return .pink
        case .vlm: return .teal
        case .llm: return NovaTheme.Colors.accent
        case nil: return NovaTheme.Colors.textTertiary
        }
    }()
    return Text(label)
        .font(.system(size: 10, weight: .bold))
        .foregroundColor(color)
        .frame(width: 36, alignment: .center)
}

@ViewBuilder
private func statusIcon(_ status: RequestStatus) -> some View {
    switch status {
    case .success:
        Image(systemName: "checkmark.circle.fill")
            .font(.system(size: 11))
            .foregroundColor(NovaTheme.Colors.statusOK)
    case .error:
        Image(systemName: "xmark.circle.fill")
            .font(.system(size: 11))
            .foregroundColor(NovaTheme.Colors.statusError)
    case .cancelled:
        Image(systemName: "minus.circle.fill")
            .font(.system(size: 11))
            .foregroundColor(NovaTheme.Colors.textTertiary)
    case .pending:
        ProgressView()
            .scaleEffect(0.5)
            .frame(width: 11, height: 11)
            .foregroundColor(NovaTheme.Colors.textTertiary)
    }
}

private func methodColor(_ method: String) -> Color {
    switch method.uppercased() {
    case "GET": return NovaTheme.Colors.accent
    case "POST": return NovaTheme.Colors.statusOK
    case "DELETE": return NovaTheme.Colors.statusError
    case "PUT": return NovaTheme.Colors.statusWarn
    default: return NovaTheme.Colors.textSecondary
    }
}

private func relativeTime(_ date: Date) -> String {
    let interval = Date().timeIntervalSince(date)
    if interval < 5 { return "now" }
    if interval < 60 { return "\(Int(interval))s" }
    if interval < 3600 { return "\(Int(interval / 60))m" }
    return "\(Int(interval / 3600))h"
}

private func durationLabel(_ ms: Double) -> String {
    if ms < 1000 { return "\(Int(ms))ms" }
    if ms < 60_000 { return String(format: "%.1fs", ms / 1000) }
    return String(format: "%.0fm", ms / 60_000)
}
