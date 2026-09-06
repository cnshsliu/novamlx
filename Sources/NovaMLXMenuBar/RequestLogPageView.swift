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
/// Click to expand the request body detail panel.
private struct ActiveRequestRow: View {
    let entry: RequestLogEntry
    let now: Date
    @State private var isExpanded: Bool = false

    private var elapsed: TimeInterval { now.timeIntervalSince(entry.startedAt) }

    var body: some View {
        VStack(spacing: 0) {
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

                chevron
            }
            .padding(.horizontal, NovaTheme.Spacing.md)
            .padding(.vertical, NovaTheme.Spacing.sm + 1)
            .background(NovaTheme.Colors.rowBackground)
            .contentShape(Rectangle())
            .onTapGesture { withAnimation(.easeInOut(duration: 0.15)) { isExpanded.toggle() } }
            .overlay(
                RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                    .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
            )

            if isExpanded {
                RequestDetailPanel(entry: entry)
                    .padding(.top, 1)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }

    private var chevron: some View {
        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
            .font(.system(size: 9, weight: .semibold))
            .foregroundColor(NovaTheme.Colors.textTertiary)
            .frame(width: 12)
    }
}

/// A single completed request row. Click to expand the request body detail.
private struct CompletedRequestRow: View {
    let entry: RequestLogEntry
    @State private var isExpanded: Bool = false

    var body: some View {
        VStack(spacing: 0) {
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

                kindBadge(entry.kind, placeholder: entry.kind == nil ? Color.clear : nil)

                Text(entry.model ?? "")
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

                chevron
            }
            .padding(.horizontal, NovaTheme.Spacing.md)
            .padding(.vertical, NovaTheme.Spacing.sm)
            .background(NovaTheme.Colors.rowBackground)
            .contentShape(Rectangle())
            .onTapGesture { withAnimation(.easeInOut(duration: 0.15)) { isExpanded.toggle() } }
            .overlay(
                RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                    .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
            )

            if isExpanded {
                RequestDetailPanel(entry: entry)
                    .padding(.top, 1)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.md))
    }

    private var chevron: some View {
        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
            .font(.system(size: 9, weight: .semibold))
            .foregroundColor(NovaTheme.Colors.textTertiary)
            .frame(width: 12)
    }
}

/// Expandable detail panel shown below a row when clicked. Shows HTTP status,
/// content type, error message, and pretty-printed request body.
private struct RequestDetailPanel: View {
    let entry: RequestLogEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            metaRow
            if let err = entry.error, !err.isEmpty {
                errorRow(err)
            }
            bodySection
        }
        .padding(NovaTheme.Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                .fill(NovaTheme.Colors.rowBackground.opacity(0.5))
                .overlay(
                    RoundedRectangle(cornerRadius: NovaTheme.Radius.md)
                        .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
                )
        )
    }

    private var metaRow: some View {
        HStack(spacing: 12) {
            if let status = entry.responseStatus {
                statusBadge(status)
            }
            if let ct = entry.requestContentType, !ct.isEmpty {
                metaTag("Content-Type", value: ct)
            }
            if let pt = entry.promptTokens {
                metaTag("prompt", value: "\(pt) tok")
            }
            if let ct = entry.completionTokens {
                metaTag("completion", value: "\(ct) tok")
            }
            if let tps = entry.tps, tps > 0 {
                metaTag("speed", value: String(format: "%.1f tok/s", tps))
            }
            Spacer()
        }
    }

    private func statusBadge(_ code: Int) -> some View {
        let color: Color = {
            if (200..<300).contains(code) { return NovaTheme.Colors.statusOK }
            if (400..<500).contains(code) { return NovaTheme.Colors.statusWarn }
            return NovaTheme.Colors.statusError
        }()
        return Text("HTTP \(code)")
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundColor(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 3))
    }

    private func metaTag(_ label: String, value: String) -> some View {
        HStack(spacing: 3) {
            Text(label)
                .font(.system(size: 9.5))
                .foregroundColor(NovaTheme.Colors.textTertiary)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textSecondary)
        }
    }

    private func errorRow(_ err: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 9))
                .foregroundColor(NovaTheme.Colors.statusError)
            Text(err)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.statusError)
                .textSelection(.enabled)
        }
        .padding(6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(NovaTheme.Colors.statusError.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    @ViewBuilder
    private var bodySection: some View {
        if let note = entry.requestBodyNote {
            Text(note)
                .font(.system(size: 10.5, design: .monospaced))
                .foregroundColor(NovaTheme.Colors.textTertiary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(NovaTheme.Colors.rowBackground)
                .clipShape(RoundedRectangle(cornerRadius: 4))
        } else if let body = entry.requestBody, !body.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Request body")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(NovaTheme.Colors.textTertiary)
                    Spacer()
                    Button(action: {
                        #if canImport(AppKit)
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setData(body, forType: .string)
                        #endif
                    }) {
                        Label("Copy", systemImage: "doc.on.doc")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
                Text(prettyBody(body))
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.textPrimary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(NovaTheme.Colors.rowBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .lineLimit(40)
            }
        } else {
            Text("No request body")
                .font(.system(size: 10.5))
                .foregroundColor(NovaTheme.Colors.textTertiary)
        }
    }

    /// Pretty-print JSON bodies; fall back to raw UTF-8 for non-JSON.
    private func prettyBody(_ data: Data) -> String {
        // Only attempt JSON reformatting if the content type suggests JSON.
        let isJSON = (entry.requestContentType ?? "").lowercased().contains("json")
            || String(data: data.prefix(1), encoding: .utf8) == "{"
            || String(data: data.prefix(1), encoding: .utf8) == "["
        if isJSON,
           let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
           let pretty = try? JSONSerialization.data(withJSONObject: parsed, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: pretty, encoding: .utf8) {
            return str
        }
        return String(data: data, encoding: .utf8) ?? "<\(data.count) bytes binary>"
    }
}

// MARK: - Shared small components

private func kindBadge(_ kind: InferenceKind?, placeholder: Color? = nil) -> some View {
    let label = kind?.label ?? ""
    let color: Color = {
        switch kind {
        case .asr: return NovaTheme.Colors.statusWarn
        case .tts: return .purple
        case .image: return .pink
        case .vlm: return .teal
        case .llm: return NovaTheme.Colors.accent
        case nil: return placeholder ?? NovaTheme.Colors.textTertiary
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
