# Dark Mode GUI Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all hardcoded/NSColor-bridged colors in the NovaMLX GUI with a centralized "Control Room / Deep Space" dark-mode theme system.

**Architecture:** Two new files (`NovaTheme.swift` for tokens, `NovaComponents.swift` for shared components) + edits to all 8 view files. No behavior changes — purely visual.

**Tech Stack:** SwiftUI on macOS 15, .ultraThinMaterial, ColorScheme-aware tokens.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `Sources/NovaMLXMenuBar/NovaTheme.swift` | All color, spacing, radius tokens |
| Create | `Sources/NovaMLXMenuBar/NovaComponents.swift` | Shared view components + modifiers |
| Modify | `Sources/NovaMLXMenuBar/NovaAppView.swift` | Sidebar + navigation frame |
| Modify | `Sources/NovaMLXMenuBar/StatusPageView.swift` | Status hero, metrics, device info |
| Modify | `Sources/NovaMLXMenuBar/ModelsPageView.swift` | Extract components, apply theme |
| Modify | `Sources/NovaMLXMenuBar/ChatPageView.swift` | Chat bubbles, input bar |
| Modify | `Sources/NovaMLXMenuBar/SettingsPageView.swift` | Section cards, TQ buttons |
| Modify | `Sources/NovaMLXMenuBar/DashboardView.swift` | Stats grid, model lists |
| Modify | `Sources/NovaMLXMenuBar/StatusMenuView.swift` | Menu bar popover status |
| Modify | `Sources/NovaMLXMenuBar/ModelsMenuView.swift` | Menu bar popover models |
| Modify | `Sources/NovaMLXMenuBar/MenuBarController.swift` | Quit button color |

---

## Task 1: NovaTheme.swift — Design Tokens

**Files:**
- Create: `Sources/NovaMLXMenuBar/NovaTheme.swift`

- [ ] **Step 1: Create the theme file**

```swift
import SwiftUI

/// Centralized design tokens for the NovaMLX "Control Room" aesthetic.
/// Dark mode is the primary design target. Light mode adapts to lighter values.
enum NovaTheme {
    // MARK: - Never instantiated
    @available(*, unavailable)
    init() {}
}

// MARK: - Color Tokens

extension NovaTheme {
    enum Colors {
        private static func adapt(dark: Color, light: Color) -> Color {
            // Use a resolved color that adapts to the current color scheme.
            // This works in SwiftUI views without needing @Environment.
            Color(light: light, dark: dark)
        }

        /// Base window background
        static let background = adapt(
            dark: Color(red: 0.039, green: 0.055, blue: 0.102),  // #0a0e1a
            light: Color(nsColor: .windowBackgroundColor)
        )

        /// Section card background (slightly elevated)
        static let cardBackground = adapt(
            dark: Color(red: 0.06, green: 0.075, blue: 0.12),
            light: Color(nsColor: .controlBackgroundColor)
        )

        /// Inner row background within cards
        static let rowBackground = adapt(
            dark: Color(red: 0.08, green: 0.09, blue: 0.15),
            light: Color(nsColor: .textBackgroundColor)
        )

        /// Luminous card border
        static let cardBorder = adapt(
            dark: Color.white.opacity(0.06),
            light: Color.black.opacity(0.08)
        )

        /// Electric blue accent (#00d4ff)
        static let accent = adapt(
            dark: Color(red: 0, green: 0.831, blue: 1.0),
            light: Color(red: 0, green: 0.6, blue: 0.85)
        )

        /// Accent at dim opacity (highlights, badges, selected items)
        static let accentDim = adapt(
            dark: Color(red: 0, green: 0.831, blue: 1.0).opacity(0.12),
            light: Color(red: 0, green: 0.6, blue: 0.85).opacity(0.1)
        )

        /// Green — status OK (#00ff88)
        static let statusOK = adapt(
            dark: Color(red: 0, green: 1.0, blue: 0.533),
            light: Color(red: 0, green: 0.75, blue: 0.4)
        )

        /// Amber — warning (#ffaa00)
        static let statusWarn = adapt(
            dark: Color(red: 1.0, green: 0.667, blue: 0),
            light: Color(red: 0.85, green: 0.55, blue: 0)
        )

        /// Red — error (#ff4757)
        static let statusError = adapt(
            dark: Color(red: 1.0, green: 0.278, blue: 0.341),
            light: Color(red: 0.85, green: 0.2, blue: 0.25)
        )

        /// Primary text
        static let textPrimary = adapt(
            dark: .white,
            light: .black
        )

        /// Secondary text
        static let textSecondary = adapt(
            dark: .white.opacity(0.5),
            light: .black.opacity(0.55)
        )

        /// Tertiary text
        static let textTertiary = adapt(
            dark: .white.opacity(0.35),
            light: .black.opacity(0.35)
        )
    }
}

// MARK: - Spacing Tokens

extension NovaTheme {
    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
        static let xxl: CGFloat = 24
    }
}

// MARK: - Radius Tokens

extension NovaTheme {
    enum Radius {
        static let sm: CGFloat = 6
        static let md: CGFloat = 8
        static let lg: CGFloat = 10
        static let xl: CGFloat = 12
    }
}

// MARK: - Adaptive Color Helper

/// A Color that adapts between light and dark appearances.
/// Uses SwiftUI's built-in color scheme resolution.
private extension Color {
    init(light: Color, dark: Color) {
        // Use the SwiftUI environment-aware color adaptation.
        // This works because SwiftUI resolves @Environment(\.colorScheme)
        // at the view body evaluation time.
        self.init(.displayP3, white: 0, opacity: 1) // placeholder
        // Actually, the simplest cross-platform way is:
        #if os(macOS)
        // Use NSColor with name-based adaptation
        self = Color(nsColor: .textColor) // will be overridden below
        #endif
    }
}
```

Wait — the `adapt()` helper needs a different approach since SwiftUI `Color` doesn't natively support light/dark variants without Asset catalogs or environment reads. Let me use the standard approach:

- [ ] **Step 1 (revised): Create the theme file with proper adaptive colors**

The file needs to use a different technique. In SwiftUI, the cleanest way is an `Environment`-based approach or `Color.init(uiColor:)` on iOS / using `NSColor` dynamic colors on macOS.

For macOS 15 SwiftUI, the most reliable approach is to define an `@ViewBuilder`-based color resolver or use `NSColor(name:dynamicProvider:)` for dynamic AppKit colors.

**Final approach:**

```swift
import SwiftUI
import AppKit

/// Centralized design tokens for the NovaMLX "Control Room" aesthetic.
enum NovaTheme {
    @available(*, unavailable) init() {}
}

// MARK: - Color Tokens

extension NovaTheme {
    enum Colors {
        /// Base window background
        static let background = Color(nsColor: NSColor(name: "nova.background") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 0.039, green: 0.055, blue: 0.102, alpha: 1)  // #0a0e1a
                : NSColor.windowBackgroundColor
        })

        /// Section card background
        static let cardBackground = Color(nsColor: NSColor(name: "nova.card") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 0.06, green: 0.075, blue: 0.12, alpha: 1)
                : NSColor.controlBackgroundColor
        })

        /// Inner row background within cards
        static let rowBackground = Color(nsColor: NSColor(name: "nova.row") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 0.08, green: 0.09, blue: 0.15, alpha: 1)
                : NSColor.textBackgroundColor
        })

        /// Luminous card border
        static let cardBorder = Color(nsColor: NSColor(name: "nova.border") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(white: 1, alpha: 0.06)
                : NSColor(white: 0, alpha: 0.08)
        })

        /// Electric blue accent (#00d4ff)
        static let accent = Color(nsColor: NSColor(name: "nova.accent") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 0, green: 0.831, blue: 1.0, alpha: 1)
                : NSColor(red: 0, green: 0.6, blue: 0.85, alpha: 1)
        })

        /// Accent at dim opacity
        static let accentDim = Color(nsColor: NSColor(name: "nova.accentDim") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 0, green: 0.831, blue: 1.0, alpha: 0.12)
                : NSColor(red: 0, green: 0.6, blue: 0.85, alpha: 0.1)
        })

        /// Green — status OK (#00ff88)
        static let statusOK = Color(nsColor: NSColor(name: "nova.ok") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 0, green: 1.0, blue: 0.533, alpha: 1)
                : NSColor(red: 0.15, green: 0.75, blue: 0.35, alpha: 1)
        })

        /// Amber — warning (#ffaa00)
        static let statusWarn = Color(nsColor: NSColor(name: "nova.warn") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 1.0, green: 0.667, blue: 0, alpha: 1)
                : NSColor(red: 0.85, green: 0.55, blue: 0, alpha: 1)
        })

        /// Red — error (#ff4757)
        static let statusError = Color(nsColor: NSColor(name: "nova.error") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(red: 1.0, green: 0.278, blue: 0.341, alpha: 1)
                : NSColor(red: 0.85, green: 0.2, blue: 0.25, alpha: 1)
        })

        /// Primary text (white in dark, black in light)
        static let textPrimary = Color(nsColor: NSColor(name: "nova.textPrimary") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor.white
                : NSColor.textColor
        })

        /// Secondary text
        static let textSecondary = Color(nsColor: NSColor(name: "nova.textSecondary") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(white: 1, alpha: 0.5)
                : NSColor(white: 0, alpha: 0.55)
        })

        /// Tertiary text
        static let textTertiary = Color(nsColor: NSColor(name: "nova.textTertiary") { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(white: 1, alpha: 0.35)
                : NSColor(white: 0, alpha: 0.35)
        })
    }
}

// MARK: - Spacing Tokens

extension NovaTheme {
    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
        static let xxl: CGFloat = 24
    }
}

// MARK: - Radius Tokens

extension NovaTheme {
    enum Radius {
        static let sm: CGFloat = 6
        static let md: CGFloat = 8
        static let lg: CGFloat = 10
        static let xl: CGFloat = 12
    }
}
```

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete with no errors from NovaTheme.swift

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/NovaTheme.swift
git commit -m "feat(gui): add NovaTheme design token system for dark mode"
```

---

## Task 2: NovaComponents.swift — Shared Components

**Files:**
- Create: `Sources/NovaMLXMenuBar/NovaComponents.swift`
- Read: `Sources/NovaMLXMenuBar/ModelsPageView.swift` (lines 932-947, 545-556, 1025-1044)

Extract `sectionCard()`, `rowCard()`, `sectionHeader()`, `CopyIDButton` from ModelsPageView. Add new `StatusBadge` and `MetricCard` components.

- [ ] **Step 1: Create the shared components file**

```swift
import SwiftUI
import NovaMLXCore

// MARK: - View Modifiers

extension View {
    /// Section-level card with material background, border, padding, radius.
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
            .padding(.vertical, NovaTheme.Spacing.sm)
            .background(NovaTheme.Colors.rowBackground)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.sm))
    }
}

// MARK: - Section Header

/// Unified section header with icon and optional count badge.
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
                .padding(.horizontal, 6)
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

/// Stat card with title, value, and optional subtitle.
struct MetricCard: View {
    let title: String
    let value: String
    var subtitle: String? = nil
    var valueColor: Color = NovaTheme.Colors.textPrimary

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 9))
                .foregroundColor(NovaTheme.Colors.textTertiary)
                .textCase(.uppercase)
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
        }
        .buttonStyle(.plain)
    }
}
```

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete (note: ModelsPageView still has its own definitions, so there will be duplicate symbol errors. We'll resolve this in Task 4.)

Actually — since `sectionCard()`, `rowCard()`, and `CopyIDButton` are currently in ModelsPageView.swift, adding them to NovaComponents will cause duplicate symbol errors. The correct approach is:

1. Create NovaComponents.swift with ALL shared components
2. Immediately remove them from ModelsPageView.swift in the same step

- [ ] **Step 1 (revised): Create NovaComponents.swift AND remove duplicates from ModelsPageView.swift**

1. Create `Sources/NovaMLXMenuBar/NovaComponents.swift` with the full code above.

2. In `Sources/NovaMLXMenuBar/ModelsPageView.swift`:
   - Delete lines 932-947 (the `private extension View` with `sectionCard()` and `rowCard()`)
   - Delete lines 1025-1044 (the `CopyIDButton` struct)
   - Delete lines 545-556 (the `sectionHeader` function) — but only the one with 3 params (title, icon, count)
   - Keep `FlowLayout`, `HFSearchResult`, `ModelCardData`, `ModelCardFile` in ModelsPageView

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete with no errors

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/NovaComponents.swift Sources/NovaMLXMenuBar/ModelsPageView.swift
git commit -m "feat(gui): extract shared components to NovaComponents.swift"
```

---

## Task 3: NovaAppView.swift — Sidebar + Navigation

**Files:**
- Modify: `Sources/NovaMLXMenuBar/NovaAppView.swift`

**Current state (key lines):**
- Line 72: `.background(Color(nsColor: .controlBackgroundColor))` — sidebar bg
- Line 102: `.background(selectedPage == page ? Color.accentColor.opacity(0.15) : Color.clear)` — selected item
- Line 93: `.foregroundColor(.white)` — badge count
- Line 122: `.foregroundColor(.accentColor)` — "MLX" text
- Line 126: `.fill(appState.isServerRunning ? Color.green : Color.red)` — status dot

- [ ] **Step 1: Replace all color usages**

In `Sources/NovaMLXMenuBar/NovaAppView.swift`, make these replacements:

**Line 72** — sidebar background:
```swift
// BEFORE:
.background(Color(nsColor: .controlBackgroundColor))
// AFTER:
.background(.ultraThinMaterial)
.background(NovaTheme.Colors.background.opacity(0.85))
```

**Line 102** — selected nav item:
```swift
// BEFORE:
.background(selectedPage == page ? Color.accentColor.opacity(0.15) : Color.clear)
// AFTER:
.background(selectedPage == page ? NovaTheme.Colors.accentDim : Color.clear)
```

Add a left accent bar to the selected item by inserting after line 102:
```swift
.overlay(alignment: .leading) {
    if selectedPage == page {
        RoundedRectangle(cornerRadius: 1)
            .fill(NovaTheme.Colors.accent)
            .frame(width: 2)
    }
}
```

**Line 93** — badge count text color stays `.white` (on accent background, correct in both modes).

**Line 122** — "MLX" text:
```swift
// BEFORE:
.foregroundColor(.accentColor)
// AFTER:
.foregroundColor(NovaTheme.Colors.accent)
```

**Line 126** — status dot:
```swift
// BEFORE:
.fill(appState.isServerRunning ? Color.green : Color.red)
// AFTER:
.fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
```

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/NovaAppView.swift
git commit -m "feat(gui): apply theme to NovaAppView sidebar"
```

---

## Task 4: StatusPageView.swift — Status Hero + Metrics

**Files:**
- Modify: `Sources/NovaMLXMenuBar/StatusPageView.swift`

**Changes needed:**
- Lines 61, 94, 113, 170: `Color(nsColor: .controlBackgroundColor)` → `NovaTheme.Colors.cardBackground` (via sectionCard modifier or direct)
- Line 163: `Color(nsColor: .textBackgroundColor)` → `NovaTheme.Colors.rowBackground`
- Line 29: `Color.green.opacity(0.15)` / `Color.red.opacity(0.15)` → use `statusOK`/`statusError` with opacity
- Line 33: `.foregroundColor(.green/.red)` → `NovaTheme.Colors.statusOK`/`statusError`
- Lines 139, 159: `.foregroundColor(.green)` → `NovaTheme.Colors.statusOK`
- Lines 152-153: `Color.purple.opacity(0.15)` / `.foregroundColor(.purple)` → `NovaTheme.Colors.accent`
- Replace inline `metricCard` function with `MetricCard` component
- Replace status text with `StatusBadge`

- [ ] **Step 1: Replace all color usages and apply themed components**

Replace the `serverStatusHero` section background:
```swift
// Line 61 — BEFORE:
.background(Color(nsColor: .controlBackgroundColor))
// AFTER:
.background(NovaTheme.Colors.cardBackground)
.overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
.clipShape(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg))
```

Replace status dot fill and bolt icon color:
```swift
// Line 29 — the hero circle fill:
// BEFORE: .fill(appState.isServerRunning ? Color.green.opacity(0.15) : Color.red.opacity(0.15))
// AFTER:
.fill(appState.isServerRunning ? NovaTheme.Colors.statusOK.opacity(0.2) : NovaTheme.Colors.statusError.opacity(0.2))

// Line 33 — bolt icon:
// BEFORE: .foregroundColor(appState.isServerRunning ? .green : .red)
// AFTER:
.foregroundColor(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
```

Replace the `metricCard` helper function with calls to `MetricCard`:
```swift
// Replace the private func metricCard with usage of MetricCard component:
// BEFORE: metricCard(title:value:subtitle:)
// AFTER: MetricCard(title:value:subtitle:)
```

For each of the 8 metric cards in the grid (lines ~75-91), change from:
```swift
metricCard(title: "Models", value: "\(appState.loadedModels.count)", subtitle: "loaded")
```
to:
```swift
MetricCard(title: "Models", value: "\(appState.loadedModels.count)", subtitle: "loaded")
```

And remove the old `metricCard` private function.

Replace device section and loaded models section backgrounds:
```swift
// Lines 113, 170 — BEFORE:
.background(Color(nsColor: .controlBackgroundColor))
// AFTER: use sectionCard() modifier
```

Replace loaded model row background:
```swift
// Line 163 — BEFORE:
.background(Color(nsColor: .textBackgroundColor))
// AFTER:
.background(NovaTheme.Colors.rowBackground)
```

Replace green color usages:
```swift
// Line 139 — BEFORE: .foregroundColor(.green)
// AFTER: .foregroundColor(NovaTheme.Colors.statusOK)

// Line 159 — BEFORE: .foregroundColor(.green)
// AFTER: .foregroundColor(NovaTheme.Colors.statusOK)
```

Replace purple TurboQuant badge:
```swift
// Lines 152-153 — BEFORE:
.background(Color.purple.opacity(0.15))
.foregroundColor(.purple)
// AFTER:
.background(NovaTheme.Colors.accentDim)
.foregroundColor(NovaTheme.Colors.accent)
```

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/StatusPageView.swift
git commit -m "feat(gui): apply theme to StatusPageView"
```

---

## Task 5: ModelsPageView.swift — Extract + Theme

**Files:**
- Modify: `Sources/NovaMLXMenuBar/ModelsPageView.swift`

This is the largest file (1044 lines). After Task 2, the shared components have been extracted. Now apply theme tokens to all remaining inline colors.

- [ ] **Step 1: Replace all Color(nsColor:) usages**

```swift
// Line 98 — subTabBar background:
// BEFORE: .background(Color(nsColor: .controlBackgroundColor))
// AFTER: .background(NovaTheme.Colors.cardBackground)

// Line 143 — search bar input:
// BEFORE: .background(Color(nsColor: .textBackgroundColor))
// AFTER: .background(NovaTheme.Colors.rowBackground)

// Line 779 — cardSection background:
// BEFORE: .background(Color(nsColor: .textBackgroundColor))
// AFTER: .background(NovaTheme.Colors.rowBackground)
```

- [ ] **Step 2: Replace all .accentColor.opacity() usages**

```swift
// Line 92 — sub-tab highlight:
// BEFORE: .background(subTab == tab ? Color.accentColor.opacity(0.1) : Color.clear)
// AFTER: .background(subTab == tab ? NovaTheme.Colors.accentDim : Color.clear)

// Line 188 — search result tag:
// BEFORE: .background(Color.accentColor.opacity(0.1))
// AFTER: .background(NovaTheme.Colors.accentDim)

// Line 691 — model card tag:
// BEFORE: .background(Color.accentColor.opacity(0.1))
// AFTER: .background(NovaTheme.Colors.accentDim)
```

- [ ] **Step 3: Replace semantic color usages**

```swift
// Line 206 — "Downloaded" label:
.foregroundColor(NovaTheme.Colors.statusOK)

// Lines 358, 361, 362 — failed file:
.foregroundColor(NovaTheme.Colors.statusWarn)

// Line 405 — download status icon:
.foregroundColor(task.status == .completed ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)

// Line 561 — model row status dot:
.fill(isLoaded ? NovaTheme.Colors.statusOK : Color.clear)

// Line 568, 178, 503 — accentColor references:
.foregroundColor(NovaTheme.Colors.accent)

// Line 663 — model card header icon:
.foregroundColor(NovaTheme.Colors.accent)

// Line 83, 550 — .white on accent badges: keep as .white (correct on colored bg)
// Line 552 — .background(Color.accentColor): keep or change to NovaTheme.Colors.accent
```

- [ ] **Step 4: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXMenuBar/ModelsPageView.swift
git commit -m "feat(gui): apply theme to ModelsPageView"
```

---

## Task 6: ChatPageView.swift — Bubbles + Input

**Files:**
- Modify: `Sources/NovaMLXMenuBar/ChatPageView.swift`

- [ ] **Step 1: Replace all color usages**

```swift
// Line 51 — chatToolbar background:
// BEFORE: .background(Color(nsColor: .controlBackgroundColor))
// AFTER: .background(NovaTheme.Colors.cardBackground)
//         .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)

// Line 113 — user message bubble:
// BEFORE: Color.accentColor.opacity(0.12)
// AFTER: NovaTheme.Colors.accentDim

// Line 114 — assistant message bubble:
// BEFORE: Color(nsColor: .textBackgroundColor)
// AFTER: NovaTheme.Colors.cardBackground

// Line 137 — inputBar background:
// BEFORE: .background(Color(nsColor: .controlBackgroundColor))
// AFTER: .background(NovaTheme.Colors.cardBackground)
//         .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)
```

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/ChatPageView.swift
git commit -m "feat(gui): apply theme to ChatPageView"
```

---

## Task 7: SettingsPageView.swift — Section Cards

**Files:**
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`

- [ ] **Step 1: Replace all Color(nsColor:) usages**

All 7 usages (`controlBackgroundColor` on lines 60, 119, 178, 314, 343; `textBackgroundColor` on lines 208, 308):
```swift
// For section-level containers (lines 60, 119, 178, 314, 343):
// BEFORE: .background(Color(nsColor: .controlBackgroundColor))
//         .clipShape(RoundedRectangle(cornerRadius: 10))
// AFTER: apply .sectionCard() modifier instead of manual background + clipShape

// For row-level items (lines 208, 308):
// BEFORE: .background(Color(nsColor: .textBackgroundColor))
//         .clipShape(RoundedRectangle(cornerRadius: 6))
// AFTER: apply .rowCard() modifier instead of manual background + clipShape
```

- [ ] **Step 2: Replace the duplicated sectionHeader**

```swift
// Lines 349-352 — remove the local sectionHeader function.
// It's now provided by NovaComponents.swift with the same signature.
```

- [ ] **Step 3: Replace semantic colors**

```swift
// Line 73 — config path:
.foregroundColor(NovaTheme.Colors.accent)

// Line 104 — "Installed" label:
.foregroundColor(NovaTheme.Colors.statusOK)

// Line 115 — install message:
.foregroundColor(cliInstalled ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)

// Line 278 — Clear All button:
.foregroundColor(NovaTheme.Colors.statusError)
```

- [ ] **Step 4: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXMenuBar/SettingsPageView.swift
git commit -m "feat(gui): apply theme to SettingsPageView"
```

---

## Task 8: DashboardView.swift — Stats Grid

**Files:**
- Modify: `Sources/NovaMLXMenuBar/DashboardView.swift`

- [ ] **Step 1: Replace all color usages**

```swift
// Line 35 — main background:
// BEFORE: .background(Color(nsColor: .windowBackgroundColor))
// AFTER: .background(NovaTheme.Colors.background)

// Lines 101, 120, 180, 249 — card backgrounds:
// BEFORE: .background(Color(nsColor: .controlBackgroundColor))
// AFTER: .background(NovaTheme.Colors.cardBackground)
//         (or apply sectionCard() modifier for outer containers)
```

Replace inline `statCard` function with `MetricCard`:
```swift
// Replace private func statCard calls with MetricCard
// The old statCard function can be removed entirely
```

```swift
// Line 44 — "MLX" text:
.foregroundColor(NovaTheme.Colors.accent)

// Line 55 — status dot:
.fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)

// Lines 152, 222 — green icons:
.foregroundColor(NovaTheme.Colors.statusOK)
```

- [ ] **Step 2: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/DashboardView.swift
git commit -m "feat(gui): apply theme to DashboardView"
```

---

## Task 9: Menu Bar Popover Views

**Files:**
- Modify: `Sources/NovaMLXMenuBar/StatusMenuView.swift`
- Modify: `Sources/NovaMLXMenuBar/ModelsMenuView.swift`
- Modify: `Sources/NovaMLXMenuBar/MenuBarController.swift`

- [ ] **Step 1: StatusMenuView.swift**

```swift
// Line 13 — status dot:
// BEFORE: .fill(appState.isServerRunning ? Color.green : Color.red)
// AFTER: .fill(appState.isServerRunning ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusError)
```

- [ ] **Step 2: ModelsMenuView.swift**

```swift
// Line 24 — cpu icon:
// BEFORE: .foregroundColor(.green)
// AFTER: .foregroundColor(NovaTheme.Colors.statusOK)
```

- [ ] **Step 3: MenuBarController.swift**

```swift
// Line 73 — Quit button:
// BEFORE: .foregroundColor(.red)
// AFTER: .foregroundColor(NovaTheme.Colors.statusError)
```

- [ ] **Step 4: Build to verify**

Run: `swift build 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXMenuBar/StatusMenuView.swift Sources/NovaMLXMenuBar/ModelsMenuView.swift Sources/NovaMLXMenuBar/MenuBarController.swift
git commit -m "feat(gui): apply theme to menu bar popover views"
```

---

## Task 10: Final Build + Verification

- [ ] **Step 1: Clean build**

Run: `swift build 2>&1 | tail -20`
Expected: Build complete with 0 errors

- [ ] **Step 2: Verify no leftover hardcoded colors**

Run: `grep -n "nsColor:" Sources/NovaMLXMenuBar/*.swift`
Expected: 0 results (except inside NovaTheme.swift itself which uses NSColor for the adaptive color mechanism)

Run: `grep -n "\.accentColor" Sources/NovaMLXMenuBar/Nova*.swift Sources/NovaMLXMenuBar/Status*.swift Sources/NovaMLXMenuBar/Chat*.swift Sources/NovaMLXMenuBar/Settings*.swift Sources/NovaMLXMenuBar/Dashboard*.swift Sources/NovaMLXMenuBar/Models*.swift Sources/NovaMLXMenuBar/Menu*.swift`
Expected: 0 results

- [ ] **Step 3: Visual verification**

Launch the app. Verify:
1. Dark mode: deep space background, electric blue accents, green status indicators, luminous borders
2. Light mode: coherent light theme with same accent colors
3. All navigation, buttons, chat, settings work identically
4. Menu bar popover renders correctly

---

## Self-Review Checklist

**Spec coverage:**
- [x] NovaTheme.swift with all 11 color tokens — Task 1
- [x] NovaComponents.swift with sectionCard, rowCard, sectionHeader, StatusBadge, MetricCard, CopyIDButton — Task 2
- [x] NovaAppView sidebar + navigation — Task 3
- [x] StatusPageView status hero + metrics — Task 4
- [x] ModelsPageView extract + theme — Task 5
- [x] ChatPageView bubbles + input — Task 6
- [x] SettingsPageView section cards — Task 7
- [x] DashboardView stats grid — Task 8
- [x] StatusMenuView + ModelsMenuView + MenuBarController — Task 9
- [x] Final build + verify — Task 10

**Placeholder scan:** No TBD/TODO. All steps have complete code.

**Type consistency:** `NovaTheme.Colors.xxx` tokens used consistently. `sectionCard()` / `rowCard()` return `some View`. `MetricCard` takes `(title:value:subtitle:)`. `StatusBadge` takes `(text:color:)`. `CopyIDButton` takes `(id:)`.
