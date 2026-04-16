# Dark Mode GUI Upgrade — Design Spec

**Date:** 2026-04-15
**Aesthetic:** Control Room — Deep Space variant
**Accent:** Electric Blue (#00d4ff)
**Scope:** All 10 SwiftUI view files in NovaMLXMenuBar

## Problem

The GUI uses `Color(nsColor: .controlBackgroundColor)` and opacity-based accents (`.accentColor.opacity(0.1)`) with zero dark-mode awareness. Opacity accents become invisible in dark mode. No theme system exists — colors are scattered inline across all views.

## Design Decisions

- **Aesthetic:** Control Room / Deep Space — near-black backgrounds, luminous borders, electric blue accent, green status indicators. Cockpit-grade readability.
- **Materials:** SwiftUI `.ultraThinMaterial` as base layer, overlaid with themed color at ~60% opacity. Frosted glass feel with controlled dark palette.
- **Light mode:** Supported but secondary. Theme detects `colorScheme == .light` and shifts to slate backgrounds with darker text. The control-room feel softens but stays coherent.
- **Accent color:** Electric Blue (#00d4ff) for primary interactive elements. System accent color is NOT used — this is a deliberate brand choice.
- **Scope:** All 10 view files. No new views added. No behavior changes — purely visual.

## Architecture

### New File: `NovaTheme.swift`

Single source of truth for all visual tokens. No hardcoded colors anywhere in view code.

```
NovaTheme (enum — never instantiated)
├── Colors
│   ├── background          — base window bg
│   ├── cardBackground      — section card fill
│   ├── cardBorder          — luminous card border (semi-transparent white)
│   ├── accent              — electric blue #00d4ff
│   ├── accentDim           — accent at 12% (highlights, badges, selected items)
│   ├── statusOK            — green #00ff88
│   ├── statusWarn          — amber #ffaa00
│   ├── statusError         — red #ff4757
│   ├── textPrimary         — white
│   ├── textSecondary       — 50% opacity white
│   └── textTertiary        — 35% opacity white
├── Spacing — xs(4), sm(8), md(12), lg(16), xl(20), xxl(24)
├── Radius — sm(6), md(8), lg(10), xl(12)
└── adapt() — internal helper that switches color values based on colorScheme
```

Each color property reads `@Environment(\.colorScheme)` via a static helper and returns the appropriate variant.

### New File: `NovaComponents.swift`

Extracted shared view components and modifiers, promoted from private/file-scoped to internal access.

| Component | Source | Description |
|-----------|--------|-------------|
| `CopyIDButton` | ModelsPageView.swift:1025 | Copy-to-clipboard button, now themed |
| `sectionCard()` | ModelsPageView.swift:933 (private ext) | View modifier: material bg + border + padding + radius |
| `rowCard()` | ModelsPageView.swift:940 (private ext) | View modifier: inner row style within sections |
| `sectionHeader()` | Duplicated in ModelsPageView + SettingsPageView | Unified: icon + title, themed typography |
| `StatusBadge` | New | Running/Stopped/Error badge with luminous bg |
| `MetricCard` | Patterns from DashboardView + StatusPageView | title + value + subtitle stat card |

`FlowLayout` (custom Layout from ModelsPageView:984) stays in ModelsPageView — it's a layout algorithm, not a visual component.

### Modified Files (8 views)

All views swap `Color(nsColor:)` → `NovaTheme.Colors.xxx` and inline patterns → `NovaComponents`.

## Color Token Mapping

| Current | Count | New Token |
|---------|-------|-----------|
| `Color(nsColor: .controlBackgroundColor)` | 18 | `Colors.cardBackground` |
| `Color(nsColor: .textBackgroundColor)` | 7 | `Colors.cardBackground` (subtle variant) |
| `Color(nsColor: .windowBackgroundColor)` | 1 | `Colors.background` |
| `.accentColor.opacity(0.1/0.12/0.15)` | 6 | `Colors.accentDim` |
| `.foregroundColor(.green)` status | 12 | `Colors.statusOK` |
| `.foregroundColor(.red)` errors | 7 | `Colors.statusError` |
| `.foregroundColor(.purple)` | 1 | `Colors.accent` |
| `.foregroundColor(.orange)` | 3 | `Colors.statusWarn` |
| `.secondary.opacity(0.3/0.5/0.6)` | 3 | `Colors.textTertiary` |

## Per-View Changes

### NovaAppView.swift (162 lines)
- Sidebar: `.ultraThinMaterial` + `Colors.background` overlay
- Selected nav item: `Colors.accentDim` fill + 2px left accent bar
- Brand "NovaMLX": white + accent split
- Main content area: `Colors.background`

### StatusPageView.swift (216 lines)
- Status hero: radial gradient glow (`statusOK`/`statusError`) instead of flat opacity
- 8 metric cards → `MetricCard` component
- Status text → `StatusBadge`
- Device section → `sectionCard()`
- All inline colors → theme tokens

### ModelsPageView.swift (1044 lines)
- **Extract to NovaComponents:** `CopyIDButton`, `sectionCard()`, `rowCard()`, `sectionHeader()`
- Search bar: dark bg, cardBorder outline, accent focus ring
- Model tags: `accentDim` background
- Download progress: `accent` fill on `cardBackground` track
- Model card sheet: themed layers
- All `Color(nsColor:)` → theme tokens

### ChatPageView.swift (186 lines)
- Toolbar/input: `cardBackground` + top border
- User bubbles: `accentDim` background
- AI bubbles: `cardBackground` + subtle cardBorder
- Input field: dark bg, accent border on focus

### SettingsPageView.swift (400 lines)
- Section cards → `sectionCard()` modifier
- TurboQuant buttons: active = `accent` filled, inactive = bordered
- CLI status: `statusOK` color
- Sessions: `rowCard()` per entry

### DashboardView.swift (275 lines)
- Stats grid → `MetricCard`
- Window bg → `Colors.background`
- All inline colors → theme tokens

### StatusMenuView.swift (45 lines) + ModelsMenuView.swift (49 lines)
- Compact themed cards, reduced spacing for popover
- Same color language as main window

### MenuBarAppState.swift (207 lines) + MenuBarController.swift (85 lines)
- No changes — state/controller only, no visual code

## Implementation Order

1. `NovaTheme.swift` — color/spacing/radius tokens (other files depend on this)
2. `NovaComponents.swift` — shared components (depends on NovaTheme)
3. `NovaAppView.swift` — sidebar + window frame
4. `StatusPageView.swift` — metrics + status
5. `ModelsPageView.swift` — extract shared components, apply theme
6. `ChatPageView.swift` — chat bubbles + input
7. `SettingsPageView.swift` — section cards
8. `DashboardView.swift` — stats grid
9. `StatusMenuView.swift` + `ModelsMenuView.swift` — popover views
10. Build + verify

## Testing

- **Compile test:** `swift build` passes
- **Visual test:** Launch app in dark mode, verify all views render correctly
- **Light mode test:** Switch to light mode, verify no broken colors
- **Popover test:** Click menu bar icon, verify popover views
- **No behavior changes:** All buttons, navigation, downloads, chat still work identically

## Out of Scope

- No new features or behavior changes
- No new views or navigation changes
- No changes to inference engine, API server, or model management logic
- No custom fonts (keeping system font)
- No animations or transitions (could be a follow-up)
