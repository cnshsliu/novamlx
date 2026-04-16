import SwiftUI
import AppKit

/// Centralized design tokens for the NovaMLX "Control Room" aesthetic.
/// Dark mode uses deep space backgrounds with electric blue accents.
/// Light mode adapts to lighter variants automatically.
/// This enum is never instantiated — it serves as a namespace.
enum NovaTheme {}

// MARK: - Color Tokens

extension NovaTheme {
    enum Colors {
        /// Base window background — deep space in dark, system window in light
        static let background = Color(nsColor: NSColor(name: "nova.background") { appearance in
            appearance.isDark
                ? NSColor(red: 0.039, green: 0.055, blue: 0.102, alpha: 1)
                : NSColor.windowBackgroundColor
        })

        /// Section card background — slightly elevated
        static let cardBackground = Color(nsColor: NSColor(name: "nova.card") { appearance in
            appearance.isDark
                ? NSColor(red: 0.06, green: 0.075, blue: 0.12, alpha: 1)
                : NSColor.controlBackgroundColor
        })

        /// Inner row background within cards
        static let rowBackground = Color(nsColor: NSColor(name: "nova.row") { appearance in
            appearance.isDark
                ? NSColor(red: 0.08, green: 0.09, blue: 0.15, alpha: 1)
                : NSColor.textBackgroundColor
        })

        /// Luminous card border
        static let cardBorder = Color(nsColor: NSColor(name: "nova.border") { appearance in
            appearance.isDark
                ? NSColor(white: 1, alpha: 0.06)
                : NSColor(white: 0, alpha: 0.08)
        })

        /// Electric blue accent (#00d4ff)
        static let accent = Color(nsColor: NSColor(name: "nova.accent") { appearance in
            appearance.isDark
                ? NSColor(red: 0, green: 0.831, blue: 1.0, alpha: 1)
                : NSColor(red: 0, green: 0.6, blue: 0.85, alpha: 1)
        })

        /// Accent at dim opacity — highlights, badges, selected items
        static let accentDim = Color(nsColor: NSColor(name: "nova.accentDim") { appearance in
            appearance.isDark
                ? NSColor(red: 0, green: 0.831, blue: 1.0, alpha: 0.12)
                : NSColor(red: 0, green: 0.6, blue: 0.85, alpha: 0.1)
        })

        /// Green — status OK (#00ff88)
        static let statusOK = Color(nsColor: NSColor(name: "nova.ok") { appearance in
            appearance.isDark
                ? NSColor(red: 0, green: 1.0, blue: 0.533, alpha: 1)
                : NSColor(red: 0.15, green: 0.75, blue: 0.35, alpha: 1)
        })

        /// Amber — warning (#ffaa00)
        static let statusWarn = Color(nsColor: NSColor(name: "nova.warn") { appearance in
            appearance.isDark
                ? NSColor(red: 1.0, green: 0.667, blue: 0, alpha: 1)
                : NSColor(red: 0.85, green: 0.55, blue: 0, alpha: 1)
        })

        /// Red — error (#ff4757)
        static let statusError = Color(nsColor: NSColor(name: "nova.error") { appearance in
            appearance.isDark
                ? NSColor(red: 1.0, green: 0.278, blue: 0.341, alpha: 1)
                : NSColor(red: 0.85, green: 0.2, blue: 0.25, alpha: 1)
        })

        /// Primary text — white in dark, label in light
        static let textPrimary = Color(nsColor: NSColor(name: "nova.text1") { appearance in
            appearance.isDark ? NSColor.white : NSColor.labelColor
        })

        /// Secondary text — 50% white in dark, secondary label in light
        static let textSecondary = Color(nsColor: NSColor(name: "nova.text2") { appearance in
            appearance.isDark
                ? NSColor(white: 1, alpha: 0.5)
                : NSColor.secondaryLabelColor
        })

        /// Tertiary text — 35% white in dark, tertiary label in light
        static let textTertiary = Color(nsColor: NSColor(name: "nova.text3") { appearance in
            appearance.isDark
                ? NSColor(white: 1, alpha: 0.35)
                : NSColor.tertiaryLabelColor
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

// MARK: - Appearance Helper

private extension NSAppearance {
    /// Whether this appearance is a dark variant
    var isDark: Bool {
        bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
    }
}
