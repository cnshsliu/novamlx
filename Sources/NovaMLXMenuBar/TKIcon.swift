import SwiftUI
import AppKit

/// Custom TK icon view - rounded rectangle with TK text
struct TKIcon: View {
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Rounded rectangle background
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.primary)

                // TK text
                Text("TK")
                    .font(.system(size: min(geometry.size.width, geometry.size.height) * 0.55, weight: .heavy, design: .rounded))
                    .foregroundColor(Color(NSColor.textBackgroundColor))
            }
        }
    }
}

/// Creates an NSImage from the TK icon for use in MenuBarExtra
public func createTKIconImage(size: NSSize = NSSize(width: 20, height: 20)) -> NSImage {
    let iconView = TKIcon()
        .frame(width: size.width, height: size.height)

    let hostingController = NSHostingController(rootView: iconView)
    hostingController.view.setBoundsSize(size)

    let image = NSImage(size: size)
    image.lockFocus()
    defer { image.unlockFocus() }

    let bounds = NSRect(origin: .zero, size: size)
    hostingController.view.draw(bounds)

    // Set as template image for proper menu bar appearance
    image.isTemplate = true

    return image
}

#Preview {
    TKIcon()
        .frame(width: 20, height: 20)
        .padding()
        .background(Color.gray.opacity(0.2))
}
