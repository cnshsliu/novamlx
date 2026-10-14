import TknetPeer

import AppKit

/// Menu-bar mark from the app icon: two peer nodes joined by a beam through a
/// tunnel portal. Drawn as a template image so macOS tints it for light and
/// dark menu bars. The beam and nodes carry the tunnel state.
enum MenuBarIcon {
    private static let canvas = NSSize(width: 18, height: 18)

    static func image(for status: TunnelStatus?) -> NSImage {
        if case .upgradeRequired = status {
            let symbol = NSImage(
                systemSymbolName: "exclamationmark.triangle",
                accessibilityDescription: "Upgrade required"
            )
            symbol?.isTemplate = true
            return symbol ?? NSImage()
        }
        let image = NSImage(size: canvas, flipped: false) { rect in
            draw(in: rect, status: status)
            return true
        }
        image.isTemplate = true
        return image
    }

    private static func draw(in rect: NSRect, status: TunnelStatus?) {
        var beamAlpha: CGFloat = 1
        var arcAlpha: CGFloat = 1
        var beamDashed = false
        var arcDashed = false
        var nodesFilled = false
        var drawBeam = true

        switch status {
        case .connected:
            nodesFilled = true
        case .connecting:
            beamDashed = true
        case .backingOff:
            beamDashed = true
            arcDashed = true
        case .stopping:
            nodesFilled = true
            beamAlpha = 0.45
            arcAlpha = 0.45
        case .idle, .upgradeRequired, .none:
            drawBeam = false
            arcAlpha = 0.55
        }

        let mid = NSPoint(x: rect.midX, y: rect.midY)
        let nodeRadius: CGFloat = 1.85
        let left = NSPoint(x: 2.7, y: mid.y)
        let right = NSPoint(x: rect.maxX - 2.7, y: mid.y)

        NSColor.black.withAlphaComponent(arcAlpha).setStroke()
        for (startAngle, endAngle) in [(38.0, 142.0), (218.0, 322.0)] {
            let arc = NSBezierPath()
            arc.appendArc(withCenter: mid, radius: 5.1, startAngle: startAngle, endAngle: endAngle)
            arc.lineWidth = 1.25
            arc.lineCapStyle = .round
            if arcDashed { arc.setLineDash([2.0, 1.6], count: 2, phase: 0) }
            arc.stroke()
        }

        if drawBeam {
            NSColor.black.withAlphaComponent(beamAlpha).setStroke()
            let beam = NSBezierPath()
            beam.move(to: NSPoint(x: left.x + nodeRadius + 0.2, y: mid.y))
            beam.line(to: NSPoint(x: right.x - nodeRadius - 0.2, y: mid.y))
            beam.lineWidth = 1.35
            beam.lineCapStyle = .round
            if beamDashed { beam.setLineDash([2.4, 1.7], count: 2, phase: 0) }
            beam.stroke()
        }

        NSColor.black.withAlphaComponent(drawBeam ? 1 : arcAlpha).set()
        for center in [left, right] {
            let node = NSBezierPath(ovalIn: NSRect(
                x: center.x - nodeRadius,
                y: center.y - nodeRadius,
                width: nodeRadius * 2,
                height: nodeRadius * 2
            ))
            if nodesFilled {
                node.fill()
            } else {
                node.lineWidth = 1.25
                node.stroke()
            }
        }
    }
}
