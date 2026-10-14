import TknetPeer

import AppKit

/// Dock icon: the full-color tunnel mark, tinted by tunnel state. Visible only
/// while a Peer window promotes the app to the regular activation policy.
///
/// Assigning `applicationIconImage` snapshots the NSImage to a CGImage on the
/// main thread, so every frame must come from a bitmap rep, not a custom
/// drawing handler — snapshotting a handler re-rasterizes the gradients each
/// tick (~20% CPU). Frames are prerendered once per (status, phase) into
/// bitmaps; the phase cycles through 8 values, so steady-state animation is a
/// cache hit plus a property assignment.
enum DockIcon {
    /// Redrawing is the dominant cost of the traffic animation, so skip it
    /// while the app runs as an accessory and the Dock isn't showing the
    /// icon at all.
    static func apply(status: TunnelStatus?, trafficPhase: Int? = nil) {
        guard NSApp.activationPolicy() == .regular else { return }
        NSApp.applicationIconImage = frameImage(for: status, trafficPhase: trafficPhase)
    }

    // MARK: - Cached frames

    private static let frameCacheLock = NSLock()
    /// Access only under `frameCacheLock`. Frames of the previous status are
    /// dropped when the status changes.
    nonisolated(unsafe) private static var frameCache: [String: NSImage] = [:]

    private static func frameImage(for status: TunnelStatus?, trafficPhase: Int?) -> NSImage {
        let statusKey = String(describing: status)
        let key = statusKey + ":" + String(describing: trafficPhase)
        frameCacheLock.lock()
        if let cached = frameCache[key] {
            frameCacheLock.unlock()
            return cached
        }
        frameCache = frameCache.filter { $0.key.hasPrefix(statusKey + ":") }
        frameCacheLock.unlock()

        // Render at 128pt: the Dock tile is ~64pt, and NSDockTile.display
        // re-composites the image on our main thread at every assignment, so
        // the bitmap size directly drives the per-frame cost. The drawing code
        // uses 512 coordinates; scale them down.
        let size = NSSize(width: 128, height: 128)
        let rect = NSRect(origin: .zero, size: NSSize(width: 512, height: 512))
        let image = NSImage(size: size)
        image.lockFocus()
        let scale = NSAffineTransform()
        scale.scale(by: 0.25)
        scale.concat()
        drawStatic(in: rect, status: status)
        if drawBeam(for: status) {
            if let phase = trafficPhase {
                let xs: [CGFloat] = [0.0, 1.0 / 3.0, 2.0 / 3.0].map { offset in
                    let t = (CGFloat(phase) / 8 + offset).truncatingRemainder(dividingBy: 1)
                    return 71 + t * 370
                }
                drawPackets(at: xs, radius: 15, status: status)
            } else {
                drawPackets(at: [131, 381], radius: 11, status: status)
            }
        }
        image.unlockFocus()

        frameCacheLock.lock()
        frameCache[key] = image
        frameCacheLock.unlock()
        return image
    }

    private static let cyan = NSColor(calibratedRed: 0.133, green: 0.827, blue: 0.933, alpha: 1)
    private static let indigo = NSColor(calibratedRed: 0.506, green: 0.549, blue: 0.973, alpha: 1)
    private static let bgTop = NSColor(calibratedRed: 0.075, green: 0.106, blue: 0.243, alpha: 1)
    private static let bgBottom = NSColor(calibratedRed: 0.031, green: 0.047, blue: 0.118, alpha: 1)
    private static let hole = NSColor(calibratedRed: 0.039, green: 0.055, blue: 0.133, alpha: 1)
    private static let amber = NSColor(calibratedRed: 0.961, green: 0.620, blue: 0.043, alpha: 1)

    // MARK: - Static base

    private static func drawBeam(for status: TunnelStatus?) -> Bool {
        switch status {
        case .idle, .none: false
        default: true
        }
    }

    private static func drawStatic(in rect: NSRect, status: TunnelStatus?) {
        var beamAlpha: CGFloat = 1
        var arcAlpha: CGFloat = 1
        var beamDashed = false
        var badge = false

        switch status {
        case .connected:
            break
        case .connecting:
            beamDashed = true
        case .backingOff:
            beamDashed = true
            arcAlpha = 0.55
        case .stopping:
            beamAlpha = 0.45
            arcAlpha = 0.45
        case .upgradeRequired:
            badge = true
        case .idle, .none:
            arcAlpha = 0.55
        }

        let center = NSPoint(x: rect.midX, y: rect.midY)
        let drawBeam = drawBeam(for: status)

        // Background
        let bg = NSBezierPath(roundedRect: rect, xRadius: 114, yRadius: 114)
        NSGradient(colors: [bgTop, bgBottom])?.draw(in: bg, angle: -60)

        // Portal glow
        let glowRect = NSRect(x: center.x - 170, y: center.y - 170, width: 340, height: 340)
        NSGradient(colors: [cyan.withAlphaComponent(0.35), cyan.withAlphaComponent(0)])?
            .draw(in: glowRect, relativeCenterPosition: .zero)

        // Beam between the peers, drawn as one gradient bar or dashed segments
        if drawBeam {
            if beamDashed {
                var x: CGFloat = 60
                while x < 452 {
                    let width = min(26.0, 452 - x)
                    let segment = NSBezierPath(roundedRect: NSRect(x: x, y: 249.5, width: width, height: 13),
                                               xRadius: 6.5, yRadius: 6.5)
                    blend(from: cyan, to: indigo, at: (x - 60) / 392, alpha: beamAlpha).setFill()
                    segment.fill()
                    x += 44
                }
            } else {
                let beam = NSBezierPath(roundedRect: NSRect(x: 60, y: 249.5, width: 392, height: 13),
                                        xRadius: 6.5, yRadius: 6.5)
                NSGradient(colors: [cyan.withAlphaComponent(beamAlpha),
                                    indigo.withAlphaComponent(beamAlpha)])?.draw(in: beam, angle: 0)
            }
        }

        // Portal arcs, top and bottom
        strokeArc(center: center, startAngle: 140, endAngle: 40, alpha: arcAlpha)
        strokeArc(center: center, startAngle: 320, endAngle: 220, alpha: arcAlpha)

        // Peer nodes: colored ring with a dark center
        let nodeAlpha = drawBeam ? beamAlpha : arcAlpha
        node(at: 70, centerY: center.y, color: cyan, alpha: nodeAlpha)
        node(at: 442, centerY: center.y, color: indigo, alpha: nodeAlpha)

        // Warning badge for the terminal upgrade state
        if badge {
            amber.setFill()
            NSBezierPath(ovalIn: NSRect(x: 369, y: 37, width: 110, height: 110)).fill()
            NSColor.white.setFill()
            NSBezierPath(roundedRect: NSRect(x: 415, y: 88, width: 18, height: 44),
                         xRadius: 9, yRadius: 9).fill()
            NSBezierPath(ovalIn: NSRect(x: 415, y: 60, width: 18, height: 18)).fill()
        }
    }

    // MARK: - Per-frame packets

    private static func drawPackets(at xs: [CGFloat], radius: CGFloat, status: TunnelStatus?) {
        let beamAlpha: CGFloat
        switch status {
        case .stopping: beamAlpha = 0.45
        default: beamAlpha = 1
        }
        for x in xs {
            blend(from: cyan, to: indigo, at: (x - 60) / 392, alpha: beamAlpha).setFill()
            NSBezierPath(ovalIn: NSRect(x: x - radius, y: 256 - radius,
                                        width: radius * 2, height: radius * 2)).fill()
        }
    }

    private static func strokeArc(center: NSPoint, startAngle: CGFloat, endAngle: CGFloat, alpha: CGFloat) {
        let arc = NSBezierPath()
        arc.appendArc(withCenter: center, radius: 130, startAngle: startAngle, endAngle: endAngle, clockwise: true)
        let outline = NSBezierPath(cgPath: arc.cgPath.copy(
            strokingWithWidth: 20, lineCap: .round, lineJoin: .miter, miterLimit: 10
        ))
        NSGraphicsContext.current?.saveGraphicsState()
        outline.addClip()
        NSGradient(colors: [cyan.withAlphaComponent(alpha), indigo.withAlphaComponent(alpha)])?
            .draw(in: outline.bounds, angle: 0)
        NSGraphicsContext.current?.restoreGraphicsState()
    }

    private static func node(at x: CGFloat, centerY: CGFloat, color: NSColor, alpha: CGFloat) {
        color.withAlphaComponent(alpha).setFill()
        NSBezierPath(ovalIn: NSRect(x: x - 27, y: centerY - 27, width: 54, height: 54)).fill()
        hole.setFill()
        NSBezierPath(ovalIn: NSRect(x: x - 13, y: centerY - 13, width: 26, height: 26)).fill()
    }

    private static func blend(from: NSColor, to: NSColor, at t: CGFloat, alpha: CGFloat) -> NSColor {
        let t = max(0, min(1, t))
        return NSColor(
            calibratedRed: from.redComponent + (to.redComponent - from.redComponent) * t,
            green: from.greenComponent + (to.greenComponent - from.greenComponent) * t,
            blue: from.blueComponent + (to.blueComponent - from.blueComponent) * t,
            alpha: alpha
        )
    }
}
