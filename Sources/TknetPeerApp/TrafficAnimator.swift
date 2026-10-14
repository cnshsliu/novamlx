import AppKit

/// Drives the Dock-icon packet animation.
///
/// Two deliberate constraints keep the animation at zero cost when it can't
/// be seen:
/// - The clock only starts while the app runs as `.regular` — as an accessory
///   there is no Dock icon, so there is nothing to animate.
/// - Nothing observes the phase. The menu-bar mark is static; only the Dock
///   icon animates, updated through `onTick`. Publishing the phase from the
///   app-wide state object used to invalidate every SwiftUI subscriber per
///   frame, including the closed-but-alive Peer window's view graph.
@MainActor
final class TrafficAnimator {
    /// Non-nil while traffic flowed within the last second. The real signal
    /// plugs into `noteTraffic`.
    private(set) var phase: Int?

    /// Fired on every phase change; PeerAppState wires this to the Dock icon.
    var onTick: ((Int?) -> Void)?

    private var lastTrafficAt = Date.distantPast
    private var timer: Timer?

    func noteTraffic() {
        lastTrafficAt = Date()
        guard timer == nil else { return }
        guard NSApp.activationPolicy() == .regular else { return }
        phase = 0
        onTick?(phase)
        let timer = Timer(timeInterval: 0.22, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.advance() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    private func advance() {
        guard NSApp.activationPolicy() == .regular,
              Date().timeIntervalSince(lastTrafficAt) < 1.0
        else {
            timer?.invalidate()
            timer = nil
            phase = nil
            onTick?(nil)
            return
        }
        phase = ((phase ?? 0) + 1) % 8
        onTick?(phase)
    }
}
