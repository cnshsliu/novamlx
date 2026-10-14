import Foundation

/// Drives the menu-bar packet animation. Kept out of `PeerAppState` on
/// purpose: the phase ticks at ~4.5 Hz, and as an `@Published` on the
/// app-wide state object it invalidated every `@EnvironmentObject` subscriber
/// per frame — including the closed-but-alive Peer window's view graph, which
/// dominated the animation's CPU cost. Only the menu-bar label observes this
/// object, so a frame re-renders nothing but the status item.
@MainActor
final class TrafficAnimator: ObservableObject {
    /// Non-nil while traffic flowed within the last second. The real signal
    /// plugs into `noteTraffic`.
    @Published private(set) var phase: Int?

    private var lastTrafficAt = Date.distantPast
    private var timer: Timer?

    func noteTraffic() {
        lastTrafficAt = Date()
        if phase == nil { phase = 0 }
        guard timer == nil else { return }
        let timer = Timer(timeInterval: 0.22, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.advance() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    private func advance() {
        guard Date().timeIntervalSince(lastTrafficAt) < 1.0 else {
            timer?.invalidate()
            timer = nil
            phase = nil
            return
        }
        phase = ((phase ?? 0) + 1) % 8
    }
}
