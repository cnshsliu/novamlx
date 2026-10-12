import Foundation

public enum TunnelLaunchDecision: Equatable, Sendable {
    /// Keep counting launch failures. Manual Start never enters this path.
    case keepWatching
    case connected
    /// The initial dial plus `maxLaunchRetries` failed. The caller should
    /// `stopTunnel()` only. Do not clear `tunnelWanted` — the next launch
    /// tries again. Only the Stop control clears the flag.
    case abandon
    case upgradeRequired
}

/// Launch-path dial cap. The menu-bar app arms this on cold start only.
///
/// The first `.backingOff` is the initial attempt. The next
/// `maxLaunchRetries` (3) are the retries. A fourth `.backingOff` abandons
/// the launch watcher. The tunnel client's own reconnect loop is unchanged
/// for a manual Start, which does not call `begin()`.
public struct TunnelLaunchPolicy: Sendable, Equatable {
    public static let maxLaunchRetries = 3

    public private(set) var isWatching = false
    public private(set) var launchFailures = 0

    public init() {}

    public mutating func begin() {
        isWatching = true
        launchFailures = 0
    }

    public mutating func stopWatching() {
        isWatching = false
    }

    public mutating func note(_ status: TunnelStatus) -> TunnelLaunchDecision {
        guard isWatching else { return .keepWatching }
        switch status {
        case .connected:
            isWatching = false
            launchFailures = 0
            return .connected
        case .backingOff:
            launchFailures += 1
            let retriesUsed = launchFailures - 1
            if retriesUsed >= Self.maxLaunchRetries {
                isWatching = false
                return .abandon
            }
            return .keepWatching
        case .upgradeRequired:
            isWatching = false
            return .upgradeRequired
        case .idle, .connecting, .stopping:
            return .keepWatching
        }
    }
}

/// `UserDefaults` flag shared with installs that used NovaMLX's peer page.
/// Absent means the user never pressed Stop, so the next launch auto-dials.
public enum TunnelWantedPreference {
    public static let key = "tknetPeer.tunnelWanted"

    public static func read(from defaults: UserDefaults) -> Bool {
        if defaults.object(forKey: key) == nil { return true }
        return defaults.bool(forKey: key)
    }

    public static func write(_ wanted: Bool, to defaults: UserDefaults) {
        defaults.set(wanted, forKey: key)
    }
}
