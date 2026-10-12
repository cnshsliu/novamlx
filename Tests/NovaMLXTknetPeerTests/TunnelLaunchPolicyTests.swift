import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("Tunnel launch policy")
struct TunnelLaunchPolicyTests {
    @Test("abandons after the initial dial and 3 retries")
    func abandonsAfterThreeRetries() {
        var policy = TunnelLaunchPolicy()
        policy.begin()
        #expect(policy.note(.connecting) == .keepWatching)
        #expect(policy.note(.backingOff(seconds: 1)) == .keepWatching)
        #expect(policy.note(.backingOff(seconds: 2)) == .keepWatching)
        #expect(policy.note(.backingOff(seconds: 4)) == .keepWatching)
        #expect(policy.note(.backingOff(seconds: 8)) == .abandon)
        #expect(policy.isWatching == false)
        #expect(policy.note(.backingOff(seconds: 8)) == .keepWatching)
    }

    @Test("connected stops the watcher")
    func connectedStopsWatching() {
        var policy = TunnelLaunchPolicy()
        policy.begin()
        _ = policy.note(.backingOff(seconds: 1))
        #expect(policy.note(.connected) == .connected)
        #expect(policy.launchFailures == 0)
        #expect(policy.note(.backingOff(seconds: 1)) == .keepWatching)
    }

    @Test("upgrade required is terminal for the launch watcher")
    func upgradeRequired() {
        var policy = TunnelLaunchPolicy()
        policy.begin()
        #expect(policy.note(.upgradeRequired) == .upgradeRequired)
        #expect(policy.isWatching == false)
    }

    @Test("notes are ignored until begin")
    func idleUntilBegin() {
        var policy = TunnelLaunchPolicy()
        #expect(policy.note(.backingOff(seconds: 1)) == .keepWatching)
        #expect(policy.launchFailures == 0)
    }

    @Test("absent tunnelWanted defaults to auto-dial and Stop persists false")
    func tunnelWantedDefault() {
        let name = "tknet-peer-wanted-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(TunnelWantedPreference.read(from: defaults))
        TunnelWantedPreference.write(false, to: defaults)
        #expect(TunnelWantedPreference.read(from: defaults) == false)
        TunnelWantedPreference.write(true, to: defaults)
        #expect(TunnelWantedPreference.read(from: defaults))
    }

    @Test("retry exhaustion does not clear tunnelWanted")
    func exhaustionLeavesWanted() {
        let name = "tknet-peer-wanted-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        defer { defaults.removePersistentDomain(forName: name) }
        TunnelWantedPreference.write(true, to: defaults)
        var policy = TunnelLaunchPolicy()
        policy.begin()
        var decision = TunnelLaunchDecision.keepWatching
        for _ in 0..<4 {
            decision = policy.note(.backingOff(seconds: 1))
        }
        #expect(decision == .abandon)
        #expect(TunnelWantedPreference.read(from: defaults))
    }
}
