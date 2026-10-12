import AppKit
import SwiftUI

/// Peer menu-bar app. This target links TknetPeer only — no MLX,
/// NovaMLXCore, NovaMLXDB, or NovaMLXMenuBar.
///
/// `LSUIElement` stays true, so launch is a menu-bar agent. `PeerWindowActivation`
/// promotes to `.regular` only while a Peer window exists (Cmd+Tab and Dock),
/// then returns to `.accessory` when the last one closes. The status item stays.
@main
struct TknetPeerApp: App {
    @StateObject private var appState: PeerAppState
    @NSApplicationDelegateAdaptor(PeerAppDelegate.self) private var delegate

    init() {
        let state = PeerAppState()
        _appState = StateObject(wrappedValue: state)
        PeerAppDelegate.appState = state
    }

    var body: some Scene {
        MenuBarExtra {
            PeerMenu()
                .environmentObject(appState)
        } label: {
            PeerMenuBarLabel(symbol: appState.connectionSymbol)
        }
        .menuBarExtraStyle(.menu)

        Window("Tknet Peer", id: "peer") {
            PeerRootView()
                .environmentObject(appState)
        }
        .defaultSize(width: 760, height: 640)
    }
}

/// Toggles `NSApplication.ActivationPolicy` around the Peer window.
/// Launch and "no window" stay `.accessory`. An open or miniaturized window is `.regular`.
@MainActor
enum PeerWindowActivation {
    static let windowTitle = "Tknet Peer"
    static let windowIdentifier = "peer"

    /// `openWindow` captured from the always-mounted menu-bar label, so a Dock
    /// click can recreate the SwiftUI window after the menu content is gone.
    static var openWindow: (() -> Void)?

    /// Bumped by every show. A deferred demote from an older close no-ops.
    private static var showGeneration = 0

    static func isPeerWindow(_ window: NSWindow) -> Bool {
        if window is NSPanel { return false }
        guard window.canBecomeMain else { return false }
        if window.title == windowTitle { return true }
        return window.identifier?.rawValue == windowIdentifier
    }

    static func show(openIfNeeded: () -> Void) {
        showGeneration += 1
        NSApp.setActivationPolicy(.regular)
        if let existing = peerWindows(excluding: nil).first {
            if existing.isMiniaturized {
                existing.deminiaturize(nil)
            }
            existing.makeKeyAndOrderFront(nil)
        } else {
            openIfNeeded()
        }
        NSApp.activate()
    }

    /// `willClose` still reports the window visible, so `closing` is excluded.
    /// The hop lets that close finish; a later `show` bumps `showGeneration` and
    /// cancels this demote. Miniaturized windows still count as open.
    static func scheduleDemote(excluding closing: NSWindow) {
        let generation = showGeneration
        let closingID = ObjectIdentifier(closing)
        Task { @MainActor in
            guard generation == showGeneration else { return }
            guard peerWindows(excluding: closingID).isEmpty else { return }
            NSApp.setActivationPolicy(.accessory)
        }
    }

    private static func peerWindows(excluding: ObjectIdentifier?) -> [NSWindow] {
        NSApp.windows.filter { candidate in
            if let excluding, ObjectIdentifier(candidate) == excluding { return false }
            return isPeerWindow(candidate) && (candidate.isVisible || candidate.isMiniaturized)
        }
    }
}

@MainActor
final class PeerAppDelegate: NSObject, NSApplicationDelegate {
    static var appState: PeerAppState?

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowWillClose(_:)),
            name: NSWindow.willCloseNotification,
            object: nil
        )
        let state = Self.appState
        Task { await state?.launch() }
    }

    /// Dock click while promoted. Accessory launch has no Dock icon, so this
    /// only runs after a window has been shown.
    nonisolated func applicationShouldHandleReopen(
        _ sender: NSApplication,
        hasVisibleWindows flag: Bool
    ) -> Bool {
        MainActor.assumeIsolated {
            PeerWindowActivation.show {
                PeerWindowActivation.openWindow?()
            }
        }
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let state = Self.appState else { return .terminateNow }
        Task {
            await state.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    @objc func windowWillClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        guard PeerWindowActivation.isPeerWindow(window) else { return }
        PeerWindowActivation.scheduleDemote(excluding: window)
    }
}

/// Menu-bar label stays mounted for the life of the status item, unlike menu
/// contents, so it can hold the SwiftUI `openWindow` action for Dock reopen.
private struct PeerMenuBarLabel: View {
    let symbol: String
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Label("Tknet Peer", systemImage: symbol)
            .onAppear {
                let open = openWindow
                PeerWindowActivation.openWindow = { open(id: "peer") }
            }
    }
}

struct PeerMenu: View {
    @EnvironmentObject private var state: PeerAppState
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Open Tknet Peer…") {
            let open = openWindow
            PeerWindowActivation.openWindow = { open(id: "peer") }
            PeerWindowActivation.show {
                open(id: "peer")
            }
        }
        Divider()
        Button("Gateway \(state.gatewayLabel)") {}
            .disabled(true)
        Button(state.connectionLabel) {}
            .disabled(true)
        if state.isTunnelActive {
            Button("Stop tunnel") {
                Task { await state.stopTunnelFromUser() }
            }
            .disabled(state.lockError != nil)
        } else {
            Button("Start tunnel") {
                Task { await state.startTunnelFromUser() }
            }
            .disabled(state.lockError != nil || state.isBusy)
            if state.tunnelWanted {
                Button("Stop auto-dial") {
                    Task { await state.stopTunnelFromUser() }
                }
                .disabled(state.lockError != nil)
            }
        }
        Divider()
        Button("Quit Tknet Peer") {
            NSApp.terminate(nil)
        }
    }
}
