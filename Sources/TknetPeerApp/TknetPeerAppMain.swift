import AppKit
import SwiftUI

/// Peer menu-bar app. This target links NovaMLXTknetPeer only — no MLX,
/// NovaMLXCore, NovaMLXDB, or NovaMLXMenuBar.
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
            Label("Tknet Peer", systemImage: appState.connectionSymbol)
        }
        .menuBarExtraStyle(.menu)

        Window("Tknet Peer", id: "peer") {
            PeerRootView()
                .environmentObject(appState)
        }
        .defaultSize(width: 760, height: 640)
    }
}

@MainActor
final class PeerAppDelegate: NSObject, NSApplicationDelegate {
    static var appState: PeerAppState?

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let state = Self.appState
        Task { await state?.launch() }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let state = Self.appState else { return .terminateNow }
        Task {
            await state.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

struct PeerMenu: View {
    @EnvironmentObject private var state: PeerAppState
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Open Tknet Peer…") {
            NSApp.activate()
            openWindow(id: "peer")
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
            .disabled(state.lockError != nil || state.isBusy)
        } else {
            Button("Start tunnel") {
                Task { await state.startTunnelFromUser() }
            }
            .disabled(state.lockError != nil || state.isBusy)
            if state.tunnelWanted {
                Button("Stop auto-dial") {
                    Task { await state.stopTunnelFromUser() }
                }
                .disabled(state.lockError != nil || state.isBusy)
            }
        }
        Divider()
        Button("Quit Tknet Peer") {
            NSApp.terminate(nil)
        }
    }
}
