import XCTest

/// Debug probe: dump the accessibility hierarchy of the running TknetPeerApp
/// so we can see what XCUITest can reach (status item, menu, window).
final class HierarchyProbeTests: XCTestCase {

    func testDumpHierarchy() throws {
        // Diagnostic only: re-enable (remove the skip) when debugging
        // accessibility queries; prints the full element subtree.
        throw XCTSkip("hierarchy probe; re-enable manually when debugging AX queries")
        #if false
        let home = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-probe-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: home) }

        let app = XCUIApplication(url: URL(fileURLWithPath:
            "/Users/lucas/orca/workspaces/novamlx/conger/dist/TknetPeer.app"))
        app.launchEnvironment["TKNETPEER_CONFIG_DIR"] = home.appendingPathComponent(".config/tknet-peer").path
        app.launchArguments += ["--start-tab=members"]
        app.launch()
        sleep(8)

        print("===== DEBUG DESCRIPTION =====")
        print(app.debugDescription)
        print("===== STATUS ITEMS =====")
        print(app.statusItems.debugDescription)
        app.terminate()
        #endif
    }
}
