import XCTest

/// Read-only GUI verification for the Members Factory two-pane feature
/// (2026-10-15). Synthetic events (clicks/typing) are deliberately NOT used:
/// the user declined UI-testing automation authorization. The app is launched
/// with `--show-window --start-tab=members` and each test only READS the
/// accessibility hierarchy. CRUD logic is covered by the unit suite
/// (FactoryConfigTests); these tests verify the rendered UI end to end.
final class TknetPeerMembersUITests: XCTestCase {

    private var app: XCUIApplication!
    private var home: URL!

    override func setUpWithError() throws {
        continueAfterFailure = false
        home = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-uitest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
    }

    private func launchApp() {
        app = XCUIApplication(url: URL(fileURLWithPath:
            "/Users/lucas/orca/workspaces/novamlx/conger/dist/TknetPeer.app"))
        app.launchEnvironment["TKNETPEER_CONFIG_DIR"] = home.appendingPathComponent(".config/tknet-peer").path
        app.launchArguments += ["--show-window", "--start-tab=members"]
        app.launch()
    }

    override func tearDownWithError() throws {
        app?.terminate()
        if let home { try? FileManager.default.removeItem(at: home) }
    }

    // MARK: - Helpers

    private func waitFor(_ element: XCUIElement, _ message: String, timeout: TimeInterval = 15) {
        XCTAssertTrue(element.waitForExistence(timeout: timeout), message)
    }

    /// First static text whose value begins with `prefix`. SwiftUI Text rows
    /// carry their content as the accessibility value, not the label.
    private func textBeginning(with prefix: String) -> XCUIElement {
        app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH %@", prefix)).firstMatch
    }

    private func writeConfig(_ json: String) throws {
        let dir = home.appendingPathComponent(".config/tknet-peer", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try json.write(to: dir.appendingPathComponent("peer.json"), atomically: true, encoding: .utf8)
    }

    // MARK: - Tests

    /// No factories: the left pane shows "All" plus a non-selectable "(none)".
    func testEmptyStateShowsAllAndNone() throws {
        launchApp()
        waitFor(app.staticTexts["All"].firstMatch, "Factory pane should show the All row")
        waitFor(app.staticTexts["(none)"].firstMatch,
                "No factories: the (none) row is visible")
        waitFor(textBeginning(with: "No members yet"),
                "Empty right pane explains you need a Factory first")
    }

    /// A seeded pre-factory config (no factories key) is auto-grouped by
    /// host:port at load: one factory per base, local NovaMLX ungrouped.
    func testLegacyConfigMigrationRendersFactories() throws {
        try writeConfig("""
        {
          "serverURL": "wss://tknet.ai",
          "members": [
            {"id": "oa1", "name": "oa-one", "type": "openaiCompatible",
             "endpoint": "https://api.one.com/v1", "apiKeyRef": "src/oa1", "upstreamModel": "gpt-x"},
            {"id": "oa2", "name": "oa-two", "type": "openaiCompatible",
             "endpoint": "https://api.one.com/v2", "apiKeyRef": "src/oa2", "upstreamModel": "gpt-y"},
            {"id": "ob1", "name": "ob-one", "type": "anthropic",
             "endpoint": "http://two.example.com:8080/x", "apiKeyRef": "src/ob1", "upstreamModel": "claude"},
            {"id": "loc1", "name": "loc-one", "type": "localNovaMLX",
             "endpoint": "http://127.0.0.1:6590/v1", "apiKeyRef": "src/loc1", "upstreamModel": "qwen"}
          ]
        }
        """)
        launchApp()

        waitFor(app.staticTexts["api.one.com"].firstMatch,
                "Migration derives a factory named by host")
        waitFor(app.staticTexts["two.example.com:8080"].firstMatch,
                "Non-default ports appear in the factory name")
        XCTAssertFalse(app.staticTexts["(none)"].firstMatch.exists,
                       "(none) disappears once factories exist")

        // "All" is the initial selection: every member renders, including
        // the ungrouped local NovaMLX row and the resolved endpoints.
        waitFor(app.staticTexts["oa-one"].firstMatch, "All shows every member")
        waitFor(app.staticTexts["https://api.one.com/v1"].firstMatch,
                "Member shows its resolved endpoint")
        waitFor(app.staticTexts["loc-one"].firstMatch,
                "Ungrouped local NovaMLX member still appears under All")
        waitFor(textBeginning(with: "Local NovaMLX"),
                "Type caption renders")
    }

    /// A config with explicit factories renders the two panes from disk.
    func testExplicitFactoriesRender() throws {
        try writeConfig("""
        {
          "serverURL": "wss://tknet.ai",
          "factories": [
            {"id": "f1", "name": "OpenAI", "baseUrl": "https://api.openai.com"},
            {"id": "f2", "name": "Local Host", "baseUrl": "http://127.0.0.1:9000"}
          ],
          "members": [
            {"id": "oa1", "name": "oa-one", "type": "openaiCompatible",
             "endpoint": "https://api.openai.com/v1", "apiKeyRef": "src/oa1",
             "upstreamModel": "gpt-x", "factoryId": "f1"},
            {"id": "h1", "name": "h-one", "type": "openaiCompatible",
             "endpoint": "http://127.0.0.1:9000/v1", "apiKeyRef": "src/h1",
             "upstreamModel": "m2", "factoryId": "f2"}
          ]
        }
        """)
        launchApp()

        waitFor(app.staticTexts["OpenAI"].firstMatch, "Factory row renders")
        waitFor(app.staticTexts["Local Host"].firstMatch, "Second factory renders")
        waitFor(app.staticTexts["oa-one"].firstMatch, "Members render under All")
        waitFor(app.staticTexts["http://127.0.0.1:9000/v1"].firstMatch,
                "Second factory member endpoint renders")
    }
}
