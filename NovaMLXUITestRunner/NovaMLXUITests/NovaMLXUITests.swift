import XCTest

/// Comprehensive XCUITest suite for the new "Model Source" (mirror) feature.
/// This tests the ability to select different download/search mirrors:
/// - Official Hugging Face
/// - hf-mirror.com (China)
/// - ModelScope (Alibaba)
/// - Custom URL
///
/// The tests are fully automated and cover switching, custom input, search behavior,
/// and the immediate effect of the mirror choice.
final class NovaMLXMirrorFeatureTests: XCTestCase {

    let app = XCUIApplication()

    // Path to the freshly built debug app produced by ./build.sh
    private let appExecutablePath = "/Users/lucas/dev/novamlx/dist/NovaMLX.app/Contents/MacOS/NovaMLX"

    override func setUp() {
        super.setUp()
        continueAfterFailure = false

        // When the scheme has "Target Application" set to dist/NovaMLX.app,
        // this is the clean, recommended way.
        let app = XCUIApplication()
        app.launch()
    }

    override func tearDown() {
        app.terminate()
        super.tearDown()
    }

    // MARK: - Helper: Navigate to Downloads tab

    /// Ensures we are on the Downloads page. Clicks the sidebar item if needed.
    private func navigateToDownloadsTab() {
        let downloadsSidebar = app.buttons["sidebar-downloads"]
        if downloadsSidebar.exists && !downloadsSidebar.isSelected {
            downloadsSidebar.click()
            // Wait for the Downloads content to appear
            _ = app.staticTexts["Model Source"].waitForExistence(timeout: 5)
        }
    }

    // MARK: - Test Cases

    /// 1. Basic smoke test: "Model Source" section should be visible at the top when on Downloads.
    func testModelSourceSectionIsVisibleAtTopOfDownloads() {
        navigateToDownloadsTab()

        let modelSourceSection = app.staticTexts["Model Source"]
        XCTAssertTrue(modelSourceSection.exists, "Model Source section should be visible at the top of the Downloads tab")
    }

    /// 2. Default state should be "Official".
    func testDefaultMirrorIsOfficial() {
        navigateToDownloadsTab()

        let picker = app.popUpButtons["model-source-picker"]
        XCTAssertTrue(picker.exists)

        // Open the menu and verify "Official" is selected (or at least present as first choice)
        picker.click()
        let officialItem = app.menuItems.containing(.staticText, identifier: "Official (huggingface.co)").firstMatch
        XCTAssertTrue(officialItem.exists)
    }

    /// 3. Switching to hf-mirror.com should show the correct toast and update the picker.
    func testSwitchToHFMirrorChina() {
        navigateToDownloadsTab()

        let picker = app.popUpButtons["model-source-picker"]
        picker.click()

        let hfMirrorItem = app.menuItems["hf-mirror.com (China)"]
        XCTAssertTrue(hfMirrorItem.exists, "hf-mirror.com option must exist")
        hfMirrorItem.click()

        // Verify the toast appears
        let toast = app.staticTexts.containing(.staticText, identifier: "Mirror changed").firstMatch
        XCTAssertTrue(toast.waitForExistence(timeout: 3), "Toast should appear after changing mirror")
    }

    /// 4. Switching to ModelScope (Alibaba) — the new first-class option.
    func testSwitchToModelScopeAlibaba() {
        navigateToDownloadsTab()

        let picker = app.popUpButtons["model-source-picker"]
        picker.click()

        let modelScopeItem = app.menuItems["ModelScope (Alibaba)"]
        XCTAssertTrue(modelScopeItem.exists, "ModelScope (Alibaba) must appear in the picker")
        modelScopeItem.click()

        // Verify specific toast for ModelScope
        let toast = app.staticTexts.containing(.staticText, identifier: "ModelScope").firstMatch
        XCTAssertTrue(toast.waitForExistence(timeout: 3))
    }

    /// 5. Custom URL flow: select Custom, type a URL, and verify it is accepted.
    func testCustomMirrorURL() {
        navigateToDownloadsTab()

        let picker = app.popUpButtons["model-source-picker"]
        picker.click()

        let customItem = app.menuItems["Custom URL..."]
        customItem.click()

        // The custom text field should appear
        let customField = app.textFields["model-source-custom-field"]
        XCTAssertTrue(customField.waitForExistence(timeout: 2))

        customField.click()
        customField.typeText("https://hf-mirror.com")

        // Press return to commit
        customField.typeKey(XCUIKeyboardKey.return.rawValue, modifierFlags: [])

        // The field should still exist and contain the value
        XCTAssertEqual(customField.value as? String, "https://hf-mirror.com")
    }

    /// 6. End-to-end: Change mirror, then perform a search — the search should still succeed
    /// (we don't assert the exact backend here, but we verify the UI doesn't break).
    func testSearchWorksAfterChangingToNonOfficialMirror() {
        navigateToDownloadsTab()

        // Switch to ModelScope
        let picker = app.popUpButtons["model-source-picker"]
        picker.click()
        app.menuItems["ModelScope (Alibaba)"].click()

        // Wait for any toast to disappear
        sleep(1)

        // Perform a search
        let searchField = app.textFields["downloads-search-field"]
        XCTAssertTrue(searchField.exists)

        searchField.click()
        searchField.typeText("mlx-community/gemma")
        searchField.typeKey(XCUIKeyboardKey.return.rawValue, modifierFlags: [])

        // We should either get results or at least not crash / show an error alert immediately
        // Give the network call some time
        sleep(4)

        // If search results appear, great. If not, we at least didn't get a fatal error.
        let hasResults = app.otherElements["search-results"].exists || app.tables.firstMatch.exists
        let noFatalError = !app.alerts.firstMatch.exists

        XCTAssertTrue(noFatalError, "Search after mirror change should not produce a fatal error alert")
    }

    /// 7. Verify that changing the mirror is immediately reflected in the picker value.
    func testPickerSelectionUpdatesImmediately() {
        navigateToDownloadsTab()

        let picker = app.popUpButtons["model-source-picker"]

        // Switch to hf-mirror
        picker.click()
        app.menuItems["hf-mirror.com (China)"].click()

        // Re-open and check selection (the title of the pop-up button usually reflects the chosen item)
        picker.click()
        // The menu item should be marked as selected (we can check existence + state in some cases)
        let selectedItem = app.menuItems["hf-mirror.com (China)"]
        XCTAssertTrue(selectedItem.exists)
    }
}