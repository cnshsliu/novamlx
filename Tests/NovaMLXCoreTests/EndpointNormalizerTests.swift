import XCTest
@testable import NovaMLXCore

/// Tokenhub accepts any endpoint from any provider. Every form a user
/// can paste must produce the URL the proxy actually calls.
final class EndpointNormalizerTests: XCTestCase {

    private func assertURL(_ endpoint: String, _ suffix: String, _ expected: String,
                           file: StaticString = #filePath, line: UInt = #line) {
        guard let url = EndpointNormalizer.url(endpoint: endpoint, suffix: suffix) else {
            return XCTFail("nil for \(endpoint)", file: file, line: line)
        }
        XCTAssertEqual(url.absoluteString, expected, file: file, line: line)
    }

    func testBaseURLAppendsSuffix() {
        assertURL("https://api.openai.com/v1", "chat/completions",
                  "https://api.openai.com/v1/chat/completions")
    }

    func testTrailingSlashTolerated() {
        assertURL("https://api.deepseek.com/", "chat/completions",
                  "https://api.deepseek.com/chat/completions")
    }

    func testBareHost() {
        assertURL("https://api.deepseek.com", "models",
                  "https://api.deepseek.com/models")
    }

    func testFullRequestURLUsedAsIs() {
        // User pasted the complete chat URL — must NOT double the suffix.
        assertURL("https://gateway.example.com/v1/chat/completions", "chat/completions",
                  "https://gateway.example.com/v1/chat/completions")
        assertURL("https://api.anthropic.com/v1/messages", "messages",
                  "https://api.anthropic.com/v1/messages")
    }

    func testQueryStringPreserved() {
        // Azure-style / gateway-style endpoints carry auth or version in
        // the query. appendingPathComponent would corrupt these.
        assertURL("https://res.example.com/proxy?key=abc123", "chat/completions",
                  "https://res.example.com/proxy/chat/completions?key=abc123")
        assertURL("https://res.example.com/openai/deployments/gpt?api-version=2026-01-01", "models",
                  "https://res.example.com/openai/deployments/gpt/models?api-version=2026-01-01")
    }

    func testFullURLWithQueryUnchanged() {
        assertURL("https://gw.example.com/v1/chat/completions?session=x", "chat/completions",
                  "https://gw.example.com/v1/chat/completions?session=x")
    }

    func testSchemeLessEndpointDefaultsToHTTPS() {
        assertURL("api.groq.com/openai/v1", "chat/completions",
                  "https://api.groq.com/openai/v1/chat/completions")
    }

    func testWhitespaceTrimmed() {
        assertURL("  https://api.x.ai/v1\n", "chat/completions",
                  "https://api.x.ai/v1/chat/completions")
    }

    func testNestedProxyPath() {
        // Corporate proxies with path prefixes.
        assertURL("https://corp.example.com/llm/gateway/v1", "messages",
                  "https://corp.example.com/llm/gateway/v1/messages")
    }

    func testInvalidInputsReturnNil() {
        for bad in ["", "   ", "not a url at all ://", "ftp://files.example.com/v1"] {
            XCTAssertNil(EndpointNormalizer.url(endpoint: bad, suffix: "chat/completions"),
                         "expected nil for \(bad)")
        }
    }
}
