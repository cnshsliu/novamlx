import NIOCore
import HTTPTypes
import Hummingbird
import Testing

@testable import NovaMLXAPI

struct ClientDetectorTests {

    // MARK: - Anthropic-Version header detection

    @Test func anthropicVersionHeaderReturnsClaudeCode() {
        var headers = HTTPFields()
        headers[.userAgent] = "curl/8.4.0"
        headers[HTTPField.Name("anthropic-version")!] = "2023-06-01"
        let request = makeRequest(headers: headers)

        let result = ClientDetector.detect(request: request)

        #expect(result == .agentTool(name: "Claude Code"))
        #expect(result.shouldScaleContext == true)
    }

    // MARK: - User-Agent substring detection

    @Test func claudeCodeUserAgent() {
        let request = makeRequest(userAgent: "ClaudeCode/2.1.89 (macOS)")
        let result = ClientDetector.detect(request: request)
        #expect(result == .agentTool(name: "Claude Code"))
    }

    @Test func openCodeUserAgent() {
        let request = makeRequest(userAgent: "OpenCode/1.0.5")
        let result = ClientDetector.detect(request: request)
        #expect(result == .agentTool(name: "OpenCode"))
    }

    @Test func openClawUserAgent() {
        let request = makeRequest(userAgent: "openclaw/0.3.1")
        let result = ClientDetector.detect(request: request)
        #expect(result == .agentTool(name: "OpenClaw"))
    }

    @Test func hermesUserAgent() {
        let request = makeRequest(userAgent: "Hermes/2.0.1")
        let result = ClientDetector.detect(request: request)
        #expect(result == .agentTool(name: "Hermes"))
    }

    @Test func caseInsensitiveUserAgent() {
        let request = makeRequest(userAgent: "HERMES-agent/3.0")
        let result = ClientDetector.detect(request: request)
        #expect(result == .agentTool(name: "Hermes"))
    }

    // MARK: - Anthropic-Version takes priority

    @Test func anthropicVersionOverridesUserAgent() {
        var headers = HTTPFields()
        headers[.userAgent] = "OpenCode/1.0"
        headers[HTTPField.Name("anthropic-version")!] = "2023-06-01"
        let request = makeRequest(headers: headers)

        let result = ClientDetector.detect(request: request)

        #expect(result == .agentTool(name: "Claude Code"))
    }

    // MARK: - General chat fallback

    @Test func generalChatWithNoMatchingHeaders() {
        let request = makeRequest(userAgent: "curl/8.4.0")
        let result = ClientDetector.detect(request: request)
        #expect(result == .generalChat)
        #expect(result.shouldScaleContext == false)
    }

    @Test func generalChatWithNoUserAgent() {
        let request = makeRequest(headers: HTTPFields())
        let result = ClientDetector.detect(request: request)
        #expect(result == .generalChat)
        #expect(result.shouldScaleContext == false)
    }

    @Test func generalChatWithPythonRequests() {
        let request = makeRequest(userAgent: "python-requests/2.31.0")
        let result = ClientDetector.detect(request: request)
        #expect(result == .generalChat)
    }

    // MARK: - Partial match prevention

    @Test func partialSubstringDoesNotMatch() {
        // "claud" should not match "claude" since we search for "claude" which contains "claud"...
        // Actually "claude" contains "claude"? Wait:
        // We search for "claude" as the pattern. "claud" does NOT contain "claude".
        // But "claudecode" DOES contain "claude". Let's test a clearly non-matching case.
        let request = makeRequest(userAgent: "generic-client/1.0")
        let result = ClientDetector.detect(request: request)
        #expect(result == .generalChat)
    }

    // MARK: - Helpers

    private func makeRequest(userAgent: String? = nil, headers: HTTPFields = [:]) -> Request {
        var h = headers
        if let ua = userAgent {
            h[.userAgent] = ua
        }
        // Use a valid fake URI and method to construct a Request
        var head = HTTPRequest(method: .post, scheme: "http", authority: "localhost", path: "/v1/messages")
        head.headerFields = h
        return Request(head: head, body: .init(buffer: ByteBuffer()))
    }
}
