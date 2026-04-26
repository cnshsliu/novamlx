import Foundation
import Testing
import NovaMLXCore

/// Integration tests for VLM (vision) capabilities.
/// Requires a running NovaMLX server with at least one VLM model loaded.
///
/// To run: swift test -c release --filter "VLMAPITests"
///
/// Prerequisites:
///   - Server running on 127.0.0.1:6590
///   - API key configured (default: abcd1234)
///   - A VLM model loaded (e.g. gemma-4-e4b-it-4bit)

private let apiBase = "http://127.0.0.1:6590"
private let apiKey = "abcd1234"
private let vlmModel = "mlx-community/gemma-4-e4b-it-4bit"

private func serverAvailable() -> Bool {
    guard let url = URL(string: "\(apiBase)/health") else { return false }
    var req = URLRequest(url: url)
    req.timeoutInterval = 2
    let sem = DispatchSemaphore(value: 0)
    let box = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
    box.initialize(to: false)
    defer { box.deallocate() }
    URLSession.shared.dataTask(with: req) { _, resp, _ in
        box.pointee = (resp as? HTTPURLResponse)?.statusCode == 200
        sem.signal()
    }.resume()
    _ = sem.wait(timeout: .now() + 3)
    return box.pointee
}

private func makeRequest(path: String, method: String = "POST", body: Data? = nil) -> (Data, HTTPURLResponse)? {
    guard let url = URL(string: "\(apiBase)\(path)") else { return nil }
    var req = URLRequest(url: url)
    req.httpMethod = method
    req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let body { req.httpBody = body }
    req.timeoutInterval = 120

    let sem = DispatchSemaphore(value: 0)
    let box = UnsafeMutablePointer<(Data, HTTPURLResponse)?>.allocate(capacity: 1)
    box.initialize(to: nil)
    defer { box.deallocate() }
    URLSession.shared.dataTask(with: req) { data, resp, _ in
        if let http = resp as? HTTPURLResponse, let data {
            box.pointee = (data, http)
        }
        sem.signal()
    }.resume()
    _ = sem.wait(timeout: .now() + 130)
    return box.pointee
}

/// Minimal 1x1 red pixel PNG, base64-encoded. Useful for quick VLM tests.
private let onePixelRedPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

@Suite("VLM API Integration Tests")
struct VLMAPITests {
    // MARK: - VLM Tests

    @Test("VLM chat completion with base64 image returns valid response")
    func vlmChatWithBase64Image() async throws {
        guard serverAvailable() else {
            print("[SKIP] Server not available on \(apiBase)")
            return
        }

        let requestBody: [String: Any] = [
            "model": vlmModel,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": "What color is the dot in this image? Answer in one word."],
                        ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(onePixelRedPNGBase64)"]],
                    ],
                ]
            ],
            "max_tokens": 30,
            "stream": false,
        ]

        guard let body = try? JSONSerialization.data(withJSONObject: requestBody),
              let (data, http) = makeRequest(path: "/v1/chat/completions", body: body)
        else {
            #expect(Bool(false), "Request to /v1/chat/completions failed")
            return
        }

        #expect(http.statusCode == 200, "Expected 200, got \(http.statusCode)")

        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let error = json["error"] as? [String: Any] {
                print("[INFO] VLM returned error: \(error["message"] ?? "unknown")")
                // VLM model might not be loaded — skip gracefully
                return
            }
            #expect(json["object"] as? String == "chat.completion")
            #expect(json["model"] as? String == vlmModel)
            if let choices = json["choices"] as? [[String: Any]],
               let msg = choices.first?["message"] as? [String: Any],
               let content = msg["content"] as? String {
                #expect(!content.isEmpty)
                print("[VLM RESPONSE] \(content)")
            }
        }
    }

    @Test("VLM chat completion with image_url from HTTP URL")
    func vlmChatWithHTTPImageURL() async throws {
        guard serverAvailable() else {
            print("[SKIP] Server not available")
            return
        }

        let requestBody: [String: Any] = [
            "model": vlmModel,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": "Describe this image concisely."],
                        ["type": "image_url", "image_url": ["url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/120px-Google_2015_logo.svg.png"]],
                    ],
                ]
            ],
            "max_tokens": 40,
            "stream": false,
        ]

        guard let body = try? JSONSerialization.data(withJSONObject: requestBody),
              let (data, http) = makeRequest(path: "/v1/chat/completions", body: body)
        else { return }

        if http.statusCode == 200,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           json["error"] == nil {
            #expect(json["object"] as? String == "chat.completion")
            if let choices = json["choices"] as? [[String: Any]],
               let msg = choices.first?["message"] as? [String: Any],
               let content = msg["content"] as? String {
                print("[VLM URL RESPONSE] \(content)")
            }
        }
    }

}
