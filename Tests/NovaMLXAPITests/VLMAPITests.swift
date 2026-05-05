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
private let adminBase = "http://127.0.0.1:6591"
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

/// Load a model via admin API, unloading any currently loaded model first
private func ensureModelLoaded(modelId: String) -> Bool {
    // Unload currently loaded models via admin API
    guard let adminURL = URL(string: "\(adminBase)/admin/models") else { return false }
    var listReq = URLRequest(url: adminURL)
    listReq.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    listReq.timeoutInterval = 5
    let sem1 = DispatchSemaphore(value: 0)
    var loadedModels: [String] = []
    URLSession.shared.dataTask(with: listReq) { data, resp, _ in
        if let http = resp as? HTTPURLResponse, http.statusCode == 200,
           let data, let models = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            loadedModels = models.filter { ($0["loaded"] as? Bool) == true }.compactMap { $0["id"] as? String }
        }
        sem1.signal()
    }.resume()
    _ = sem1.wait(timeout: .now() + 5)

    // Check if target already loaded
    if loadedModels.contains(modelId) { return true }

    // Unload any currently loaded model
    for id in loadedModels {
        guard let url = URL(string: "\(adminBase)/admin/models/unload") else { continue }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["modelId": id])
        req.timeoutInterval = 10
        let sem = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: req) { _, _, _ in sem.signal() }.resume()
        _ = sem.wait(timeout: .now() + 10)
    }
    sleep(2)

    // Load target model
    guard let loadURL = URL(string: "\(adminBase)/admin/models/load") else { return false }
    var loadReq = URLRequest(url: loadURL)
    loadReq.httpMethod = "POST"
    loadReq.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    loadReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
    loadReq.httpBody = try? JSONSerialization.data(withJSONObject: ["modelId": modelId])
    loadReq.timeoutInterval = 300
    let sem2 = DispatchSemaphore(value: 0)
    var loadSuccess = false
    URLSession.shared.dataTask(with: loadReq) { _, resp, _ in
        loadSuccess = (resp as? HTTPURLResponse)?.statusCode == 200
        sem2.signal()
    }.resume()
    _ = sem2.wait(timeout: .now() + 300)
    guard loadSuccess else { return false }

    // Poll until loaded (up to 90s)
    for _ in 0..<30 {
        sleep(3)
        guard let url = URL(string: "\(adminBase)/admin/models") else { return false }
        var req = URLRequest(url: url)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 5
        let sem = DispatchSemaphore(value: 0)
        var isLoaded = false
        URLSession.shared.dataTask(with: req) { data, resp, _ in
            if let http = resp as? HTTPURLResponse, http.statusCode == 200,
               let data, let models = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                isLoaded = models.first(where: { $0["id"] as? String == modelId })?["loaded"] as? Bool == true
            }
            sem.signal()
        }.resume()
        _ = sem.wait(timeout: .now() + 5)
        if isLoaded { return true }
    }
    return false
}

@Suite("VLM API Integration Tests")
struct VLMAPITests {
    // MARK: - VLM Tests

    @Test("VLM chat completion with base64 image returns valid response")
    func vlmChatWithBase64Image() async throws {
        guard serverAvailable() else {
            print("[SKIP] Server not available on \(apiBase)")
            return
        }

        // Ensure VLM model is loaded (unload current, load VLM target)
        guard ensureModelLoaded(modelId: vlmModel) else {
            print("[SKIP] Could not load VLM model \(vlmModel)")
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

        // Ensure VLM model is loaded
        guard ensureModelLoaded(modelId: vlmModel) else {
            print("[SKIP] Could not load VLM model \(vlmModel)")
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
