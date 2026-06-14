import Foundation
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXUtils

// MARK: - Image Preprocessing for Text-Only Providers

/// Three-tier image preprocessing: local VLM → anthropic-proxy → companion model → placeholder.
/// When a request contains images and the target provider doesn't support vision,
/// images are transparently converted to text descriptions before forwarding.
enum ImagePreprocessor {

    // MARK: - Backend Types

    enum VisionBackend: Sendable {
        case localVLM(modelId: String)
        case anthropicProxy(endpoint: String, apiKey: String, model: String)
        case companion(endpoint: String, apiKey: String, model: String)
        case none
    }

    struct PreprocessResult: @unchecked Sendable {
        let messages: [[String: Any]]
        let imagesProcessed: Int
    }

    // MARK: - Content Extraction

    /// Separate text and image URLs from ResponseMessageContent.
    static func extractContent(_ content: ResponseMessageContent) -> (text: String, imageURLs: [String]) {
        return (content.textValue, content.imageURLs)
    }

    // MARK: - Backend Resolution

    /// Pick the best vision backend based on available resources and provider config.
    static func resolveBackend(
        provider: TokenhubProvider,
        inference: InferenceService,
        defaultVisionModel: String? = nil
    ) -> VisionBackend {
        // Tier 1: Local VLM
        let vlmId = findLocalVLM(inference: inference, preferred: defaultVisionModel)
        if let vlmId {
            NovaMLXLog.info("[ImagePreprocessor] Using local VLM: \(vlmId)")
            return .localVLM(modelId: vlmId)
        }

        // Tier 2: Anthropic proxy (provider's own internal routing, e.g., Zhipu GLM)
        if provider.visionStrategy == "anthropic-proxy",
           let endpoint = provider.anthropicEndpoint {
            let apiKey = provider.tags.contains("managed") ? (AuthCache.loadSession() ?? "") : provider.apiKey
            NovaMLXLog.info("[ImagePreprocessor] Using anthropic-proxy: \(endpoint)")
            return .anthropicProxy(endpoint: endpoint, apiKey: apiKey, model: provider.remoteModel)
        }

        // Tier 3: Companion vision model
        if provider.visionStrategy == "companion",
           let companionModel = provider.visionCompanionModel {
            let apiKey = provider.tags.contains("managed") ? (AuthCache.loadSession() ?? "") : provider.apiKey
            NovaMLXLog.info("[ImagePreprocessor] Using companion model: \(companionModel)")
            return .companion(endpoint: provider.endpoint, apiKey: apiKey, model: companionModel)
        }

        NovaMLXLog.warning("[ImagePreprocessor] No vision backend available for \(provider.id)")
        return .none
    }

    // MARK: - Batch Preprocessing

    /// Preprocess messages: describe images and inject text into message content.
    /// - Parameters:
    ///   - messages: Chat Completions format messages (mutated in place via copy)
    ///   - imageBlocks: messageIndex → [imageURLs] extracted from Responses API input
    ///   - backend: Where to send images for description
    ///   - inference: For local VLM path
    /// - Returns: Updated messages and count of images processed
    static func preprocess(
        messages: [[String: Any]],
        imageBlocks: [Int: [String]],
        backend: VisionBackend,
        inference: InferenceService
    ) async -> PreprocessResult {
        guard !imageBlocks.isEmpty else {
            return PreprocessResult(messages: messages, imagesProcessed: 0)
        }

        var result = messages
        var processed = 0

        for (msgIdx, imageURLs) in imageBlocks {
            guard msgIdx < result.count else { continue }

            var descriptions: [String] = []
            for url in imageURLs.prefix(10) {
                let desc = await describeImage(url: url, backend: backend, inference: inference)
                if !desc.isEmpty {
                    descriptions.append(desc)
                } else {
                    NovaMLXLog.warning("[ImagePreprocessor] Empty description returned for \(truncateURL(url))")
                }
                processed += 1
            }

            if !descriptions.isEmpty {
                let existingContent = result[msgIdx]["content"] as? String ?? ""
                let imageBlock = descriptions.map { "[Image description: \($0)]" }.joined(separator: "\n")
                let newContent = existingContent.isEmpty
                    ? imageBlock
                    : existingContent + "\n\n" + imageBlock
                result[msgIdx]["content"] = newContent
            }
        }

        return PreprocessResult(messages: result, imagesProcessed: processed)
    }

    // MARK: - Single Image Description

    /// Describe a single image using the specified backend.
    private static func describeImage(
        url: String,
        backend: VisionBackend,
        inference: InferenceService
    ) async -> String {
        switch backend {
        case .localVLM(let modelId):
            return await describeViaLocalVLM(url: url, modelId: modelId, inference: inference)
        case .anthropicProxy(let endpoint, let apiKey, let model):
            return await describeViaAnthropicProxy(url: url, endpoint: endpoint, apiKey: apiKey, model: model)
        case .companion(let endpoint, let apiKey, let model):
            return await describeViaCompanion(url: url, endpoint: endpoint, apiKey: apiKey, model: model)
        case .none:
            return "[Image: \(truncateURL(url))]"
        }
    }

    // MARK: - Tier 1: Local VLM

    private static func describeViaLocalVLM(
        url: String,
        modelId: String,
        inference: InferenceService
    ) async -> String {
        guard inference.isModelLoaded(modelId) else {
            NovaMLXLog.warning("[ImagePreprocessor] Local VLM \(modelId) not loaded, skipping")
            return "[Image: \(truncateURL(url))]"
        }

        let request = InferenceRequest(
            model: modelId,
            messages: [
                ChatMessage(role: .user, content: "Describe this image in detail.", images: [url])
            ],
            stream: false
        )

        do {
            let result = try await inference.generate(request)
            let desc = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            NovaMLXLog.info("[ImagePreprocessor] Local VLM described image: \(desc.prefix(100))...")
            return desc
        } catch {
            NovaMLXLog.error("[ImagePreprocessor] Local VLM failed: \(error)")
            return "[Image: \(truncateURL(url)) - description failed]"
        }
    }

    // MARK: - Tier 2: Anthropic Proxy

    /// Send image to provider's Anthropic-compatible endpoint.
    /// The provider (e.g., Zhipu) handles internal routing to their vision model.
    private static func describeViaAnthropicProxy(
        url: String,
        endpoint: String,
        apiKey: String,
        model: String
    ) async -> String {
        let baseURL = endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let requestURL = URL(string: "\(baseURL)/v1/messages") else {
            return "[Image: invalid endpoint]"
        }

        // Build Anthropic Messages API format with image content block
        var body: [String: Any] = [
            "model": model,
            "max_tokens": 1024,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": "Describe this image in detail."],
                        buildImageContentBlock(url: url)
                    ]
                ]
            ]
        ]

        guard let bodyData = try? JSONSerialization.data(withJSONObject: body) else {
            return "[Image: serialization failed]"
        }

        var urlRequest = URLRequest(url: requestURL)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        if !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "x-api-key")
        }
        urlRequest.timeoutInterval = 15
        urlRequest.httpBody = bodyData

        do {
            let (data, response) = try await URLSession.shared.data(for: urlRequest)
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                NovaMLXLog.warning("[ImagePreprocessor] Anthropic proxy failed HTTP \(status)")
                return "[Image: proxy returned \(status)]"
            }

            // Parse Anthropic response: { content: [{ type: "text", text: "..." }] }
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let content = json["content"] as? [[String: Any]],
                  let textBlock = content.first(where: { $0["type"] as? String == "text" }),
                  let text = textBlock["text"] as? String else {
                return "[Image: unexpected response format]"
            }

            NovaMLXLog.info("[ImagePreprocessor] Anthropic proxy described image: \(text.prefix(100))...")
            return text
        } catch {
            NovaMLXLog.error("[ImagePreprocessor] Anthropic proxy error: \(error)")
            return "[Image: proxy error]"
        }
    }

    // MARK: - Tier 3: Companion Model

    /// Call a separate vision model via Chat Completions API.
    private static func describeViaCompanion(
        url: String,
        endpoint: String,
        apiKey: String,
        model: String
    ) async -> String {
        let baseURL = endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let requestURL = URL(string: "\(baseURL)/chat/completions") else {
            return "[Image: invalid endpoint]"
        }

        var body: [String: Any] = [
            "model": model,
            "max_tokens": 1024,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": "Describe this image in detail."],
                        ["type": "image_url", "image_url": ["url": url]]
                    ]
                ]
            ]
        ]

        guard let bodyData = try? JSONSerialization.data(withJSONObject: body) else {
            return "[Image: serialization failed]"
        }

        var urlRequest = URLRequest(url: requestURL)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.timeoutInterval = 15
        urlRequest.httpBody = bodyData

        do {
            let (data, response) = try await URLSession.shared.data(for: urlRequest)
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                NovaMLXLog.warning("[ImagePreprocessor] Companion model failed HTTP \(status)")
                return "[Image: companion returned \(status)]"
            }

            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any],
                  let text = message["content"] as? String else {
                return "[Image: unexpected response format]"
            }

            NovaMLXLog.info("[ImagePreprocessor] Companion model described image: \(text.prefix(100))...")
            return text
        } catch {
            NovaMLXLog.error("[ImagePreprocessor] Companion model error: \(error)")
            return "[Image: companion error]"
        }
    }

    // MARK: - Helpers

    /// Find a loaded local VLM model. Prefer configured default, else first loaded VLM.
    private static func findLocalVLM(
        inference: InferenceService,
        preferred: String?
    ) -> String? {
        let loadedIds = inference.engine.pool.loadedModelIds

        // Check preferred first
        if let preferred, loadedIds.contains(preferred),
           let container = inference.engine.pool.get(preferred),
           container.config.modelType == .vlm {
            return preferred
        }

        // First loaded VLM
        return loadedIds.first { id in
            guard let container = inference.engine.pool.get(id) else { return false }
            return container.config.modelType == .vlm
        }
    }

    /// Build an Anthropic-format image content block.
    /// Handles both data: URLs and HTTPS URLs.
    private static func buildImageContentBlock(url: String) -> [String: Any] {
        if url.hasPrefix("data:") {
            // data:image/png;base64,... → extract media type and base64 data
            let parts = url.split(separator: ",", maxSplits: 1)
            let mediaType = parts.first?.replacingOccurrences(of: "data:", with: "")
                .replacingOccurrences(of: ";base64", with: "") ?? "image/png"
            let base64 = parts.count > 1 ? String(parts[1]) : ""
            return [
                "type": "image",
                "source": [
                    "type": "base64",
                    "media_type": mediaType,
                    "data": base64
                ]
            ]
        } else {
            // HTTPS URL — Anthropic supports url source type
            return [
                "type": "image",
                "source": [
                    "type": "url",
                    "url": url
                ]
            ]
        }
    }

    private static func truncateURL(_ url: String) -> String {
        if url.hasPrefix("data:") {
            return "base64 image (\(url.count) chars)"
        }
        return String(url.prefix(100))
    }
}
