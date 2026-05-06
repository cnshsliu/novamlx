import Testing
import Foundation
import HTTPTypes
import Hummingbird
import NovaMLXCore

import NovaMLXModelManager
@testable import NovaMLXAPI

@Suite("API Types Tests")
struct APITypesTests {
    @Test("OpenAI request encoding")
    func openAIRequestEncoding() throws {
        let req = OpenAIRequest(
            model: "gpt-4",
            messages: [OpenAIChatMessage(role: "user", content: "Hello")],
            temperature: 0.7,
            maxTokens: 100
        )
        let data = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(OpenAIRequest.self, from: data)
        #expect(decoded.model == "gpt-4")
        #expect(decoded.temperature == 0.7)
        #expect(decoded.maxTokens == 100)
    }

    @Test("OpenAI response encoding")
    func openAIResponseEncoding() throws {
        let response = OpenAIResponse(
            id: "chatcmpl-test",
            model: "test",
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(role: "assistant", content: "Hi there!"),
                    finishReason: "stop"
                )
            ],
            usage: OpenAIUsage(promptTokens: 10, completionTokens: 5)
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAIResponse.self, from: data)
        #expect(decoded.object == "chat.completion")
        #expect(decoded.choices.count == 1)
        #expect(decoded.usage?.totalTokens == 15)
    }

    @Test("OpenAI stream chunk")
    func openAIStreamChunk() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-test",
            model: "test",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "Hello"))]
        )
        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        #expect(decoded.object == "chat.completion.chunk")
        #expect(decoded.choices.first?.delta.content == "Hello")
    }

    @Test("OpenAI models response")
    func openAIModelsResponse() throws {
        let response = OpenAIModelsResponse(data: [OpenAIModel(id: "test-1"), OpenAIModel(id: "test-2")])
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
        #expect(decoded.object == "list")
        #expect(decoded.data.count == 2)
    }

    @Test("Anthropic request encoding")
    func anthropicRequestEncoding() throws {
        let req = AnthropicRequest(
            model: "claude-3",
            messages: [AnthropicMessage(role: "user", content: "Hello")],
            maxTokens: 2048,
            system: .text("You are helpful")
        )
        let data = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(AnthropicRequest.self, from: data)
        #expect(decoded.model == "claude-3")
        #expect(decoded.maxTokens == 2048)
        #expect(decoded.system == .text("You are helpful"))
    }

    @Test("Anthropic response encoding")
    func anthropicResponseEncoding() throws {
        let response = AnthropicResponse(
            id: "msg-test",
            model: "claude-3",
            content: [AnthropicContentBlock(text: "Hello!")],
            stopReason: "end_turn",
            usage: AnthropicUsage(inputTokens: 10, outputTokens: 5)
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(AnthropicResponse.self, from: data)
        #expect(decoded.type == "message")
        #expect(decoded.content.first?.text == "Hello!")
        #expect(decoded.usage.outputTokens == 5)
    }

    @Test("Anthropic stream events")
    func anthropicStreamEvents() throws {
        let start = AnthropicStreamEvent.messageStart(id: "msg-test", model: "claude-3")
        #expect(start.type == "message_start")
        #expect(start.message?.id == "msg-test")

        let blockStart = AnthropicStreamEvent.contentBlockStart(index: 0)
        #expect(blockStart.type == "content_block_start")

        let delta = AnthropicStreamEvent.textDelta("Hello")
        #expect(delta.type == "content_block_delta")
        #expect(delta.delta?.text == "Hello")

        let stop = AnthropicStreamEvent.messageStop()
        #expect(stop.type == "message_stop")
    }

    @Test("OpenAI chat message roles")
    func openAIChatMessageRoles() {
        let msgs = [
            OpenAIChatMessage(role: "system", content: "sys"),
            OpenAIChatMessage(role: "user", content: "usr"),
            OpenAIChatMessage(role: "assistant", content: "ast"),
        ]
        #expect(msgs[0].role == "system")
        #expect(msgs[1].role == "user")
        #expect(msgs[2].role == "assistant")
    }

    @Test("OpenAI error response encoding")
    func openAIErrorResponseEncoding() throws {
        let detail = OpenAIErrorDetail(
            message: "Model not found: test-model",
            type: "not_found_error",
            code: "model_not_found"
        )
        let response = OpenAIErrorResponse(error: detail)
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAIErrorResponse.self, from: data)
        #expect(decoded.error.message == "Model not found: test-model")
        #expect(decoded.error.type == "not_found_error")
        #expect(decoded.error.code == "model_not_found")
        #expect(decoded.error.param == nil)
    }

    @Test("OpenAI error response with param")
    func openAIErrorResponseWithParam() throws {
        let detail = OpenAIErrorDetail(
            message: "Invalid temperature",
            type: "invalid_request_error",
            param: "temperature",
            code: "invalid_value"
        )
        let response = OpenAIErrorResponse(error: detail)
        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let errorObj = json?["error"] as? [String: Any]
        #expect(errorObj?["param"] as? String == "temperature")
    }

    @Test("NovaMLXError HTTP status mapping")
    func novaMLXErrorHTTPStatus() {
        #expect(NovaMLXError.modelNotFound("x").httpStatus == .notFound)
        #expect(NovaMLXError.modelLoadFailed("x", underlying: NSError(domain: "", code: 0)).httpStatus == .serviceUnavailable)
        #expect(NovaMLXError.inferenceFailed("x").httpStatus == .internalServerError)
        #expect(NovaMLXError.configurationError("x").httpStatus == .internalServerError)
        #expect(NovaMLXError.cacheError("x").httpStatus == .internalServerError)
        #expect(NovaMLXError.apiError("x").httpStatus == .badRequest)
        #expect(NovaMLXError.downloadFailed("x", underlying: NSError(domain: "", code: 0)).httpStatus == .badGateway)
        #expect(NovaMLXError.unsupportedModel("x").httpStatus == .badRequest)
    }

    @Test("NovaMLXError error type and code mapping")
    func novaMLXErrorTypeAndCode() {
        let notFound = NovaMLXError.modelNotFound("test")
        #expect(notFound.apiErrorType == "not_found_error")
        #expect(notFound.apiErrorCode == "model_not_found")

        let apiErr = NovaMLXError.apiError("bad")
        #expect(apiErr.apiErrorType == "invalid_request_error")
        #expect(apiErr.apiErrorCode == "api_error")

        let unsupported = NovaMLXError.unsupportedModel("bad")
        #expect(unsupported.apiErrorType == "invalid_request_error")
        #expect(unsupported.apiErrorCode == "unsupported_model")
    }

    @Test("NovaMLXError produces valid error detail")
    func novaMLXErrorDetail() {
        let error = NovaMLXError.modelNotFound("llama-3")
        let detail = OpenAIErrorDetail(
            message: error.errorDescription ?? "Unknown error",
            type: error.apiErrorType,
            code: error.apiErrorCode
        )
        let response = OpenAIErrorResponse(error: detail)
        let data = try! JSONEncoder().encode(response)
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        let errorObj = json["error"] as! [String: Any]
        #expect(errorObj["message"] as? String == "Model not found: llama-3")
        #expect(errorObj["type"] as? String == "not_found_error")
        #expect(errorObj["code"] as? String == "model_not_found")
    }

    @Test("Admin model status encoding")
    func adminModelStatusEncoding() throws {
        let status = AdminModelStatus(
            id: "test-model",
            family: "llama",
            downloaded: true,
            loaded: false,
            sizeBytes: 4_800_000_000,
            downloadedAt: Date()
        )
        let data = try JSONEncoder().encode(status)
        let decoded = try JSONDecoder().decode(AdminModelStatus.self, from: data)
        #expect(decoded.id == "test-model")
        #expect(decoded.family == "llama")
        #expect(decoded.downloaded == true)
        #expect(decoded.loaded == false)
        #expect(decoded.sizeBytes == 4_800_000_000)
        #expect(decoded.downloadedAt != nil)
    }

    @Test("Admin model status without download date")
    func adminModelStatusNoDate() throws {
        let status = AdminModelStatus(
            id: "new-model",
            family: "other",
            downloaded: false,
            loaded: false,
            sizeBytes: 0,
            downloadedAt: nil
        )
        let data = try JSONEncoder().encode(status)
        let decoded = try JSONDecoder().decode(AdminModelStatus.self, from: data)
        #expect(decoded.downloadedAt == nil)
        #expect(decoded.downloaded == false)
    }

    @Test("Admin download request encoding")
    func adminDownloadRequestEncoding() throws {
        let req = AdminDownloadRequest(modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit")
        let data = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(AdminDownloadRequest.self, from: data)
        #expect(decoded.modelId == "mlx-community/Qwen2.5-7B-Instruct-4bit")
    }

    @Test("Admin load request encoding")
    func adminLoadRequestEncoding() throws {
        let req = AdminLoadRequest(modelId: "test-model")
        let data = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(AdminLoadRequest.self, from: data)
        #expect(decoded.modelId == "test-model")
    }

    @Test("Admin download progress encoding")
    func adminDownloadProgressEncoding() throws {
        let progress = DownloadProgress(
            bytesDownloaded: 1_000_000,
            totalBytes: 4_800_000_000,
            bytesPerSecond: 5_000_000
        )
        let adminProgress = AdminDownloadProgress(modelId: "test-model", progress: progress)
        let data = try JSONEncoder().encode(adminProgress)
        let decoded = try JSONDecoder().decode(AdminDownloadProgress.self, from: data)
        #expect(decoded.modelId == "test-model")
        #expect(decoded.bytesDownloaded == 1_000_000)
        #expect(decoded.totalBytes == 4_800_000_000)
        #expect(decoded.fractionCompleted > 0)
        #expect(decoded.status == "downloading")
    }

    @Test("ServerConfig includes admin port")
    func serverConfigAdminPort() {
        let config = ServerConfig()
        #expect(config.adminPort == 6591)
        #expect(config.port == 6590)

        let custom = ServerConfig(port: 9090, adminPort: 9091, apiKeys: ["secret"])
        #expect(custom.port == 9090)
        #expect(custom.adminPort == 9091)
        #expect(custom.apiKeys == ["secret"])
    }

    @Test("Embedding input decoding from string")
    func embeddingInputString() throws {
        let json = """
        {"model":"test","input":"hello"}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: data)
        #expect(req.model == "test")
        #expect(req.input.texts == ["hello"])
    }

    @Test("Embedding input decoding from array")
    func embeddingInputArray() throws {
        let json = """
        {"model":"test","input":["hello","world"]}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: data)
        #expect(req.model == "test")
        #expect(req.input.texts == ["hello", "world"])
    }

    @Test("Embedding response encoding")
    func embeddingResponseEncoding() throws {
        let response = OpenAIEmbeddingResponse(
            model: "test-embed",
            data: [
                OpenAIEmbeddingData(index: 0, embedding: [0.1, 0.2, 0.3]),
                OpenAIEmbeddingData(index: 1, embedding: [0.4, 0.5, 0.6]),
            ],
            usage: OpenAIEmbeddingUsage(promptTokens: 4, totalTokens: 4)
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAIEmbeddingResponse.self, from: data)
        #expect(decoded.object == "list")
        #expect(decoded.data.count == 2)
        #expect(decoded.data[0].index == 0)
        #expect(decoded.data[0].object == "embedding")
        #expect(decoded.usage.promptTokens == 4)
        #expect(decoded.usage.totalTokens == 4)
    }

    @Test("Embedding request with encoding format")
    func embeddingRequestWithFormat() throws {
        let req = OpenAIEmbeddingRequest(
            model: "text-embedding-3-small",
            input: .multiple(["hello", "world"]),
            encodingFormat: "base64"
        )
        let data = try JSONEncoder().encode(req)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["encoding_format"] as? String == "base64")
        #expect(json?["input"] as? [String] == ["hello", "world"])
    }

    // MARK: - VLM (Vision) Content Decoding

    @Test("OpenAI chat message with image_url content part")
    func openAIChatMessageWithImageURL() throws {
        let json = """
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let msg = try JSONDecoder().decode(OpenAIChatMessage.self, from: data)
        #expect(msg.role == "user")

        guard let content = msg.content else {
            #expect(Bool(false), "content should not be nil")
            return
        }

        // Verify text is extracted
        #expect(content.textValue == "What is in this image?")

        // Verify image parts
        let imageParts = content.imageParts
        #expect(imageParts.count == 1)
        if case .imageUrl(let img) = imageParts[0] {
            #expect(img.url.hasPrefix("data:image/png;base64,"))
        } else {
            #expect(Bool(false), "First part should be image_url")
        }
    }

    @Test("OpenAI chat message with multiple images")
    func openAIChatMessageWithMultipleImages() throws {
        let json = """
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img1.jpg"}},
                {"type": "text", "text": "Compare these images"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img2.jpg"}}
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let msg = try JSONDecoder().decode(OpenAIChatMessage.self, from: data)

        guard let content = msg.content else {
            #expect(Bool(false), "content should not be nil")
            return
        }

        #expect(content.textValue == "Compare these images")
        #expect(content.imageParts.count == 2)

        let urls = content.imageParts.compactMap { part -> String? in
            if case .imageUrl(let img) = part { return img.url } else { return nil }
        }
        #expect(urls.count == 2)
        #expect(urls[0] == "https://example.com/img1.jpg")
        #expect(urls[1] == "https://example.com/img2.jpg")
    }

    @Test("VLM ChatMessage maps images from OpenAIChatMessage")
    func vlmChatMessageImageMapping() throws {
        let json = """
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,FAKEBASE64=="}}
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let msg = try JSONDecoder().decode(OpenAIChatMessage.self, from: data)

        let imageURLs = msg.content?.imageParts.compactMap { part -> String? in
            if case .imageUrl(let img) = part { return img.url } else { return nil }
        }
        let chatMsg = ChatMessage(
            role: .user,
            content: msg.content?.textValue,
            images: (imageURLs?.isEmpty ?? true) ? nil : imageURLs
        )

        #expect(chatMsg.content == "Describe this")
        #expect(chatMsg.images?.count == 1)
        #expect(chatMsg.images?[0] == "data:image/png;base64,FAKEBASE64==")
    }

    @Test("ImageURL decodes detail field")
    func imageURLWithDetail() throws {
        let json = """
        {"url": "https://example.com/photo.jpg", "detail": "high"}
        """
        let data = json.data(using: .utf8)!
        let img = try JSONDecoder().decode(ImageURL.self, from: data)
        #expect(img.url == "https://example.com/photo.jpg")
        #expect(img.detail == "high")
    }

    // MARK: - reasoning_effort mapping

    @Test("reasoning_effort decodes from JSON")
    func reasoningEffortDecodes() throws {
        let json = """
        {"model":"test","messages":[],"reasoning_effort":"high"}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIRequest.self, from: data)
        #expect(req.reasoningEffort == "high")
    }

    @Test("reasoning_effort maps to thinking budget")
    func reasoningEffortMapping() throws {
        let cases: [(String?, Int?)] = [
            ("high", 32768),
            ("medium", 8192),
            ("low", 1024),
            ("none", 0),
            ("HIGH", 32768), // case-insensitive
            (nil, nil),
            ("unknown", nil),
        ]
        for (effort, expected) in cases {
            let req = OpenAIRequest(
                model: "test",
                messages: [],
                reasoningEffort: effort
            )
            #expect(req.resolvedThinkingBudget == expected,
                    "reasoning_effort=\(effort ?? "nil") should map to \(String(describing: expected))")
        }
    }

    @Test("explicit thinking_budget wins over reasoning_effort")
    func thinkingBudgetPrecedence() throws {
        let req = OpenAIRequest(
            model: "test",
            messages: [],
            thinkingBudget: 2048,
            reasoningEffort: "high"
        )
        #expect(req.resolvedThinkingBudget == 2048,
                "Explicit thinking_budget should take precedence over reasoning_effort")
    }

    // MARK: - Logprobs

    @Test("OpenAI request with logprobs decoding")
    func openAIRequestLogprobs() throws {
        let json = """
        {"model":"test","messages":[],"logprobs":true,"top_logprobs":5}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIRequest.self, from: data)
        #expect(req.logprobs == true)
        #expect(req.topLogprobs == 5)
    }

    @Test("OpenAI request without logprobs defaults to nil")
    func openAIRequestNoLogprobs() throws {
        let json = """
        {"model":"test","messages":[]}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIRequest.self, from: data)
        #expect(req.logprobs == nil)
        #expect(req.topLogprobs == nil)
    }

    @Test("OpenAI logprob response types encoding")
    func openAILogprobResponseEncoding() throws {
        let topLogprobs = [
            OpenAITopLogprob(token: " Hello", logprob: -0.999),
            OpenAITopLogprob(token: " Hi", logprob: -2.5),
        ]
        let entry = OpenAILogprobEntry(
            token: " Hello",
            logprob: -0.999,
            topLogprobs: topLogprobs
        )
        let logprobs = OpenAILogprobs(content: [entry])

        let response = OpenAIResponse(
            id: "chatcmpl-test",
            model: "test",
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(role: "assistant", content: "Hello!"),
                    finishReason: "stop",
                    logprobs: logprobs
                )
            ]
        )

        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAIResponse.self, from: data)

        #expect(decoded.choices[0].logprobs != nil)
        let lp = decoded.choices[0].logprobs!
        #expect(lp.content?.count == 1)
        #expect(lp.content?[0].token == " Hello")
        #expect(lp.content?[0].logprob == -0.999)
        #expect(lp.content?[0].topLogprobs.count == 2)
        #expect(lp.content?[0].topLogprobs[0].token == " Hello")
        #expect(lp.content?[0].topLogprobs[1].token == " Hi")
    }

    @Test("OpenAI stream chunk with logprobs")
    func openAIStreamChunkLogprobs() throws {
        let entry = OpenAILogprobEntry(token: " Hello", logprob: -0.5)
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-test",
            model: "test",
            choices: [
                OpenAIStreamChoice(
                    index: 0,
                    delta: OpenAIDelta(content: "Hello"),
                    logprobs: OpenAILogprobs(content: [entry])
                )
            ]
        )
        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        #expect(decoded.choices[0].logprobs?.content?.count == 1)
        #expect(decoded.choices[0].logprobs?.content?[0].token == " Hello")
    }

    @Test("OpenAI choice with nil logprobs")
    func openAIChoiceNilLogprobs() throws {
        let response = OpenAIResponse(
            id: "chatcmpl-test",
            model: "test",
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(role: "assistant", content: "Hi"),
                    finishReason: "stop"
                )
            ]
        )
        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let choice = (json["choices"] as! [[String: Any]])[0]
        // logprobs should be absent (nil) not present with null
        #expect(choice["logprobs"] == nil)
    }

    @Test("TopLogprob encoding/decoding")
    func topLogprobEncoding() throws {
        let tl = TopLogprob(tokenId: 42, tokenText: " hello", logprob: -1.5)
        let data = try JSONEncoder().encode(tl)
        let decoded = try JSONDecoder().decode(TopLogprob.self, from: data)
        #expect(decoded.tokenId == 42)
        #expect(decoded.tokenText == " hello")
        #expect(decoded.logprob == -1.5)
    }

    @Test("Token with logprobs")
    func tokenWithLogprobs() throws {
        let token = Token(
            id: 0,
            text: "hello",
            logprob: -0.5,
            topLogprobs: [
                TopLogprob(tokenId: 1, tokenText: "hello", logprob: -0.5),
                TopLogprob(tokenId: 2, tokenText: " world", logprob: -2.0)
            ]
        )
        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(Token.self, from: data)
        #expect(decoded.logprob == -0.5)
        #expect(decoded.topLogprobs?.count == 2)
        #expect(decoded.topLogprobs?[0].tokenId == 1)
        #expect(decoded.topLogprobs?[1].logprob == -2.0)
    }

    @Test("Token without logprobs backward compat")
    func tokenWithoutLogprobs() throws {
        let token = Token(id: 0, text: "hello")
        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(Token.self, from: data)
        #expect(decoded.logprob == nil)
        #expect(decoded.topLogprobs == nil)
    }

    @Test("InferenceRequest includes logprobs fields")
    func inferenceRequestLogprobs() {
        let req = InferenceRequest(
            model: "test",
            messages: [],
            includeLogprobs: true,
            topLogprobsCount: 5
        )
        #expect(req.includeLogprobs == true)
        #expect(req.topLogprobsCount == 5)
    }

    @Test("InferenceRequest logprobs defaults to false")
    func inferenceRequestLogprobsDefaults() {
        let req = InferenceRequest(model: "test", messages: [])
        #expect(req.includeLogprobs == false)
        #expect(req.topLogprobsCount == nil)
    }

    @Test("InferenceResult includes tokenLogprobs")
    func inferenceResultLogprobs() {
        let result = InferenceResult(
            id: UUID(),
            model: "test",
            text: "hello world",
            tokensPerSecond: 100,
            promptTokens: 5,
            completionTokens: 2,
            finishReason: .stop,
            tokenLogprobs: [
                Token(id: 0, text: "hello", logprob: -0.5),
                Token(id: 1, text: " world", logprob: -1.0)
            ]
        )
        #expect(result.tokenLogprobs?.count == 2)
        #expect(result.tokenLogprobs?[0].logprob == -0.5)
        #expect(result.tokenLogprobs?[1].logprob == -1.0)
    }

    // MARK: - Logprobs Semantic Invariants

    @Test("buildLogprobs preserves token order and counts")
    func buildLogprobsAggregation() {
        // Aggregator must produce one entry per token-with-logprob, in order,
        // and skip tokens that have no logprob (e.g., from non-logprob requests).
        let tokens: [Token] = [
            Token(id: 0, text: "A", logprob: -0.1),
            Token(id: 1, text: "B", logprob: nil),                 // skipped
            Token(id: 2, text: "C", logprob: -0.3),
        ]
        let lp = NovaMLXAPIServer.buildLogprobs(from: tokens)
        #expect(lp?.content?.count == 2, "Tokens without logprob must be skipped")
        #expect(lp?.content?[0].token == "A")
        #expect(lp?.content?[1].token == "C")
        #expect(lp?.content?[0].logprob == -0.1)
        #expect(lp?.content?[1].logprob == -0.3)
    }

    @Test("buildLogprobs returns nil when no token has logprobs")
    func buildLogprobsEmpty() {
        let tokens = [Token(id: 0, text: "A"), Token(id: 1, text: "B")]
        #expect(NovaMLXAPIServer.buildLogprobs(from: tokens) == nil)
    }

    @Test("Logprob entries populate UTF-8 bytes per OpenAI spec")
    func logprobBytesPopulated() {
        // ASCII single byte
        let asciiToken = Token(id: 0, text: "A", logprob: -0.5,
                               topLogprobs: [TopLogprob(tokenId: 0, tokenText: "A", logprob: -0.5)])
        let asciiEntry = NovaMLXAPIServer.tokenToLogprobEntry(asciiToken)
        #expect(asciiEntry?.bytes == [65])
        #expect(asciiEntry?.topLogprobs.first?.bytes == [65])

        // Multi-byte UTF-8 (CJK char "中" → 0xE4 0xB8 0xAD)
        let cjkToken = Token(id: 1, text: "中", logprob: -1.0,
                             topLogprobs: [TopLogprob(tokenId: 1, tokenText: "中", logprob: -1.0)])
        let cjkEntry = NovaMLXAPIServer.tokenToLogprobEntry(cjkToken)
        #expect(cjkEntry?.bytes == [0xE4, 0xB8, 0xAD])
        #expect(cjkEntry?.topLogprobs.first?.bytes == [0xE4, 0xB8, 0xAD])
    }

    @Test("Logprob ordering invariant: top_logprobs descending")
    func logprobTopKOrdering() {
        // The scheduler emits top_logprobs sorted by logprob descending so that
        // index 0 is the most likely alternative. Verify the contract.
        let token = Token(
            id: 0,
            text: "X",
            logprob: -0.5,
            topLogprobs: [
                TopLogprob(tokenId: 1, tokenText: "X", logprob: -0.5),
                TopLogprob(tokenId: 2, tokenText: "Y", logprob: -1.2),
                TopLogprob(tokenId: 3, tokenText: "Z", logprob: -2.4),
            ]
        )
        let entry = NovaMLXAPIServer.tokenToLogprobEntry(token)
        let lps = entry?.topLogprobs.map { $0.logprob } ?? []
        for i in 1..<lps.count {
            #expect(lps[i - 1] >= lps[i],
                    "top_logprobs must be in descending order at position \(i)")
        }
        // Logprobs must be ≤ 0 (they are log-probabilities)
        #expect(entry?.logprob ?? 0 <= 0)
        for tp in entry?.topLogprobs ?? [] {
            #expect(tp.logprob <= 0, "log-probabilities must be non-positive")
        }
    }
}
