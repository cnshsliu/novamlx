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
        #expect(config.adminPort == 8081)
        #expect(config.port == 8080)

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
}
