import AVFoundation
import Foundation
import Testing
import NovaMLXCore
import NovaMLXUtils
@testable import NovaMLXAPI
@testable import NovaMLXEngine

@Suite("E2E Rate Limiter Tests")
struct E2ERateLimiterTests {
    @Test("Rate limiter stats tracking")
    func rateLimiterStats() {
        let limiter = RateLimiter(config: RateLimitConfig(requestsPerSecond: 100, burstSize: 100))
        for _ in 0..<10 {
            _ = limiter.allow(key: "test-key")
        }
        let stats = limiter.getStats()
        #expect(stats["total_allowed"] as? UInt64 == 10)
        #expect(stats["total_rejected"] as? UInt64 == 0)
        #expect(stats["active_buckets"] as? Int == 1)
    }

    @Test("Rate limiter rejection tracking")
    func rateLimiterRejection() {
        let limiter = RateLimiter(config: RateLimitConfig(requestsPerSecond: 0.1, burstSize: 2, enabled: true))
        #expect(limiter.allow(key: "k1") == true)
        #expect(limiter.allow(key: "k1") == true)
        #expect(limiter.allow(key: "k1") == false)
        let stats = limiter.getStats()
        #expect(stats["total_allowed"] as? UInt64 == 2)
        #expect(stats["total_rejected"] as? UInt64 == 1)
    }

    @Test("Rate limiter disabled allows all")
    func rateLimiterDisabled() {
        let limiter = RateLimiter(config: RateLimitConfig.disabled)
        for _ in 0..<1000 {
            #expect(limiter.allow(key: "k1") == true)
        }
        let stats = limiter.getStats()
        #expect(stats["total_allowed"] as? UInt64 == 1000)
    }

    @Test("Rate limiter per-key isolation")
    func rateLimiterPerKeyIsolation() {
        let limiter = RateLimiter(config: RateLimitConfig(requestsPerSecond: 0.1, burstSize: 1, enabled: true))
        #expect(limiter.allow(key: "key-a") == true)
        #expect(limiter.allow(key: "key-a") == false)
        #expect(limiter.allow(key: "key-b") == true)
    }

    @Test("Token bucket consume and depletion")
    func tokenBucketDepletion() {
        let bucket = TokenBucket(maxTokens: 3, refillRate: 0)
        #expect(bucket.consume() == true)
        #expect(bucket.consume() == true)
        #expect(bucket.consume() == true)
        #expect(bucket.consume() == false)
    }

    @Test("Token bucket current tokens tracking")
    func tokenBucketCurrentTokens() {
        let bucket = TokenBucket(maxTokens: 5, refillRate: 0)
        let initial = bucket.currentTokens
        #expect(initial == 5.0)
        _ = bucket.consume()
        #expect(bucket.currentTokens == 4.0)
    }

    @Test("Rate limit config defaults")
    func rateLimitConfigDefaults() {
        let config = RateLimitConfig()
        #expect(config.requestsPerSecond == 10)
        #expect(config.burstSize == 20)
        #expect(config.enabled == true)
    }

    @Test("Rate limiter cleanup removes excess buckets")
    func rateLimiterCleanup() {
        let limiter = RateLimiter(config: RateLimitConfig(requestsPerSecond: 100, burstSize: 100))
        for i in 0..<20 {
            _ = limiter.allow(key: "key-\(i)")
        }
        limiter.cleanup(maxBuckets: 10)
        let stats = limiter.getStats()
        #expect(stats["active_buckets"] as? Int ?? 0 <= 10)
    }
}

@Suite("E2E API Serialization Tests")
struct E2EAPISerializationTests {
    @Test("OpenAI request with all optional fields")
    func openAIRequestAllFields() throws {
        let req = OpenAIRequest(
            model: "llama-3",
            messages: [
                OpenAIChatMessage(role: "system", content: "You are helpful"),
                OpenAIChatMessage(role: "user", content: "Hello"),
            ],
            temperature: 0.5,
            topP: 0.9,
            topK: 50,
            minP: 0.05,
            maxTokens: 256,
            stream: true,
            stop: ["STOP"],
            frequencyPenalty: 0.1,
            presencePenalty: 0.2,
            repetitionPenalty: 1.1,
            seed: 42,
            responseFormat: OpenAIResponseFormat(type: "json_object"),
            thinkingBudget: 10000
        )
        let data = try JSONEncoder().encode(req)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["model"] as? String == "llama-3")
        #expect(json?["temperature"] as? Double == 0.5)
        #expect(json?["max_tokens"] as? Int == 256)
        #expect(json?["top_k"] as? Int == 50)
        #expect(json?["min_p"] as? Double == 0.05)
        #expect(json?["seed"] as? Int == 42)
        #expect(json?["stream"] as? Bool == true)
    }

    @Test("OpenAI response with tool calls serialization")
    func openAIResponseToolCalls() throws {
        let toolCall = OpenAIToolCall(
            id: "call_abc",
            function: OpenAIFunctionCall(name: "get_weather", arguments: "{\"location\":\"NYC\"}")
        )
        let message = OpenAIChatMessage(role: "assistant", content: String?.none, toolCalls: [toolCall])
        let response = OpenAIResponse(
            id: "chatcmpl-123",
            model: "test",
            choices: [
                OpenAIChoice(index: 0, message: message, finishReason: "tool_calls")
            ],
            usage: OpenAIUsage(promptTokens: 50, completionTokens: 20)
        )
        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choice = (json?["choices"] as? [[String: Any]])?.first
        let msg = choice?["message"] as? [String: Any]
        let toolCalls = msg?["tool_calls"] as? [[String: Any]]
        #expect(toolCalls?.count == 1)
        #expect(toolCalls?.first?["id"] as? String == "call_abc")
        let function = toolCalls?.first?["function"] as? [String: Any]
        let funcName = function?["name"] as? String
        #expect(funcName == "get_weather")
    }

    @Test("OpenAI completion request and response")
    func completionRoundTrip() throws {
        let reqJSON = """
        {"model":"test","prompt":"Once upon a time","temperature":0.7,"max_tokens":100}
        """
        let reqData = reqJSON.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(OpenAICompletionRequest.self, from: reqData)
        #expect(decoded.model == "test")
        #expect(decoded.prompt == "Once upon a time")

        let response = OpenAICompletionResponse(
            id: "cmpl-123",
            model: "test",
            choices: [OpenAICompletionChoice(index: 0, text: " there was a dragon.", finishReason: "length")],
            usage: OpenAIUsage(promptTokens: 5, completionTokens: 5)
        )
        let responseData = try JSONEncoder().encode(response)
        let decodedResp = try JSONDecoder().decode(OpenAICompletionResponse.self, from: responseData)
        #expect(decodedResp.object == "text_completion")
        #expect(decodedResp.choices.first?.text == " there was a dragon.")
    }

    @Test("ResponseFormat regex and gbnf serialization")
    func responseFormatRegexGbnf() throws {
        let regexFmt = OpenAIResponseFormat(type: "regex", regex: "\\d{3}-\\d{4}")
        let data = try JSONEncoder().encode(regexFmt)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["type"] as? String == "regex")
        #expect(json?["regex"] as? String == "\\d{3}-\\d{4}")

        let gbnfFmt = OpenAIResponseFormat(type: "gbnf", gbnf: "root ::= [a-z]+")
        let gbnfData = try JSONEncoder().encode(gbnfFmt)
        let gbnfJson = try JSONSerialization.jsonObject(with: gbnfData) as? [String: Any]
        #expect(gbnfJson?["type"] as? String == "gbnf")
        #expect(gbnfJson?["gbnf"] as? String == "root ::= [a-z]+")
    }

    @Test("ServerConfig with TLS fields round-trip")
    func serverConfigTLSRoundTrip() throws {
        let config = ServerConfig(
            host: "0.0.0.0",
            port: 443,
            tlsCertPath: "/certs/server.p12",
            tlsKeyPassword: "pass123",
            maxRequestSizeMB: 50
        )
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(ServerConfig.self, from: data)
        #expect(decoded.host == "0.0.0.0")
        #expect(decoded.port == 443)
        #expect(decoded.tlsCertPath == "/certs/server.p12")
        #expect(decoded.tlsKeyPassword == "pass123")
        #expect(decoded.maxRequestSizeMB == 50)
        #expect(decoded.isTLSEnabled)
    }

    @Test("ServerConfig defaults no TLS")
    func serverConfigDefaultsNoTLS() {
        let config = ServerConfig()
        #expect(!config.isTLSEnabled)
        #expect(config.maxRequestSizeMB == 100)
        #expect(config.tlsCertPath == nil)
        #expect(config.tlsKeyPath == nil)
    }

    @Test("InferenceRequest with all constraint fields")
    func inferenceRequestConstraints() {
        let req = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "Say a phone number")],
            temperature: 0.0,
            maxTokens: 50,
            responseFormat: .jsonObject,
            regexPattern: "\\d{3}-\\d{4}",
            gbnfGrammar: nil,
            thinkingBudget: 5000
        )
        #expect(req.regexPattern == "\\d{3}-\\d{4}")
        #expect(req.gbnfGrammar == nil)
        #expect(req.thinkingBudget == 5000)
        #expect(req.responseFormat == .jsonObject)
    }

    @Test("Anthropic full request round-trip")
    func anthropicFullRoundTrip() throws {
        let req = AnthropicRequest(
            model: "claude-3",
            messages: [
                AnthropicMessage(role: "user", content: "Hello"),
                AnthropicMessage(role: "assistant", content: "Hi!"),
                AnthropicMessage(role: "user", content: "What is 2+2?"),
            ],
            maxTokens: 1024,
            system: .text("You are a math tutor."),
            temperature: 0.5,
            topP: 0.9,
            stopSequences: ["END"]
        )
        let data = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(AnthropicRequest.self, from: data)
        #expect(decoded.messages.count == 3)
        #expect(decoded.system == .text("You are a math tutor."))
        #expect(decoded.stopSequences?.count == 1)
    }
}

@Suite("E2E Audio Service Tests")
struct E2EAudioTests {
    @Test("Supported voices structure")
    func supportedVoicesStructure() {
        let voices = AudioService.supportedVoices()
        #expect(!voices.isEmpty)
        let first = voices.first!
        #expect(first["identifier"] != nil)
        #expect(first["name"] != nil)
        #expect(first["language"] != nil)
        #expect(first["quality"] != nil)
    }

    @Test("Supported languages contains major languages")
    func supportedLanguagesMajor() {
        let langs = AudioService.supportedLanguages()
        #expect(!langs.isEmpty)
        #expect(langs.contains("en-US"))
        #expect(langs.contains(where: { $0.hasPrefix("zh") }))
        #expect(langs.contains(where: { $0.hasPrefix("es") }))
        #expect(langs.contains(where: { $0.hasPrefix("ja") }))
    }

    @Test("SynthesisRequest coding round-trip")
    func synthesisRequestCoding() throws {
        let req = AudioService.SynthesisRequest(
            text: "Hello world",
            voice: "com.apple.voice.compact.en-US.Samantha",
            speed: 0.8,
            language: "en-US"
        )
        let data = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(AudioService.SynthesisRequest.self, from: data)
        #expect(decoded.text == "Hello world")
        #expect(decoded.voice == "com.apple.voice.compact.en-US.Samantha")
        #expect(decoded.speed == 0.8)
        #expect(decoded.language == "en-US")
    }

    @Test("TranscriptionResult coding round-trip with segments")
    func transcriptionResultCoding() throws {
        let result = AudioService.TranscriptionResult(
            text: "Hello world",
            language: "en",
            duration: 2.5,
            segments: [
                AudioService.TranscriptionResult.Segment(text: "Hello", start: 0.0, end: 0.5),
                AudioService.TranscriptionResult.Segment(text: " world", start: 0.5, end: 1.0),
            ]
        )
        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(AudioService.TranscriptionResult.self, from: data)
        #expect(decoded.text == "Hello world")
        #expect(decoded.language == "en")
        #expect(decoded.duration == 2.5)
        #expect(decoded.segments?.count == 2)
        #expect(decoded.segments?.first?.text == "Hello")
    }

    @Test("TranscriptionResult without segments")
    func transcriptionResultNoSegments() throws {
        let result = AudioService.TranscriptionResult(
            text: "Hello",
            language: "en",
            duration: nil,
            segments: nil
        )
        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(AudioService.TranscriptionResult.self, from: data)
        #expect(decoded.text == "Hello")
        #expect(decoded.duration == nil)
        #expect(decoded.segments == nil)
    }

    @Test("AudioChunk creation and properties")
    func audioChunkProperties() {
        let chunk = AudioService.AudioChunk(data: Data([0x01, 0x02, 0x03]), isFinal: false)
        #expect(chunk.data.count == 3)
        #expect(!chunk.isFinal)

        let finalChunk = AudioService.AudioChunk(data: Data(), isFinal: true)
        #expect(finalChunk.data.isEmpty)
        #expect(finalChunk.isFinal)
    }
}
