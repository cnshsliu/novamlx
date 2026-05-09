import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXEngine
@testable import NovaMLXAPI

// ────────────────────────────────────────────────────────────
// Harmony (GPT-OSS) Streaming Protocol Tests
// ────────────────────────────────────────────────────────────

@Suite("Harmony Channel Detection Tests")
struct HarmonyChannelDetectionTests {

    @Test("Detect analysis channel from accumulated text")
    func detectAnalysisChannel() {
        let text = "<|channel|>analysis<|message|>Let me think about this"
        let channel = MLXEngine.detectHarmonyChannel(in: text)
        #expect(channel == "analysis")
    }

    @Test("Detect final channel from accumulated text")
    func detectFinalChannel() {
        let text = "<|channel|>final<|message|>The answer is 42"
        let channel = MLXEngine.detectHarmonyChannel(in: text)
        #expect(channel == "final")
    }

    @Test("Detect commentary channel from accumulated text")
    func detectCommentaryChannel() {
        let text = "<|channel|>commentary<|message|>This is interesting"
        let channel = MLXEngine.detectHarmonyChannel(in: text)
        #expect(channel == "commentary")
    }

    @Test("Returns latest channel when multiple present")
    func detectLatestChannel() {
        let text = "<|channel|>analysis<|message|>Thinking...<|end|><|start|>assistant<|channel|>final<|message|>Answer"
        let channel = MLXEngine.detectHarmonyChannel(in: text)
        #expect(channel == "final")
    }

    @Test("Returns nil when no channel marker present")
    func detectNoChannel() {
        let text = "Just regular text without any markers"
        let channel = MLXEngine.detectHarmonyChannel(in: text)
        #expect(channel == nil)
    }

    @Test("Returns nil for empty string")
    func detectEmptyString() {
        let channel = MLXEngine.detectHarmonyChannel(in: "")
        #expect(channel == nil)
    }
}

@Suite("Harmony SSE Streaming Format Tests")
struct HarmonyStreamingFormatTests {

    @Test("Stream chunk includes nova.channels for Harmony model")
    func streamChunkIncludesChannels() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-harmony",
            model: "gpt-oss-120b",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "Hello"))],
            novaChannels: [NovaHarmonyChannel(channel: "final", text: "Hello")]
        )
        let data = try JSONEncoder().encode(chunk)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        let channels = json?["nova.channels"] as? [[String: String]]
        #expect(channels != nil)
        #expect(channels?.count == 1)
        #expect(channels?[0]["channel"] == "final")
        #expect(channels?[0]["text"] == "Hello")
    }

    @Test("Stream chunk omits nova.channels when nil")
    func streamChunkOmitsChannelsWhenNil() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-regular",
            model: "llama-3.1-8b",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "Hello"))]
        )
        let data = try JSONEncoder().encode(chunk)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        #expect(json?["nova.channels"] == nil)
    }

    @Test("Stream chunk supports multiple channel entries")
    func streamChunkMultipleChannels() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-multi",
            model: "gpt-oss-120b",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "Hello"))],
            novaChannels: [
                NovaHarmonyChannel(channel: "analysis", text: "Let me"),
                NovaHarmonyChannel(channel: "final", text: "Hello")
            ]
        )
        let data = try JSONEncoder().encode(chunk)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let channels = json?["nova.channels"] as? [[String: String]]
        #expect(channels?.count == 2)
        #expect(channels?[0]["channel"] == "analysis")
        #expect(channels?[1]["channel"] == "final")
    }

    @Test("OpenAIStreamChunk decodes with nova.channels")
    func decodeChunkWithChannels() throws {
        let json = """
        {
            "id": "chatcmpl-decode",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-oss-120b",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            "nova.channels": [{"channel": "final", "text": "Hello"}]
        }
        """
        let data = json.data(using: .utf8)!
        let chunk = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        #expect(chunk.novaChannels?.count == 1)
        #expect(chunk.novaChannels?[0].channel == "final")
        #expect(chunk.novaChannels?[0].text == "Hello")
    }
}

@Suite("Harmony Token Channel Tests")
struct HarmonyTokenChannelTests {

    @Test("Token with channels encodes and decodes")
    func tokenWithChannels() throws {
        let token = Token(
            id: 0,
            text: "Hello",
            channels: [HarmonyChannel(channel: "final", text: "Hello")]
        )
        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(Token.self, from: data)
        #expect(decoded.channels?.count == 1)
        #expect(decoded.channels?[0].channel == "final")
        #expect(decoded.channels?[0].text == "Hello")
    }

    @Test("Token without channels has nil channels field")
    func tokenWithoutChannels() throws {
        let token = Token(id: 0, text: "Hello")
        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(Token.self, from: data)
        #expect(decoded.channels == nil)
    }
}

@Suite("Harmony No-Channels For Non-Harmony Models Tests")
struct HarmonyNoChannelsForOtherModelsTests {

    @Test("Llama model token should not have channels")
    func llamaTokenNoChannels() throws {
        let token = Token(id: 0, text: "Hello")
        #expect(token.channels == nil)
    }

    @Test("Qwen model token should not have channels")
    func qwenTokenNoChannels() throws {
        let token = Token(id: 0, text: "Hello")
        #expect(token.channels == nil)
    }

    @Test("Stream chunk for Llama omits nova.channels")
    func llamaStreamChunkNoChannels() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-llama",
            model: "llama-3.1-8b",
            choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(content: "Hello"))]
        )
        #expect(chunk.novaChannels == nil)
    }
}
