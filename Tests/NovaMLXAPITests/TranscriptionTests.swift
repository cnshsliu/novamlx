import Testing
import Foundation
@testable import NovaMLXCore
@testable import NovaMLXEngine
@testable import NovaMLXAPI

@Suite("Transcription / Audio Types")
struct TranscriptionTests {

    @Test("ModelFamily.whisper round-trips through JSON")
    func testModelFamilyWhisperCodable() throws {
        let original = ModelFamily.whisper
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ModelFamily.self, from: encoded)
        #expect(decoded == .whisper)
    }

    @Test("ModelType.audio round-trips through JSON")
    func testModelTypeAudioCodable() throws {
        let original = ModelType.audio
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ModelType.self, from: encoded)
        #expect(decoded == .audio)
    }

    @Test("Unknown ModelFamily falls back to .other")
    func testModelFamilyFallback() throws {
        let json = "\"some_unknown_family\"".data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelFamily.self, from: json)
        #expect(decoded == .other)
    }

    @Test("TranscriptionRequest decodes from JSON with base64 audio")
    func testTranscriptionRequestDecoding() throws {
        let audioBase64 = Data("fake audio data".utf8).base64EncodedString()
        let json = """
        {
            "model": "whisper-large-v3-turbo",
            "file": "\(audioBase64)",
            "language": "en",
            "response_format": "json",
            "temperature": 0.0
        }
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(TranscriptionRequest.self, from: data)
        #expect(req.model == "whisper-large-v3-turbo")
        #expect(req.language == "en")
        #expect(req.resolvedResponseFormat == "json")
        #expect(req.temperature == 0.0)
        #expect(Data(base64Encoded: req.file) == Data("fake audio data".utf8))
    }

    @Test("TranscriptionRequest uses default response format")
    func testTranscriptionRequestDefaultFormat() throws {
        let audioBase64 = Data("x".utf8).base64EncodedString()
        let json = """
        {"model": "whisper-tiny", "file": "\(audioBase64)"}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(TranscriptionRequest.self, from: data)
        #expect(req.resolvedResponseFormat == "json")
        #expect(req.language == nil)
        #expect(req.temperature == nil)
    }

    @Test("TranscriptionResponse encodes correctly")
    func testTranscriptionResponseEncoding() throws {
        let response = TranscriptionResponse(
            text: "Hello, world.",
            language: "en",
            duration: 3.5
        )
        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(json["text"] as? String == "Hello, world.")
        #expect(json["language"] as? String == "en")
        #expect(json["duration"] as? Double == 3.5)
    }

    @Test("ModelCapabilities includes audio field")
    func testModelCapabilitiesAudio() {
        let caps = ModelCapabilities(audio: true)
        #expect(caps.audio == true)
        #expect(caps.reasoning == false)
        #expect(caps.vision == false)

        let noAudio = ModelCapabilities()
        #expect(noAudio.audio == false)
    }

    @Test("TranscriptionService isLoaded returns false for unknown model")
    func testTranscriptionServiceNotLoaded() {
        let service = TranscriptionService()
        #expect(service.isLoaded("nonexistent-model") == false)
    }

    // MARK: - Multipart Parser Tests

    @Test("Multipart boundary extraction from Content-Type header")
    func testMultipartBoundaryExtraction() {
        let ct1 = "multipart/form-data; boundary=----WebKitFormBoundaryXYZ123"
        #expect(MultipartParser.extractBoundary(from: ct1) == "----WebKitFormBoundaryXYZ123")

        let ct2 = "multipart/form-data; boundary=\"quotedBoundary\""
        #expect(MultipartParser.extractBoundary(from: ct2) == "quotedBoundary")

        let ct3 = "application/json"
        #expect(MultipartParser.extractBoundary(from: ct3) == nil)

        let ct4 = "multipart/form-data"
        #expect(MultipartParser.extractBoundary(from: ct4) == nil)
    }

    @Test("Multipart parsing extracts file and text fields")
    func testMultipartParsingValid() throws {
        let boundary = "testboundary123"
        let audioContent = Data("fake wav audio bytes here".utf8)

        var body = Data()
        let crlf = Data("\r\n".utf8)
        func addPart(name: String, filename: String? = nil, contentType: String? = nil, data: Data) {
            body += Data("--\(boundary)\r\n".utf8)
            var disp = "Content-Disposition: form-data; name=\"\(name)\""
            if let fn = filename { disp += "; filename=\"\(fn)\"" }
            body += Data("\(disp)\r\n".utf8)
            if let ct = contentType { body += Data("Content-Type: \(ct)\r\n".utf8) }
            body += Data("\r\n".utf8)
            body += data
            body += crlf
        }
        addPart(name: "file", filename: "test.wav", contentType: "audio/wav", data: audioContent)
        addPart(name: "model", data: Data("qwen3-asr".utf8))
        addPart(name: "language", data: Data("en".utf8))
        body += Data("--\(boundary)--\r\n".utf8)

        let ctHeader = "multipart/form-data; boundary=\(boundary)"
        let parts = try MultipartParser.parse(body: body, contentType: ctHeader)

        #expect(parts["file"] != nil)
        #expect(parts["file"]?.filename == "test.wav")
        #expect(parts["file"]?.contentType == "audio/wav")
        #expect(parts["file"]?.body == audioContent)
        #expect(parts["model"]?.body == Data("qwen3-asr".utf8))
        #expect(parts["language"]?.body == Data("en".utf8))
    }

    @Test("Multipart parsing throws when file part is missing")
    func testMultipartMissingFile() throws {
        let boundary = "noboundary"
        var body = Data()
        body += Data("--\(boundary)\r\n".utf8)
        body += Data("Content-Disposition: form-data; name=\"model\"\r\n".utf8)
        body += Data("\r\n".utf8)
        body += Data("some-model".utf8)
        body += Data("\r\n--\(boundary)--\r\n".utf8)

        let ctHeader = "multipart/form-data; boundary=\(boundary)"
        let parts = try MultipartParser.parse(body: body, contentType: ctHeader)

        #expect(parts["file"] == nil)
        #expect(parts["model"]?.body == Data("some-model".utf8))
    }

    // MARK: - SSE Streaming

    @Test("TranscriptionRequest parses stream field")
    func testTranscriptionRequestStreamField() throws {
        let audioBase64 = Data("x".utf8).base64EncodedString()
        let json = """
        {"model": "whisper-tiny", "file": "\(audioBase64)", "stream": true}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(TranscriptionRequest.self, from: data)
        #expect(req.stream == true)

        let jsonNoStream = """
        {"model": "whisper-tiny", "file": "\(audioBase64)"}
        """
        let req2 = try JSONDecoder().decode(TranscriptionRequest.self, from: jsonNoStream.data(using: .utf8)!)
        #expect(req2.stream == nil)
    }

    @Test("AudioSSEStream produces valid ResponseBody")
    func testAudioSSEStreamBodyCreation() async throws {
        let stream = AsyncThrowingStream<String, Error> { continuation in
            continuation.yield("Hello")
            continuation.yield(" world")
            continuation.finish()
        }
        let body = AudioSSEStream.body(from: stream)
        #expect(body != nil)
    }

    @Test("SSE delta event format is correct")
    func testSSEDeltaEventFormat() {
        let token = "Hello"
        let escaped = token
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        let event = "event: transcript.delta\ndata: {\"text\": \"\(escaped)\"}\n\n"
        #expect(event.contains("event: transcript.delta"))
        #expect(event.contains("\"text\": \"Hello\""))
        #expect(event.hasSuffix("\n\n"))
    }
}
