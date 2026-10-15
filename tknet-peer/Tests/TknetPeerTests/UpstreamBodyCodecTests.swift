import Foundation
import Testing
@testable import TknetPeer

@Suite("Upstream body codec")
struct UpstreamBodyCodecTests {
    @Test("json model is replaced and the member path keeps slashes")
    func jsonAndURL() throws {
        let body = Data(#"{"model":"pool","input":"hi"}"#.utf8)
        let (rewritten, type) = try UpstreamBodyCodec.prepare(
            body: body, contentType: "application/json", upstreamModel: "upstream")
        let object = try JSONSerialization.jsonObject(with: rewritten) as? [String: Any]
        #expect(object?["model"] as? String == "upstream")
        #expect(object?["input"] as? String == "hi")
        #expect(type == "application/json")
        let url = UpstreamBodyCodec.upstreamURL(
            endpoint: URL(string: "http://10.0.0.2:8080/v1")!, path: "images/generations")
        #expect(url == "http://10.0.0.2:8080/v1/images/generations")
    }

    @Test("multipart model is replaced and the file bytes stay")
    func multipartKeepsFile() throws {
        let file = Data([0, 1, 2, 255, 10, 13, 34])
        var body = Data()
        body.append(Data("--bound\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\npool\r\n".utf8))
        body.append(Data("--bound\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n".utf8))
        body.append(file)
        body.append(Data("\r\n--bound--\r\n".utf8))
        let type = "multipart/form-data; boundary=bound"
        #expect(UpstreamBodyCodec.poolName(body: body, contentType: type) == "pool")
        let (rewritten, sendType) = try UpstreamBodyCodec.prepare(
            body: body, contentType: type, upstreamModel: "upstream-model")
        #expect(sendType == type)
        #expect(UpstreamBodyCodec.poolName(body: rewritten, contentType: type) == "upstream-model")
        #expect(rewritten.range(of: file) != nil)
    }
}
