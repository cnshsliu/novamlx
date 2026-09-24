import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Frame codec")
struct FrameCodecTests {
    @Test("hello round-trips with capabilities and prices")
    func helloRoundTrip() throws {
        let cap = Capability(
            demandId: "d1", model: "qwen-3-8b", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0.5, priceOut: 1.0
        )
        let frame = Frame.hello(nodeId: "node-1", capabilities: [cap])
        let encoded = FrameCodec.encode(frame)
        let decoded = try FrameCodec.decode(encoded)
        #expect(decoded == frame)
    }

    @Test("request round-trips raw JSON body bytes")
    func requestRoundTrip() throws {
        let body = #"{"model":"x","messages":[]}"#.data(using: .utf8)!
        let frame = Frame.request(RequestFrame(
            reqId: "r1", model: "x", apiFormat: .openai, body: body
        ))
        let decoded = try FrameCodec.decode(FrameCodec.encode(frame))
        #expect(decoded == frame)
    }

    @Test("responseEnd carries full telemetry")
    func responseEndRoundTrip() throws {
        let result = RequestResult(
            status: .completed, ttftMs: 120, totalMs: 4_000,
            promptTokens: 30, completionTokens: 200, upstreamStatus: 200, errorMessage: nil
        )
        let decoded = try FrameCodec.decode(FrameCodec.encode(.responseEnd(reqId: "r1", result: result)))
        #expect(decoded == .responseEnd(reqId: "r1", result: result))
    }

    @Test("demandUpdate marks the lifecycle path")
    func demandUpdateRoundTrip() throws {
        let entries = [DemandEntry(demandId: "d1", model: "qwen-3-8b", modality: "language", note: nil)]
        let decoded = try FrameCodec.decode(FrameCodec.encode(.demandUpdate(entries)))
        #expect(decoded == .demandUpdate(entries))
    }

    @Test("unknown frame type throws")
    func unknownTypeThrows() {
        #expect(throws: FrameError.self) {
            _ = try FrameCodec.decode(#"{"type":"wat","data":{}}"#)
        }
    }

    @Test("malformed JSON throws")
    func malformedThrows() {
        #expect(throws: FrameError.self) {
            _ = try FrameCodec.decode("not json")
        }
    }
}
