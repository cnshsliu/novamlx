import Foundation
import Hummingbird
import NovaMLXCore

struct AudioSSEStream {
    static func body(
        from stream: AsyncThrowingStream<String, Error>
    ) -> ResponseBody {
        ResponseBody { writer in
            do {
                for try await tokenText in stream {
                    let escaped = tokenText
                        .replacingOccurrences(of: "\\", with: "\\\\")
                        .replacingOccurrences(of: "\"", with: "\\\"")
                        .replacingOccurrences(of: "\n", with: "\\n")
                    try await writer.write(ByteBuffer(string: "event: transcript.delta\ndata: {\"text\": \"\(escaped)\"}\n\n"))
                }
                try await writer.write(ByteBuffer(string: "event: done\ndata: [DONE]\n\n"))
                try await writer.finish(nil)
            } catch {
                let msg = String(describing: error)
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\"", with: "\\\"")
                try? await writer.write(ByteBuffer(string: "event: error\ndata: {\"message\": \"\(msg)\"}\n\n"))
                try? await writer.finish(nil)
            }
        }
    }
}
