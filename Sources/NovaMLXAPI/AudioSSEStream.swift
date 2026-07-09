import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXUtils

struct AudioSSEStream {
    /// - Parameter onComplete: invoked once the SSE stream finishes (success or
    ///   error) so callers can finalize their request-log entry. Captures the
    ///   request id + model synchronously before the response is returned.
    static func body(
        from stream: AsyncThrowingStream<String, Error>,
        onComplete: (@Sendable (_ error: Error?) -> Void)? = nil
    ) -> ResponseBody {
        ResponseBody { writer in
            var count = 0
            var streamError: Error?
            do {
                for try await tokenText in stream {
                    count += 1
                    let escaped = tokenText
                        .replacingOccurrences(of: "\\", with: "\\\\")
                        .replacingOccurrences(of: "\"", with: "\\\"")
                        .replacingOccurrences(of: "\n", with: "\\n")
                    try await writer.write(ByteBuffer(string: "event: transcript.delta\ndata: {\"text\": \"\(escaped)\"}\n\n"))
                }
                try await writer.write(ByteBuffer(string: "event: done\ndata: [DONE]\n\n"))
                try await writer.finish(nil)
            } catch {
                streamError = error
                let msg = String(describing: error)
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\"", with: "\\\"")
                try? await writer.write(ByteBuffer(string: "event: error\ndata: {\"message\": \"\(msg)\"}\n\n"))
                try? await writer.finish(nil)
            }
            // Finalize the request-log entry (runs after the body closes so the
            // entry flip-flips from active → completed cleanly).
            onComplete?(streamError)
        }
    }
}
