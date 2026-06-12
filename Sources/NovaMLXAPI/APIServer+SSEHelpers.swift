import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXEngine
import NovaMLXUtils

// MARK: - SSE Keep-Alive & Error Helpers
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    enum SSEKeepAliveEvent: Sendable {
        case token(Token)
        case keepAlive
        case done
    }

    static func withSSEKeepAlive(
        _ stream: AsyncThrowingStream<Token, Error>,
        interval: Duration = .seconds(10),
        reqTag: String = "unknown"
    ) -> AsyncThrowingStream<SSEKeepAliveEvent, Error> {
        AsyncThrowingStream { continuation in
            // Shared guard prevents double-yield/finish when onTermination
            // races with the inference consumer or heartbeat tasks.
            let guard_ = FinishGuard()

            let task = Task {
                do {
                    guard !guard_.isDone else { return }
                    continuation.yield(.keepAlive)
                    for try await token in stream {
                        if Task.isCancelled {
                            NovaMLXLog.debug("[SSE:\(reqTag)] Inference stream consumer cancelled")
                            break
                        }
                        guard !guard_.isDone else { return }
                        continuation.yield(.token(token))
                    }
                    NovaMLXLog.debug("[SSE:\(reqTag)] Inference stream finished normally")
                    if guard_.tryMarkFinished() {
                        continuation.finish()
                    }
                } catch {
                    NovaMLXLog.error("[SSE:\(reqTag)] Inference stream error: \(error)")
                    if guard_.tryMarkFinished() {
                        continuation.finish(throwing: error)
                    }
                }
            }
            let heartbeat = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: interval)
                    guard !Task.isCancelled else { break }
                    guard !guard_.isDone else { return }
                    continuation.yield(.keepAlive)
                }
            }
            continuation.onTermination = { reason in
                // Command-009: finished(nil) is the normal AsyncThrowingStream completion
                // (no error thrown). Only WARN on real failures; normal close is DEBUG.
                if case .finished(let error?) = reason {
                    NovaMLXLog.warning("[SSE:\(reqTag)] SSE connection terminated with error: \(error)")
                } else {
                    NovaMLXLog.debug("[SSE:\(reqTag)] SSE connection terminated: \(reason)")
                }
                task.cancel()
                heartbeat.cancel()
            }
        }
    }

    static func streamErrorFields(_ error: Error) -> (message: String, type: String, code: String) {
        if let error = error as? NovaMLXError {
            return (error.errorDescription ?? "Unknown error", error.apiErrorType, error.apiErrorCode)
        }
        return (error.localizedDescription, "internal_error", "internal_error")
    }

    /// Anthropic-format error fields for SSE error events.
    /// Returns (message, type) matching the Anthropic API error schema.
    static func anthropicStreamErrorFields(_ error: Error) -> (message: String, type: String) {
        if let error = error as? NovaMLXError {
            return (error.errorDescription ?? "Unknown error", error.apiErrorType)
        }
        return (error.localizedDescription, "api_error")
    }
}
