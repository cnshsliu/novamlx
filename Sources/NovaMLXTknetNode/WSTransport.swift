import Foundation
import HummingbirdWSClient
import Logging
import os

public enum WSTransportError: Error, Equatable {
    case invalidServerURL
    /// send() after close(); TunnelClient reconnects on thrown errors, so a
    /// closed transport must never silently swallow frames.
    case closed
    /// The connection came down before the dial completed.
    case connectionClosedDuringDial
}

/// Production transport: outbound WebSocket carrying FrameCodec text frames
/// (one frame per WS text message). The node token rides the Authorization
/// header of the WS upgrade request — it is loaded fresh from the SecretStore
/// on every dial so a re-registered token works without a restart, and it is
/// never logged.
///
/// hummingbird-websocket's client is handler-shaped: `WebSocketClient.run()`
/// drives a `(inbound, outbound, context)` closure for the lifetime of the
/// connection (there is no persistent socket object). The handler forwards
/// decoded frames into `inbound` and hands the outbound writer back to this
/// transport via a once-guarded continuation, so the factory returns only
/// after the WS handshake succeeded.
public final class WSTransport: TunnelTransport, @unchecked Sendable {
    /// Inbound text-message cap. Decode failures are dropped (one bad frame
    /// must not kill the tunnel); a peer exceeding this kills the connection
    /// via the library's messageTooLarge close path.
    private static let maxInboundMessageBytes = 64 * 1024 * 1024

    public let inbound: AsyncStream<Frame>

    private struct State {
        var writer: WebSocketOutboundWriter?
        var runTask: Task<Void, Never>?
        var closed = false
    }

    private let continuation: AsyncStream<Frame>.Continuation
    private let state = OSAllocatedUnfairLock(initialState: State())

    private init() {
        var continuation: AsyncStream<Frame>.Continuation!
        self.inbound = AsyncStream { continuation = $0 }
        self.continuation = continuation
    }

    /// `TransportFactory` for the CLI `serve` command and the Mac page.
    /// Dials `<server>/api/node/tunnel` with `Bearer <token>` on the upgrade
    /// request. `ws`/`wss` schemes pass through; `http`/`https` are converted
    /// to `ws`/`wss`. TLS for `wss` uses the client library's default client
    /// TLS configuration.
    public static func factory(server: URL, tokenRef: String, secrets: any SecretStore) -> TransportFactory {
        {
            // Fresh token each dial: a re-registered token works without restart.
            let token = (try? secrets.load(tokenRef)) ?? ""
            return try await dial(
                url: server.appendingPathComponent("api/node/tunnel"),
                authorization: "Bearer \(token)")
        }
    }

    /// Dials and returns only after the WebSocket handshake completed and the
    /// first handler invocation is live. Throws if the connection fails or
    /// drops before then.
    static func dial(url: URL, authorization: String) async throws -> WSTransport {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let host = components.host, !host.isEmpty else {
            throw WSTransportError.invalidServerURL
        }
        switch components.scheme?.lowercased() {
        case "http": components.scheme = "ws"
        case "https": components.scheme = "wss"
        case "ws", "wss": break
        default: throw WSTransportError.invalidServerURL
        }
        guard let dialURL = components.url else { throw WSTransportError.invalidServerURL }

        var configuration = WebSocketClientConfiguration()
        configuration.additionalHeaders[.authorization] = authorization

        let transport = WSTransport()
        return try await withCheckedThrowingContinuation { (dialContinuation: CheckedContinuation<WSTransport, Error>) in
            let once = Once()
            let client = WebSocketClient(
                url: dialURL.absoluteString,
                configuration: configuration,
                logger: Logger(label: "tknet.node.ws-transport")
            ) { inboundStream, outbound, _ in
                transport.activate(writer: outbound)
                once.run { dialContinuation.resume(returning: transport) }
                // Handler return closes the socket cleanly — the library sends
                // the close frame itself (WSCore/WebSocketHandler.swift).
                do {
                    for try await message in inboundStream.messages(maxSize: Self.maxInboundMessageBytes) {
                        if case .text(let text) = message,
                           let frame = try? FrameCodec.decode(text) {
                            transport.continuation.yield(frame)
                        }
                    }
                } catch {}
                transport.continuation.finish()
            }
            let task = Task<Void, Never> {
                do {
                    _ = try await client.run()
                    once.run { dialContinuation.resume(throwing: WSTransportError.connectionClosedDuringDial) }
                } catch {
                    once.run { dialContinuation.resume(throwing: error) }
                }
            }
            transport.setRunTask(task)
        }
    }

    public func send(_ frame: Frame) async throws {
        let (writer, closed) = state.withLock { s in (s.writer, s.closed) }
        guard !closed, let writer else { throw WSTransportError.closed }
        // writeTextMessage (not write) fragments payloads above maxFrameSize —
        // responseChunk frames carry base64 SSE bytes that can exceed 16 KiB.
        try await writer.writeTextMessage(FrameCodec.encode(frame))
    }

    public func close() async {
        let task = state.withLock { s -> Task<Void, Never>? in
            guard !s.closed else { return nil }
            s.closed = true
            s.writer = nil
            return s.runTask
        }
        guard let task else { return }
        // Finish first so the session loop notices immediately, then cancel
        // the run task: the library's cancellation handler closes the input
        // side and the handler returns through the clean-close path.
        continuation.finish()
        task.cancel()
        _ = await task.value
    }

    // MARK: - Called from the client handler task

    private func activate(writer: WebSocketOutboundWriter) {
        state.withLock { $0.writer = writer }
    }

    private func setRunTask(_ task: Task<Void, Never>) {
        state.withLock { $0.runTask = task }
    }

    /// Runs the closure exactly once so the dial continuation is never
    /// resumed twice (handler-start and run()-completion race).
    private final class Once: @unchecked Sendable {
        private let lock = NSLock()
        private var done = false
        func run(_ body: () -> Void) {
            lock.lock(); defer { lock.unlock() }
            guard !done else { return }
            done = true
            body()
        }
    }
}
