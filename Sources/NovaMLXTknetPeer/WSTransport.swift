import Foundation
import HummingbirdWSClient
import Logging
import NIOWebSocket
public enum WSTransportError: Error, Equatable {
    case invalidServerURL
    /// send() after close(); TunnelClient reconnects on thrown errors, so a
    /// closed transport must never silently swallow frames.
    case closed
    /// The connection came down before the dial completed.
    case connectionClosedDuringDial
}

/// Production transport: outbound WebSocket carrying FrameCodec text frames
/// (one frame per WS text message). The peer token rides the Authorization
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
        var closeCode: UInt16?
        var closeReason: String?
    }

    private let continuation: AsyncStream<Frame>.Continuation
    private let state = PeerLock(initialState: State())

    private init() {
        var continuation: AsyncStream<Frame>.Continuation!
        self.inbound = AsyncStream { continuation = $0 }
        self.continuation = continuation
    }

    /// `TransportFactory` for the CLI `serve` command and the Mac page.
    /// Dials `<server>/api/peer/tunnel` with `Bearer <token>` on the upgrade
    /// request. `ws`/`wss` schemes pass through; `http`/`https` are converted
    /// to `ws`/`wss`. TLS for `wss` uses the client library's default client
    /// TLS configuration.
    /// `server` is read on every dial so a registration that switches servers
    /// takes effect without rebuilding the service (the gateway stays up).
    public static func factory(
        server: @escaping @Sendable () -> URL,
        tokenRef: String,
        secrets: any SecretStore
    ) -> TransportFactory {
        {
            // Fresh token each dial: a re-registered token works without restart.
            let token = (try? secrets.load(tokenRef)) ?? ""
            return try await dial(
                url: server().appendingPathComponent("api/peer/tunnel"),
                authorization: "Bearer \(token)")
        }
    }

    public static func factory(server: URL, tokenRef: String, secrets: any SecretStore) -> TransportFactory {
        factory(server: { server }, tokenRef: tokenRef, secrets: secrets)
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
        // Dial must resume on Task cancellation: stopTunnel cancels the client
        // run task while WebSocketClient.run() may still be hung, and without
        // this handler stopTunnel never returns (UI stuck on Connecting + grey Stop).
        let gate = DialGate()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (dialContinuation: CheckedContinuation<WSTransport, Error>) in
                gate.arm(dialContinuation)
                let client = WebSocketClient(
                    url: dialURL.absoluteString,
                    configuration: configuration,
                    logger: Logger(label: "tknet.peer.ws-transport")
                ) { inboundStream, outbound, _ in
                    transport.activate(writer: outbound)
                    gate.succeed(transport)
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
                        // The close frame (code + reason) is the upgrade-required
                        // contract: 4005 means stop reconnecting and tell the user.
                        let closeFrame = try await client.run()
                        transport.recordClose(code: closeFrame.map { Self.rawCode($0.closeCode) }, reason: closeFrame?.reason)
                        gate.fail(WSTransportError.connectionClosedDuringDial)
                    } catch {
                        gate.fail(error)
                    }
                }
                transport.setRunTask(task)
                gate.attach(task)
                if Task.isCancelled {
                    gate.cancel()
                }
            }
        } onCancel: {
            gate.cancel()
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

    /// Close code the server sent (nil when the connection died without one).
    public var closeCode: UInt16? {
        state.withLock { $0.closeCode }
    }

    /// Close reason string the server sent, if any.
    public var closeReason: String? {
        state.withLock { $0.closeReason }
    }

    // MARK: - Called from the client handler task

    private func recordClose(code: UInt16?, reason: String?) {
        state.withLock { s in
            s.closeCode = code
            s.closeReason = reason
        }
    }

    private func activate(writer: WebSocketOutboundWriter) {
        state.withLock { $0.writer = writer }
    }

    private func setRunTask(_ task: Task<Void, Never>) {
        state.withLock { $0.runTask = task }
    }

    /// Resumes the dial continuation exactly once. Cancel stops the WS run
    /// task and fails the dial so stopTunnel does not hang.
    private final class DialGate: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<WSTransport, Error>?
        private var task: Task<Void, Never>?
        private var finished = false

        func arm(_ continuation: CheckedContinuation<WSTransport, Error>) {
            lock.lock(); defer { lock.unlock() }
            self.continuation = continuation
        }

        func attach(_ task: Task<Void, Never>) {
            lock.lock(); defer { lock.unlock() }
            self.task = task
        }

        func succeed(_ transport: WSTransport) {
            finish { $0.resume(returning: transport) }
        }

        func fail(_ error: Error) {
            finish { $0.resume(throwing: error) }
        }

        func cancel() {
            let task: Task<Void, Never>? = {
                lock.lock(); defer { lock.unlock() }
                let task = self.task
                if !finished, let continuation {
                    finished = true
                    self.continuation = nil
                    continuation.resume(throwing: CancellationError())
                }
                return task
            }()
            task?.cancel()
        }

        private func finish(_ body: (CheckedContinuation<WSTransport, Error>) -> Void) {
            lock.lock()
            defer { lock.unlock() }
            guard !finished, let continuation else { return }
            finished = true
            self.continuation = nil
            body(continuation)
        }
    }
}

extension WSTransport {
    /// NIO names the well-known codes and boxes the rest as .unknown(raw) —
    /// 4005 (upgrade-required) is ours, so it always rides .unknown.
    static func rawCode(_ code: WebSocketErrorCode) -> UInt16 {
        switch code {
        case .normalClosure: 1000
        case .goingAway: 1001
        case .protocolError: 1002
        case .unacceptableData: 1003
        case .dataInconsistentWithMessage: 1007
        case .policyViolation: 1008
        case .messageTooLarge: 1009
        case .missingExtension: 1010
        case .unexpectedServerError: 1011
        case .unknown(let raw): raw
        @unknown default: 0
        }
    }
}
