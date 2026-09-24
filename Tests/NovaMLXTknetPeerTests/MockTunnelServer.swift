import Foundation
import Hummingbird
import HummingbirdWebSocket
import os
@testable import NovaMLXTknetPeer

/// Real local tknet.ai tunnel stand-in over a real WebSocket upgrade.
/// Accepts any Authorization header (records it for assertions), decodes
/// every inbound text frame via FrameCodec, and keeps the live outbound
/// writer so tests can `push(_:)` server-originated frames (Task 12).
/// Shares MockTknetServer's bind machinery (port 0 + onServerRunning
/// continuation + Once guard) so `port` is the real bound port.
final class MockTunnelServer: @unchecked Sendable {
    private struct State {
        var port = 0
        var serverTask: Task<Void, Never>?
        var outbound: WebSocketOutboundWriter?
        var lastAuthorization: String?
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    let receivedFrames = FrameRecorder()

    /// Actual bound port; valid after `start()` returns.
    var port: Int { state.withLock { $0.port } }

    /// Authorization header of the most recent WS upgrade request, if any.
    var lastAuthorization: String? { state.withLock { $0.lastAuthorization } }

    /// The upgrade path `WSTransport.factory` dials (`<server>/api/peer/tunnel`).
    static let tunnelPath = "/api/peer/tunnel"

    /// Thread-safe frame recorder. New subscribers first see every frame
    /// recorded so far (replay), then live frames.
    final class FrameRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var frames: [Frame] = []
        private var continuations: [UUID: AsyncStream<Frame>.Continuation] = [:]

        func record(_ frame: Frame) {
            lock.lock(); defer { lock.unlock() }
            frames.append(frame)
            continuations.values.forEach { $0.yield(frame) }
        }

        var all: [Frame] {
            lock.lock(); defer { lock.unlock() }; return frames
        }

        var stream: AsyncStream<Frame> {
            AsyncStream { continuation in
                lock.lock(); defer { lock.unlock() }
                for frame in frames { continuation.yield(frame) }
                let id = UUID()
                continuations[id] = continuation
                continuation.onTermination = { [weak self] _ in
                    self?.removeContinuation(id)
                }
            }
        }

        private func removeContinuation(_ id: UUID) {
            lock.lock(); defer { lock.unlock() }
            continuations[id] = nil
        }
    }

    /// Starts on an OS-assigned ephemeral port (bind 0) and returns once the
    /// listener is up.
    func start() async throws {
        let port = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Int, Error>) in
            let once = Once()
            let app = Application(
                router: Router(),
                server: .http1WebSocketUpgrade { [self] request, _, _ in
                    guard request.path == Self.tunnelPath else { return .dontUpgrade }
                    self.recordAuthorization(request.headerFields[.authorization])
                    return .upgrade([:]) { [self] inbound, outbound, _ in
                        self.setOutbound(outbound)
                        // Returning closes the WebSocket cleanly (the library
                        // sends the close frame); mirrors Hummingbird's examples.
                        do {
                            for try await message in inbound.messages(maxSize: .max) {
                                if case .text(let text) = message,
                                   let frame = try? FrameCodec.decode(text) {
                                    self.receivedFrames.record(frame)
                                }
                            }
                        } catch {}
                        self.clearOutbound()
                    }
                },
                configuration: .init(address: .hostname("127.0.0.1", port: 0)),
                onServerRunning: { channel in
                    once.run { cont.resume(returning: channel.localAddress?.port ?? 0) }
                }
            )
            let task = Task<Void, Never> {
                do {
                    try await app.run()
                    once.run { cont.resume(throwing: MockTunnelServerError.serverStoppedEarly) }
                } catch {
                    once.run { cont.resume(throwing: error) }
                }
            }
            state.withLock { $0.serverTask = task }
        }
        state.withLock { $0.port = port }
    }

    /// Cancels the server task and waits for the listener to go down.
    func stop() async {
        let task = state.withLock { s -> Task<Void, Never>? in
            let task = s.serverTask
            s.serverTask = nil
            return task
        }
        guard let task else { return }
        task.cancel()
        _ = await task.value
    }

    /// Writes a server-originated frame to the live tunnel connection.
    /// No-op if no connection is currently open (Task 12 drives relays with it).
    func push(_ frame: Frame) async {
        let outbound = state.withLock { $0.outbound }
        try? await outbound?.writeTextMessage(FrameCodec.encode(frame))
    }

    // MARK: - Internals

    private enum MockTunnelServerError: Error { case serverStoppedEarly }

    /// Runs the closure exactly once so a CheckedContinuation is never
    /// resumed twice (server-running and run()-failure race).
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

    private func recordAuthorization(_ value: String?) {
        state.withLock { $0.lastAuthorization = value }
    }

    private func setOutbound(_ outbound: WebSocketOutboundWriter) {
        state.withLock { $0.outbound = outbound }
    }

    private func clearOutbound() {
        state.withLock { $0.outbound = nil }
    }
}
