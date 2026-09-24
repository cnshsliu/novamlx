import Foundation
import os

/// One tunnel connection carrying frames. WS in production (Task 10),
/// slow-poll in Phase 2, in-memory in tests. FrameCodec stays transport-agnostic.
public protocol TunnelTransport: Sendable {
    var inbound: AsyncStream<Frame> { get }
    func send(_ frame: Frame) async throws
    func close() async
}

/// Factory so clients can dial reconnects; tests inject the in-memory pair.
public typealias TransportFactory = @Sendable () async throws -> TunnelTransport

/// Two transports wired to each other. `nodeSide` is what TunnelClient holds;
/// `serverSide` is what the test drives as a fake tknet.ai.
public final class InMemoryTransportPair: @unchecked Sendable {
    public let nodeSide: TunnelTransport
    public let serverSide: TunnelTransport

    private final class Side: TunnelTransport, @unchecked Sendable {
        let inbound: AsyncStream<Frame>
        private let continuation: AsyncStream<Frame>.Continuation
        private let otherContinuation: AsyncStream<Frame>.Continuation
        private let closed = OSAllocatedUnfairLock(initialState: false)

        init(mine: AsyncStream<Frame>, mineCont: AsyncStream<Frame>.Continuation,
             otherCont: AsyncStream<Frame>.Continuation) {
            self.inbound = mine
            self.continuation = mineCont
            self.otherContinuation = otherCont
        }

        func send(_ frame: Frame) async throws {
            otherContinuation.yield(frame)
        }

        func close() async {
            let alreadyClosed = closed.withLock { state -> Bool in
                if state { return true }
                state = true
                return false
            }
            guard !alreadyClosed else { return }
            continuation.finish()
            otherContinuation.finish()
        }
    }

    public init() {
        var nodeCont: AsyncStream<Frame>.Continuation?
        var serverCont: AsyncStream<Frame>.Continuation?
        let nodeStream = AsyncStream<Frame> { nodeCont = $0 }
        let serverStream = AsyncStream<Frame> { serverCont = $0 }
        guard let nc = nodeCont, let sc = serverCont else { fatalError("stream init") }
        self.nodeSide = Side(mine: nodeStream, mineCont: nc, otherCont: sc)
        self.serverSide = Side(mine: serverStream, mineCont: sc, otherCont: nc)
    }
}

/// Test helper: async single-item read with timeout via task group race.
public extension AsyncStream where Element == Frame {
    func next(timeout: TimeInterval = 5) async -> Frame? {
        await withTaskGroup(of: Frame?.self) { group in
            group.addTask {
                var it = self.makeAsyncIterator()
                return await it.next()
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }
}
