import Foundation
import Hummingbird
import NIOCore

/// OpenAI-compatible loopback gateway. `POST /v1/chat/completions` treats the
/// JSON `model` field as a pool name. `GET /v1/models` lists enabled pools.
public final class LocalGateway: @unchecked Sendable {
    private let lock = NSLock()
    private var serverTask: Task<Void, Never>?
    private var boundPort = 0
    private var listening = false

    public init() {}

    public var port: Int {
        lock.lock(); defer { lock.unlock() }
        return boundPort
    }

    public var isListening: Bool {
        lock.lock(); defer { lock.unlock() }
        return listening
    }

    public func start(host: String, port: Int, forwarder: PoolForwarder) async throws {
        if isListening { return }
        let router = Self.makeRouter(forwarder: forwarder)
        let bound = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Int, Error>) in
            let once = Once()
            let app = Application(
                router: router,
                configuration: .init(address: .hostname(host, port: port)),
                onServerRunning: { channel in
                    once.run { cont.resume(returning: channel.localAddress?.port ?? port) }
                }
            )
            let task = Task<Void, Never> {
                do {
                    try await app.run()
                    once.run { cont.resume(throwing: LocalGatewayError.stoppedBeforeBind) }
                } catch {
                    once.run { cont.resume(throwing: error) }
                }
            }
            self.setTask(task)
        }
        setListening(port: bound)
    }

    public func stop() async {
        let task = clearTask()
        guard let task else { return }
        task.cancel()
        _ = await task.value
        setStopped()
    }

    // MARK: - Routes

    private static func makeRouter(forwarder: PoolForwarder) -> Router<BasicRequestContext> {
        let router = Router()
        router.get("/v1/models") { _, _ -> Response in
            let models = forwarder.currentConfig.pools.filter(\.isEnabled).map { pool in
                [
                    "id": pool.name,
                    "object": "model",
                    "owned_by": "tknet-peer",
                ]
            }
            return json(["object": "list", "data": models], status: .ok)
        }
        router.post("/v1/chat/completions") { request, _ -> Response in
            let buffer = try await request.body.collect(upTo: 32 * 1024 * 1024)
            let raw = Data(buffer.readableBytesView)
            guard let object = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any],
                  let model = object["model"] as? String, !model.isEmpty else {
                return errorResponse(
                    status: .badRequest,
                    message: "missing model",
                    type: "invalid_request_error")
            }
            let timeout = forwarder.currentConfig.requestTimeoutSeconds
            let outcome = await forwarder.forward(poolName: model, body: raw, timeoutSeconds: timeout)
            return response(for: outcome, model: model)
        }
        return router
    }

    private static func response(for outcome: PoolForwardOutcome, model: String) -> Response {
        switch outcome {
        case .success(let upstream):
            var headers = HTTPFields()
            headers[.contentType] = upstream.contentType ?? "application/json"
            let status = HTTPResponse.Status(code: upstream.status)
            return Response(status: status, headers: headers, body: .init(contentLength: nil) { writer in
                for await chunk in upstream.body {
                    var buffer = ByteBuffer()
                    buffer.writeBytes(chunk)
                    try await writer.write(buffer)
                }
                try await writer.finish(nil)
            })
        case .unknownPool:
            return errorResponse(
                status: .notFound,
                message: "Unknown pool: \(model)",
                type: "invalid_request_error")
        case .poolDisabled:
            return errorResponse(
                status: .serviceUnavailable,
                message: "Pool '\(model)' is disabled",
                type: "invalid_request_error")
        case .noHealthyMembers:
            return errorResponse(
                status: .serviceUnavailable,
                message: "No healthy members in pool '\(model)'",
                type: "server_error")
        case .allMembersFailed(let lastError, _, _):
            return errorResponse(
                status: .badGateway,
                message: "All members of pool '\(model)' failed; last error: \(lastError)",
                type: "server_error")
        case .cancelled:
            return errorResponse(
                status: .serviceUnavailable,
                message: "Pool '\(model)' request was cancelled",
                type: "server_error")
        }
    }

    private static func errorResponse(status: HTTPResponse.Status, message: String, type: String) -> Response {
        json(["error": ["message": message, "type": type]], status: status)
    }

    private static func json(_ object: Any, status: HTTPResponse.Status) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: buffer))
    }

    // MARK: - State

    private func setTask(_ task: Task<Void, Never>) {
        lock.lock(); defer { lock.unlock() }
        serverTask = task
    }

    private func clearTask() -> Task<Void, Never>? {
        lock.lock(); defer { lock.unlock() }
        let task = serverTask
        serverTask = nil
        return task
    }

    private func setListening(port: Int) {
        lock.lock(); defer { lock.unlock() }
        boundPort = port
        listening = true
    }

    private func setStopped() {
        lock.lock(); defer { lock.unlock() }
        listening = false
    }

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

public enum LocalGatewayError: Error {
    case stoppedBeforeBind
}
