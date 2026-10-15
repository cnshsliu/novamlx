import AsyncHTTPClient
import Foundation
import Logging
import NIOCore
import NIOHTTP1

/// One upstream HTTP response. The body is raw bytes (JSON or SSE) and is
/// not merged with a later attempt.
public enum UpstreamStreamError: Error, Sendable, Equatable {
    case cancelled
    case timedOut
    case failed(String)
}

/// Side channel for a mid-body failure. The byte stream itself does not throw,
/// so a 200 that dies halfway is not retried and is not stitched onto the next member.
public final class UpstreamTerminal: @unchecked Sendable {
    private let lock = NSLock()
    private var error: UpstreamStreamError?

    func set(_ error: UpstreamStreamError) {
        lock.lock(); defer { lock.unlock() }
        self.error = error
    }

    public func current() -> UpstreamStreamError? {
        lock.lock(); defer { lock.unlock() }
        return error
    }
}

public struct UpstreamBody: Sendable {
    public var status: Int
    public var contentType: String?
    public var body: AsyncStream<Data>
    public var terminal: UpstreamTerminal

    public init(status: Int, contentType: String?, body: AsyncStream<Data>, terminal: UpstreamTerminal) {
        self.status = status
        self.contentType = contentType
        self.body = body
        self.terminal = terminal
    }
}

/// Result of routing one request through a pool. Mirrors `LBProxyOutcome`.
public enum PoolForwardOutcome: Sendable {
    case success(UpstreamBody)
    case unknownPool
    case poolDisabled
    case noHealthyMembers
    /// Path is not in the pool's interface — rejected before member pick.
    case pathNotAllowed(path: String)
    case allMembersFailed(lastError: String, lastStatus: Int, timedOut: Bool)
    case cancelled
}

/// Shared HTTP forwarder for the local gateway and the tunnel relay.
/// Selection is `PoolRouter`; retry rules match `LBProxy.handle`.
public final class PoolForwarder: Sendable {
    private let holder: Relay.ConfigHolder
    private let secrets: any SecretStore
    private let client: HTTPClient
    private let stats: MemberStatsStore
    private let envFile: URL
    private let shutDown = PeerLock(initialState: false)
    private let logger = Logger(label: "TknetPeer.PoolForwarder")

    public init(
        config: PeerConfig, secrets: any SecretStore, stats: MemberStatsStore = MemberStatsStore(),
        envFile: URL = EnvKeyFile.defaultURL
    ) {
        var config = config
        config.migrateLegacyPoolsIfNeeded()
        self.holder = Relay.ConfigHolder(config)
        self.secrets = secrets
        self.stats = stats
        self.envFile = envFile
        self.client = HTTPClient(eventLoopGroupProvider: .singleton)
    }

    /// Test seam and gateway reads. Already migrated.
    public var currentConfig: PeerConfig { holder.value }

    public func updateConfig(_ config: PeerConfig) { holder.update(config) }

    /// Idempotent. AsyncHTTPClient traps in debug builds if dropped un-shutdown.
    public func shutdown() async throws {
        let already = shutDown.withLock { flag -> Bool in
            if flag { return true }
            flag = true
            return false
        }
        if already { return }
        try await client.shutdown()
    }

    public func forward(
        poolId: String, body: Data, timeoutSeconds: Double,
        upstreamPath: String = "chat/completions", contentType: String? = nil,
        interface: APIContract? = nil
    ) async -> PoolForwardOutcome {
        let config = holder.value
        guard let pool = config.pool(id: poolId) else { return .unknownPool }
        return await dispatch(
            pool: pool, config: config, body: body, timeoutSeconds: timeoutSeconds,
            upstreamPath: upstreamPath, contentType: contentType, interface: interface)
    }

    public func forward(
        poolName: String, body: Data, timeoutSeconds: Double,
        upstreamPath: String = "chat/completions", contentType: String? = nil,
        interface: APIContract? = nil
    ) async -> PoolForwardOutcome {
        let config = holder.value
        guard let pool = config.pool(named: poolName) else { return .unknownPool }
        return await dispatch(
            pool: pool, config: config, body: body, timeoutSeconds: timeoutSeconds,
            upstreamPath: upstreamPath, contentType: contentType, interface: interface)
    }

    /// Builds the upstream POST. `upstreamPath` is the path under the member's
    /// `/v1` base, such as `chat/completions` or `embeddings`. `X-No-Autoload: 1`
    /// is set only for loopback and `localNovaMLX` members. The API key is a
    /// header, never logged.
    static func makeUpstreamRequest(
        member: MemberConfig, body: Data, apiKey: String?,
        upstreamPath: String = "chat/completions", contentType: String? = nil
    ) throws -> HTTPClientRequest {
        let (rewritten, sendType) = try UpstreamBodyCodec.prepare(
            body: body, contentType: contentType, upstreamModel: member.upstreamModel)
        var request = HTTPClientRequest(
            url: UpstreamBodyCodec.upstreamURL(endpoint: member.endpoint, path: upstreamPath))
        request.method = .POST
        request.headers.add(name: "content-type", value: sendType)
        if MemberAddress.isLocalTier(member) {
            request.headers.add(name: "X-No-Autoload", value: "1")
        }
        if let apiKey, !apiKey.isEmpty {
            request.headers.add(name: "authorization", value: "Bearer \(apiKey)")
        }
        request.body = .bytes(rewritten)
        return request
    }

    /// Stored-key resolution: own key, else the factory's shared key. An
    /// ALL-UPPERCASE stored value names a variable in the env file.
    enum KeyResolution: Equatable {
        case key(String?)
        case missingReference(String)
    }

    func apiKey(for member: MemberConfig, config: PeerConfig) -> KeyResolution {
        var stored: String?
        if let own = try? secrets.load(member.apiKeyRef), !own.isEmpty {
            stored = own
        } else if let factoryId = member.factoryId,
                  let ref = config.factory(id: factoryId)?.apiKeyRef,
                  let shared = try? secrets.load(ref), !shared.isEmpty {
            stored = shared
        }
        guard let stored else { return .key(nil) }
        guard EnvKeyFile.isReference(stored) else { return .key(stored) }
        guard let value = EnvKeyFile.load(url: envFile)[stored], !value.isEmpty else {
            return .missingReference(stored)
        }
        return .key(value)
    }

    // MARK: - Dispatch

    private func dispatch(
        pool: PoolConfig, config: PeerConfig, body: Data, timeoutSeconds: Double,
        upstreamPath: String, contentType: String?, interface: APIContract?
    ) async -> PoolForwardOutcome {
        let gatewayPath = upstreamPath.hasPrefix("/v1/") ? upstreamPath : "/v1/\(upstreamPath)"
        let effectiveApi = interface ?? pool.api
        guard SupplyAPI.allows(effectiveApi, gatewayPath: gatewayPath) else {
            return .pathNotAllowed(path: gatewayPath)
        }
        guard pool.isEnabled else { return .poolDisabled }
        let members = config.members(in: pool)
        let stats = await stats.snapshot()
        let candidates = PoolRouter.plan(PoolRouterInput(pool: pool, members: members, stats: stats))
        guard !candidates.isEmpty else { return .noHealthyMembers }

        if UpstreamBodyCodec.poolName(body: body, contentType: contentType) == nil,
           UpstreamBodyCodec.boundary(in: contentType) != nil || jsonObject(body) == nil {
            return .allMembersFailed(
                lastError: unreadableBodyMessage(contentType), lastStatus: 0, timedOut: false)
        }

        holder.bumpRequestCount(poolId: pool.id)
        var lastError = "no attempts made"
        var lastStatus = 0
        var timedOut = false
        let maxAttempts = max(1, min(pool.maxRetries, candidates.count))
        let timeout = TimeAmount.seconds(max(1, Int64(timeoutSeconds.rounded())))

        for member in candidates.prefix(maxAttempts) {
            if Task.isCancelled { return .cancelled }
            if member.type == .anthropic {
                lastError = "anthropic source forwarding not yet supported"
                lastStatus = 0
                timedOut = false
                await record(memberId: member.id, succeeded: false, latencyMs: 0,
                             httpStatus: 0, errorMessage: lastError)
                continue
            }

            let started = ContinuousClock.now
            let resolvedKey: String?
            switch apiKey(for: member, config: config) {
            case .key(let value):
                resolvedKey = value
            case .missingReference(let name):
                lastError = "API key \(name) is not defined in \(envFile.path)"
                lastStatus = 0
                timedOut = false
                await record(memberId: member.id, succeeded: false, latencyMs: 0,
                             httpStatus: 0, errorMessage: lastError)
                continue
            }
            do {
                let request = try Self.makeUpstreamRequest(
                    member: member, body: body, apiKey: resolvedKey,
                    upstreamPath: upstreamPath, contentType: contentType)
                let response = try await client.execute(request, timeout: timeout)
                let status = Int(response.status.code)
                let latency = Self.latencyMs(since: started)
                if status < 500 {
                    await record(memberId: member.id, succeeded: true, latencyMs: latency,
                                 httpStatus: status, errorMessage: nil)
                    let contentType = response.headers.first(name: "content-type")
                    let (stream, terminal) = Self.stream(response)
                    return .success(UpstreamBody(
                        status: status, contentType: contentType, body: stream, terminal: terminal))
                }
                let payload = await Self.collect(response, limit: 1 << 20)
                lastError = UpstreamErrorText.message(from: payload, status: status)
                lastStatus = status
                timedOut = false
                await record(memberId: member.id, succeeded: false, latencyMs: latency,
                             httpStatus: status, errorMessage: lastError)
                logger.warning("member \(member.id) returned HTTP \(status); trying next")
            } catch is CancellationError {
                return .cancelled
            } catch UpstreamRequestError.bodyNotJSON {
                return .allMembersFailed(
                    lastError: unreadableBodyMessage(contentType), lastStatus: 0, timedOut: false)
            } catch UpstreamRequestError.rewriteFailed {
                lastError = "model rewrite failed"
                lastStatus = 0
                timedOut = false
                await record(memberId: member.id, succeeded: false, latencyMs: Self.latencyMs(since: started),
                             httpStatus: 0, errorMessage: lastError)
            } catch let error as HTTPClientError where error == .deadlineExceeded {
                lastError = "timeout"
                lastStatus = 0
                timedOut = true
                await record(memberId: member.id, succeeded: false, latencyMs: Self.latencyMs(since: started),
                             httpStatus: 0, errorMessage: nil)
                logger.warning("member \(member.id) timed out; trying next")
            } catch {
                lastError = String(String(describing: error).prefix(200))
                lastStatus = 0
                timedOut = false
                await record(memberId: member.id, succeeded: false, latencyMs: Self.latencyMs(since: started),
                             httpStatus: 0, errorMessage: lastError)
                logger.warning("member \(member.id) failed: \(lastError)")
            }
        }
        return .allMembersFailed(lastError: lastError, lastStatus: lastStatus, timedOut: timedOut)
    }

    private func unreadableBodyMessage(_ contentType: String?) -> String {
        if UpstreamBodyCodec.boundary(in: contentType) != nil {
            return "multipart form has no model field"
        }
        return "request body is not a JSON object"
    }

    private func jsonObject(_ body: Data) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
    }

    private func record(memberId: String, succeeded: Bool, latencyMs: Int64,
                        httpStatus: Int, errorMessage: String?) async {
        await stats.record(memberId: memberId, succeeded: succeeded, latencyMs: latencyMs,
                           httpStatus: httpStatus, errorMessage: errorMessage)
    }

    private static func latencyMs(since start: ContinuousClock.Instant) -> Int64 {
        let elapsed = ContinuousClock.now - start
        let ms = Double(elapsed.components.seconds) * 1_000
            + Double(elapsed.components.attoseconds) / 1e15
        return Int64(ms.rounded())
    }

    private static func collect(_ response: HTTPClientResponse, limit: Int) async -> Data {
        guard let buffer = try? await response.body.collect(upTo: limit) else { return Data() }
        return Data(buffer.readableBytesView)
    }

    private static func stream(_ response: HTTPClientResponse) -> (AsyncStream<Data>, UpstreamTerminal) {
        let terminal = UpstreamTerminal()
        let stream = AsyncStream<Data> { continuation in
            let task = Task {
                do {
                    for try await buffer in response.body {
                        if Task.isCancelled {
                            terminal.set(.cancelled)
                            break
                        }
                        continuation.yield(Data(buffer.readableBytesView))
                    }
                } catch is CancellationError {
                    terminal.set(.cancelled)
                } catch let error as HTTPClientError where error == .deadlineExceeded {
                    terminal.set(.timedOut)
                } catch {
                    terminal.set(.failed(String(String(describing: error).prefix(200))))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return (stream, terminal)
    }
}

enum UpstreamRequestError: Error {
    case bodyNotJSON
    case rewriteFailed
}

enum UpstreamErrorText {
    /// NovaMLX nests `error.message`; some peers send `error` as a string.
    static func message(from data: Data, status: Int) -> String {
        let fallback = "upstream status \(status)"
        guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let error = object["error"] else { return fallback }
        if let nested = error as? [String: Any], let message = nested["message"] as? String, !message.isEmpty {
            return message
        }
        if let plain = error as? String, !plain.isEmpty { return plain }
        return fallback
    }
}
