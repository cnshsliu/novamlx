import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdRouter
import NovaMLXCore
import NovaMLXUtils

public struct RateLimitConfig: Codable, Sendable {
    public let requestsPerSecond: Double
    public let burstSize: Int
    public let enabled: Bool

    public init(
        requestsPerSecond: Double = 10,
        burstSize: Int = 20,
        enabled: Bool = true
    ) {
        self.requestsPerSecond = requestsPerSecond
        self.burstSize = burstSize
        self.enabled = enabled
    }

    public static let `default` = RateLimitConfig()
    public static let disabled = RateLimitConfig(enabled: false)
}

final class TokenBucket: @unchecked Sendable {
    private let maxTokens: Int
    private let refillRate: Double
    private var tokens: Double
    private var lastRefill: Date
    private let lock = NovaMLXLock()

    init(maxTokens: Int, refillRate: Double) {
        self.maxTokens = maxTokens
        self.refillRate = refillRate
        self.tokens = Double(maxTokens)
        self.lastRefill = Date()
    }

    func consume() -> Bool {
        lock.withLock {
            let now = Date()
            let elapsed = now.timeIntervalSince(lastRefill)
            tokens = min(Double(maxTokens), tokens + elapsed * refillRate)
            lastRefill = now

            if tokens >= 1.0 {
                tokens -= 1.0
                return true
            }
            return false
        }
    }

    var currentTokens: Double {
        lock.withLock {
            let elapsed = Date().timeIntervalSince(lastRefill)
            return min(Double(maxTokens), tokens + elapsed * refillRate)
        }
    }
}

public final class RateLimiter: @unchecked Sendable {
    public struct BucketStats: Codable, Sendable {
        public let key: String
        public let currentTokens: Double
        public let maxTokens: Int
        public let refillRate: Double
    }

    private let config: RateLimitConfig
    private var buckets: [String: TokenBucket] = [:]
    private let lock = NovaMLXLock()
    private var totalAllowed: UInt64 = 0
    private var totalRejected: UInt64 = 0

    public init(config: RateLimitConfig) {
        self.config = config
    }

    func allow(key: String) -> Bool {
        guard config.enabled else {
            lock.withLock { totalAllowed += 1 }
            return true
        }

        let bucket: TokenBucket
        let existing = lock.withLock { buckets[key] }
        if let existing {
            bucket = existing
        } else {
            bucket = TokenBucket(maxTokens: config.burstSize, refillRate: config.requestsPerSecond)
            lock.withLock { buckets[key] = bucket }
        }

        let allowed = bucket.consume()
        lock.withLock {
            if allowed { totalAllowed += 1 }
            else { totalRejected += 1 }
        }
        return allowed
    }

    public func getStats() -> [String: Any] {
        let (allowed, rejected, bucketCount) = lock.withLock {
            (totalAllowed, totalRejected, buckets.count)
        }
        return [
            "enabled": config.enabled,
            "requests_per_second": config.requestsPerSecond,
            "burst_size": config.burstSize,
            "total_allowed": allowed,
            "total_rejected": rejected,
            "active_buckets": bucketCount
        ]
    }

    func cleanup(maxBuckets: Int = 10000) {
        lock.withLock {
            if buckets.count > maxBuckets {
                let keys = Array(buckets.keys.prefix(buckets.count - maxBuckets / 2))
                for k in keys { buckets.removeValue(forKey: k) }
            }
        }
    }
}

struct RateLimitMiddleware: RouterMiddleware, @unchecked Sendable {
    typealias Context = AppContext

    private let limiter: RateLimiter
    private let keyExtractor: @Sendable (Request) -> String

    init(limiter: RateLimiter, extractKey: @escaping @Sendable (Request) -> String) {
        self.limiter = limiter
        self.keyExtractor = extractKey
    }

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let key = keyExtractor(request)
        guard limiter.allow(key: key) else {
            let detail = OpenAIErrorDetail(
                message: "Rate limit exceeded. Please retry after a moment.",
                type: "rate_limit_error",
                code: "rate_limit_exceeded"
            )
            var response = NovaMLXErrorMiddleware.jsonError(status: .tooManyRequests, detail: detail)
            response.headers[HTTPField.Name("Retry-After")!] = "1"
            return response
        }
        return try await next(request, context)
    }

    static func perAPIKey(limiter: RateLimiter) -> Self {
        Self(limiter: limiter) { request in
            if let auth = request.headers[.authorization], auth.hasPrefix("Bearer ") {
                return String(auth.dropFirst(7))
            }
            return "anonymous"
        }
    }

    static func perIP(limiter: RateLimiter) -> Self {
        let xffName = HTTPField.Name("X-Forwarded-For")!
        return Self(limiter: limiter) { request in
            if let xff = request.headers[xffName],
               let first = xff.components(separatedBy: ",").first?.trimmingCharacters(in: .whitespaces),
               !first.isEmpty {
                return first
            }
            return "unknown"
        }
    }
}

struct RequestSizeLimitMiddleware: RouterMiddleware {
    typealias Context = AppContext

    private let maxBodySize: Int

    init(maxBytes: Int) {
        self.maxBodySize = maxBytes
    }

    init(maxMB: Double) {
        self.maxBodySize = Int(maxMB * 1024 * 1024)
    }

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        if let contentLength = request.headers[.contentLength],
           let size = Int(contentLength),
           size > maxBodySize {
            let detail = OpenAIErrorDetail(
                message: "Request body too large (\(size) bytes). Maximum allowed: \(maxBodySize) bytes.",
                type: "invalid_request_error",
                code: "request_too_large"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .contentTooLarge, detail: detail)
        }

        let response = try await next(request, context)

        if let contentLength = response.headers[.contentLength],
           let reportedSize = Int(contentLength),
           reportedSize > maxBodySize {
            let detail = OpenAIErrorDetail(
                message: "Response body too large (\(reportedSize) bytes). Maximum allowed: \(maxBodySize) bytes.",
                type: "invalid_request_error",
                code: "response_too_large"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .contentTooLarge, detail: detail)
        }

        return response
    }
}

struct TimeoutMiddleware: RouterMiddleware {
    typealias Context = AppContext

    private let timeout: TimeInterval

    init(timeout: TimeInterval) {
        self.timeout = timeout
    }

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let deadline = ContinuousClock.now.advanced(by: .seconds(timeout))
        let response = try await next(request, context)
        if ContinuousClock.now > deadline {
            context.logger.warning("Request exceeded timeout of \(self.timeout)s")
        }
        return response
    }
}

struct SecurityHeadersMiddleware: RouterMiddleware {
    typealias Context = AppContext

    private static let xFrameOptions = HTTPField.Name("X-Frame-Options")!
    private static let referrerPolicy = HTTPField.Name("Referrer-Policy")!

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        var response = try await next(request, context)
        response.headers[.xContentTypeOptions] = "nosniff"
        response.headers[.strictTransportSecurity] = "max-age=31536000; includeSubDomains"
        response.headers[Self.xFrameOptions] = "SAMEORIGIN"
        response.headers[Self.referrerPolicy] = "strict-origin-when-cross-origin"
        return response
    }
}
