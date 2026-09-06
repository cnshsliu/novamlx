import Foundation
import NovaMLXCore
import NovaMLXDB

/// Lifecycle of a single logged API request.
public enum RequestStatus: Sendable {
    case pending
    case success
    case error
    case cancelled
}

/// One row in the request log. Identified by the request id generated at the
/// HTTP layer (see `RequestLogMiddleware`).
public struct RequestLogEntry: Identifiable, Sendable {
    public let id: String
    public let method: String
    public let path: String
    public var model: String?
    public var kind: InferenceKind?
    public let apiKeyId: String?
    public let apiKeyName: String?
    public let startedAt: Date
    public var finishedAt: Date?
    public var durationMs: Double?
    public var status: RequestStatus
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var tps: Double?
    public var error: String?

    /// Raw request body bytes (only set for small text/JSON payloads).
    public var requestBody: Data?
    /// `Content-Type` of the request body (e.g. "application/json").
    public var requestContentType: String?
    /// Why the request body was not captured (e.g. "[audio/wav · 2.3 MB — body not captured]").
    public var requestBodyNote: String?
    /// HTTP status code of the response (200, 404, 401, …).
    public var responseStatus: Int?

    public init(
        id: String,
        method: String,
        path: String,
        model: String? = nil,
        kind: InferenceKind? = nil,
        apiKeyId: String? = nil,
        apiKeyName: String? = nil,
        startedAt: Date = Date(),
        requestBody: Data? = nil,
        requestContentType: String? = nil,
        requestBodyNote: String? = nil
    ) {
        self.id = id
        self.method = method
        self.path = path
        self.model = model
        self.kind = kind
        self.apiKeyId = apiKeyId
        self.apiKeyName = apiKeyName
        self.startedAt = startedAt
        self.status = .pending
        self.requestBody = requestBody
        self.requestContentType = requestContentType
        self.requestBodyNote = requestBodyNote
    }

    /// Short path label for the UI (e.g. "/v1/chat/completions").
    public var endpoint: String {
        if path.hasPrefix("/v1/chat/completions") { return "/v1/chat/completions" }
        if path.hasPrefix("/v1/messages") { return "/v1/messages" }
        if path.hasPrefix("/v1/audio/transcriptions") { return "/v1/audio/transcriptions" }
        if path.hasPrefix("/v1/audio/speech") { return "/v1/audio/speech" }
        if path.hasPrefix("/v1/images/generations") { return "/v1/images/generations" }
        return path
    }
}

/// Central, thread-safe request log. The HTTP middleware records every inbound
/// request here on entry; the inference layer enriches + finalizes entries on
/// completion with model/kind/timing/tokens. The Requests page renders it.
public final class RequestLogStore: @unchecked Sendable {
    public static let shared = RequestLogStore()

    private let lock = NovaMLXLock()
    /// Requests currently in flight, keyed by request id.
    private var active: [String: RequestLogEntry] = [:]
    /// Index from model id -> request ids, so the inference layer (which knows
    /// the model but not the HTTP request id) can finalize the right entry.
    private var activeByModel: [String: [String]] = [:]
    /// Completed requests, most-recent-first, capped.
    private var recent: [RequestLogEntry] = []
    private let maxRecent: Int

    public init(maxRecent: Int = 300) {
        self.maxRecent = maxRecent
    }

    // MARK: - HTTP layer (middleware)

    /// Record the start of an inbound request. Resolves the API key name from
    /// the raw bearer/x-api-key token so the log shows which key was used.
    public func start(
        id: String,
        method: String,
        path: String,
        apiKeyToken: String?,
        model: String? = nil,
        requestBody: Data? = nil,
        requestContentType: String? = nil,
        requestBodyNote: String? = nil
    ) {
        let keyInfo = Self.resolveApiKey(token: apiKeyToken)
        let entry = RequestLogEntry(
            id: id,
            method: method,
            path: path,
            model: model,
            apiKeyId: keyInfo?.id,
            apiKeyName: keyInfo?.displayName,
            requestBody: requestBody,
            requestContentType: requestContentType,
            requestBodyNote: requestBodyNote
        )
        lock.withLock {
            active[id] = entry
        }
    }

    /// Attach the HTTP response status to an in-flight entry (so the detail
    /// panel can show 200 / 401 / 404 etc.). No-op if the entry is already
    /// gone (e.g. finalized by the inference layer first).
    public func recordResponse(id: String, status: Int) {
        lock.withLock {
            guard var entry = active[id] else { return }
            entry.responseStatus = status
            active[id] = entry
        }
    }

    /// Finalize an entry directly from the HTTP layer. Used for non-inference
    /// responses and for requests that fail before reaching the engine
    /// (auth errors, 404s, etc.).
    public func finishHTTP(
        id: String,
        status: RequestStatus,
        error: String? = nil,
        durationMs: Double? = nil,
        responseStatus: Int? = nil
    ) {
        lock.withLock {
            guard var entry = active[id] else { return }
            if let rs = responseStatus { entry.responseStatus = rs }
            finalizeLocked(&entry, status: status, error: error, durationMs: durationMs)
            active.removeValue(forKey: id)
            if let idx = activeByModel[entry.model ?? ""]?.firstIndex(of: id) {
                var list = activeByModel[entry.model ?? ""] ?? []
                list.remove(at: idx)
                activeByModel[entry.model ?? ""] = list.isEmpty ? nil : list
            }
        }
    }

    /// Update the model/kind for an already-started entry (the HTTP layer only
    /// learns the true model after modelfile/LB resolution).
    public func update(id: String, model: String, kind: InferenceKind) {
        lock.withLock {
            guard var entry = active[id] else { return }
            let oldModel = entry.model
            entry.model = model
            entry.kind = kind
            active[id] = entry
            // Reindex by model.
            if let old = oldModel, old != model,
               let idx = activeByModel[old]?.firstIndex(of: id) {
                var list = activeByModel[old] ?? []
                list.remove(at: idx)
                activeByModel[old] = list.isEmpty ? nil : list
            }
            if model != "" {
                var list = activeByModel[model] ?? []
                if !list.contains(id) { list.append(id) }
                activeByModel[model] = list
            }
        }
    }

    // MARK: - Inference layer

    /// Enrich in-flight timing as generation progresses (called periodically
    /// during streaming) so the active-requests panel updates live.
    public func update(id: String, tps: Double, promptTokens: Int, completionTokens: Int, durationMs: Double) {
        lock.withLock {
            guard var entry = active[id] else { return }
            entry.tps = tps
            entry.promptTokens = promptTokens
            entry.completionTokens = completionTokens
            entry.durationMs = durationMs
            active[id] = entry
        }
    }

    /// Finish the most recent in-flight entry for `model`. Called by the
    /// inference layer on completion; it knows the model but not the HTTP
    /// request id, so we match on model + recency. Pass the request id when
    /// the caller has it (preferred).
    public func finish(
        model: String,
        kind: InferenceKind,
        status: RequestStatus,
        tps: Double,
        promptTokens: Int,
        completionTokens: Int,
        durationMs: Double,
        error: String? = nil,
        requestId: String? = nil
    ) {
        lock.withLock {
            let id: String?
            if let rid = requestId, active[rid] != nil {
                id = rid
            } else {
                id = activeByModel[model]?.last
            }
            guard let id, var entry = active[id] else { return }
            entry.kind = kind
            entry.model = model
            entry.tps = tps
            entry.promptTokens = promptTokens
            entry.completionTokens = completionTokens
            finalizeLocked(&entry, status: status, error: error, durationMs: durationMs)
            active.removeValue(forKey: id)
            if let idx = activeByModel[model]?.firstIndex(of: id) {
                var list = activeByModel[model] ?? []
                list.remove(at: idx)
                activeByModel[model] = list.isEmpty ? nil : list
            }
        }
    }

    /// Finalize an in-flight entry from the engine's result. Looks up the entry
    /// by the HTTP request id carried on the `InferenceRequest` (preferred) and
    /// falls back to model + recency. Used at every LLM/VLM completion site.
    public func finish(
        request: InferenceRequest,
        model: String,
        kind: InferenceKind,
        tps: Double,
        promptTokens: Int,
        completionTokens: Int,
        durationMs: Double,
        finishReason: String? = nil,
        error: String? = nil
    ) {
        let status: RequestStatus = error != nil ? .error : .success
        finish(
            model: model,
            kind: kind,
            status: status,
            tps: tps,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            durationMs: durationMs,
            error: error ?? finishReason.flatMap { "finished: \($0)" },
            requestId: request.httpRequestId
        )
    }

    /// Cancel any entries still in flight older than `age` (safety net so stale
    /// rows from crashed requests don't linger forever). Returns count cleared.
    @discardableResult
    public func cancelStale(olderThan age: TimeInterval = 120) -> Int {
        let cutoff = Date().addingTimeInterval(-age)
        return lock.withLock {
            let stale = active.filter { $0.value.startedAt < cutoff }.map { $0.key }
            for id in stale {
                guard var entry = active[id] else { continue }
                finalizeLocked(&entry, status: .cancelled, error: "timeout", durationMs: age * 1000)
                active.removeValue(forKey: id)
                if let model = entry.model, let idx = activeByModel[model]?.firstIndex(of: id) {
                    var list = activeByModel[model] ?? []
                    list.remove(at: idx)
                    activeByModel[model] = list.isEmpty ? nil : list
                }
            }
            return stale.count
        }
    }

    // MARK: - Read

    /// Snapshot of in-flight requests, oldest-first (stable display order).
    public var activeRequests: [RequestLogEntry] {
        lock.withLock { active.values.sorted { $0.startedAt < $1.startedAt } }
    }

    /// Snapshot of completed requests, most-recent-first.
    public var completedRequests: [RequestLogEntry] {
        lock.withLock { recent }
    }

    /// Both combined: active first (sorted by start), then completed.
    public var allRequests: [RequestLogEntry] {
        lock.withLock {
            active.values.sorted { $0.startedAt < $1.startedAt } + recent
        }
    }

    public func clearCompleted() {
        lock.withLock { recent.removeAll() }
    }

    // MARK: - Helpers

    private static func resolveApiKey(token: String?) -> (id: String, displayName: String)? {
        guard let token, !token.isEmpty else {
            // No auth — treat as local/open access.
            return ("", "no-key")
        }
        if let rec = try? NovaDB.shared.apiKeyStore.findByRawKey(token) {
            let name = rec.name.isEmpty ? rec.keyPrefix : rec.name
            return (rec.id, "\(name) (\(rec.keyPrefix))")
        }
        return ("?", "unknown")
    }

    private func finalizeLocked(_ entry: inout RequestLogEntry, status: RequestStatus, error: String?, durationMs: Double?) {
        entry.status = status
        entry.error = error
        entry.finishedAt = Date()
        if let d = durationMs {
            entry.durationMs = d
        } else {
            entry.durationMs = entry.finishedAt!.timeIntervalSince(entry.startedAt) * 1000
        }
        recent.insert(entry, at: 0)
        if recent.count > maxRecent {
            recent.removeLast(recent.count - maxRecent)
        }
    }
}
