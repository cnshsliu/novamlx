import Foundation

public enum APIFormat: String, Codable, Sendable, Equatable { case openai, anthropic }

public enum SourceType: String, Codable, Sendable, Equatable { case openaiCompatible, anthropic, localNovaMLX }

/// Which demand entries this peer serves, at what price. Sent in `hello`
/// and `capabilities.update`; prices are per 1k tokens.
public struct Capability: Codable, Equatable, Sendable {
    public var demandId: String
    public var model: String
    public var sourceId: String
    public var sourceType: SourceType
    public var priceIn: Double
    public var priceOut: Double

    public init(demandId: String, model: String, sourceId: String,
                sourceType: SourceType, priceIn: Double, priceOut: Double) {
        self.demandId = demandId; self.model = model; self.sourceId = sourceId
        self.sourceType = sourceType; self.priceIn = priceIn; self.priceOut = priceOut
    }
}

/// One entry of tknet.ai's demand list.
public struct DemandEntry: Codable, Equatable, Sendable {
    public var demandId: String
    public var model: String
    public var modality: String   // "language" | "image" | "audio" | "video"
    public var note: String?

    public init(demandId: String, model: String, modality: String, note: String?) {
        self.demandId = demandId; self.model = model; self.modality = modality; self.note = note
    }
}

public struct Heartbeat: Codable, Equatable, Sendable {
    public var activeReq: Int
    public var queueDepth: Int
    public var avgTtftMs: Double?
    public var avgTokPerSec: Double?

    public init(activeReq: Int, queueDepth: Int, avgTtftMs: Double? = nil, avgTokPerSec: Double? = nil) {
        self.activeReq = activeReq; self.queueDepth = queueDepth
        self.avgTtftMs = avgTtftMs; self.avgTokPerSec = avgTokPerSec
    }
}

public struct RequestFrame: Codable, Equatable, Sendable {
    public var reqId: String
    public var model: String            // demand model name, NOT the upstream model
    public var apiFormat: APIFormat
    public var body: Data               // raw request JSON from the end user

    public init(reqId: String, model: String, apiFormat: APIFormat, body: Data) {
        self.reqId = reqId; self.model = model; self.apiFormat = apiFormat; self.body = body
    }
}

public enum RelayStatus: String, Codable, Equatable, Sendable {
    case completed, failed, cancelled, timeout
}

/// Telemetry the server bills and scores reputation on.
public struct RequestResult: Codable, Equatable, Sendable {
    public var status: RelayStatus
    public var ttftMs: Double
    public var totalMs: Double
    public var promptTokens: Int
    public var completionTokens: Int
    public var upstreamStatus: Int
    public var errorMessage: String?

    public init(status: RelayStatus, ttftMs: Double, totalMs: Double,
                promptTokens: Int, completionTokens: Int,
                upstreamStatus: Int, errorMessage: String?) {
        self.status = status; self.ttftMs = ttftMs; self.totalMs = totalMs
        self.promptTokens = promptTokens; self.completionTokens = completionTokens
        self.upstreamStatus = upstreamStatus; self.errorMessage = errorMessage
    }
}

/// Tunnel frame. The peer token never appears in any frame — it rides the
/// WebSocket upgrade's `Authorization: Bearer` header (see `WSTransport`);
/// source API keys never appear in any frame either.
public enum Frame: Equatable, Sendable {
    case hello(peerId: String, capabilities: [Capability])
    case capabilitiesUpdate([Capability])
    case heartbeat(Heartbeat)
    case responseChunk(reqId: String, payload: Data)     // raw SSE/JSON bytes
    case responseEnd(reqId: String, result: RequestResult)
    case request(RequestFrame)
    case requestCancel(reqId: String)
    case demandUpdate([DemandEntry])
    case error(String)
}

extension Frame {
    /// True when this frame carries upstream payload bytes.
    public var isChunk: Bool {
        if case .responseChunk = self { return true }
        return false
    }

    /// True when this is a terminal `responseEnd` that completed successfully.
    public var isCompletedEnd: Bool {
        if case .responseEnd(_, let result) = self { return result.status == .completed }
        return false
    }
}
