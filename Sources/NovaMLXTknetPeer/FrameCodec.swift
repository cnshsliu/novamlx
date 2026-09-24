import Foundation

public enum FrameError: Error, Equatable {
    case unknownType(String)
    case malformed(String)
}

/// JSON envelope: {"type":"hello","data":{...}} — one frame per WS text message.
public enum FrameCodec {
    /// `JSONSerialization` only accepts property-list types, so Codable payloads
    /// must be routed through `JSONEncoder` before embedding in the envelope.
    private static func jsonValue<T: Encodable>(_ value: T) -> Any {
        let data = try! JSONEncoder().encode(value)
        return try! JSONSerialization.jsonObject(with: data)
    }

    public static func encode(_ frame: Frame) -> String {
        let obj: [String: Any]
        switch frame {
        case .hello(let peerId, let caps):
            obj = ["type": "hello", "data": jsonValue(HelloPayload(peerId: peerId, capabilities: caps, version: TknetPeer.version))]
        case .capabilitiesUpdate(let caps):
            obj = ["type": "capabilities.update", "data": ["capabilities": jsonValue(caps)]]
        case .heartbeat(let hb):
            obj = ["type": "heartbeat", "data": jsonValue(hb)]
        case .responseChunk(let reqId, let payload):
            obj = ["type": "response.chunk", "data": ["reqId": reqId, "payload": payload.base64EncodedString()]]
        case .responseEnd(let reqId, let result):
            obj = ["type": "response.end", "data": ["reqId": reqId, "result": jsonValue(result)]]
        case .request(let req):
            obj = ["type": "request", "data": [
                "reqId": req.reqId, "model": req.model,
                "apiFormat": req.apiFormat.rawValue,
                "body": req.body.base64EncodedString(),
            ]]
        case .requestCancel(let reqId):
            obj = ["type": "request.cancel", "data": ["reqId": reqId]]
        case .demandUpdate(let entries):
            obj = ["type": "demand.update", "data": ["entries": jsonValue(entries)]]
        case .error(let message):
            obj = ["type": "error", "data": ["message": message]]
        }
        let data = try! JSONSerialization.data(withJSONObject: obj)
        return String(data: data, encoding: .utf8)!
    }

    public static func decode(_ text: String) throws -> Frame {
        guard let data = text.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let type = obj["type"] as? String,
              let raw = obj["data"] else {
            throw FrameError.malformed(text.prefix(120).description)
        }
        func decodePayload<T: Decodable>(_ type: T.Type) throws -> T {
            try JSONDecoder().decode(T.self, from: JSONSerialization.data(withJSONObject: raw))
        }
        switch type {
        case "hello":
            let p = try decodePayload(HelloPayload.self)
            return .hello(peerId: p.peerId, capabilities: p.capabilities)
        case "capabilities.update":
            let p = try decodePayload(CapabilitiesPayload.self)
            return .capabilitiesUpdate(p.capabilities)
        case "heartbeat":
            return .heartbeat(try decodePayload(Heartbeat.self))
        case "response.chunk":
            let p = try decodePayload(ChunkPayload.self)
            return .responseChunk(reqId: p.reqId, payload: Data(base64Encoded: p.payload) ?? Data())
        case "response.end":
            let p = try decodePayload(EndPayload.self)
            return .responseEnd(reqId: p.reqId, result: p.result)
        case "request":
            let p = try decodePayload(RequestPayload.self)
            return .request(RequestFrame(
                reqId: p.reqId, model: p.model,
                apiFormat: APIFormat(rawValue: p.apiFormat) ?? .openai,
                body: Data(base64Encoded: p.body) ?? Data()
            ))
        case "request.cancel":
            let p = try decodePayload(CancelPayload.self)
            return .requestCancel(reqId: p.reqId)
        case "demand.update":
            let p = try decodePayload(DemandPayload.self)
            return .demandUpdate(p.entries)
        case "error":
            let p = try decodePayload(ErrorPayload.self)
            return .error(p.message)
        default:
            throw FrameError.unknownType(type)
        }
    }
}

struct HelloPayload: Codable { var peerId: String; var capabilities: [Capability]; var version: String }
struct CapabilitiesPayload: Codable { var capabilities: [Capability] }
struct ChunkPayload: Codable { var reqId: String; var payload: String }
struct EndPayload: Codable { var reqId: String; var result: RequestResult }
struct RequestPayload: Codable { var reqId: String; var model: String; var apiFormat: String; var body: String }
struct CancelPayload: Codable { var reqId: String }
struct DemandPayload: Codable { var entries: [DemandEntry] }
struct ErrorPayload: Codable { var message: String }
