import Foundation

enum SafetensorsHeaderError: Error {
    case shortRead
    case badJSON
    case invalidLength
}

/// Reads ONLY the safetensors header (metadata JSON) without touching tensor data.
/// Format: first 8 bytes = little-endian u64 header length, next N bytes = JSON.
/// We only care about the "__metadata__" key — the per-tensor descriptors are skipped.
func readSafetensorsMetadata(url: URL) throws -> [String: String] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else {
        throw SafetensorsHeaderError.shortRead
    }
    let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
    guard headerLen > 0, headerLen < 16 * 1024 * 1024 else {
        throw SafetensorsHeaderError.invalidLength
    }
    guard let jsonData = try handle.read(upToCount: Int(headerLen)),
          jsonData.count == Int(headerLen) else {
        throw SafetensorsHeaderError.shortRead
    }
    guard let parsed = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
          let meta = parsed["__metadata__"] as? [String: String] else {
        return [:]
    }
    return meta
}
