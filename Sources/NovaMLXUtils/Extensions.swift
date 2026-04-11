import Foundation

extension String {
    public var nilIfEmpty: String? {
        isEmpty ? nil : self
    }

    public func truncated(to length: Int, suffix: String = "...") -> String {
        if count <= length { return self }
        return String(prefix(length - suffix.count)) + suffix
    }

    public var sanitized: String {
        let invalid = CharacterSet(charactersIn: "/\\?%*|\"<>")
        return components(separatedBy: invalid).joined(separator: "_")
    }
}

extension Double {
    public func rounded(_ decimals: Int) -> Double {
        let multiplier = pow(10.0, Double(decimals))
        return (self * multiplier).rounded() / multiplier
    }
}

extension Int {
    public var bytesFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(self), countStyle: .file)
    }
}

extension UInt64 {
    public var bytesFormatted: String {
        ByteCountFormatter.string(fromByteCount: Int64(self), countStyle: .file)
    }
}

extension URL {
    public var fileName: String {
        deletingPathExtension().lastPathComponent
    }

    public var fileExists: Bool {
        FileManager.default.fileExists(atPath: path)
    }

    public var directoryExists: Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
    }
}

extension Date {
    public var iso8601String: String {
        ISO8601DateFormatter().string(from: self)
    }

    public static func - (lhs: Date, rhs: Date) -> TimeInterval {
        lhs.timeIntervalSinceReferenceDate - rhs.timeIntervalSinceReferenceDate
    }
}

extension FileManager {
    public func directorySize(at url: URL) -> UInt64 {
        guard let enumerator = enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += UInt64(size)
            }
        }
        return total
    }
}
