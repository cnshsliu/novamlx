import Foundation
import Logging
import NovaMLXCore

public enum NovaMLXLog {
    public static let logger = Logger(label: "com.novamlx")

    private static let logFileURL: URL = NovaMLXPaths.logFile

    private static let logQueue = DispatchQueue(label: "com.novamlx.logfile")
    private static nonisolated(unsafe) var fileHandle: FileHandle?

    /// Clear log file — call once at app startup.
    public static func rotateLogFile() {
        logQueue.sync {
            fileHandle?.closeFile()
            fileHandle = nil
            let fm = FileManager.default
            try? fm.createDirectory(at: logFileURL.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            fm.createFile(atPath: logFileURL.path, contents: nil)
            fileHandle = try? FileHandle(forWritingTo: logFileURL)
        }
    }

    private static func writeToFile(_ level: String, _ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "[\(timestamp)] [\(level)] \(message)\n"
        guard let data = line.data(using: .utf8) else { return }
        logQueue.async {
            if let handle = fileHandle {
                handle.write(data)
            } else {
                // Lazy init if rotateLogFile wasn't called or handle went bad
                let fm = FileManager.default
                if !fm.fileExists(atPath: logFileURL.path) {
                    try? fm.createDirectory(at: logFileURL.deletingLastPathComponent(),
                                           withIntermediateDirectories: true)
                    fm.createFile(atPath: logFileURL.path, contents: nil)
                }
                if let handle = try? FileHandle(forWritingTo: logFileURL) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    // Don't close — keep for next write
                    fileHandle = handle
                }
            }
        }
    }

    public static func debug(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.debug(Logger.Message(stringLiteral: message), metadata: metadata)
    }

    public static func info(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.info(Logger.Message(stringLiteral: message), metadata: metadata)
        writeToFile("INFO", message)
    }

    public static func warning(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.warning(Logger.Message(stringLiteral: message), metadata: metadata)
        writeToFile("WARN", message)
    }

    public static func error(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.error(Logger.Message(stringLiteral: message), metadata: metadata)
        writeToFile("ERROR", message)
    }

    public static func request(_ requestId: String, _ message: String) {
        info("[\(requestId)] \(message)")
    }
}
