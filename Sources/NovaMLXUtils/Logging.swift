import Foundation
import Logging
import NovaMLXCore

public enum NovaMLXLog {
    public static let logger = Logger(label: "com.novamlx")

    private static let logFileURL: URL = NovaMLXPaths.logFile

    /// How many old log files to keep during rotation.
    public nonisolated(unsafe) static var maxRotatedFiles: Int = 5

    /// Minimum log level that writes to file. Default: .info.
    /// Set to .debug to capture all debug output in the log file.
    public nonisolated(unsafe) static var fileLogLevel: LogLevel = .info

    private static let logQueue = DispatchQueue(label: "com.novamlx.logfile")
    private static nonisolated(unsafe) var fileHandle: FileHandle?

    public enum LogLevel: Int, Comparable, Sendable {
        case debug = 0
        case info = 1
        case warning = 2
        case error = 3

        public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    /// Rotate log file — call once at app startup.
    /// Keeps up to `maxRotatedFiles` old logs as `.1`, `.2`, etc.
    public static func rotateLogFile() {
        logQueue.sync {
            fileHandle?.closeFile()
            fileHandle = nil
            let fm = FileManager.default

            // Rotate: .4 → .5, .3 → .4, ..., .1 → .2
            let base = logFileURL
            for i in stride(from: maxRotatedFiles - 1, through: 1, by: -1) {
                let old = base.appendingPathExtension(".\(i)")
                let next = base.appendingPathExtension(".\(i + 1)")
                try? fm.removeItem(at: next)
                _ = try? fm.moveItem(at: old, to: next)
            }
            // Current → .1
            let first = base.appendingPathExtension(".1")
            try? fm.removeItem(at: first)
            _ = try? fm.moveItem(at: base, to: first)

            // Create fresh log file
            try? fm.createDirectory(at: base.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            fm.createFile(atPath: base.path, contents: nil)
            fileHandle = try? FileHandle(forWritingTo: base)
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
                let fm = FileManager.default
                if !fm.fileExists(atPath: logFileURL.path) {
                    try? fm.createDirectory(at: logFileURL.deletingLastPathComponent(),
                                           withIntermediateDirectories: true)
                    fm.createFile(atPath: logFileURL.path, contents: nil)
                }
                if let handle = try? FileHandle(forWritingTo: logFileURL) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    fileHandle = handle
                }
            }
        }
    }

    public static func debug(_ message: String) {
        logger.debug(Logger.Message(stringLiteral: message))
        if fileLogLevel <= .debug {
            writeToFile("DEBUG", message)
        }
    }

    public static func info(_ message: String) {
        logger.info(Logger.Message(stringLiteral: message))
        if fileLogLevel <= .info {
            writeToFile("INFO", message)
        }
    }

    public static func warning(_ message: String) {
        logger.warning(Logger.Message(stringLiteral: message))
        if fileLogLevel <= .warning {
            writeToFile("WARN", message)
        }
    }

    public static func error(_ message: String) {
        logger.error(Logger.Message(stringLiteral: message))
        if fileLogLevel <= .error {
            writeToFile("ERROR", message)
        }
    }

    public static func request(_ requestId: String, _ message: String) {
        info("[\(requestId)] \(message)")
    }
}
