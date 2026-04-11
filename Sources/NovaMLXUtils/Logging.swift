import Foundation
import Logging
import NovaMLXCore

public enum NovaMLXLog {
    public static let logger = Logger(label: "com.novamlx")

    public static func debug(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.debug(Logger.Message(stringLiteral: message), metadata: metadata)
    }

    public static func info(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.info(Logger.Message(stringLiteral: message), metadata: metadata)
    }

    public static func warning(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.warning(Logger.Message(stringLiteral: message), metadata: metadata)
    }

    public static func error(_ message: String, metadata: Logger.Metadata? = nil) {
        logger.error(Logger.Message(stringLiteral: message), metadata: metadata)
    }

    public static func request(_ requestId: String, _ message: String) {
        info("[\(requestId)] \(message)")
    }
}
