import Foundation

/// Which binary this process actually is. Debug and release copies share the
/// marketing version, and both `/Applications` and `dist/` can be running, so
/// the path and the executable's mtime are what tell them apart.
public struct AppBuildIdentity: Equatable, Sendable {
    public enum Configuration: String, Sendable {
        case debug
        case release

        public var label: String {
            switch self {
            case .debug: return "Debug"
            case .release: return "Release"
            }
        }
    }

    public let configuration: Configuration
    /// `.app` bundle path, or the executable directory when not bundled.
    public let bundlePath: String
    /// Modification time of the running executable. `nil` when it cannot be read.
    public let builtAt: Date?

    public var configurationLabel: String { configuration.label }

    /// `yyyy-MM-dd HH:mm:ss` in the given time zone, or `"Unknown"`.
    public var stamp: String { Self.stamp(for: builtAt) }

    /// One line for the menu bar: version, Debug/Release, and the build clock.
    public var summaryLine: String {
        "v\(version) · \(configurationLabel) · \(stamp)"
    }

    public static let current: AppBuildIdentity = {
        let path = Bundle.main.bundleURL.standardizedFileURL.path
        #if DEBUG
        let configuration = Configuration.debug
        #else
        let configuration = Configuration.release
        #endif
        return AppBuildIdentity(
            configuration: configuration,
            bundlePath: path,
            builtAt: modificationDate(of: Bundle.main.executableURL)
        )
    }()

    public static func stamp(for date: Date?, timeZone: TimeZone = .current) -> String {
        guard let date else { return "Unknown" }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter.string(from: date)
    }

    static func modificationDate(of url: URL?) -> Date? {
        guard let url else { return nil }
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return attrs?[.modificationDate] as? Date
    }
}
