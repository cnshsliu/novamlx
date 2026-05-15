import Foundation

/// Safely locates SPM resource bundles without crashing.
///
/// SPM's generated `Bundle.module` accessor hardcodes the build directory
/// as a fallback search path. When the binary is deployed to a different
/// machine (e.g. via rsync to a Worker node), both `mainPath` and `buildPath`
/// may fail, causing a `fatalError` that kills the process.
///
/// This utility searches standard macOS app bundle locations instead:
/// 1. `Contents/Resources/` — standard macOS resource location
/// 2. `Contents/MacOS/` — alongside the executable
/// 3. `.app` root — SPM's preferred location (codesign-hostile)
public enum ResourceBundleLocator {

    /// Safely find a resource bundle by its SPM-generated name.
    /// Returns nil if not found (no crash).
    public static func find(bundleName: String) -> Bundle? {
        let candidates: [URL] = [
            Bundle.main.resourceURL,
            Bundle.main.executableURL?.deletingLastPathComponent(),
            Bundle.main.bundleURL,
        ].compactMap { $0 }

        for candidate in candidates {
            let bundleURL = candidate.appendingPathComponent("\(bundleName).bundle")
            if let bundle = Bundle(url: bundleURL) {
                return bundle
            }
        }
        return nil
    }

    /// Find a resource URL from a named bundle, searching safely.
    public static func url(
        forResource resource: String,
        withExtension ext: String?,
        subdirectory: String? = nil,
        inBundle bundleName: String
    ) -> URL? {
        find(bundleName: bundleName)?
            .url(forResource: resource, withExtension: ext, subdirectory: subdirectory)
    }
}
