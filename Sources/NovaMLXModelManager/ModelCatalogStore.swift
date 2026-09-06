import Foundation
import NovaMLXCore
import NovaMLXUtils

public protocol CatalogTransport: Sendable {
    func data(from url: URL) async throws -> Data
}

public struct URLSessionCatalogTransport: CatalogTransport {
    public init() {}
    public func data(from url: URL) async throws -> Data {
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false) ?? URLComponents()
        var items = components.queryItems ?? []
        items.append(URLQueryItem(name: "_", value: String(Int(Date().timeIntervalSince1970 * 1000))))
        components.queryItems = items
        var request = URLRequest(url: components.url ?? url)
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")
        request.setValue("no-cache", forHTTPHeaderField: "Pragma")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}

public final class ModelCatalogStore: @unchecked Sendable {
    /// Live list of downloadable models. Author at `catalog/models.json` in this
    /// repo; the bundled snapshot in NovaMLXUtils is the offline fallback.
    public static let defaultRemoteURL = URL(
        string: "https://raw.githubusercontent.com/cnshsliu/novamlx/main/catalog/models.json"
    )!

    private let remoteURL: URL
    private let cacheURL: URL
    private let bundleURL: URL?
    private let transport: CatalogTransport
    private let lock = NSLock()
    private var _models: [CatalogEntry] = []

    public var models: [CatalogEntry] {
        lock.withLock { _models }
    }

    public init(
        remoteURL: URL = ModelCatalogStore.defaultRemoteURL,
        cacheURL: URL = NovaMLXPaths.catalogCacheFile,
        bundleURL: URL? = ModelCatalogStore.bundledSnapshotURL(),
        transport: CatalogTransport = URLSessionCatalogTransport()
    ) {
        self.remoteURL = remoteURL
        self.cacheURL = cacheURL
        self.bundleURL = bundleURL
        self.transport = transport
    }

    public static func bundledSnapshotURL() -> URL? {
        // SPM may pack Resources/catalog/ as "catalog", "Resources/catalog",
        // or flatten the file next to other resources depending on host layout.
        let subdirs: [String?] = ["catalog", "Resources/catalog", nil]
        for subdir in subdirs {
            if let url = ResourceBundleLocator.url(
                forResource: "models",
                withExtension: "json",
                subdirectory: subdir,
                inBundle: "NovaMLX_NovaMLXUtils"
            ) {
                return url
            }
        }
        return nil
    }

    public func refresh() async {
        struct Stamped {
            let entry: CatalogEntry
            let stamp: Date
        }

        var map: [String: Stamped] = [:]
        var order: [String] = []
        var sources: [String] = []
        var newestStamp = Date.distantPast
        var newestStampRaw: String?

        func ingest(_ file: CatalogFile, source: String) {
            sources.append("\(source):\(file.models.count)")
            let stamp = Self.timestamp(file.updatedAt)
            if stamp >= newestStamp {
                newestStamp = stamp
                newestStampRaw = file.updatedAt ?? newestStampRaw
            }
            for entry in file.models {
                if map[entry.id] == nil { order.append(entry.id) }
                if let existing = map[entry.id] {
                    if stamp >= existing.stamp {
                        map[entry.id] = Stamped(entry: entry, stamp: stamp)
                    }
                } else {
                    map[entry.id] = Stamped(entry: entry, stamp: stamp)
                }
            }
        }

        if let data = try? await transport.data(from: remoteURL),
           let file = try? CatalogFile.decode(data) {
            ingest(file, source: "remote")
        }
        if let data = try? Data(contentsOf: cacheURL),
           let file = try? CatalogFile.decode(data) {
            ingest(file, source: "cache")
        }
        if let bundleURL,
           let data = try? Data(contentsOf: bundleURL),
           let file = try? CatalogFile.decode(data) {
            ingest(file, source: "bundle")
        }

        let models = order.compactMap { map[$0]?.entry }
        guard !models.isEmpty else {
            NovaMLXLog.error("[Catalog] no catalog available")
            return
        }

        lock.withLock { _models = models }

        let envelope = CatalogFile(
            schemaVersion: 1,
            updatedAt: newestStampRaw,
            models: models
        )
        if let data = try? JSONEncoder().encode(envelope) {
            try? FileManager.default.createDirectory(
                at: cacheURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? data.write(to: cacheURL, options: .atomic)
        }
        NovaMLXLog.info("[Catalog] loaded \(models.count) models (merged \(sources.joined(separator: "+")))")
    }

    /// Replace the in-memory list after a local catalog-admin save.
    public func applyLocal(_ file: CatalogFile) {
        lock.withLock { _models = file.models }
    }

    public func containsExactId(_ id: String) -> Bool {
        lock.withLock { _models.contains { $0.id == id } }
    }

    /// ISO-8601 `updatedAt`; missing / unparseable sorts as oldest.
    static func timestamp(_ raw: String?) -> Date {
        guard let raw, !raw.isEmpty else { return .distantPast }
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        if let date = iso.date(from: raw) { return date }
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return iso.date(from: raw) ?? .distantPast
    }
}
