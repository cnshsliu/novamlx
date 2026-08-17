import Foundation
import NovaMLXCore
import NovaMLXUtils

public protocol CatalogTransport: Sendable {
    func data(from url: URL) async throws -> Data
}

public struct URLSessionCatalogTransport: CatalogTransport {
    public init() {}
    public func data(from url: URL) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}

public final class ModelCatalogStore: @unchecked Sendable {
    public static let defaultRemoteURL = URL(string: "https://novamlx.ai/catalog/models.json")!

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
        ResourceBundleLocator.url(
            forResource: "models",
            withExtension: "json",
            subdirectory: "catalog",
            inBundle: "NovaMLX_NovaMLXUtils"
        )
    }

    public func refresh() async {
        if let data = try? await transport.data(from: remoteURL),
           let file = try? CatalogFile.decode(data) {
            lock.withLock { _models = file.models }
            try? FileManager.default.createDirectory(
                at: cacheURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? data.write(to: cacheURL, options: .atomic)
            NovaMLXLog.info("[Catalog] loaded \(file.models.count) models from remote")
            return
        }
        if let data = try? Data(contentsOf: cacheURL),
           let file = try? CatalogFile.decode(data) {
            lock.withLock { _models = file.models }
            NovaMLXLog.info("[Catalog] loaded \(file.models.count) models from cache")
            return
        }
        if let bundleURL,
           let data = try? Data(contentsOf: bundleURL),
           let file = try? CatalogFile.decode(data) {
            lock.withLock { _models = file.models }
            NovaMLXLog.info("[Catalog] loaded \(file.models.count) models from bundle")
            return
        }
        NovaMLXLog.error("[Catalog] no catalog available")
    }
}
