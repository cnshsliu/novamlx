import Foundation
import Logging

// MARK: - WeightDistributorError

/// Errors raised by ``WeightDistributor`` operations.
public enum WeightDistributorError: Error, Sendable, Equatable {
    /// The requested model was not found on the coordinator.
    case modelNotFound(String)
    /// The download from the coordinator failed.
    case downloadFailed(String)
    /// The coordinator is not reachable.
    case coordinatorUnavailable
}

extension WeightDistributorError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let modelId):
            "Model not found on coordinator: \(modelId)"
        case .downloadFailed(let reason):
            "Download failed: \(reason)"
        case .coordinatorUnavailable:
            "Coordinator is not reachable"
        }
    }
}

// MARK: - DownloadProgress

/// Tracks the progress of a model weight download from the coordinator.
public struct DownloadProgress: Sendable, Equatable {
    /// Identifier of the model being downloaded.
    public let modelId: String
    /// Number of bytes downloaded so far.
    public var bytesDownloaded: UInt64
    /// Total expected size in bytes. Updated once Content-Length is known.
    public var totalBytes: UInt64
    /// Whether the download has completed.
    public var isComplete: Bool

    /// Fraction of the download completed, in the range `[0.0, 1.0]`.
    public var fraction: Double {
        guard totalBytes > 0 else { return 0.0 }
        return min(Double(bytesDownloaded) / Double(totalBytes), 1.0)
    }

    public init(
        modelId: String,
        bytesDownloaded: UInt64 = 0,
        totalBytes: UInt64 = 0,
        isComplete: Bool = false
    ) {
        self.modelId = modelId
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
        self.isComplete = isComplete
    }
}

// MARK: - WeightDistributor

/// Manages model weight file availability on worker nodes.
///
/// When a worker needs a model that is not yet present locally, the
/// ``WeightDistributor`` downloads it from the coordinator node and tracks
/// download progress for admin API observability.
///
/// Thread safety: all mutable state is guarded by ``queue``. The class is
/// marked `@unchecked Sendable` because the serial queue serialises access.
public final class WeightDistributor: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = WeightDistributor()

    // MARK: - Public properties

    /// Snapshot of currently active downloads (thread-safe copy).
    public var activeDownloadsSnapshot: [String: DownloadProgress] {
        queue.sync { activeDownloads }
    }

    // MARK: - Private state

    /// All mutable state is accessed exclusively on this serial queue.
    private let queue = DispatchQueue(
        label: "com.novamlx.weight-distributor",
        qos: .userInitiated
    )

    /// Currently active downloads keyed by model ID.
    private var activeDownloads: [String: DownloadProgress] = [:]

    /// Logger instance.
    private let logger = Logger(label: "NovaMLXDistributed.WeightDistributor")

    /// Chunk size for streaming downloads (1 MB).
    private let downloadChunkSize = 1024 * 1024

    // MARK: - Lifecycle

    private init() {}

    // MARK: - Public API

    /// Ensure a model is available locally, downloading from the coordinator if needed.
    ///
    /// - Path A: If the file already exists at `expectedPath`, returns immediately.
    /// - Path B: If the file is missing, downloads from the coordinator's model
    ///   distribution endpoint.
    ///
    /// - Parameters:
    ///   - modelId: Identifier of the model (e.g. `"Qwen2.5-7B"`).
    ///   - expectedPath: Local path where the model files should reside.
    ///   - coordinatorHost: Hostname or IP of the coordinator node.
    ///   - coordinatorPort: Admin API port on the coordinator.
    /// - Returns: The `expectedPath` if the model is available or was successfully downloaded.
    /// - Throws: ``WeightDistributorError`` if the coordinator is unreachable or the
    ///   download fails.
    public func ensureModelAvailable(
        modelId: String,
        expectedPath: String,
        coordinatorHost: String,
        coordinatorPort: Int
    ) async throws -> String {
        // Path A: file already exists locally.
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: expectedPath, isDirectory: &isDir), isDir.boolValue {
            logger.info("Model \(modelId) already available at \(expectedPath)")
            return expectedPath
        }

        logger.info("Model \(modelId) not found at \(expectedPath) — downloading from coordinator")

        // Path B: download from coordinator.
        return try await downloadFromCoordinator(
            modelId: modelId,
            to: expectedPath,
            host: coordinatorHost,
            port: coordinatorPort
        )
    }

    /// Download model weights from the coordinator.
    ///
    /// Streams the download in 1 MB chunks, updating ``activeDownloads`` as
    /// data arrives. Writes to a temporary file first, then atomically moves
    /// to the target path on success.
    ///
    /// - Parameters:
    ///   - modelId: Identifier of the model to download.
    ///   - to targetPath: Destination directory for the model files.
    ///   - host: Coordinator hostname or IP.
    ///   - port: Coordinator admin API port.
    /// - Returns: The `targetPath` on successful download.
    /// - Throws: ``WeightDistributorError`` on network or I/O failures.
    public func downloadFromCoordinator(
        modelId: String,
        to targetPath: String,
        host: String,
        port: Int
    ) async throws -> String {
        let urlString = "http://\(host):\(port)/admin/api/cluster/models/\(modelId)/download"
        guard let url = URL(string: urlString) else {
            throw WeightDistributorError.coordinatorUnavailable
        }

        // Register active download entry.
        queue.sync {
            activeDownloads[modelId] = DownloadProgress(
                modelId: modelId,
                bytesDownloaded: 0,
                totalBytes: 0,
                isComplete: false
            )
        }

        // Ensure cleanup on all exit paths.
        defer {
            _ = queue.sync {
                activeDownloads.removeValue(forKey: modelId)
            }
        }

        logger.info("Starting download of \(modelId) from \(host):\(port)")

        // Perform streaming download.
        do {
            _ = try await streamDownload(url: url, modelId: modelId, targetPath: targetPath)
        } catch let error as WeightDistributorError {
            throw error
        } catch {
            throw WeightDistributorError.downloadFailed(error.localizedDescription)
        }

        // Mark download complete before defer removes it.
        queue.sync {
            activeDownloads[modelId]?.isComplete = true
        }

        logger.info("Download of \(modelId) complete — saved to \(targetPath)")
        return targetPath
    }

    /// Get the current download progress for a model, if one is active.
    ///
    /// Used by the admin API to report download status to the coordinator.
    ///
    /// - Parameter modelId: The model identifier to look up.
    /// - Returns: Current ``DownloadProgress``, or `nil` if no download is active.
    public func syncStatus(modelId: String) -> DownloadProgress? {
        queue.sync {
            activeDownloads[modelId]
        }
    }

    // MARK: - Private: Streaming download

    /// Stream a download from the coordinator, writing chunks to a temporary file.
    ///
    /// On completion, atomically moves the temp file to the target directory.
    private func streamDownload(
        url: URL,
        modelId: String,
        targetPath: String
    ) async throws -> URL {
        let tempDir = NSTemporaryDirectory()
        let tempFileURL = URL(fileURLWithPath: tempDir)
            .appendingPathComponent("novamlx-download-\(modelId)-\(UUID().uuidString).tmp")

        // Create target directory if needed.
        let targetURL = URL(fileURLWithPath: targetPath)
        try FileManager.default.createDirectory(
            at: targetURL,
            withIntermediateDirectories: true
        )

        // Stream the response.
        let (asyncBytes, response) = try await URLSession.shared.bytes(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw WeightDistributorError.coordinatorUnavailable
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if httpResponse.statusCode == 404 {
                throw WeightDistributorError.modelNotFound(modelId)
            }
            throw WeightDistributorError.downloadFailed(
                "Coordinator returned HTTP \(httpResponse.statusCode)"
            )
        }

        // Total size from Content-Length header (may be absent for chunked transfers).
        let totalBytes: UInt64
        if let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
           let parsed = UInt64(contentLength)
        {
            totalBytes = parsed
        } else {
            totalBytes = 0
        }

        // Update total bytes in progress.
        queue.sync {
            activeDownloads[modelId]?.totalBytes = totalBytes
        }

        // Stream to temp file in 1 MB chunks.
        let fileManager = FileManager.default
        fileManager.createFile(atPath: tempFileURL.path, contents: nil)
        let fileHandle = try FileHandle(forWritingTo: tempFileURL)
        defer { try? fileHandle.close() }

        var bytesDownloaded: UInt64 = 0
        let chunkSize = downloadChunkSize
        var buffer = Data(capacity: chunkSize)

        for try await byte in asyncBytes {
            buffer.append(byte)

            if buffer.count >= chunkSize {
                try fileHandle.write(contentsOf: buffer)
                bytesDownloaded += UInt64(buffer.count)
                buffer.removeAll(keepingCapacity: true)

                queue.sync {
                    activeDownloads[modelId]?.bytesDownloaded = bytesDownloaded
                }
            }
        }

        // Flush remaining bytes.
        if !buffer.isEmpty {
            try fileHandle.write(contentsOf: buffer)
            bytesDownloaded += UInt64(buffer.count)
        }

        // Final progress update.
        queue.sync {
            activeDownloads[modelId]?.bytesDownloaded = bytesDownloaded
        }

        logger.debug("Streamed \(bytesDownloaded) bytes for \(modelId)")

        // Atomic move to final location.
        let finalFileURL = targetURL.appendingPathComponent(tempFileURL.lastPathComponent)
        try fileManager.moveItem(at: tempFileURL, to: finalFileURL)

        return finalFileURL
    }
}
