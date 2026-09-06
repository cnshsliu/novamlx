import Darwin
import Foundation
import NovaMLXCore
import NovaMLXUtils

/// One file to fetch via `aria2c`. NovaMLX never streams model bytes itself.
public struct Aria2DownloadFile: Sendable, Equatable {
    public let url: URL
    public let relativePath: String
    public let expectedSize: Int64

    public init(url: URL, relativePath: String, expectedSize: Int64 = 0) {
        self.url = url
        self.relativePath = relativePath
        self.expectedSize = expectedSize
    }
}

public struct Aria2FileProgress: Sendable, Equatable {
    public let relativePath: String
    public let downloadedBytes: Int64
    public let totalBytes: Int64
    public let isComplete: Bool
}

public struct Aria2ProgressSnapshot: Sendable, Equatable {
    public let files: [Aria2FileProgress]
    public let downloadedBytes: Int64
    public let totalBytes: Int64
}

public enum Aria2Error: Error, LocalizedError, Equatable {
    case binaryNotFound
    case failed(exitCode: Int32, message: String)

    public var errorDescription: String? {
        switch self {
        case .binaryNotFound:
            return "aria2c not found. Install it with: brew install aria2"
        case .failed(let code, let message):
            return "aria2c exited \(code): \(message)"
        }
    }
}

/// Spawns the system `aria2c` binary. File listing / registry stay in NovaMLX;
/// the HTTP transfer is entirely aria2's job.
public enum Aria2Downloader {
    public static func locateBinary() -> URL? {
        let fm = FileManager.default
        var candidates: [String] = [
            "/opt/homebrew/bin/aria2c",
            "/usr/local/bin/aria2c",
            "/opt/local/bin/aria2c",
        ]
        if let path = ProcessInfo.processInfo.environment["PATH"] {
            for dir in path.split(separator: ":") {
                candidates.append("\(dir)/aria2c")
            }
        }
        var seen = Set<String>()
        for path in candidates where seen.insert(path).inserted {
            if fm.isExecutableFile(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }

    /// Writes an aria2 `-i` input file. Exposed for tests.
    public static func writeInputList(
        files: [Aria2DownloadFile],
        destination: URL,
        to listURL: URL
    ) throws {
        var body = ""
        for file in files {
            body += file.url.absoluteString
            body += "\n  out=\(file.relativePath)\n  dir=\(destination.path)\n"
        }
        try body.data(using: .utf8)!.write(to: listURL, options: .atomic)
    }

    public static func diskSnapshot(files: [Aria2DownloadFile], destination: URL) -> Aria2ProgressSnapshot {
        let fm = FileManager.default
        var rows: [Aria2FileProgress] = []
        var downloaded: Int64 = 0
        var total: Int64 = 0
        for file in files {
            let dest = destination.appendingPathComponent(file.relativePath)
            let sidecar = URL(fileURLWithPath: dest.path + ".aria2")
            let sizes = sizesOnDisk(at: dest)
            let hasSidecar = fm.fileExists(atPath: sidecar.path)
            let complete = fm.fileExists(atPath: dest.path)
                && !hasSidecar
                && (file.expectedSize == 0 || sizes.logical == file.expectedSize)
            var progressBytes = hasSidecar ? sizes.allocated : sizes.logical
            let expected = file.expectedSize > 0 ? file.expectedSize : progressBytes
            if file.expectedSize > 0 {
                progressBytes = min(progressBytes, file.expectedSize)
            }
            rows.append(Aria2FileProgress(
                relativePath: file.relativePath,
                downloadedBytes: progressBytes,
                totalBytes: expected,
                isComplete: complete
            ))
            downloaded += progressBytes
            total += expected
        }
        return Aria2ProgressSnapshot(files: rows, downloadedBytes: downloaded, totalBytes: total)
    }

    public static func download(
        files: [Aria2DownloadFile],
        destination: URL,
        authorization: String? = nil,
        userAgent: String? = nil,
        onProgress: (@Sendable (Aria2ProgressSnapshot) -> Void)? = nil
    ) async throws {
        guard !files.isEmpty else { return }
        guard let binary = locateBinary() else { throw Aria2Error.binaryNotFound }

        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)
        for file in files {
            let parent = destination.appendingPathComponent(file.relativePath).deletingLastPathComponent()
            try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        }

        let listURL = destination.appendingPathComponent(".aria2.input.txt")
        let logURL = destination.appendingPathComponent(".aria2.log")
        try writeInputList(files: files, destination: destination, to: listURL)
        fm.createFile(atPath: logURL.path, contents: nil)

        var args: [String] = [
            "-c",
            "-x", "16",
            "-s", "16",
            "-j", "6",
            "--max-tries=0",
            "--retry-wait=3",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--summary-interval=0",
            "--console-log-level=error",
            "--quiet=true",
            "--stop-with-process=\(ProcessInfo.processInfo.processIdentifier)",
            "--log=\(logURL.path)",
            "--log-level=notice",
            "--dir=\(destination.path)",
            "-i", listURL.path,
        ]
        if let authorization, !authorization.isEmpty {
            args.append("--header=Authorization: \(authorization)")
        }
        if let userAgent, !userAgent.isEmpty {
            args.append("--user-agent=\(userAgent)")
        }

        NovaMLXLog.info("[aria2] spawning \(binary.path) for \(files.count) files → \(destination.path)")

        let process = Process()
        process.executableURL = binary
        process.arguments = args
        process.currentDirectoryURL = destination
        let logHandle = try FileHandle(forWritingTo: logURL)
        process.standardOutput = logHandle
        process.standardError = logHandle

        let box = ProcessBox(process)
        try box.run()
        defer {
            try? logHandle.close()
            try? fm.removeItem(at: listURL)
        }

        do {
            while box.isRunning {
                try Task.checkCancellation()
                onProgress?(diskSnapshot(files: files, destination: destination))
                try await Task.sleep(for: .milliseconds(400))
            }
        } catch is CancellationError {
            box.terminate()
            throw CancellationError()
        }

        onProgress?(diskSnapshot(files: files, destination: destination))

        let status = box.terminationStatus
        if status != 0 {
            let tail = (try? String(contentsOf: logURL, encoding: .utf8))
                .flatMap { String($0.suffix(1500)) } ?? "see \(logURL.path)"
            throw Aria2Error.failed(exitCode: status, message: tail)
        }

        var missing: [String] = []
        for file in files {
            let dest = destination.appendingPathComponent(file.relativePath)
            guard fm.fileExists(atPath: dest.path) else {
                missing.append(file.relativePath)
                continue
            }
            if file.expectedSize > 0 {
                let actual = Int64(fm.fileSize(at: dest) ?? 0)
                if actual != file.expectedSize {
                    missing.append("\(file.relativePath) [size \(actual) != \(file.expectedSize)]")
                }
            }
        }
        if !missing.isEmpty {
            throw Aria2Error.failed(
                exitCode: status,
                message: "Missing/corrupt: \(missing.prefix(5).joined(separator: ", "))"
            )
        }
        NovaMLXLog.info("[aria2] completed \(files.count) files in \(destination.path)")
    }

    static func sizesOnDisk(at url: URL) -> (logical: Int64, allocated: Int64) {
        let values = try? url.resourceValues(forKeys: [
            .totalFileAllocatedSizeKey,
            .fileAllocatedSizeKey,
            .fileSizeKey,
        ])
        let logical = Int64(values?.fileSize ?? 0)
        let allocated = Int64(values?.totalFileAllocatedSize ?? values?.fileAllocatedSize ?? values?.fileSize ?? 0)
        return (logical, allocated)
    }
}

private final class ProcessBox: @unchecked Sendable {
    private let process: Process
    private let lock = NSLock()

    init(_ process: Process) {
        self.process = process
    }

    func run() throws {
        try process.run()
    }

    var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return process.isRunning
    }

    var terminationStatus: Int32 {
        lock.lock(); defer { lock.unlock() }
        return process.terminationStatus
    }

    func terminate() {
        lock.lock()
        let running = process.isRunning
        let pid = process.processIdentifier
        lock.unlock()
        guard running else { return }
        process.terminate()
        var waited = 0
        while process.isRunning && waited < 20 {
            usleep(100_000)
            waited += 1
        }
        if process.isRunning {
            kill(pid, SIGKILL)
        }
    }
}
