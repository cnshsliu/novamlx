import Foundation

/// Process execution that can NEVER hang (Lucas: 卡死在任何情况下都绝对不可接受).
///
/// The classic `waitUntilExit() → readDataToEndOfFile()` shape deadlocks when
/// the child's output exceeds the 64KB pipe buffer: the child blocks writing
/// (stuck in exit's `__sflush` forever) while the parent blocks waiting for
/// exit. Sixteen zombie `arp -a` processes and ~13h of 97% CPU in production
/// were exactly this (2026-10-05, ClusterPageView).
///
/// This helper makes the safe shape the DEFAULT for the whole codebase:
/// - stdout (and stderr when captured) drain on concurrent side queues
/// - the exit wait is BOUNDED — on deadline the child is terminated
/// - a run that times out or fails degrades to an error/empty result, never a hang
public enum SafeProcess {

    public struct Result: Sendable {
        public let status: Int32
        public let stdout: Data
        public let stderr: Data
        public var stdoutText: String { String(data: stdout, encoding: .utf8) ?? "" }
        public var stderrText: String { String(data: stderr, encoding: .utf8) ?? "" }
        public var succeeded: Bool { status == 0 }
        /// True when the child was killed at the deadline instead of exiting.
        public let timedOut: Bool
    }

    public enum Failure: Error, LocalizedError {
        case launchFailed(String)
        case timedOut(Double)

        public var errorDescription: String? {
            switch self {
            case .launchFailed(let why): return "process launch failed: \(why)"
            case .timedOut(let seconds): return "process did not exit within \(seconds)s — terminated"
            }
        }
    }

    /// Run a process to completion with both pipes concurrently drained and
    /// a bounded wait. `deadline` caps the TOTAL wall time; on expiry the
    /// child is terminated and `.timedOut` is thrown (unless `allowTimeout`,
    /// in which case the partial result is returned with `timedOut: true`).
    ///
    /// - Parameters:
    ///   - url: executable path
    ///   - arguments: argv
    ///   - currentDirectory: working directory (nil = inherit)
    ///   - environment: nil = inherit; set to override
    ///   - mergeStderrIntoStdout: route stderr to the same pipe as stdout
    ///     (single combined output — use for tools whose messages split
    ///     across both streams, e.g. git)
    ///   - deadline: hard wall-time cap in seconds
    ///   - allowTimeout: return a partial result on deadline instead of throwing
    @discardableResult
    public static func run(
        _ url: String,
        arguments: [String],
        currentDirectory: URL? = nil,
        environment: [String: String]? = nil,
        mergeStderrIntoStdout: Bool = false,
        deadline: TimeInterval = 60,
        allowTimeout: Bool = false
    ) throws -> Result {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: url)
        process.arguments = arguments
        if let currentDirectory { process.currentDirectoryURL = currentDirectory }
        if let environment { process.environment = environment }

        let outPipe = Pipe()
        process.standardOutput = outPipe
        if mergeStderrIntoStdout {
            process.standardError = outPipe
        } else {
            process.standardError = Pipe()
        }
        let errPipe = process.standardError as? Pipe

        // Concurrent drains — this is what makes large outputs safe.
        let outBox = DataBox()
        let errBox = DataBox()
        let drain = DispatchQueue(label: "nova.safeproc.drain", qos: .utility)
        drain.async { outBox.set(outPipe.fileHandleForReading.readDataToEndOfFile()) }
        if let errPipe {
            drain.async { errBox.set(errPipe.fileHandleForReading.readDataToEndOfFile()) }
        }

        do {
            try process.run()
        } catch {
            throw Failure.launchFailed(error.localizedDescription)
        }

        // Bounded wait on a side queue; terminate the child at the deadline.
        let exited = SignalBox()
        let wait = DispatchQueue(label: "nova.safeproc.wait", qos: .utility)
        wait.async {
            process.waitUntilExit()
            exited.signal()
        }
        let finished = exited.wait(deadline: deadline)
        var timedOut = false
        if !finished {
            timedOut = true
            process.terminate()
            // Give it a beat to die on SIGTERM, then force-kill.
            if !exited.wait(deadline: 2) { kill(process.processIdentifier, SIGKILL) }
            NovaMLXLogSafe.warn("SafeProcess[\(url)] exceeded \(deadline)s — terminated")
            if !allowTimeout { throw Failure.timedOut(deadline) }
        }

        // The child is gone; EOF has arrived on both pipes. Bounded take so
        // even a pathological kernel state cannot wedge us.
        let out = outBox.take(timeout: 5) ?? Data()
        let err = mergeStderrIntoStdout ? Data() : (errBox.take(timeout: 5) ?? Data())
        try? outPipe.fileHandleForReading.close()
        if let errPipe { try? errPipe.fileHandleForReading.close() }

        return Result(status: process.terminationStatus, stdout: out, stderr: err, timedOut: timedOut)
    }

    /// Convenience: success-or-nil for probe-style calls (`which`, `say -v ?`).
    public static func runForText(
        _ url: String,
        arguments: [String],
        currentDirectory: URL? = nil,
        deadline: TimeInterval = 60
    ) -> String? {
        guard let result = try? run(url, arguments: arguments, currentDirectory: currentDirectory, deadline: deadline),
              result.succeeded, !result.timedOut else { return nil }
        return result.stdoutText
    }
}

/// Signal-once mailbox used by the bounded wait.
private final class SignalBox: @unchecked Sendable {
    private let cond = NSCondition()
    private var signaled = false
    func signal() {
        cond.lock(); signaled = true; cond.broadcast(); cond.unlock()
    }
    /// Returns true if signaled before the deadline.
    func wait(deadline: TimeInterval) -> Bool {
        cond.lock(); defer { cond.unlock() }
        if signaled { return true }
        return cond.wait(until: Date().addingTimeInterval(deadline))
    }
}

/// One-shot Data mailbox for the pipe drains.
private final class DataBox: @unchecked Sendable {
    private let cond = NSCondition()
    private var value: Data?
    func set(_ v: Data) {
        cond.lock(); value = v; cond.broadcast(); cond.unlock()
    }
    func take(timeout: TimeInterval) -> Data? {
        cond.lock(); defer { cond.unlock() }
        if value == nil { _ = cond.wait(until: Date().addingTimeInterval(timeout)) }
        return value
    }
}

/// Minimal logging without importing the full logger (this file is in
/// NovaMLXUtils; the real logger lives there too but keep the dependency
/// one-way and light).
private enum NovaMLXLogSafe {
    static func warn(_ message: String) {
        NSLog("[NovaMLX] %@", message)
    }
}
