import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

public enum PeerProcessLockError: Error, Equatable, Sendable, CustomStringConvertible {
    /// Another live owner holds the lock, or the lock file names a live pid.
    /// `ownerHint` is that pid when the file contained one.
    case alreadyRunning(ownerHint: Int32?)
    case io(String)

    public var description: String {
        switch self {
        case .alreadyRunning(let pid?):
            return "Tknet Peer is already running (pid \(pid)). Stop the other Tknet Peer or `tknet-peer serve` first."
        case .alreadyRunning(nil):
            return "Tknet Peer is already running. Stop the other Tknet Peer or `tknet-peer serve` first."
        case .io(let message):
            return message
        }
    }
}

/// Exclusive lock shared by `tknet-peer serve` and the Peer menu-bar app.
///
/// The lock file is `peer.lock` beside the config (default
/// `~/.config/tknet-peer/peer.lock`). Acquire is non-blocking. A dead pid is
/// replaced only when `flock` is free and `kill(pid, 0)` reports the pid is
/// gone (`ESRCH`). A live pid is never stolen, even if that process forgot
/// to hold `flock`. The same process cannot acquire twice: Darwin's `flock`
/// is per-process, so a second open in-process is rejected via an in-memory
/// set as well as the pid check.
public final class PeerProcessLock: @unchecked Sendable {
    public static var defaultURL: URL {
        let config = URL(fileURLWithPath: CLIConfig.expandTilde(CLIConfig.defaultConfigPath))
        return url(besideConfig: config)
    }

    public static func url(besideConfig configURL: URL) -> URL {
        configURL.deletingLastPathComponent().appendingPathComponent("peer.lock")
    }

    private static let held = PeerLock(initialState: Set<String>())

    private let fd: Int32
    private let path: String
    private let flag = PeerLock(initialState: false)

    private init(fd: Int32, path: String) {
        self.fd = fd
        self.path = path
    }

    public static func acquire(at url: URL = defaultURL) throws -> PeerProcessLock {
        let directory = url.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw PeerProcessLockError.io("Could not create \(directory.path): \(error)")
        }
        let fd = POSIX.open(url.path, O_CREAT | O_RDWR | O_CLOEXEC, mode_t(0o600))
        guard fd >= 0 else {
            throw PeerProcessLockError.io("Could not open \(url.path): \(POSIX.errorString())")
        }
        let existing = POSIX.readPID(fd)
        if POSIX.flockExclusive(fd) != 0 {
            _ = POSIX.close(fd)
            throw PeerProcessLockError.alreadyRunning(ownerHint: existing)
        }
        if let existing, existing != POSIX.currentPID(), POSIX.isAlive(existing) {
            _ = POSIX.flockUnlock(fd)
            _ = POSIX.close(fd)
            throw PeerProcessLockError.alreadyRunning(ownerHint: existing)
        }
        let path = url.standardizedFileURL.path
        let inserted = held.withLock { paths -> Bool in
            if paths.contains(path) { return false }
            paths.insert(path)
            return true
        }
        if !inserted {
            _ = POSIX.flockUnlock(fd)
            _ = POSIX.close(fd)
            throw PeerProcessLockError.alreadyRunning(ownerHint: POSIX.currentPID())
        }
        POSIX.writePID(fd, POSIX.currentPID())
        return PeerProcessLock(fd: fd, path: path)
    }

    public func unlock() {
        let shouldRelease = flag.withLock { released -> Bool in
            if released { return false }
            released = true
            return true
        }
        guard shouldRelease else { return }
        Self.held.withLock { $0.remove(path) }
        _ = POSIX.flockUnlock(fd)
        _ = POSIX.close(fd)
    }

    deinit { unlock() }
}

#if canImport(Darwin) || canImport(Glibc) || canImport(Musl)
private enum POSIX {
    static func open(_ path: String, _ flags: Int32, _ mode: mode_t) -> Int32 {
        path.withCString { pointer in
            #if canImport(Darwin)
            Darwin.open(pointer, flags, mode)
            #elseif canImport(Glibc)
            Glibc.open(pointer, flags, mode)
            #else
            Musl.open(pointer, flags, mode)
            #endif
        }
    }

    static func close(_ fd: Int32) -> Int32 {
        #if canImport(Darwin)
        Darwin.close(fd)
        #elseif canImport(Glibc)
        Glibc.close(fd)
        #else
        Musl.close(fd)
        #endif
    }

    static func flockExclusive(_ fd: Int32) -> Int32 {
        #if canImport(Darwin)
        // Darwin.flock is the `struct flock` type; call the C function unqualified.
        return flock(fd, LOCK_EX | LOCK_NB)
        #elseif canImport(Glibc)
        return Glibc.flock(fd, LOCK_EX | LOCK_NB)
        #else
        return Musl.flock(fd, LOCK_EX | LOCK_NB)
        #endif
    }

    static func flockUnlock(_ fd: Int32) -> Int32 {
        #if canImport(Darwin)
        return flock(fd, LOCK_UN)
        #elseif canImport(Glibc)
        return Glibc.flock(fd, LOCK_UN)
        #else
        return Musl.flock(fd, LOCK_UN)
        #endif
    }

    static func currentPID() -> Int32 {
        #if canImport(Darwin)
        Darwin.getpid()
        #elseif canImport(Glibc)
        Glibc.getpid()
        #else
        Musl.getpid()
        #endif
    }

    static func isAlive(_ pid: Int32) -> Bool {
        if pid <= 0 { return false }
        let result: Int32
        #if canImport(Darwin)
        result = Darwin.kill(pid, 0)
        #elseif canImport(Glibc)
        result = Glibc.kill(pid, 0)
        #else
        result = Musl.kill(pid, 0)
        #endif
        if result == 0 { return true }
        // ESRCH: no such process. EPERM means it exists but we cannot signal it.
        return errno != ESRCH
    }

    static func readPID(_ fd: Int32) -> Int32? {
        var buffer = [UInt8](repeating: 0, count: 64)
        let count: Int = buffer.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return 0 }
            #if canImport(Darwin)
            _ = Darwin.lseek(fd, 0, SEEK_SET)
            return Int(Darwin.read(fd, base, 63))
            #elseif canImport(Glibc)
            _ = Glibc.lseek(fd, 0, SEEK_SET)
            return Int(Glibc.read(fd, base, 63))
            #else
            _ = Musl.lseek(fd, 0, SEEK_SET)
            return Int(Musl.read(fd, base, 63))
            #endif
        }
        guard count > 0 else { return nil }
        let text = String(decoding: buffer.prefix(count), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return Int32(text)
    }

    static func writePID(_ fd: Int32, _ pid: Int32) {
        let text = Data("\(pid)\n".utf8)
        #if canImport(Darwin)
        _ = Darwin.ftruncate(fd, 0)
        _ = Darwin.lseek(fd, 0, SEEK_SET)
        #elseif canImport(Glibc)
        _ = Glibc.ftruncate(fd, 0)
        _ = Glibc.lseek(fd, 0, SEEK_SET)
        #else
        _ = Musl.ftruncate(fd, 0)
        _ = Musl.lseek(fd, 0, SEEK_SET)
        #endif
        text.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            #if canImport(Darwin)
            _ = Darwin.write(fd, base, raw.count)
            #elseif canImport(Glibc)
            _ = Glibc.write(fd, base, raw.count)
            #else
            _ = Musl.write(fd, base, raw.count)
            #endif
        }
    }

    static func errorString() -> String {
        String(cString: strerror(errno))
    }
}
#endif
