import Foundation
import Testing
@testable import TknetPeer

@Suite("Peer process lock")
struct PeerProcessLockTests {
    private func temporaryLockURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-lock-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("peer.lock")
    }

    @Test("a second acquire on the same path fails")
    func secondAcquireFails() throws {
        let url = try temporaryLockURL()
        let first = try PeerProcessLock.acquire(at: url)
        defer { first.unlock() }
        do {
            let second = try PeerProcessLock.acquire(at: url)
            second.unlock()
            Issue.record("second acquire should fail")
        } catch let error as PeerProcessLockError {
            guard case .alreadyRunning(let pid) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(pid == Int32(ProcessInfo.processInfo.processIdentifier))
            #expect(error.description.contains("already running"))
        }
    }

    @Test("unlock lets the next acquire succeed")
    func unlockThenAcquire() throws {
        let url = try temporaryLockURL()
        let first = try PeerProcessLock.acquire(at: url)
        first.unlock()
        let second = try PeerProcessLock.acquire(at: url)
        second.unlock()
    }

    @Test("a dead pid is replaced")
    func deadPIDIsReplaced() throws {
        let url = try temporaryLockURL()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sleep")
        process.arguments = ["30"]
        try process.run()
        let pid = process.processIdentifier
        process.terminate()
        process.waitUntilExit()
        try Data("\(pid)\n".utf8).write(to: url)
        let lock = try PeerProcessLock.acquire(at: url)
        lock.unlock()
    }

    @Test("a live foreign pid is not stolen")
    func livePIDIsNotStolen() throws {
        let url = try temporaryLockURL()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sleep")
        process.arguments = ["30"]
        try process.run()
        defer {
            if process.isRunning {
                process.terminate()
                process.waitUntilExit()
            }
        }
        try Data("\(process.processIdentifier)\n".utf8).write(to: url)
        do {
            let stolen = try PeerProcessLock.acquire(at: url)
            stolen.unlock()
            Issue.record("live pid was stolen")
        } catch let error as PeerProcessLockError {
            guard case .alreadyRunning(let pid) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(pid == process.processIdentifier)
        }
        process.terminate()
        process.waitUntilExit()
        let lock = try PeerProcessLock.acquire(at: url)
        lock.unlock()
    }

    @Test("default lock file sits beside the CLI config")
    func defaultURLMatchesConfig() {
        let config = URL(fileURLWithPath: CLIConfig.expandTilde(CLIConfig.defaultConfigPath))
        #expect(PeerProcessLock.defaultURL == PeerProcessLock.url(besideConfig: config))
        #expect(PeerProcessLock.defaultURL.lastPathComponent == "peer.lock")
    }
}
