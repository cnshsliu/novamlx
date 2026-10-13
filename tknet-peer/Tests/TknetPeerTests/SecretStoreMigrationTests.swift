import Foundation
import Testing
@testable import TknetPeer

@Suite("Secret store migration")
struct SecretStoreMigrationTests {
    private final class MemorySecretStore: SecretStore, @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String: String] = [:]

        func save(_ secret: String, for ref: String) {
            lock.lock()
            defer { lock.unlock() }
            values[ref] = secret
        }

        func load(_ ref: String) -> String? {
            lock.lock()
            defer { lock.unlock() }
            return values[ref]
        }

        func delete(_ ref: String) {
            lock.lock()
            defer { lock.unlock() }
            values[ref] = nil
        }
    }

    private func makePair() -> (dir: URL, file: FileSecretStore, primary: MemorySecretStore, mirror: MirroringSecretStore) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-mirror-\(UUID().uuidString)", isDirectory: true)
        let file = FileSecretStore(directory: dir)
        let primary = MemorySecretStore()
        let mirror = MirroringSecretStore(primary: primary, file: file)
        return (dir, file, primary, mirror)
    }

    @Test("existing file secrets move into the primary and stay readable by the CLI")
    func migratesFileWithoutDeletingIt() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        pair.file.save("conn-key", for: "peer/token")
        pair.file.save("conn-key", for: "peer/token/tknet.ai")
        pair.file.save("sk-member", for: "src/lan-box")

        try pair.mirror.reconcile()

        #expect(try pair.mirror.load("peer/token") == "conn-key")
        #expect(try pair.mirror.load("peer/token/tknet.ai") == "conn-key")
        #expect(try pair.mirror.load("src/lan-box") == "sk-member")
        #expect(pair.file.load("peer/token") == "conn-key")
        #expect(pair.file.load("src/lan-box") == "sk-member")
        #expect(pair.file.snapshot()[MirroringSecretStore.syncStampRef] == nil)

        let cli = FileSecretStore(directory: pair.dir)
        #expect(cli.load("peer/token") == "conn-key")
        let raw = try Data(contentsOf: cli.fileURL)
        let decoded = try JSONDecoder().decode([String: String].self, from: raw)
        #expect(decoded["peer/token"] == "conn-key")
        #expect(decoded["src/lan-box"] == "sk-member")
        let mode = try FileManager.default.attributesOfItem(atPath: cli.fileURL.path)[.posixPermissions] as? Int
        #expect(mode == 0o600)
    }

    @Test("app writes update the file the CLI reads")
    func saveMirrorsToFile() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        try pair.mirror.reconcile()
        try pair.mirror.save("from-app", for: "peer/token")
        try pair.mirror.save("sk-new", for: "src/box")

        #expect(pair.file.load("peer/token") == "from-app")
        #expect(pair.file.load("src/box") == "sk-new")
        #expect(pair.primary.load("peer/token") == "from-app")
        #expect(pair.file.snapshot().keys.contains(MirroringSecretStore.syncStampRef) == false)

        let cli = FileSecretStore(directory: pair.dir)
        #expect(cli.load("peer/token") == "from-app")
        #expect(cli.load("src/box") == "sk-new")
    }

    @Test("a newer CLI file replaces the primary and keeps primary-only refs")
    func cliWriteWinsOnNextReconcile() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        try pair.mirror.save("from-app", for: "peer/token")
        pair.primary.save("session-only", for: "user/session/tknet.ai")

        pair.file.save("from-cli", for: "peer/token")
        try pair.mirror.reconcile()

        #expect(try pair.mirror.load("peer/token") == "from-cli")
        #expect(pair.primary.load("user/session/tknet.ai") == "session-only")
        #expect(pair.file.load("peer/token") == "from-cli")
    }

    @Test("unchanged file does not clobber a primary value")
    func matchingFileLeavesPrimary() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        try pair.mirror.save("from-app", for: "peer/token")
        pair.primary.save("rotated-in-keychain", for: "peer/token")

        try pair.mirror.reconcile()

        #expect(try pair.mirror.load("peer/token") == "rotated-in-keychain")
        #expect(pair.file.load("peer/token") == "from-app")
    }

    @Test("delete removes the ref from both stores")
    func deleteBoth() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        try pair.mirror.save("conn-key", for: "peer/token")
        try pair.mirror.save("sk-member", for: "src/box")
        try pair.mirror.delete("src/box")

        #expect(try pair.mirror.load("src/box") == nil)
        #expect(pair.file.load("src/box") == nil)
        #expect(pair.primary.load("src/box") == nil)
        #expect(pair.file.load("peer/token") == "conn-key")
        #expect(try pair.mirror.load("peer/token") == "conn-key")
    }

    @Test("load promotes a file secret when the primary has not reconciled")
    func loadPromotesFile() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        pair.file.save("from-cli", for: "peer/token")

        #expect(try pair.mirror.load("peer/token") == "from-cli")
        #expect(pair.primary.load("peer/token") == "from-cli")
        #expect(pair.file.load("peer/token") == "from-cli")
    }

    @Test("account session written by the app is in both stores")
    func sessionIsMirrored() throws {
        let pair = makePair()
        defer { try? FileManager.default.removeItem(at: pair.dir) }
        try pair.mirror.save("sess-1", for: "user/session/tknet.ai")
        #expect(pair.primary.load("user/session/tknet.ai") == "sess-1")
        #expect(pair.file.load("user/session/tknet.ai") == "sess-1")
        #expect(pair.file.snapshot()[MirroringSecretStore.syncStampRef] == nil)
    }

#if os(macOS)
    @Test("keychain secret store round-trips when the login keychain accepts items")
    func keychainRoundTrip() throws {
        let service = "ai.tknet.peer.tests.\(UUID().uuidString)"
        let store = KeychainSecretStore(service: service)
        let ref = "peer/token"
        do {
            try store.save("tok-test", for: ref)
        } catch {
            return
        }
        defer { try? store.delete(ref) }
        #expect(try store.load(ref) == "tok-test")
        try store.save("tok-next", for: ref)
        #expect(try store.load(ref) == "tok-next")
        try store.delete(ref)
        #expect(try store.load(ref) == nil)
    }
#endif
}
