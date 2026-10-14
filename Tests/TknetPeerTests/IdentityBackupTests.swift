import Foundation
import Testing
@testable import TknetPeer

@Suite("Identity backup")
struct IdentityBackupTests {
    private func makeLive() throws -> (dir: URL, config: URL, secrets: URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-id-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-live"
        let configURL = dir.appendingPathComponent("peer.json")
        try ConfigStore.save(config, to: configURL)
        let store = FileSecretStore(directory: dir.appendingPathComponent("secrets"))
        store.save("conn-key", for: "peer/token")
        store.save("sk-member", for: "src/m1")
        return (dir, configURL, store.fileURL)
    }

    private func permissions(at url: URL) throws -> Int {
        let value = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
        return try #require(value)
    }

    @Test("export writes both files and keeps the member key")
    func exportRoundTrip() throws {
        let live = try makeLive()
        defer { try? FileManager.default.removeItem(at: live.dir) }
        let usb = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-usb-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: usb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: usb) }

        let folder = try IdentityBackup.export(configFile: live.config, secretsFile: live.secrets, to: usb)
        #expect(folder.lastPathComponent == IdentityBackup.folderName)
        #expect(try permissions(at: folder) & 0o777 == 0o700)
        let backedConfig = folder.appendingPathComponent("peer.json")
        let backedSecrets = folder.appendingPathComponent("secrets/secrets.json")
        #expect(try permissions(at: backedConfig) == 0o600)
        #expect(try permissions(at: backedSecrets) == 0o600)
        #expect(try ConfigStore.load(from: backedConfig).peerId == "peer-live")
        let secrets = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: backedSecrets))
        #expect(secrets["peer/token"] == "conn-key")
        #expect(secrets["src/m1"] == "sk-member")

        let again = try IdentityBackup.export(configFile: live.config, secretsFile: live.secrets, to: folder)
        #expect(again.standardizedFileURL == folder.standardizedFileURL)
        #expect(!FileManager.default.fileExists(
            atPath: folder.appendingPathComponent(IdentityBackup.folderName).path))
    }

    @Test("export refuses an unregistered peer, a missing key, and a copy inside the live directory")
    func exportGuards() throws {
        let live = try makeLive()
        defer { try? FileManager.default.removeItem(at: live.dir) }
        let usb = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-usb-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: usb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: usb) }

        var blank = PeerConfig.defaultConfig()
        blank.peerId = nil
        let blankURL = live.dir.appendingPathComponent("blank.json")
        try ConfigStore.save(blank, to: blankURL)
        #expect(throws: IdentityBackup.Failure.notRegistered) {
            try IdentityBackup.export(configFile: blankURL, secretsFile: live.secrets, to: usb)
        }

        let emptySecrets = live.dir.appendingPathComponent("empty-secrets.json")
        try Data("{}".utf8).write(to: emptySecrets)
        #expect(throws: IdentityBackup.Failure.missingSecrets) {
            try IdentityBackup.export(configFile: live.config, secretsFile: emptySecrets, to: usb)
        }

        #expect(throws: IdentityBackup.Failure.destinationInsideIdentity) {
            try IdentityBackup.export(configFile: live.config, secretsFile: live.secrets, to: live.dir)
        }
    }

    @Test("restore accepts the backup folder or its parent and replaces the live identity")
    func restoreReplaces() throws {
        let live = try makeLive()
        defer { try? FileManager.default.removeItem(at: live.dir) }
        let usb = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-usb-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: usb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: usb) }
        let folder = try IdentityBackup.export(configFile: live.config, secretsFile: live.secrets, to: usb)

        var other = try ConfigStore.load(from: live.config)
        other.peerId = "peer-other"
        try ConfigStore.save(other, to: live.config)
        let store = FileSecretStore(directory: live.secrets.deletingLastPathComponent())
        store.save("other-key", for: "peer/token")

        let fromParent = try IdentityBackup.restore(from: usb, configFile: live.config, secretsFile: live.secrets)
        #expect(fromParent.peerId == "peer-live")
        #expect(store.load("peer/token") == "conn-key")
        #expect(store.load("src/m1") == "sk-member")

        store.save("other-key", for: "peer/token")
        let fromFolder = try IdentityBackup.restore(from: folder, configFile: live.config, secretsFile: live.secrets)
        #expect(fromFolder.peerId == "peer-live")
        #expect(store.load("peer/token") == "conn-key")
    }

    @Test("restore rejects a folder that is not a backup")
    func restoreMissing() throws {
        let live = try makeLive()
        defer { try? FileManager.default.removeItem(at: live.dir) }
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-empty-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(throws: IdentityBackup.Failure.backupNotFound) {
            try IdentityBackup.restore(from: empty, configFile: live.config, secretsFile: live.secrets)
        }
    }
}
