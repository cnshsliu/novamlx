import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

/// A copy of this machine's tknet identity: `peer.json` plus `secrets/secrets.json`.
///
/// The folder is `<destination>/tknet-peer-identity`. Choosing that folder
/// itself writes into it directly. Restore accepts either that folder or its
/// parent. The connection key cannot be reissued, so both files are required.
public enum IdentityBackup {
    public static let folderName = "tknet-peer-identity"

    /// Validated file bytes. Other modules can pass a snapshot back to
    /// `install` without reading the secrets.
    public struct Snapshot: Sendable {
        let configData: Data
        let secretsData: Data
    }

    public enum Failure: Error, Equatable, LocalizedError {
        case notRegistered
        case missingSecrets
        case destinationInsideIdentity
        case destinationNotFolder
        case backupNotFound
        case backupIncomplete
        case backupUnreadable

        public var errorDescription: String? {
            switch self {
            case .notRegistered:
                return "Register this peer before backing it up."
            case .missingSecrets:
                return "No connection key is stored, so there is nothing to back up."
            case .destinationInsideIdentity:
                return "Choose a folder outside ~/.config/tknet-peer."
            case .destinationNotFolder:
                return "The backup destination has to be a folder."
            case .backupNotFound:
                return "That folder has no tknet-peer-identity backup."
            case .backupIncomplete:
                return "The backup is missing the peer id or the connection key."
            case .backupUnreadable:
                return "The backup could not be read."
            }
        }
    }

    /// Copies the live identity into `destination/tknet-peer-identity`.
    /// Returns the folder that now holds the two files.
    public static func export(configFile: URL, secretsFile: URL, to destination: URL) throws -> URL {
        let snapshot = try readLive(configFile: configFile, secretsFile: secretsFile)
        let folder = outputFolder(for: destination)
        let sourceDir = configFile.deletingLastPathComponent()
        if isInside(folder, of: sourceDir) || isInside(sourceDir, of: folder) {
            throw Failure.destinationInsideIdentity
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory), !isDirectory.boolValue {
            throw Failure.destinationNotFolder
        }
        try write(snapshot, configFile: folder.appendingPathComponent("peer.json"), secretsFile: secretsFolderFile(folder))
        return folder
    }

    /// Reads a backup without touching the live files.
    public static func load(from selected: URL) throws -> Snapshot {
        guard let folder = locate(in: selected) else { throw Failure.backupNotFound }
        let configURL = folder.appendingPathComponent("peer.json")
        let secretsURL = secretsFolderFile(folder)
        do {
            let configData = try Data(contentsOf: configURL)
            let secretsData = try Data(contentsOf: secretsURL)
            _ = try validated(configData: configData, secretsData: secretsData)
            return Snapshot(configData: configData, secretsData: secretsData)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.backupUnreadable
        }
    }

    /// Replaces the live identity files. Callers stop the tunnel first.
    @discardableResult
    public static func install(_ snapshot: Snapshot, configFile: URL, secretsFile: URL) throws -> PeerConfig {
        let config = try validated(configData: snapshot.configData, secretsData: snapshot.secretsData)
        try write(snapshot, configFile: configFile, secretsFile: secretsFile)
        return config
    }

    /// Loads a backup and replaces the live files.
    @discardableResult
    public static func restore(from selected: URL, configFile: URL, secretsFile: URL) throws -> PeerConfig {
        try install(load(from: selected), configFile: configFile, secretsFile: secretsFile)
    }

    // MARK: - Layout

    static func outputFolder(for destination: URL) -> URL {
        let dest = destination.standardizedFileURL.resolvingSymlinksInPath()
        if dest.lastPathComponent == folderName { return dest }
        return dest.appendingPathComponent(folderName, isDirectory: true)
    }

    static func locate(in selected: URL) -> URL? {
        let selected = selected.standardizedFileURL
        if isBackup(selected) { return selected }
        let nested = selected.appendingPathComponent(folderName, isDirectory: true)
        if isBackup(nested) { return nested }
        return nil
    }

    // MARK: - Bytes

    private static func readLive(configFile: URL, secretsFile: URL) throws -> Snapshot {
        let configData: Data
        let secretsData: Data
        do {
            configData = try Data(contentsOf: configFile)
            secretsData = try Data(contentsOf: secretsFile)
        } catch {
            throw Failure.missingSecrets
        }
        do {
            _ = try validated(configData: configData, secretsData: secretsData)
        } catch Failure.backupIncomplete {
            let config = try? ConfigStore.load(from: configFile)
            if config?.peerId?.isEmpty != false { throw Failure.notRegistered }
            throw Failure.missingSecrets
        }
        return Snapshot(configData: configData, secretsData: secretsData)
    }

    private static func validated(configData: Data, secretsData: Data) throws -> PeerConfig {
        guard let config = try? decodeConfig(configData),
              let peerId = config.peerId, !peerId.isEmpty else {
            throw Failure.backupIncomplete
        }
        guard let secrets = try? JSONDecoder().decode([String: String].self, from: secretsData),
              hasConnectionKey(secrets) else {
            throw Failure.backupIncomplete
        }
        return config
    }

    private static func decodeConfig(_ data: Data) throws -> PeerConfig {
        try JSONDecoder().decode(PeerConfig.self, from: data)
    }

    /// `peer/token` is what the tunnel sends. Per-server copies use `peer/token/<host>`.
    private static func hasConnectionKey(_ secrets: [String: String]) -> Bool {
        secrets.contains { key, value in
            !value.isEmpty && (key == "peer/token" || key.hasPrefix("peer/token/"))
        }
    }

    private static func write(_ snapshot: Snapshot, configFile: URL, secretsFile: URL) throws {
        try replace(snapshot.secretsData, at: secretsFile)
        try replace(snapshot.configData, at: configFile)
    }

    /// Writes `data` to `url`, creating any missing parent (for example
    /// `secrets/` under a new backup folder). The directory is `0700` and the
    /// file is `0600`.
    ///
    /// The bytes are staged next to the destination and published with
    /// `rename(2)`. Linux Foundation's `replaceItemAt` renames the destination
    /// onto the temporary file, then fails with `NSCocoaErrorDomain` Code=4.
    private static func replace(_ data: Data, at url: URL) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        let temporary = directory.appendingPathComponent(
            ".\(url.lastPathComponent).\(UUID().uuidString).tmp",
            isDirectory: false
        )
        do {
            try data.write(to: temporary)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temporary.path)
            try renameReplacing(temporary, to: url)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
    }

    /// `rename(2)` replaces an existing destination on the same volume and
    /// keeps the temporary file's mode. Windows `rename` will not overwrite,
    /// so that path removes the destination first.
    private static func renameReplacing(_ source: URL, to destination: URL) throws {
        #if os(Windows)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: source, to: destination)
        #else
        let outcome: (rc: Int32, err: Int32) = source.withUnsafeFileSystemRepresentation { src in
            destination.withUnsafeFileSystemRepresentation { dst in
                guard let src, let dst else { return (-1, EINVAL) }
                let rc = platformRename(src, dst)
                return (rc, rc == 0 ? 0 : errno)
            }
        }
        guard outcome.rc == 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(outcome.err),
                userInfo: [NSFilePathErrorKey: destination.path]
            )
        }
        #endif
    }

    #if !os(Windows)
    private static func platformRename(
        _ source: UnsafePointer<CChar>,
        _ destination: UnsafePointer<CChar>
    ) -> Int32 {
        #if canImport(Darwin)
        Darwin.rename(source, destination)
        #elseif canImport(Glibc)
        Glibc.rename(source, destination)
        #else
        Musl.rename(source, destination)
        #endif
    }
    #endif

    private static func secretsFolderFile(_ folder: URL) -> URL {
        folder.appendingPathComponent("secrets", isDirectory: true).appendingPathComponent("secrets.json")
    }

    private static func isBackup(_ folder: URL) -> Bool {
        FileManager.default.fileExists(atPath: folder.appendingPathComponent("peer.json").path)
            && FileManager.default.fileExists(atPath: secretsFolderFile(folder).path)
    }

    private static func isInside(_ child: URL, of parent: URL) -> Bool {
        let childPath = child.standardizedFileURL.resolvingSymlinksInPath().path
        let parentPath = parent.standardizedFileURL.resolvingSymlinksInPath().path
        if childPath == parentPath { return true }
        let prefix = parentPath.hasSuffix("/") ? parentPath : parentPath + "/"
        return childPath.hasPrefix(prefix)
    }
}
