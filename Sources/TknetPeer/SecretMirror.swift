import Foundation

public enum SecretMirrorError: Error, Equatable, LocalizedError {
    /// The shared `secrets.json` did not keep the value. The ref is a name
    /// (`peer/token`, `src/…`), never the secret.
    case fileWriteFailed(String)

    public var errorDescription: String? {
        switch self {
        case .fileWriteFailed(let ref):
            return "Could not write \(ref) to the shared file store."
        }
    }
}

/// Keychain (or any primary) plus the CLI's `FileSecretStore`.
///
/// The Mac app reads the primary. Every connection key and member key is
/// also written to `secrets.json`, which `tknet-peer serve` still opens.
/// On launch, `reconcile()` copies the file into the primary when the file
/// changed while the app was quit (a CLI `setup` or `member` write).
/// Primary-only refs are left in place. The file is never deleted.
/// `meta/…` refs stay on the primary and are not written to the file.
public struct MirroringSecretStore: SecretStore {
    /// Primary ref that records the last file fingerprint. Not a secret.
    public static let syncStampRef = "meta/file-sync-fingerprint"

    private let primary: any SecretStore
    private let file: FileSecretStore

    public init(primary: any SecretStore, file: FileSecretStore) {
        self.primary = primary
        self.file = file
    }

    /// Copy file secrets into the primary when `secrets.json` changed.
    /// Safe to call more than once. Does not remove primary-only refs.
    public func reconcile() throws {
        let fingerprint = file.contentFingerprint()
        let stored = try primary.load(Self.syncStampRef)
        guard stored != fingerprint else { return }
        for (ref, value) in file.snapshot() where !Self.isMeta(ref) {
            if try primary.load(ref) != value {
                try primary.save(value, for: ref)
            }
        }
        try primary.save(file.contentFingerprint(), for: Self.syncStampRef)
    }

    public func save(_ secret: String, for ref: String) throws {
        if Self.isMeta(ref) {
            try primary.save(secret, for: ref)
            return
        }
        file.save(secret, for: ref)
        guard file.load(ref) == secret else {
            throw SecretMirrorError.fileWriteFailed(ref)
        }
        try primary.save(secret, for: ref)
        try primary.save(file.contentFingerprint(), for: Self.syncStampRef)
    }

    public func load(_ ref: String) throws -> String? {
        if let value = try? primary.load(ref) { return value }
        guard !Self.isMeta(ref), let value = file.load(ref) else { return nil }
        try? primary.save(value, for: ref)
        return value
    }

    public func delete(_ ref: String) throws {
        if Self.isMeta(ref) {
            try primary.delete(ref)
            return
        }
        file.delete(ref)
        guard file.load(ref) == nil else {
            throw SecretMirrorError.fileWriteFailed(ref)
        }
        try primary.delete(ref)
        try primary.save(file.contentFingerprint(), for: Self.syncStampRef)
    }

    static func isMeta(_ ref: String) -> Bool {
        ref == syncStampRef || ref.hasPrefix("meta/")
    }
}
