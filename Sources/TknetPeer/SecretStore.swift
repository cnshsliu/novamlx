import Foundation

/// Where source API keys and the peer connection key live.
///
/// The CLI uses `FileSecretStore` under `~/.config/tknet-peer/secrets`.
/// The Mac app uses `KeychainSecretStore` as its primary and
/// `MirroringSecretStore` to keep that same file in sync, so `tknet-peer
/// serve` still reads the connection key and member keys. Implementations
/// MUST NOT log values.
public protocol SecretStore: Sendable {
    func save(_ secret: String, for ref: String) throws
    func load(_ ref: String) throws -> String?
    func delete(_ ref: String) throws
}

/// JSON file in a 0700 directory — one key per source ref. CLI default.
public struct FileSecretStore: SecretStore {
    public let fileURL: URL

    public init(directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        self.fileURL = directory.appendingPathComponent("secrets.json")
    }

    private func readAll() -> [String: String] {
        guard let data = try? Data(contentsOf: fileURL),
              let dict = try? JSONDecoder().decode([String: String].self, from: data) else { return [:] }
        return dict
    }

    private func writeAll(_ dict: [String: String]) {
        if let data = try? JSONEncoder().encode(dict) {
            try? data.write(to: fileURL, options: .atomic)
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: fileURL.path)
        }
    }

    public func save(_ secret: String, for ref: String) {
        var all = readAll(); all[ref] = secret; writeAll(all)
    }

    public func load(_ ref: String) -> String? { readAll()[ref] }

    public func delete(_ ref: String) {
        var all = readAll(); all[ref] = nil; writeAll(all)
    }

    /// Every ref in `secrets.json`. Callers must not log the values.
    /// Missing or unreadable files are an empty map so a first launch can migrate.
    public func snapshot() -> [String: String] { readAll() }

    /// Stable id of the current file bytes. `"missing"` when the file is absent.
    /// Used to notice CLI writes that happened while the app was quit.
    func contentFingerprint() -> String {
        guard FileManager.default.fileExists(atPath: fileURL.path),
              let data = try? Data(contentsOf: fileURL) else {
            return "missing"
        }
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in data {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return "fnv64:\(String(hash, radix: 16))"
    }
}

/// No-op store for call sites that never persist a member key.
public struct NullSecretStore: SecretStore {
    public init() {}
    public func save(_ value: String, for ref: String) {}
    public func load(_ ref: String) -> String? { nil }
    public func delete(_ ref: String) {}
}
