import Foundation

/// Where source API keys live. Mac: Keychain-backed (provided by the app in
/// Task 11). CLI: FileSecretStore. Implementations MUST NOT log values.
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
}
