import Foundation
import Security

public enum KeychainError: Error, LocalizedError {
    case duplicateItem
    case itemNotFound
    case unexpectedData
    case unhandledOSStatus(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .duplicateItem: return "Keychain item already exists"
        case .itemNotFound: return "Keychain item not found"
        case .unexpectedData: return "Unexpected data in Keychain"
        case .unhandledOSStatus(let s): return "Keychain error: \(s)"
        }
    }
}

public enum KeychainHelper {

    private static let service = "com.novamlx.ssh-worker"

    // MARK: - SSH Credentials

    public static func saveSSHCredential(host: String, username: String, password: String) throws {
        guard let passwordData = password.data(using: .utf8) else {
            throw KeychainError.unexpectedData
        }

        // Delete existing item first (upsert semantics)
        deleteSSHCredential(host: host)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: host,
            kSecAttrLabel as String: "NovaMLX SSH: \(username)@\(host)",
            kSecValueData as String: passwordData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlocked,
            // Store username as custom attribute for retrieval
            kSecAttrComment as String: username,
        ]

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.unhandledOSStatus(status)
        }
    }

    public static func loadSSHCredential(host: String) throws -> (username: String, password: String)? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: host,
            kSecReturnData as String: true,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status != errSecItemNotFound else { return nil }
        guard status == errSecSuccess else {
            throw KeychainError.unhandledOSStatus(status)
        }

        guard let dict = result as? [String: Any],
              let passwordData = dict[kSecValueData as String] as? Data,
              let password = String(data: passwordData, encoding: .utf8),
              let username = dict[kSecAttrComment as String] as? String
        else {
            throw KeychainError.unexpectedData
        }

        return (username: username, password: password)
    }

    public static func deleteSSHCredential(host: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: host,
        ]
        SecItemDelete(query as CFDictionary)
    }

    public static func hasSSHCredential(host: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: host,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        return status == errSecSuccess
    }
}
