import Foundation
#if os(macOS)
import LocalAuthentication
import Security

public enum KeychainSecretError: Error, Equatable, LocalizedError {
    case encoding
    case status(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .encoding:
            return "Keychain value was not valid text."
        case .status(let code):
            return "Keychain error \(code)."
        }
    }
}

/// Login-keychain backend for the Mac app. Service `ai.tknet.peer`.
/// Values are never logged. Items stay on this Mac (not iCloud-synced).
public struct KeychainSecretStore: SecretStore {
    public static let peerService = "ai.tknet.peer"

    public let service: String

    public init(service: String = KeychainSecretStore.peerService) {
        self.service = service
    }

    public func save(_ secret: String, for ref: String) throws {
        SecKeychainSetUserInteractionAllowed(false)
        guard let data = secret.data(using: .utf8) else {
            throw KeychainSecretError.encoding
        }
        let query = baseQuery(ref)
        let update = [kSecValueData as String: data] as CFDictionary
        let status = SecItemUpdate(query as CFDictionary, update)
        if status == errSecSuccess { return }
        if status != errSecItemNotFound {
            throw KeychainSecretError.status(status)
        }
        var add = query
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        add[kSecAttrSynchronizable as String] = false
        add[kSecAttrLabel as String] = "Tknet Peer"
        let added = SecItemAdd(add as CFDictionary, nil)
        guard added == errSecSuccess else {
            throw KeychainSecretError.status(added)
        }
    }

    public func load(_ ref: String) throws -> String? {
        SecKeychainSetUserInteractionAllowed(false)
        var query = baseQuery(ref)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else {
            throw KeychainSecretError.status(status)
        }
        guard let data = result as? Data, let text = String(data: data, encoding: .utf8) else {
            throw KeychainSecretError.encoding
        }
        return text
    }

    public func delete(_ ref: String) throws {
        SecKeychainSetUserInteractionAllowed(false)
        let status = SecItemDelete(baseQuery(ref) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainSecretError.status(status)
        }
    }

    private func baseQuery(_ ref: String) -> [String: Any] {
        let context = LAContext()
        context.interactionNotAllowed = true
        return [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: ref,
            // A prompt during launch never appears and the call never returns.
            kSecUseAuthenticationContext as String: context,
        ]
    }
}
#endif
