import Foundation

public final class ConversationStore: @unchecked Sendable {
    public static let shared = ConversationStore()

    private var conversations: [String: String] = [:]
    private let lock = NSLock()

    private init() {}

    public func record(conversationId: String, responseId: String) {
        lock.lock()
        defer { lock.unlock() }
        conversations[conversationId] = responseId
    }

    public func lastResponseId(for conversationId: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return conversations[conversationId]
    }

    public func delete(conversationId: String) {
        lock.lock()
        defer { lock.unlock() }
        conversations.removeValue(forKey: conversationId)
    }
}
