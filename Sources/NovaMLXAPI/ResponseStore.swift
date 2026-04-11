import Foundation
import Logging

public final class ResponseStore: @unchecked Sendable {
    public static let shared = ResponseStore()

    private var store: [String: OpenAIResponseObject] = [:]
    private let lock = NSLock()

    private let maxEntries = 1024
    private var insertionOrder: [String] = []

    private init() {}

    public func put(_ response: OpenAIResponseObject) {
        lock.lock()
        defer { lock.unlock() }
        store[response.id] = response
        insertionOrder.append(response.id)
        if insertionOrder.count > maxEntries {
            let toRemove = insertionOrder.prefix(insertionOrder.count - maxEntries)
            for id in toRemove {
                store.removeValue(forKey: id)
            }
            insertionOrder.removeFirst(toRemove.count)
        }
    }

    public func get(_ id: String) -> OpenAIResponseObject? {
        lock.lock()
        defer { lock.unlock() }
        return store[id]
    }

    public func delete(_ id: String) {
        lock.lock()
        defer { lock.unlock() }
        store.removeValue(forKey: id)
        insertionOrder.removeAll { $0 == id }
    }
}
