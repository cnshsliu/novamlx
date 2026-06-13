import Foundation
import GRDB

public final class ChatStore: Sendable {
    private let db: DatabasePool

    public init(db: DatabasePool) {
        self.db = db
    }

    public func list(limit: Int = 50, offset: Int = 0) throws -> [ChatRecord] {
        try db.read { db in
            try ChatRecord
                .order(Column("updated_at").desc)
                .limit(limit, offset: offset)
                .fetchAll(db)
        }
    }

    public func get(id: String) throws -> (chat: ChatRecord, messages: [ChatMessageRecord])? {
        try db.read { db in
            guard let chat = try ChatRecord.fetchOne(db, key: id) else { return nil }
            let messages = try ChatMessageRecord
                .filter(Column("chat_id") == id)
                .order(Column("sort_order"))
                .fetchAll(db)
            return (chat, messages)
        }
    }

    public func create(id: String, model: String, title: String?, systemPrompt: String?) throws -> ChatRecord {
        let now = Date()
        let record = ChatRecord(
            id: id, title: title, model: model,
            systemPrompt: systemPrompt, createdAt: now, updatedAt: now
        )
        try db.write { db in
            try record.insert(db)
        }
        return record
    }

    public func addMessage(chatId: String, role: String, content: String?, thinkingContent: String?) throws -> ChatMessageRecord {
        let now = Date()
        let sortOrder = try db.read { db in
            (try ChatMessageRecord
                .filter(Column("chat_id") == chatId)
                .select(max(Column("sort_order")))
                .fetchOne(db) as Int?) ?? -1
        } + 1

        let record = ChatMessageRecord(
            id: UUID().uuidString, chatId: chatId, role: role,
            content: content, thinkingContent: thinkingContent,
            createdAt: now, sortOrder: sortOrder
        )
        try db.write { db in
            try record.insert(db)
            try db.execute(sql: "UPDATE chats SET updated_at = ? WHERE id = ?", arguments: [now, chatId])
        }
        return record
    }

    public func updateTitle(id: String, title: String) throws {
        try db.write { db in
            try db.execute(sql: "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?", arguments: [title, Date(), id])
        }
    }

    public func delete(id: String) throws {
        try db.write { db in
            try ChatRecord.deleteOne(db, key: id)
        }
    }

    public func search(query: String, limit: Int = 20) throws -> [ChatRecord] {
        try db.read { db in
            let pattern = "%\(query)%"
            return try ChatRecord
                .filter(sql: "id IN (SELECT chat_id FROM chat_messages WHERE content LIKE ?)", arguments: [pattern])
                .order(Column("updated_at").desc)
                .limit(limit)
                .fetchAll(db)
        }
    }

    public func count() throws -> Int {
        try db.read { db in
            try ChatRecord.fetchCount(db)
        }
    }

    /// Insert (or replace) a chat record and its messages atomically.
    /// Used by the legacy JSON importer and the cutover ChatHistoryStore.save.
    public func upsertChat(_ chat: ChatRecord, messages: [ChatMessageRecord]) throws {
        try db.write { db in
            try chat.save(db)
            // Remove existing messages for this chat before reinserting so
            // the operation is a true replace (idempotent on retry).
            _ = try ChatMessageRecord
                .filter(Column("chat_id") == chat.id)
                .deleteAll(db)
            for msg in messages {
                try msg.insert(db)
            }
        }
    }

    /// All chats ordered by updated_at desc (no pagination).
    public func listAll() throws -> [ChatRecord] {
        try db.read { db in
            try ChatRecord.order(Column("updated_at").desc).fetchAll(db)
        }
    }

    /// All messages for a chat, ordered by sort_order.
    public func messages(chatId: String) throws -> [ChatMessageRecord] {
        try db.read { db in
            try ChatMessageRecord
                .filter(Column("chat_id") == chatId)
                .order(Column("sort_order"))
                .fetchAll(db)
        }
    }
}
