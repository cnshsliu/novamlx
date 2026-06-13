import Foundation
import NovaMLXCore
import NovaMLXDB

/// Persists chat history in SQLite via `NovaDB.shared.chatStore`. The legacy
/// per-chat JSON layout under `~/.nova/chat_history/*.json` is imported once
/// on first access; after import, files are renamed to `.json.migrated`.
public final class ChatHistoryStore: Sendable {
    public static let shared = ChatHistoryStore()

    private let directory: URL
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .millisecondsSince1970
        return d
    }()

    private init(directory: URL = NovaMLXPaths.chatHistoryDir) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        importLegacyChatsIfNeeded()
    }

    // MARK: - Types (kept identical for call-site compatibility)

    public struct ChatMessage: Codable, Sendable {
        public let role: String
        public let content: String
        public let images: [String]?
        public let thinking: String?
        public let thinkingTime: String?
        public let ts: Date?
    }

    public struct ChatRecord: Codable, Sendable {
        public let id: String
        public let title: String
        public let messages: [ChatMessage]
        public let model: String?
        public let systemPrompt: String?
        public let ts: Date
    }

    public struct ChatSummary: Codable, Sendable {
        public let id: String
        public let title: String
        public let model: String?
        public let ts: Date
        public let messageCount: Int
    }

    // MARK: - CRUD

    public func list() -> [ChatSummary] {
        guard let chats = try? NovaDB.shared.chatStore.listAll() else { return [] }
        return chats.compactMap { chat -> ChatSummary? in
            guard let messages = try? NovaDB.shared.chatStore.messages(chatId: chat.id) else { return nil }
            return ChatSummary(
                id: chat.id,
                title: chat.title ?? "Untitled",
                model: chat.model.isEmpty ? nil : chat.model,
                ts: chat.updatedAt,
                messageCount: messages.count
            )
        }
    }

    public func get(id: String) -> ChatRecord? {
        let pair: (chat: NovaMLXDB.ChatRecord, messages: [NovaMLXDB.ChatMessageRecord])?
        do {
            pair = try NovaDB.shared.chatStore.get(id: id)
        } catch { return nil }
        guard let pair else { return nil }
        let messages = pair.messages.map { rec in
            ChatMessage(
                role: rec.role,
                content: rec.content ?? "",
                images: nil,
                thinking: rec.thinkingContent,
                thinkingTime: nil,
                ts: rec.createdAt
            )
        }
        return ChatRecord(
            id: pair.chat.id,
            title: pair.chat.title ?? "Untitled",
            messages: messages,
            model: pair.chat.model.isEmpty ? nil : pair.chat.model,
            systemPrompt: pair.chat.systemPrompt,
            ts: pair.chat.updatedAt
        )
    }

    public func save(_ record: ChatRecord) throws {
        let chat = NovaMLXDB.ChatRecord(
            id: record.id,
            title: record.title,
            model: record.model ?? "",
            systemPrompt: record.systemPrompt,
            createdAt: record.ts,
            updatedAt: record.ts
        )
        let messages: [NovaMLXDB.ChatMessageRecord] = record.messages.enumerated().map { idx, msg in
            NovaMLXDB.ChatMessageRecord(
                id: UUID().uuidString,
                chatId: record.id,
                role: msg.role,
                content: msg.content,
                thinkingContent: msg.thinking,
                createdAt: msg.ts ?? record.ts,
                sortOrder: idx
            )
        }
        try NovaDB.shared.chatStore.upsertChat(chat, messages: messages)
    }

    public func delete(id: String) throws {
        try NovaDB.shared.chatStore.delete(id: id)
    }

    public func search(query: String) -> [ChatSummary] {
        guard let results = try? NovaDB.shared.chatStore.search(query: query) else { return [] }
        return results.compactMap { chat -> ChatSummary? in
            guard let messages = try? NovaDB.shared.chatStore.messages(chatId: chat.id) else { return nil }
            return ChatSummary(
                id: chat.id,
                title: chat.title ?? "Untitled",
                model: chat.model.isEmpty ? nil : chat.model,
                ts: chat.updatedAt,
                messageCount: messages.count
            )
        }
    }

    // MARK: - Legacy Import

    /// One-shot import of every `~/.nova/chat_history/*.json` into the
    /// SQLite chatStore. Idempotent: skipped when the store already has
    /// rows; on success each file is renamed to `.migrated` so we never
    /// run twice.
    private func importLegacyChatsIfNeeded() {
        // Skip if store already populated — SQLite is source of truth.
        if let count = try? NovaDB.shared.chatStore.count(), count > 0 { return }

        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return }
        let jsonFiles = files.filter { $0.pathExtension == "json" && !$0.lastPathComponent.hasSuffix(".migrated") }
        guard !jsonFiles.isEmpty else { return }

        var imported = 0
        for fileURL in jsonFiles {
            guard let data = try? Data(contentsOf: fileURL),
                  let legacy = try? decoder.decode(ChatRecord.self, from: data) else {
                continue
            }
            do {
                try save(legacy)
                imported += 1
                let migrated = fileURL.appendingPathExtension("migrated")
                if fm.fileExists(atPath: migrated.path) {
                    try? fm.removeItem(at: fileURL)
                } else {
                    try? fm.moveItem(at: fileURL, to: migrated)
                }
            } catch {
                // Leave the file in place on failure — we'll retry next launch.
                continue
            }
        }
        if imported > 0 {
            NSLog("[ChatHistory] Imported \(imported) chats from legacy JSON")
        }
    }
}
