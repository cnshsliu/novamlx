import Foundation
import NovaMLXCore

public final class ChatHistoryStore: Sendable {
    public static let shared = ChatHistoryStore()

    private let directory: URL
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .millisecondsSince1970
        return e
    }()
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .millisecondsSince1970
        return d
    }()

    private init(directory: URL = NovaMLXPaths.chatHistoryDir) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: - Types

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
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return [] }
        return files.filter { $0.pathExtension == "json" }
            .compactMap { url -> ChatSummary? in
                guard let data = try? Data(contentsOf: url),
                      let record = try? decoder.decode(ChatRecord.self, from: data) else { return nil }
                return ChatSummary(id: record.id, title: record.title, model: record.model, ts: record.ts, messageCount: record.messages.count)
            }
            .sorted { $0.ts > $1.ts }
    }

    public func get(id: String) -> ChatRecord? {
        let url = fileURL(for: id)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(ChatRecord.self, from: data)
    }

    public func save(_ record: ChatRecord) throws {
        let url = fileURL(for: record.id)
        let data = try encoder.encode(record)
        try data.write(to: url, options: .atomic)
    }

    public func delete(id: String) throws {
        let url = fileURL(for: id)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    public func search(query: String) -> [ChatSummary] {
        let q = query.lowercased()
        return list().filter { $0.title.lowercased().contains(q) }
    }

    // MARK: - Private

    private func fileURL(for id: String) -> URL {
        let safeName = id.replacingOccurrences(of: "/", with: "_")
        return directory.appendingPathComponent(safeName).appendingPathExtension("json")
    }
}
