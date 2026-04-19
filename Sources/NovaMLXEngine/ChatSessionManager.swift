import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

final class SessionBox: @unchecked Sendable {
    let sessionId: String
    let modelId: String
    let createdAt: Date
    var lastAccessed: Date
    var messageCount: Int
    private let session: ChatSession
    private let lock = NovaMLXLock()

    init(sessionId: String, modelId: String, session: ChatSession) {
        self.sessionId = sessionId
        self.modelId = modelId
        self.createdAt = Date()
        self.lastAccessed = Date()
        self.messageCount = 0
        self.session = session
    }

    func streamDetails(to prompt: String) -> AsyncThrowingStream<Generation, Error> {
        lock.withLock { lastAccessed = Date() }
        messageCount += 1
        return session.streamDetails(
            to: prompt,
            images: [],
            videos: []
        )
    }

    func saveCache(to url: URL) async throws {
        try await session.saveCache(to: url)
    }

    func clear() async {
        await session.clear()
        messageCount = 0
    }
}

public struct SessionInfo: Sendable, Codable {
    public let sessionId: String
    public let modelId: String
    public let createdAt: Date
    public let lastAccessed: Date
    public let messageCount: Int

    public init(sessionId: String, modelId: String, createdAt: Date, lastAccessed: Date, messageCount: Int) {
        self.sessionId = sessionId
        self.modelId = modelId
        self.createdAt = createdAt
        self.lastAccessed = lastAccessed
        self.messageCount = messageCount
    }
}

public final class ChatSessionManager: @unchecked Sendable {
    private var sessions: [String: SessionBox] = [:]
    private let lock = NovaMLXLock()
    public let maxSessions: Int
    public let sessionTTL: TimeInterval
    public let cacheDirectory: URL

    public init(maxSessions: Int = 64, sessionTTL: TimeInterval = 1800, cacheDirectory: URL? = nil) {
        self.maxSessions = maxSessions
        self.sessionTTL = sessionTTL
        self.cacheDirectory = cacheDirectory ?? NovaMLXPaths.sessionsDir
        try? FileManager.default.createDirectory(at: self.cacheDirectory, withIntermediateDirectories: true)
    }

    func cacheURL(for sessionId: String) -> URL {
        let safeName = sessionId.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? sessionId
        return cacheDirectory.appendingPathComponent("\(safeName).safetensors")
    }

    private func metadataURL(for sessionId: String) -> URL {
        let safeName = sessionId.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? sessionId
        return cacheDirectory.appendingPathComponent("\(safeName).json")
    }

    func getOrCreate(
        sessionId: String,
        mlxContainer: MLXLMCommon.ModelContainer,
        modelId: String,
        systemPrompt: String?,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64
    ) -> SessionBox {
        lock.withLock {
            if let box = sessions[sessionId] {
                return box
            }

            var params = GenerateParameters()
            if let kvBits {
                params.kvBits = kvBits
                params.kvGroupSize = kvGroupSize
            }

            let session: ChatSession
            let cacheFile = cacheURL(for: sessionId)
            if FileManager.default.fileExists(atPath: cacheFile.path),
               let (cachedKV, _) = try? loadPromptCache(url: cacheFile) {
                session = ChatSession(
                    mlxContainer,
                    instructions: nil,
                    cache: cachedKV,
                    generateParameters: params
                )
                NovaMLXLog.info("ChatSession restored from cache: \(sessionId)")
                try? FileManager.default.removeItem(at: cacheFile)
            } else {
                session = ChatSession(
                    mlxContainer,
                    instructions: systemPrompt,
                    generateParameters: params
                )
                NovaMLXLog.info("ChatSession created: \(sessionId) for model \(modelId)")
            }

            let box = SessionBox(sessionId: sessionId, modelId: modelId, session: session)
            sessions[sessionId] = box

            if sessions.count > maxSessions {
                evictOldest()
            }

            return box
        }
    }

    func get(_ sessionId: String) -> SessionBox? {
        lock.withLock { sessions[sessionId] }
    }

    public func remove(_ sessionId: String) {
        lock.withLock {
            sessions.removeValue(forKey: sessionId)
            NovaMLXLog.info("ChatSession removed: \(sessionId)")
        }
    }

    public func saveSession(_ sessionId: String) async throws {
        let box = lock.withLock { sessions[sessionId] }
        guard let box else {
            throw NovaMLXError.cacheError("Session not found: \(sessionId)")
        }
        let url = cacheURL(for: sessionId)
        try await box.saveCache(to: url)
        let meta: [String: String] = [
            "modelId": box.modelId,
            "sessionId": box.sessionId,
        ]
        let metaData = try JSONEncoder().encode(meta)
        try metaData.write(to: metadataURL(for: sessionId))
        NovaMLXLog.info("Session saved to disk: \(sessionId)")
    }

    public func fork(
        from sourceId: String,
        into targetId: String,
        mlxContainer: MLXLMCommon.ModelContainer,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64
    ) async throws {
        let sourceURL = cacheURL(for: sourceId)

        let sourceBox = lock.withLock { sessions[sourceId] }
        if let sourceBox {
            try await sourceBox.saveCache(to: sourceURL)
        }

        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw NovaMLXError.cacheError("No cache available for session: \(sourceId)")
        }

        let (cachedKV, _) = try loadPromptCache(url: sourceURL)

        var params = GenerateParameters()
        if let kvBits {
            params.kvBits = kvBits
            params.kvGroupSize = kvGroupSize
        }
        let forkedSession = ChatSession(
            mlxContainer,
            instructions: nil,
            cache: cachedKV,
            generateParameters: params
        )

        let sourceMeta = lock.withLock { sessions[sourceId]?.modelId ?? "unknown" }
        let box = SessionBox(sessionId: targetId, modelId: sourceMeta, session: forkedSession)
        lock.withLock {
            sessions[targetId] = box
        }
        NovaMLXLog.info("Session forked: \(sourceId) -> \(targetId)")
    }

    public func removeSessions(forModel modelId: String) {
        lock.withLock {
            let toRemove = sessions.filter { $0.value.modelId == modelId }.map(\.key)
            for id in toRemove {
                sessions.removeValue(forKey: id)
            }
            if !toRemove.isEmpty {
                NovaMLXLog.info("Removed \(toRemove.count) sessions for model \(modelId)")
            }
        }
    }

    public func listSessions() -> [SessionInfo] {
        lock.withLock {
            sessions.values.map { box in
                SessionInfo(
                    sessionId: box.sessionId,
                    modelId: box.modelId,
                    createdAt: box.createdAt,
                    lastAccessed: box.lastAccessed,
                    messageCount: box.messageCount
                )
            }
        }
    }

    public func cleanupExpired() {
        let now = Date()
        lock.withLock {
            let expired = sessions.filter { now.timeIntervalSince($0.value.lastAccessed) > sessionTTL }
            for (id, _) in expired {
                sessions.removeValue(forKey: id)
            }
            if !expired.isEmpty {
                NovaMLXLog.info("Cleaned up \(expired.count) expired sessions")
            }
        }
    }

    public var sessionCount: Int {
        lock.withLock { sessions.count }
    }

    private func evictOldest() {
        let sorted = sessions.sorted { $0.value.lastAccessed < $1.value.lastAccessed }
        if let oldest = sorted.first {
            let box = oldest.value
            let url = cacheURL(for: box.sessionId)
            if let sessionBox = sessions[oldest.key] {
                Task {
                    try? await sessionBox.saveCache(to: url)
                    NovaMLXLog.info("Evicted session saved to disk: \(box.sessionId)")
                }
            }
            sessions.removeValue(forKey: oldest.key)
            NovaMLXLog.info("Evicted oldest session: \(oldest.key)")
        }
    }
}
