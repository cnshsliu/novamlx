import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXEngine

@Suite("MLXEngine Tests")
struct MLXEngineTests {
    @Test("Engine initial state")
    func engineInitialState() {
        let engine = MLXEngine()
        #expect(engine.isReady == false)
    }

    @Test("Engine with custom config")
    func engineCustomConfig() {
        let engine = MLXEngine(kvCacheCapacity: 512, maxMemoryMB: 1024, maxConcurrent: 4)
        #expect(engine.isReady == false)
    }

    @Test("Get container for unloaded model returns nil")
    func getContainerNil() {
        let engine = MLXEngine()
        let container = engine.getContainer(for: "nonexistent")
        #expect(container == nil)
    }

    @Test("Model container initial state")
    func modelContainerInitialState() {
        let id = ModelIdentifier(id: "test", family: .llama)
        let config = ModelConfig(identifier: id)
        let container = ModelContainer(identifier: id, config: config)
        #expect(container.isLoaded == false)
        #expect(container.identifier.id == "test")
    }
}

@Suite("ContinuousBatcher Tests")
struct ContinuousBatcherTests {
    @Test("Batcher initial state")
    func batcherInitialState() {
        let engine = MLXEngine()
        let batcher = ContinuousBatcher(engine: engine, maxBatchSize: 4)
        #expect(batcher.activeRequests == 0)
    }
}

@Suite("ChatSessionManager Tests")
struct ChatSessionManagerTests {
    @Test("Session manager initial state")
    func sessionManagerInitialState() {
        let manager = ChatSessionManager()
        #expect(manager.sessionCount == 0)
        #expect(manager.maxSessions == 64)
    }

    @Test("Session manager custom config")
    func sessionManagerCustomConfig() {
        let manager = ChatSessionManager(maxSessions: 32, sessionTTL: 600)
        #expect(manager.maxSessions == 32)
        #expect(manager.sessionTTL == 600)
    }

    @Test("Session manager list empty")
    func sessionManagerListEmpty() {
        let manager = ChatSessionManager()
        let sessions = manager.listSessions()
        #expect(sessions.isEmpty)
    }

    @Test("Session manager remove nonexistent")
    func sessionManagerRemoveNonexistent() {
        let manager = ChatSessionManager()
        manager.remove("nonexistent")
        #expect(manager.sessionCount == 0)
    }

    @Test("Session info is codable")
    func sessionInfoCodable() throws {
        let info = SessionInfo(
            sessionId: "test-session",
            modelId: "test-model",
            createdAt: Date(),
            lastAccessed: Date(),
            messageCount: 5
        )
        let data = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(SessionInfo.self, from: data)
        #expect(decoded.sessionId == "test-session")
        #expect(decoded.modelId == "test-model")
        #expect(decoded.messageCount == 5)
    }
}
