import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("Nova DB snapshot reader")
struct NovaDBSnapshotReaderTests {
    @Test("sqlite JSON rows become migration input")
    func decodesRows() throws {
        let providers = Data(#"""
        [{"name":"OpenAI","endpoint":"https://api.openai.com/v1","api_key":"sk-test","remote_model":"gpt-4o","is_enabled":1,"is_free":0,"provider_id":"openai","anthropic_endpoint":null}]
        """#.utf8)
        let balancers = Data(#"""
        [{"id":"lb-1","name":"Coding","slug":"coding","strategy":"tiered","max_retries":3,"is_enabled":1}]
        """#.utf8)
        let members = Data(#"""
        [{"lb_id":"lb-1","lb_slug":"coding","kind":"local","ref":"Qwen3-8B","weight":null,"is_enabled":1}]
        """#.utf8)
        let input = try NovaDBSnapshotReader.decode(
            providersJSON: providers, loadBalancersJSON: balancers, membersJSON: members)
        #expect(input.providers.first?.id == "openai")
        #expect(input.providers.first?.apiKey == "sk-test")
        #expect(input.loadBalancers.first?.slug == "coding")
        #expect(input.members.first?.kind == "local")
        #expect(input.members.first?.lbSlug == "coding")
    }

    @Test("sqlite3 file loads providers and members")
    func readsDatabase() throws {
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/sqlite3") else { return }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("nova-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let db = dir.appendingPathComponent("nova_config.db")
        let sql = """
        CREATE TABLE tokenhub_providers (
          name TEXT PRIMARY KEY, endpoint TEXT, api_key TEXT, remote_model TEXT,
          is_enabled INTEGER, is_free INTEGER, provider_id TEXT, anthropic_endpoint TEXT
        );
        CREATE TABLE load_balancers (
          id TEXT PRIMARY KEY, name TEXT, slug TEXT, strategy TEXT,
          max_retries INTEGER, is_enabled INTEGER
        );
        CREATE TABLE lb_members (
          id TEXT PRIMARY KEY, lb_id TEXT, kind TEXT, ref TEXT, weight INTEGER, is_enabled INTEGER
        );
        INSERT INTO tokenhub_providers VALUES ('OpenAI','https://api.openai.com','sk-test','gpt-4o',1,1,'openai',NULL);
        INSERT INTO load_balancers VALUES ('lb-1','Coding','coding','round_robin',2,1);
        INSERT INTO lb_members VALUES ('m1','lb-1','remote','OpenAI',NULL,1);
        """
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/sqlite3")
        process.arguments = [db.path, sql]
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        let input = try NovaDBSnapshotReader.load(database: db)
        #expect(input.providers.first?.name == "OpenAI")
        #expect(input.providers.first?.isFree == true)
        #expect(input.members.first?.ref == "OpenAI")
        #expect(input.members.first?.lbSlug == "coding")
        let result = NovaConfigMigrator.migrate(input, existing: PeerConfig.defaultConfig())
        #expect(result.config.pool(id: "coding")?.memberIds == ["th-openai"])
        #expect(result.secrets["src/th-openai"] == "sk-test")
    }
}
