import Foundation
import Testing

@testable import NovaMLXCore

/// OpenCode config generation. The shape below was verified against a live
/// OpenCode session running Flint through NovaMLX (2026-11-05) — the previous
/// generator wrote `{"provider":"custom",…}` to `config.json`, which OpenCode
/// never reads.
@Suite("AgentConfigGenerator OpenCode")
struct AgentConfigGeneratorTests {

    private func tempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("agentcfg-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func readJSON(_ path: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: path)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    @Test("writes the real opencode.json shape")
    func writesRealShape() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let error = AgentConfigGenerator.writeOpenCodeConfig(
            baseURL: "http://127.0.0.1:6590/v1",
            apiKey: nil,
            model: "RecursiveSI/flint-int4",
            directory: dir)
        #expect(error == nil)

        let config = try readJSON(dir.appendingPathComponent("opencode.json"))
        let provider = try #require(config["provider"] as? [String: Any])
        let novamlx = try #require(provider["novamlx"] as? [String: Any])
        #expect(novamlx["npm"] as? String == "@ai-sdk/openai-compatible")
        let options = try #require(novamlx["options"] as? [String: Any])
        #expect(options["baseURL"] as? String == "http://127.0.0.1:6590/v1")
        // No secret embedded: the config references the env var.
        #expect(options["apiKey"] as? String == "{env:NOVA_API_KEY}")
        #expect(config["model"] as? String == "novamlx/RecursiveSI/flint-int4")

        // The Flint-strict agent ships with the config.
        let agent = try #require(config["agent"] as? [String: Any])
        let driver = try #require(agent["flint-driver"] as? [String: Any])
        #expect(driver["mode"] as? String == "all")
        let tools = try #require(driver["tools"] as? [String: Bool])
        #expect(tools["write"] == true)
        #expect(tools["bash"] == true)
        #expect((driver["prompt"] as? String ?? "").contains("HARD RULES"))
    }

    @Test("merges instead of clobbering an existing config")
    func mergesExistingConfig() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("opencode.json")
        let existing: [String: Any] = [
            "provider": ["other": ["npm": "@ai-sdk/anthropic"]],
            "agent": ["mine": ["prompt": "keep me"]],
            "theme": "dark",
        ]
        try JSONSerialization.data(withJSONObject: existing)
            .write(to: path)

        let error = AgentConfigGenerator.writeOpenCodeConfig(
            baseURL: "http://127.0.0.1:6590/v1",
            apiKey: "sk-test",
            model: "RecursiveSI/flint-int4",
            directory: dir)
        #expect(error == nil)

        let config = try readJSON(path)
        #expect(config["theme"] as? String == "dark")
        let provider = try #require(config["provider"] as? [String: Any])
        #expect(provider["other"] != nil, "foreign providers must survive")
        #expect(provider["novamlx"] != nil)
        let agent = try #require(config["agent"] as? [String: Any])
        #expect(agent["mine"] != nil, "foreign agents must survive")
        #expect(agent["flint-driver"] != nil)
        let novamlx = try #require(provider["novamlx"] as? [String: Any])
        let options = try #require(novamlx["options"] as? [String: Any])
        #expect(options["apiKey"] as? String == "sk-test")
    }

    @Test("display text states the prerequisites and stays parseable JSON after comments")
    func displayTextPrerequisites() throws {
        let text = AgentConfigGenerator.openCodeConfigText(
            baseURL: "http://127.0.0.1:6590/v1",
            apiKey: nil,
            model: "RecursiveSI/flint-int4")
        #expect(text.contains("opencode.json"))
        #expect(text.contains("NOVA_API_KEY"))
        #expect(text.contains("bge-m3"), "must tell the user OpenCode needs an embedding model")
        // The JSON body itself (strip // comment lines) parses.
        let body = text.split(separator: "\n")
            .filter { !$0.hasPrefix("//") && !$0.hasPrefix(" //") }
            .joined(separator: "\n")
        let data = try #require(body.data(using: .utf8))
        let json = try #require(try? JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["provider"] != nil)
    }
}
