import Foundation
import Testing
@testable import TknetPeer

@Suite struct EnvKeyFileTests {
    @Test func referenceDetection() {
        #expect(EnvKeyFile.isReference("OPENROUTER_KEY"))
        #expect(EnvKeyFile.isReference("_FOO"))
        #expect(EnvKeyFile.isReference("A1_B2"))
        #expect(!EnvKeyFile.isReference("sk-abc123"))
        #expect(!EnvKeyFile.isReference("gsk_deadbeef"))
        #expect(!EnvKeyFile.isReference("MyKey"))
        #expect(!EnvKeyFile.isReference("12345"))
        #expect(!EnvKeyFile.isReference(""))
        #expect(!EnvKeyFile.isReference("HAS SPACE"))
        #expect(!EnvKeyFile.isReference("WITH-DASH"))
    }

    @Test func parseSkipsCommentsBlankAndMalformed() {
        let text = """
        # comment

        A=1
        B = spaced
        C="quoted"
        D='single'
        no-equals-line
        =novalue
        E=
        A=overrides
        """
        let map = EnvKeyFile.parse(text)
        #expect(map["A"] == "overrides")
        #expect(map["B"] == "spaced")
        #expect(map["C"] == "quoted")
        #expect(map["D"] == "single")
        #expect(map["E"] == "")
        #expect(map.count == 5)
    }

    @Test func resolvePassesLiteralsThrough() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-env-\(UUID().uuidString)")
        let url = file.appendingPathComponent("env")
        try FileManager.default.createDirectory(at: file, withIntermediateDirectories: true)
        try Data("LIVE_KEY=sk-live-9\n".utf8).write(to: url)

        #expect(EnvKeyFile.resolve("sk-literal", url: url) == "sk-literal")
        #expect(EnvKeyFile.resolve("", url: url) == nil)
        #expect(EnvKeyFile.resolve("LIVE_KEY", url: url) == "sk-live-9")
        #expect(EnvKeyFile.resolve("MISSING_KEY", url: url) == nil)
    }

    @Test func missingFileBehavesAsEmpty() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-env-\(UUID().uuidString)-nope/env")
        #expect(EnvKeyFile.load(url: url).isEmpty)
        #expect(EnvKeyFile.resolve("ANY_KEY", url: url) == nil)
        #expect(EnvKeyFile.resolve("sk-literal", url: url) == "sk-literal")
    }
}
