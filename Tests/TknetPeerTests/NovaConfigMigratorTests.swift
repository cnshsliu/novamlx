import Foundation
import Testing
@testable import TknetPeer

@Suite("Nova config migrator")
struct NovaConfigMigratorTests {
    private func localInput() -> NovaMigrationInput {
        NovaMigrationInput(
            loadBalancers: [NovaLBSnapshot(id: "lb-1", name: "Coding", slug: "coding")],
            members: [NovaLBMemberSnapshot(lbSlug: "coding", kind: "local", ref: "Qwen3-8B")])
    }

    @Test("local LB member becomes loopback NovaMLX HTTP member")
    func localMemberMaps() {
        let result = NovaConfigMigrator.migrate(localInput(), existing: PeerConfig.defaultConfig())
        let member = result.config.member(id: "local-Qwen3-8B")
        #expect(member?.type == .localNovaMLX)
        #expect(member?.endpoint.absoluteString == "http://127.0.0.1:6590/v1")
        #expect(member?.upstreamModel == "Qwen3-8B")
        let pool = result.config.pool(id: "coding")
        #expect(pool?.name == "coding")
        #expect(pool?.memberIds == ["local-Qwen3-8B"])
        #expect(result.warnings.isEmpty)
    }

    @Test("remote LB member reuses Tokenhub provider member and secret")
    func remoteReusesProvider() {
        let input = NovaMigrationInput(
            providers: [NovaProviderSnapshot(
                id: "openai", name: "OpenAI",
                endpoint: "https://api.openai.com/v1/chat/completions",
                apiKey: "sk-test", remoteModel: "gpt-4o", isFree: true)],
            loadBalancers: [NovaLBSnapshot(id: "lb-1", name: "Coding", slug: "coding", strategy: "round_robin")],
            members: [NovaLBMemberSnapshot(lbSlug: "coding", kind: "remote", ref: "OpenAI")])
        let result = NovaConfigMigrator.migrate(input, existing: PeerConfig.defaultConfig())
        let member = result.members.first { $0.id.hasPrefix("th-") }
        #expect(member?.id == "th-openai")
        #expect(member?.endpoint.absoluteString == "https://api.openai.com/v1")
        #expect(member?.upstreamModel == "gpt-4o")
        #expect(member?.isFree == true)
        #expect(member?.weight == nil)
        #expect(result.secrets[member?.apiKeyRef ?? ""] == "sk-test")
        #expect(result.config.pool(id: "coding")?.memberIds == ["th-openai"])
        #expect(result.config.pool(id: "coding")?.strategy == .roundRobin)
    }

    @Test("same local model in two LBs shares one member id")
    func sharedLocalMember() {
        let input = NovaMigrationInput(
            loadBalancers: [
                NovaLBSnapshot(id: "1", name: "A", slug: "a"),
                NovaLBSnapshot(id: "2", name: "B", slug: "b"),
            ],
            members: [
                NovaLBMemberSnapshot(lbSlug: "a", kind: "local", ref: "Qwen3-8B"),
                NovaLBMemberSnapshot(lbSlug: "b", kind: "local", ref: "Qwen3-8B"),
            ])
        let result = NovaConfigMigrator.migrate(input, existing: PeerConfig.defaultConfig())
        let locals = result.config.members.filter { $0.id == "local-Qwen3-8B" }
        #expect(locals.count == 1)
        #expect(result.config.pool(id: "a")?.memberIds == ["local-Qwen3-8B"])
        #expect(result.config.pool(id: "b")?.memberIds == ["local-Qwen3-8B"])
    }

    @Test("merge without --force leaves existing Peer pool untouched")
    func mergeRespectsExisting() {
        var existing = PeerConfig.defaultConfig()
        existing.members = [MemberConfig(
            id: "keep-me", name: "keep", type: .openaiCompatible,
            endpoint: URL(string: "http://10.0.0.1/v1")!, apiKeyRef: "src/keep-me",
            upstreamModel: "kept")]
        existing.pools = [PoolConfig(id: "coding", name: "coding", memberIds: ["keep-me"])]
        let result = NovaConfigMigrator.migrate(localInput(), existing: existing)
        #expect(result.config.pool(id: "coding")?.memberIds == ["keep-me"])
        #expect(result.config.member(id: "local-Qwen3-8B") != nil)
        #expect(result.warnings.contains { $0.contains("coding") && $0.contains("unchanged") })
    }

    @Test("Anthropic-only providers are skipped with a warning")
    func skipsAnthropicOnly() {
        let input = NovaMigrationInput(providers: [NovaProviderSnapshot(
            name: "Claude", endpoint: "", apiKey: "sk-ant", remoteModel: "claude-3",
            anthropicEndpoint: "https://api.anthropic.com")])
        let result = NovaConfigMigrator.migrate(input, existing: PeerConfig.defaultConfig())
        #expect(result.members.isEmpty)
        #expect(result.secrets.isEmpty)
        #expect(result.warnings.contains { $0.contains("Anthropic-only") })
    }

    @Test("a second load balancer with the same slug gets a -migrated id")
    func duplicateSlugSuffixes() {
        let input = NovaMigrationInput(
            loadBalancers: [
                NovaLBSnapshot(id: "1", name: "A", slug: "coding"),
                NovaLBSnapshot(id: "2", name: "B", slug: "coding"),
            ],
            members: [
                NovaLBMemberSnapshot(lbId: "1", kind: "local", ref: "Qwen3-8B"),
                NovaLBMemberSnapshot(lbId: "2", kind: "local", ref: "Other"),
            ])
        let result = NovaConfigMigrator.migrate(input, existing: PeerConfig.defaultConfig())
        #expect(result.config.pool(id: "coding")?.memberIds == ["local-Qwen3-8B"])
        #expect(result.config.pool(id: "coding-migrated")?.memberIds == ["local-Other"])
        #expect(result.warnings.contains { $0.contains("collided") && $0.contains("coding-migrated") })
    }

    @Test("bare host gains a /v1 base")
    func bareHost() {
        #expect(NovaConfigMigrator.normalizeOpenAIBase("https://api.example.com")?.absoluteString
            == "https://api.example.com/v1")
        #expect(NovaConfigMigrator.normalizeOpenAIBase("https://api.example.com/v1/")?.absoluteString
            == "https://api.example.com/v1")
    }
}
