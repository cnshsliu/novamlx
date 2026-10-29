import Testing
@testable import NovaMLXEngine

@Suite("GLM chat template jinja")
struct GlmJinjaTests {
    @Test("numeric attribute indexes become subscripts")
    func rewritesDotIndex() {
        let source = "{% if m.content.0.type == 'text' %}{{ m.content.0.text }}{% endif %} {{ value.12 }}"
        let (rewritten, count) = GlmJinja.rewriteNumericIndexes(source)
        #expect(count == 3)
        #expect(rewritten.contains("m.content[0].type"))
        #expect(rewritten.contains("m.content[0].text"))
        #expect(rewritten.contains("value[12]"))
        #expect(!rewritten.contains(".0"))
    }

    @Test("float literals stay numeric")
    func keepsFloatLiterals() {
        let source = "{% set x = 1.0 %}{{ items.0 }}"
        let (rewritten, count) = GlmJinja.rewriteNumericIndexes(source)
        #expect(count == 1)
        #expect(rewritten.contains("1.0"))
        #expect(rewritten.contains("items[0]"))
    }

    @Test("a second pass does not rewrite subscripts again")
    func idempotent() {
        let once = GlmJinja.rewriteNumericIndexes("{{ m.content.0.type }}").0
        let twice = GlmJinja.rewriteNumericIndexes(once)
        #expect(twice.0 == once)
        #expect(twice.1 == 0)
    }
}
