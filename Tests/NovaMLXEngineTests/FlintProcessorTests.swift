import Foundation
import Testing

@testable import NovaMLXEngine

@Suite("Flint chat template processor")
struct FlintProcessorTests {

    /// The model's actual chat_template.jinja, trimmed to the markers that
    /// matter for detection.
    static let flintTemplate = """
        {%- macro flint_text(t) -%}{{- t | replace("<think>", "") | replace("</think>", "") -}}{%- endmacro -%}
        {%- for m in messages -%}
        {%- if m.role == 'system' -%}<|system|>{{ flint_text(m.content) }}
        {%- elif m.role == 'user' -%}<|user|>{{ flint_text(m.content) }}
        {%- elif m.role == 'assistant' -%}<|assistant|>
        {%- if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif -%}
        {%- if m.tool_calls -%}{%- for tc in m.tool_calls -%}
        <tool_call>{{ fn.name }}<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value></tool_call>
        {%- endfor -%}{%- endif -%}
        {%- elif m.role == 'tool' -%}<|observation|><tool_response>{{ flint_text(m.content) }}</tool_response>
        {%- endif -%}
        {%- endfor -%}
        {%- if add_generation_prompt -%}<|assistant|>
        {%- endif -%}
        """

    @Test("flint format is detected from role headers and not confused with ChatML")
    func testFormatDetection() {
        let detected = ChatTemplateFormat.detectAll(from: FlintProcessorTests.flintTemplate)
        #expect(detected.first?.format == .flint)
        #expect(detected.first!.confidence >= 3)

        // ChatML must NOT detect as flint (no pipe role headers).
        let chatml = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant"
        let chatmlDetected = ChatTemplateFormat.detectAll(from: chatml)
        #expect(!chatmlDetected.contains { $0.format == .flint })
        #expect(chatmlDetected.contains { $0.format == .imStartEnd })
    }

    @Test("refineControlTokens keeps role headers, drops tool-call envelope and think tags")
    func testRefineControlTokens() {
        let processor = FlintProcessor()
        let tokens = processor.refineControlTokens(
            rawTokens: ["<|user|>", "<|assistant|>"],
            templateTokens: ["<|observation|>", "<|end_of_text|>"],
            chatTemplate: FlintProcessorTests.flintTemplate,
            addedTokens: [["content": "<think>"], ["content": "</think>"], ["content": "<|developer|>"]]
        )
        let set = Set(tokens)
        // Turn boundaries stay.
        #expect(set.contains("<|user|>"))
        #expect(set.contains("<|assistant|>"))
        #expect(set.contains("<|observation|>"))
        #expect(set.contains("<|developer|>"))
        #expect(set.contains("<|end_of_text|>"))
        #expect(set.contains("<|begin_of_text|>"))  // canonical union
        // Thinking tags are semantic — never stop/scrub.
        #expect(!set.contains("<think>"))
        #expect(!set.contains("</think>"))
        // Tool-call envelope is parser territory, not a turn boundary.
        #expect(!set.contains("<tool_call>"))
        #expect(!set.contains("</tool_call>"))
        #expect(!set.contains("<tool_response>"))
    }

    @Test("thinking model detection via template and added tokens")
    func testThinkingDetection() {
        let processor = FlintProcessor()
        #expect(processor.isThinkingModel(
            chatTemplate: FlintProcessorTests.flintTemplate, addedTokens: []))
        #expect(processor.isThinkingModel(chatTemplate: nil, addedTokens: [["content": "</think>"]]))
        #expect(!processor.isThinkingModel(chatTemplate: "<|user|>hi", addedTokens: []))
        // Flint never opens <think> for the assistant — no implicit mode.
        #expect(!processor.isImplicitThinkingModel(chatTemplate: FlintProcessorTests.flintTemplate))
    }

    @Test("registry routes flint family to FlintProcessor")
    func testRegistryRouting() {
        let processor = ChatTemplateProcessorRegistry.processor(
            family: .flint, chatTemplate: FlintProcessorTests.flintTemplate)
        #expect(processor is FlintProcessor)
    }

    @Test("scrub removes role headers from output text")
    func testScrub() {
        let processor = FlintProcessor()
        let text = "answer<|user|>\nnext question<|observation|>"
        let scrubbed = processor.scrubControlTokens(text)
        #expect(!scrubbed.contains("<|user|>"))
        #expect(!scrubbed.contains("<|observation|>"))
        #expect(scrubbed.contains("answer"))
    }
}
