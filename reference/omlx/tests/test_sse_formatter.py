# SPDX-License-Identifier: Apache-2.0
"""
Tests for SSE (Server-Sent Events) formatters.

Tests the SSEFormatter base class and concrete implementations for
OpenAI and Anthropic API formats.
"""

import json
import pytest

from omlx.api.adapters.sse_formatter import (
    AnthropicSSEFormatter,
    OpenAISSEFormatter,
    SSEFormatter,
)


class TestSSEFormatterBase:
    """Tests for SSEFormatter base class."""

    def test_sse_formatter_is_abstract(self):
        """Test that SSEFormatter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SSEFormatter()

    def test_sse_formatter_has_format_event(self):
        """Test that SSEFormatter defines format_event method."""
        assert hasattr(SSEFormatter, "format_event")

    def test_sse_formatter_has_format_end(self):
        """Test that SSEFormatter defines format_end method."""
        assert hasattr(SSEFormatter, "format_end")


class TestOpenAISSEFormatter:
    """Tests for OpenAISSEFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create OpenAISSEFormatter instance."""
        return OpenAISSEFormatter()

    def test_inherits_from_sse_formatter(self, formatter):
        """Test that OpenAISSEFormatter inherits from SSEFormatter."""
        assert isinstance(formatter, SSEFormatter)

    # =========================================================================
    # format_event Tests
    # =========================================================================

    def test_format_event_simple(self, formatter):
        """Test formatting a simple event."""
        data = {"key": "value"}

        result = formatter.format_event("ignored", data)

        assert result == 'data: {"key": "value"}\n\n'

    def test_format_event_ignores_event_type(self, formatter):
        """Test that OpenAI format ignores event_type."""
        data = {"message": "hello"}

        result1 = formatter.format_event("message_start", data)
        result2 = formatter.format_event("content_delta", data)
        result3 = formatter.format_event("", data)

        # All should produce same format (event type ignored)
        expected = 'data: {"message": "hello"}\n\n'
        assert result1 == expected
        assert result2 == expected
        assert result3 == expected

    def test_format_event_complex_data(self, formatter):
        """Test formatting event with complex data."""
        data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = formatter.format_event("", data)

        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        # Verify JSON is valid
        json_str = result[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        parsed = json.loads(json_str)

        assert parsed["id"] == "chatcmpl-abc123"
        assert parsed["choices"][0]["delta"]["content"] == "Hello"

    def test_format_event_nested_objects(self, formatter):
        """Test formatting event with deeply nested objects."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }

        result = formatter.format_event("", data)

        json_str = result[6:-2]
        parsed = json.loads(json_str)

        assert parsed["level1"]["level2"]["level3"]["value"] == "deep"

    def test_format_event_with_arrays(self, formatter):
        """Test formatting event with arrays."""
        data = {
            "items": [1, 2, 3],
            "nested": [{"a": 1}, {"b": 2}],
        }

        result = formatter.format_event("", data)

        json_str = result[6:-2]
        parsed = json.loads(json_str)

        assert parsed["items"] == [1, 2, 3]
        assert parsed["nested"][0]["a"] == 1

    def test_format_event_with_none(self, formatter):
        """Test formatting event with None values."""
        data = {
            "value": None,
            "nested": {"also_none": None},
        }

        result = formatter.format_event("", data)

        json_str = result[6:-2]
        parsed = json.loads(json_str)

        assert parsed["value"] is None
        assert parsed["nested"]["also_none"] is None

    def test_format_event_with_unicode(self, formatter):
        """Test formatting event with Unicode characters."""
        data = {
            "japanese": "こんにちは",
            "korean": "안녕하세요",
            "emoji": "Hello! 🎉",
        }

        result = formatter.format_event("", data)

        # Verify Unicode is preserved (json.dumps with ensure_ascii=False would keep them)
        json_str = result[6:-2]
        parsed = json.loads(json_str)

        assert parsed["japanese"] == "こんにちは"
        assert parsed["korean"] == "안녕하세요"
        assert parsed["emoji"] == "Hello! 🎉"

    def test_format_event_empty_dict(self, formatter):
        """Test formatting event with empty dict."""
        result = formatter.format_event("", {})

        assert result == "data: {}\n\n"

    # =========================================================================
    # format_end Tests
    # =========================================================================

    def test_format_end(self, formatter):
        """Test formatting stream end marker."""
        result = formatter.format_end()

        assert result == "data: [DONE]\n\n"

    def test_format_end_is_consistent(self, formatter):
        """Test that format_end always returns same value."""
        result1 = formatter.format_end()
        result2 = formatter.format_end()

        assert result1 == result2 == "data: [DONE]\n\n"


class TestAnthropicSSEFormatter:
    """Tests for AnthropicSSEFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create AnthropicSSEFormatter instance."""
        return AnthropicSSEFormatter()

    def test_inherits_from_sse_formatter(self, formatter):
        """Test that AnthropicSSEFormatter inherits from SSEFormatter."""
        assert isinstance(formatter, SSEFormatter)

    # =========================================================================
    # format_event Tests
    # =========================================================================

    def test_format_event_with_event_type(self, formatter):
        """Test formatting event includes event type."""
        data = {"type": "message_start"}

        result = formatter.format_event("message_start", data)

        assert result.startswith("event: message_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_format_event_message_start(self, formatter):
        """Test formatting message_start event."""
        data = {
            "type": "message_start",
            "message": {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
            },
        }

        result = formatter.format_event("message_start", data)

        # Should have event line and data line
        lines = result.strip().split("\n")
        assert lines[0] == "event: message_start"
        assert lines[1].startswith("data: ")

        # Parse data
        json_str = lines[1][6:]  # Remove "data: " prefix
        parsed = json.loads(json_str)

        assert parsed["type"] == "message_start"
        assert parsed["message"]["id"] == "msg_abc123"

    def test_format_event_content_block_start(self, formatter):
        """Test formatting content_block_start event."""
        data = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }

        result = formatter.format_event("content_block_start", data)

        assert "event: content_block_start\n" in result

        lines = result.strip().split("\n")
        json_str = lines[1][6:]
        parsed = json.loads(json_str)

        assert parsed["index"] == 0
        assert parsed["content_block"]["type"] == "text"

    def test_format_event_content_block_delta(self, formatter):
        """Test formatting content_block_delta event."""
        data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }

        result = formatter.format_event("content_block_delta", data)

        assert "event: content_block_delta\n" in result

        lines = result.strip().split("\n")
        json_str = lines[1][6:]
        parsed = json.loads(json_str)

        assert parsed["delta"]["text"] == "Hello"

    def test_format_event_content_block_stop(self, formatter):
        """Test formatting content_block_stop event."""
        data = {
            "type": "content_block_stop",
            "index": 0,
        }

        result = formatter.format_event("content_block_stop", data)

        assert "event: content_block_stop\n" in result

        lines = result.strip().split("\n")
        json_str = lines[1][6:]
        parsed = json.loads(json_str)

        assert parsed["index"] == 0

    def test_format_event_message_delta(self, formatter):
        """Test formatting message_delta event."""
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 10},
        }

        result = formatter.format_event("message_delta", data)

        assert "event: message_delta\n" in result

        lines = result.strip().split("\n")
        json_str = lines[1][6:]
        parsed = json.loads(json_str)

        assert parsed["delta"]["stop_reason"] == "end_turn"
        assert parsed["usage"]["output_tokens"] == 10

    def test_format_event_message_stop(self, formatter):
        """Test formatting message_stop event."""
        data = {"type": "message_stop"}

        result = formatter.format_event("message_stop", data)

        assert "event: message_stop\n" in result

    def test_format_event_ping(self, formatter):
        """Test formatting ping event."""
        data = {"type": "ping"}

        result = formatter.format_event("ping", data)

        assert "event: ping\n" in result

    def test_format_event_error(self, formatter):
        """Test formatting error event."""
        data = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Something went wrong",
            },
        }

        result = formatter.format_event("error", data)

        assert "event: error\n" in result

        lines = result.strip().split("\n")
        json_str = lines[1][6:]
        parsed = json.loads(json_str)

        assert parsed["error"]["type"] == "api_error"
        assert parsed["error"]["message"] == "Something went wrong"

    def test_format_event_with_unicode(self, formatter):
        """Test formatting event with Unicode characters."""
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "こんにちは"},
        }

        result = formatter.format_event("content_block_delta", data)

        lines = result.strip().split("\n")
        json_str = lines[1][6:]
        parsed = json.loads(json_str)

        assert parsed["delta"]["text"] == "こんにちは"

    # =========================================================================
    # format_end Tests
    # =========================================================================

    def test_format_end(self, formatter):
        """Test formatting stream end marker."""
        result = formatter.format_end()

        # Anthropic doesn't use [DONE] marker
        assert result == ""

    def test_format_end_is_empty(self, formatter):
        """Test that format_end returns empty string."""
        result1 = formatter.format_end()
        result2 = formatter.format_end()

        assert result1 == result2 == ""


class TestSSEFormatterComparison:
    """Tests comparing OpenAI and Anthropic SSE formatters."""

    def test_openai_format_is_data_only(self):
        """Test that OpenAI format only uses data lines."""
        formatter = OpenAISSEFormatter()
        result = formatter.format_event("any_type", {"key": "value"})

        # Should not have "event:" line
        assert "event:" not in result
        assert result.startswith("data: ")

    def test_anthropic_format_has_event_type(self):
        """Test that Anthropic format uses event type lines."""
        formatter = AnthropicSSEFormatter()
        result = formatter.format_event("message_start", {"key": "value"})

        # Should have "event:" line
        assert "event: message_start" in result
        assert "data: " in result

    def test_both_end_with_double_newline(self):
        """Test that both formatters end events with double newline."""
        openai = OpenAISSEFormatter()
        anthropic = AnthropicSSEFormatter()

        openai_result = openai.format_event("", {"key": "value"})
        anthropic_result = anthropic.format_event("test", {"key": "value"})

        assert openai_result.endswith("\n\n")
        assert anthropic_result.endswith("\n\n")

    def test_different_end_markers(self):
        """Test that formatters have different end markers."""
        openai = OpenAISSEFormatter()
        anthropic = AnthropicSSEFormatter()

        openai_end = openai.format_end()
        anthropic_end = anthropic.format_end()

        assert openai_end == "data: [DONE]\n\n"
        assert anthropic_end == ""
