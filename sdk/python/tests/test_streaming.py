"""Tests for SSE streaming parser."""

import json
import pytest
import httpx
from unittest.mock import MagicMock, patch

from novamlx._streaming import (
    parse_sse_stream,
    parse_sse_raw,
    parse_anthropic_sse,
)
from novamlx._types import ChatCompletionChunk, StreamChoice, DeltaMessage


def _make_response(lines: list[str]) -> httpx.Response:
    """Create a mock httpx.Response that yields given lines from iter_lines."""
    resp = MagicMock(spec=httpx.Response)
    resp.iter_lines.return_value = iter(lines)
    return resp


class TestSSEStreamParser:
    def test_basic_stream(self):
        chunks = [
            json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }),
            json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
            }),
            json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        lines = [f"data: {c}" for c in chunks] + ["data: [DONE]"]
        resp = _make_response(lines)

        results = list(parse_sse_stream(resp, ChatCompletionChunk))
        assert len(results) == 3
        assert results[0].choices[0].delta.content == "Hello"
        assert results[1].choices[0].delta.content == " world"
        assert results[2].choices[0].finish_reason == "stop"

    def test_skip_empty_and_comment_lines(self):
        lines = [
            "",
            ": keep-alive",
            "event: ping",
            'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"t","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}',
            "",
            "data: [DONE]",
        ]
        resp = _make_response(lines)
        results = list(parse_sse_stream(resp, ChatCompletionChunk))
        assert len(results) == 1
        assert results[0].choices[0].delta.content == "x"

    def test_raw_stream(self):
        lines = [
            'data: {"foo": "bar"}',
            'data: {"baz": 42}',
            "data: [DONE]",
        ]
        resp = _make_response(lines)
        results = list(parse_sse_raw(resp))
        assert len(results) == 2
        assert results[0] == {"foo": "bar"}
        assert results[1] == {"baz": 42}

    def test_reasoning_content_stream(self):
        chunks = [
            json.dumps({
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "t",
                "choices": [{"index": 0, "delta": {"reasoning_content": "Hmm..."}, "finish_reason": None}],
            }),
            json.dumps({
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "t",
                "choices": [{"index": 0, "delta": {"content": "42"}, "finish_reason": "stop"}],
            }),
        ]
        lines = [f"data: {c}" for c in chunks] + ["data: [DONE]"]
        resp = _make_response(lines)

        results = list(parse_sse_stream(resp, ChatCompletionChunk))
        assert results[0].choices[0].delta.reasoning_content == "Hmm..."
        assert results[1].choices[0].delta.content == "42"


class TestAnthropicSSEParser:
    def test_anthropic_events(self):
        lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","content":[],"model":"t","stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
            "",
            "event: message_stop",
            "data: {}",
        ]
        resp = _make_response(lines)
        events = list(parse_anthropic_sse(resp))

        assert len(events) == 6
        assert events[0]["_event_type"] == "message_start"
        assert events[0]["type"] == "message_start"
        assert events[2]["delta"]["text"] == "Hi"
        assert events[4]["delta"]["stop_reason"] == "end_turn"

    def test_thinking_delta(self):
        lines = [
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer."}}',
        ]
        resp = _make_response(lines)
        events = list(parse_anthropic_sse(resp))
        assert events[1]["delta"]["type"] == "thinking_delta"
        assert events[1]["delta"]["thinking"] == "Let me think..."
        assert events[3]["delta"]["type"] == "text_delta"
        assert events[3]["delta"]["text"] == "Answer."
