"""Tests for Pydantic type models."""

import json
import pytest
from novamlx._types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionChoice,
    ChatMessageContentPart,
    DeltaMessage,
    ModelCapabilities,
    ModelInfo,
    ModelListResponse,
    NovaExtension,
    StreamChoice,
    ToolCall,
    ToolCallFunction,
    Usage,
    AnthropicResponse,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    EmbeddingResponse,
    EmbeddingData,
    RerankResponse,
    RerankResult,
    HealthResponse,
)


class TestOpenAITypes:
    def test_chat_completion_parse(self):
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        resp = ChatCompletion.model_validate(data)
        assert resp.id == "chatcmpl-123"
        assert resp.model == "test-model"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 7

    def test_chat_completion_with_reasoning(self):
        data = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 0,
            "model": "thinking-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "reasoning_content": "Let me think... 6*7=42",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        resp = ChatCompletion.model_validate(data)
        assert resp.choices[0].message.reasoning_content == "Let me think... 6*7=42"
        assert resp.choices[0].message.content == "42"

    def test_chat_completion_with_tool_calls(self):
        data = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        resp = ChatCompletion.model_validate(data)
        tc = resp.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].function.name == "get_weather"
        assert tc[0].function.arguments == '{"location": "Tokyo"}'

    def test_streaming_chunk(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        chunk = ChatCompletionChunk.model_validate(data)
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello"

    def test_model_list(self):
        data = {
            "object": "list",
            "data": [
                {
                    "id": "Qwen3-8B-MLX-4bit",
                    "object": "model",
                    "created": 0,
                    "owned_by": "novamlx",
                    "nova": {
                        "capabilities": {
                            "reasoning": True,
                            "thinking": True,
                            "tools": True,
                            "vision": False,
                        }
                    },
                }
            ],
        }
        resp = ModelListResponse.model_validate(data)
        assert len(resp.data) == 1
        model = resp.data[0]
        assert model.id == "Qwen3-8B-MLX-4bit"
        assert model.nova is not None
        assert model.nova.capabilities.reasoning is True
        assert model.nova.capabilities.vision is False

    def test_usage(self):
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30

    def test_content_part_image(self):
        part = ChatMessageContentPart(
            type="image_url",
            image_url={"url": "data:image/png;base64,abc"},
        )
        assert part.type == "image_url"
        assert part.image_url is not None


class TestAnthropicTypes:
    def test_anthropic_response(self):
        data = {
            "id": "msg-123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "model": "test-model",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        resp = AnthropicResponse.model_validate(data)
        assert resp.id == "msg-123"
        assert len(resp.content) == 2
        assert resp.content[0].type == "thinking"
        assert resp.content[1].type == "text"
        assert resp.usage.input_tokens == 10
        assert resp.stop_reason == "end_turn"

    def test_anthropic_tool_use(self):
        data = {
            "id": "msg-456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "search",
                    "input": {"query": "test"},
                }
            ],
            "model": "test",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        resp = AnthropicResponse.model_validate(data)
        block = resp.content[0]
        assert block.type == "tool_use"
        assert block.name == "search"


class TestOtherTypes:
    def test_embedding_response(self):
        data = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
            ],
            "model": "embed-model",
            "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
        }
        resp = EmbeddingResponse.model_validate(data)
        assert len(resp.data) == 1
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]

    def test_rerank_response(self):
        data = {
            "model": "rerank-model",
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "doc1"}},
                {"index": 1, "relevance_score": 0.3, "document": {"text": "doc2"}},
            ],
        }
        resp = RerankResponse.model_validate(data)
        assert len(resp.results) == 2
        assert resp.results[0].relevance_score == 0.95

    def test_health_response(self):
        data = {
            "status": "ok",
            "loaded_models": ["model-1"],
            "gpu_memory_used_gb": 4.2,
            "gpu_memory_total_gb": 16.0,
            "uptime_seconds": 3600.0,
        }
        resp = HealthResponse.model_validate(data)
        assert resp.status == "ok"
        assert resp.loaded_models == ["model-1"]
