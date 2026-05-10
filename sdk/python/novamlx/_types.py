"""Pydantic models for NovaMLX API requests and responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / nested types
# ---------------------------------------------------------------------------

class ModelCapabilities(BaseModel):
    reasoning: bool = False
    thinking: bool = False
    tools: bool = False
    vision: bool = False


class NovaExtension(BaseModel):
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completion types
# ---------------------------------------------------------------------------

class ChatMessageContentPart(BaseModel):
    type: str = "text"
    text: str | None = None
    image_url: dict[str, Any] | None = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatCompletionMessage(BaseModel):
    role: str
    content: str | list[ChatMessageContentPart] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobInfo(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] | None = None


class ChatCompletionLogprobs(BaseModel):
    content: list[LogprobInfo] | None = None


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# OpenAI streaming chunk types
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: str | None = None
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[StreamChoice] = Field(default_factory=list)
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "novamlx"
    nova: NovaExtension | None = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request types (used as typed dicts for building request bodies)
# ---------------------------------------------------------------------------

class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: dict[str, Any] | None = None
    regex: str | None = None
    gbnf: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class StreamOptions(BaseModel):
    include_usage: bool = False


# ---------------------------------------------------------------------------
# Anthropic-compatible message types
# ---------------------------------------------------------------------------

class AnthropicTextBlock(BaseModel):
    type: str = "text"
    text: str


class AnthropicThinkingBlock(BaseModel):
    type: str = "thinking"
    thinking: str


class AnthropicToolUseBlock(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    type: str = "tool_result"
    tool_use_id: str
    content: str | None = None
    is_error: bool = False


AnthropicContentBlock = AnthropicTextBlock | AnthropicThinkingBlock | AnthropicToolUseBlock


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock] = Field(default_factory=list)
    model: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ---------------------------------------------------------------------------
# Anthropic streaming events
# ---------------------------------------------------------------------------

class AnthropicMessageStartEvent(BaseModel):
    type: str = "message_start"
    message: AnthropicResponse


class AnthropicContentBlockStartEvent(BaseModel):
    type: str = "content_block_start"
    index: int = 0
    content_block: AnthropicContentBlock


class AnthropicTextDelta(BaseModel):
    type: str = "text_delta"
    text: str


class AnthropicThinkingDelta(BaseModel):
    type: str = "thinking_delta"
    thinking: str


class AnthropicContentBlockDeltaEvent(BaseModel):
    type: str = "content_block_delta"
    index: int = 0
    delta: AnthropicTextDelta | AnthropicThinkingDelta


class AnthropicMessageDeltaStopReason(BaseModel):
    stop_reason: str | None = None
    stop_sequence: str | None = None


class AnthropicMessageDeltaUsage(BaseModel):
    output_tokens: int = 0


class AnthropicMessageDeltaEvent(BaseModel):
    type: str = "message_delta"
    delta: AnthropicMessageDeltaStopReason
    usage: AnthropicMessageDeltaUsage


class AnthropicMessageStopEvent(BaseModel):
    type: str = "message_stop"


AnthropicStreamEvent = (
    AnthropicMessageStartEvent
    | AnthropicContentBlockStartEvent
    | AnthropicContentBlockDeltaEvent
    | AnthropicMessageDeltaEvent
    | AnthropicMessageStopEvent
)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float] = Field(default_factory=list)


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str = ""
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

class RerankDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocument | None = None


class RerankResponse(BaseModel):
    model: str = ""
    results: list[RerankResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Admin API types
# ---------------------------------------------------------------------------

class AdminModelInfo(BaseModel):
    model_id: str
    status: str = "unknown"
    loaded: bool = False
    downloaded: bool = False
    memory_feasible: bool = True


class AdminModelListResponse(BaseModel):
    models: list[AdminModelInfo] = Field(default_factory=list)


class ModelSettings(BaseModel):
    max_context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    ttl_seconds: float | None = None
    model_alias: str | None = None
    is_pinned: bool | None = None
    is_default: bool | None = None
    display_name: str | None = None
    description: str | None = None
    thinking_budget: int | None = None
    kv_bits: int | None = None
    kv_group_size: int | None = None
    kv_memory_bytes_per_token_override: int | None = None


class SessionInfo(BaseModel):
    id: str
    model_id: str = ""
    created_at: float = 0
    last_used: float = 0
    token_count: int = 0


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo] = Field(default_factory=list)


class CacheStats(BaseModel):
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    blocks: int = 0


class DeviceInfo(BaseModel):
    chip: str = ""
    memory_gb: float = 0.0
    metal_support: str = ""


class BenchmarkRequest(BaseModel):
    model_id: str
    prompt_lengths: list[int] = Field(default_factory=lambda: [1024, 4096])
    generation_length: int = 128


class BenchmarkStatus(BaseModel):
    running: bool = False
    model_id: str | None = None
    progress: float = 0.0
    results: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    loaded_models: list[str] = Field(default_factory=list)
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    uptime_seconds: float = 0.0
