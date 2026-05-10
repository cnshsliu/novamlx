"""NovaMLX client — sync and async, OpenAI + Anthropic formats."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Literal, overload

import httpx

from ._config import discover_admin_base_url, discover_api_key, discover_base_url
from ._exceptions import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    ModelLoadTimeoutError,
    ModelNotFoundError,
    NovaMLXError,
    RateLimitError,
    ServerError,
)
from ._streaming import (
    async_parse_anthropic_sse,
    async_parse_sse_stream,
    parse_anthropic_sse,
    parse_sse_stream,
)
from ._types import (
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicMessageDeltaEvent,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
    AnthropicResponse,
    ChatCompletion,
    ChatCompletionChunk,
    EmbeddingResponse,
    HealthResponse,
    ModelListResponse,
    RerankResponse,
)

_DEFAULT_TIMEOUT = 120.0


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _handle_error(exc: httpx.HTTPStatusError) -> NovaMLXError:
    """Map HTTP status codes to typed exceptions."""
    code = exc.response.status_code
    body = ""
    try:
        body = exc.response.text
    except Exception:
        pass
    msg = f"HTTP {code}: {body}"

    if code == 401:
        return AuthenticationError(msg, status_code=code)
    if code == 404:
        return ModelNotFoundError(msg, status_code=code)
    if code == 429:
        return RateLimitError(msg, status_code=code)
    if code == 400:
        return BadRequestError(msg, status_code=code)
    if code >= 500:
        if "timeout" in body.lower() or "load" in body.lower():
            return ModelLoadTimeoutError(msg, status_code=code)
        return ServerError(msg, status_code=code)
    return NovaMLXError(msg, status_code=code)


def _build_request_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
    stream_options: dict[str, Any] | None = None,
    stop: list[str] | str | None = None,
    n: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    seed: int | None = None,
    session_id: str | None = None,
    response_format: dict[str, Any] | None = None,
    thinking_budget: int | None = None,
    enable_thinking: bool | None = None,
    preserve_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    """Build the request body, omitting None values."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if min_p is not None:
        body["min_p"] = min_p
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stream_options is not None:
        body["stream_options"] = stream_options
    if stop is not None:
        body["stop"] = stop
    if n is not None:
        body["n"] = n
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        body["repetition_penalty"] = repetition_penalty
    if seed is not None:
        body["seed"] = seed
    if session_id is not None:
        body["session_id"] = session_id
    if response_format is not None:
        body["response_format"] = response_format
    if thinking_budget is not None:
        body["thinking_budget"] = thinking_budget
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if preserve_thinking is not None:
        body["preserve_thinking"] = preserve_thinking
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if logprobs is not None:
        body["logprobs"] = logprobs
    if top_logprobs is not None:
        body["top_logprobs"] = top_logprobs
    return body


def _build_anthropic_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    system: str | list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stream: bool = False,
    stop_sequences: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    thinking_budget: int | None = None,
    enable_thinking: bool | None = None,
    preserve_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Anthropic-format request body."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if system is not None:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if stop_sequences is not None:
        body["stop_sequences"] = stop_sequences
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if thinking_budget is not None:
        body["thinking_budget"] = thinking_budget
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if preserve_thinking is not None:
        body["preserve_thinking"] = preserve_thinking
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    return body


# ---------------------------------------------------------------------------
# Nested API resource classes (for the .chat.completions.create style)
# ---------------------------------------------------------------------------

class SyncChatCompletions:
    def __init__(self, client: Client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        stream: Literal[True],
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]: ...

    @overload
    def create(
        self,
        *,
        stream: Literal[False] = False,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion: ...

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        body = _build_request_body(
            model=model, messages=messages, stream=stream, **kwargs
        )
        if stream:
            response = self._client._request_raw(
                "POST", "/chat/completions", json=body
            )
            return parse_sse_stream(response, ChatCompletionChunk)
        else:
            data = self._client._request_json("POST", "/chat/completions", json=body)
            return ChatCompletion.model_validate(data)


class AsyncChatCompletions:
    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        stream: Literal[True],
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    @overload
    async def create(
        self,
        *,
        stream: Literal[False] = False,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion: ...

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        body = _build_request_body(
            model=model, messages=messages, stream=stream, **kwargs
        )
        if stream:
            response = await self._client._request_raw(
                "POST", "/chat/completions", json=body
            )
            return async_parse_sse_stream(response, ChatCompletionChunk)
        else:
            data = await self._client._request_json("POST", "/chat/completions", json=body)
            return ChatCompletion.model_validate(data)


class SyncChat:
    def __init__(self, client: Client) -> None:
        self.completions = SyncChatCompletions(client)


class AsyncChat:
    def __init__(self, client: AsyncClient) -> None:
        self.completions = AsyncChatCompletions(client)


class SyncModels:
    def __init__(self, client: Client) -> None:
        self._client = client

    def list(self) -> ModelListResponse:
        data = self._client._request_json("GET", "/models")
        return ModelListResponse.model_validate(data)


class AsyncModels:
    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def list(self) -> ModelListResponse:
        data = await self._client._request_json("GET", "/models")
        return ModelListResponse.model_validate(data)


class SyncMessages:
    """Anthropic-format /v1/messages endpoint."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        system: str | list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> AnthropicResponse:
        body = _build_anthropic_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            stream=stream,
            **kwargs,
        )
        if stream:
            response = self._client._request_raw("POST", "/messages", json=body)
            return parse_anthropic_sse(response)
        else:
            data = self._client._request_json("POST", "/messages", json=body)
            return AnthropicResponse.model_validate(data)


class AsyncMessages:
    """Anthropic-format /v1/messages endpoint (async)."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        system: str | list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> AnthropicResponse:
        body = _build_anthropic_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            stream=stream,
            **kwargs,
        )
        if stream:
            response = await self._client._request_raw("POST", "/messages", json=body)
            return async_parse_anthropic_sse(response)
        else:
            data = await self._client._request_json("POST", "/messages", json=body)
            return AnthropicResponse.model_validate(data)


# ---------------------------------------------------------------------------
# Sync Client
# ---------------------------------------------------------------------------

class Client:
    """Synchronous NovaMLX client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (base_url or discover_base_url()).rstrip("/")
        self._api_key = api_key if api_key is not None else discover_api_key()
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=_build_headers(self._api_key),
            timeout=httpx.Timeout(timeout, read=timeout),
        )
        self.chat = SyncChat(self)
        self.models = SyncModels(self)
        self.messages = SyncMessages(self)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to {self._base_url}") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, path, **kwargs)
        return response.json()

    # Convenience methods

    def health(self) -> HealthResponse:
        data = self._request_json("GET", "/../health")
        return HealthResponse.model_validate(data)

    def embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str = "float",
    ) -> EmbeddingResponse:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        data = self._request_json("POST", "/embeddings", json=body)
        return EmbeddingResponse.model_validate(data)

    def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        top_n: int = 5,
        return_documents: bool = True,
    ) -> RerankResponse:
        body: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        data = self._request_json("POST", "/rerank", json=body)
        return RerankResponse.model_validate(data)


# ---------------------------------------------------------------------------
# Async Client
# ---------------------------------------------------------------------------

class AsyncClient:
    """Asynchronous NovaMLX client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (base_url or discover_base_url()).rstrip("/")
        self._api_key = api_key if api_key is not None else discover_api_key()
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=_build_headers(self._api_key),
            timeout=httpx.Timeout(timeout, read=timeout),
        )
        self.chat = AsyncChat(self)
        self.models = AsyncModels(self)
        self.messages = AsyncMessages(self)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to {self._base_url}") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request_raw(method, path, **kwargs)
        return response.json()

    async def health(self) -> HealthResponse:
        data = await self._request_json("GET", "/../health")
        return HealthResponse.model_validate(data)

    async def embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str = "float",
    ) -> EmbeddingResponse:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        data = await self._request_json("POST", "/embeddings", json=body)
        return EmbeddingResponse.model_validate(data)

    async def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        top_n: int = 5,
        return_documents: bool = True,
    ) -> RerankResponse:
        body: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        data = await self._request_json("POST", "/rerank", json=body)
        return RerankResponse.model_validate(data)
