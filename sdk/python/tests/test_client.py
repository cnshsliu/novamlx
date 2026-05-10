"""Tests for client request building and error handling."""

import pytest
import httpx
from unittest.mock import MagicMock, patch

from novamlx._client import (
    Client,
    _build_request_body,
    _build_anthropic_body,
    _build_headers,
    _handle_error,
)
from novamlx._exceptions import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    ModelLoadTimeoutError,
    ModelNotFoundError,
    NovaMLXError,
    RateLimitError,
    ServerError,
)


class TestBuildRequestBody:
    def test_minimal(self):
        body = _build_request_body(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is False
        assert "temperature" not in body

    def test_with_novamlx_extensions(self):
        body = _build_request_body(
            model="test",
            messages=[{"role": "user", "content": "think"}],
            thinking_budget=8192,
            enable_thinking=True,
            reasoning_effort="high",
            session_id="sess-1",
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
        )
        assert body["thinking_budget"] == 8192
        assert body["enable_thinking"] is True
        assert body["reasoning_effort"] == "high"
        assert body["session_id"] == "sess-1"
        assert body["top_k"] == 40
        assert body["min_p"] == 0.05
        assert body["repetition_penalty"] == 1.1

    def test_with_response_format(self):
        body = _build_request_body(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "regex", "regex": r"\d{3}"},
        )
        assert body["response_format"]["type"] == "regex"

    def test_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        body = _build_request_body(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
        )
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"


class TestBuildAnthropicBody:
    def test_basic(self):
        body = _build_anthropic_body(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert body["model"] == "test"
        assert body["max_tokens"] == 4096
        assert body["stream"] is False

    def test_with_system(self):
        body = _build_anthropic_body(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )
        assert body["system"] == "You are helpful."

    def test_with_extensions(self):
        body = _build_anthropic_body(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking_budget=4096,
            enable_thinking=True,
            top_k=40,
        )
        assert body["thinking_budget"] == 4096
        assert body["top_k"] == 40


class TestBuildHeaders:
    def test_no_api_key(self):
        headers = _build_headers(None)
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_with_api_key(self):
        headers = _build_headers("sk-test-123")
        assert headers["Authorization"] == "Bearer sk-test-123"


class TestErrorHandling:
    def _make_error(self, status_code: int, text: str = "error") -> httpx.HTTPStatusError:
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.text = text
        return httpx.HTTPStatusError("err", request=MagicMock(), response=response)

    def test_401_auth_error(self):
        err = _handle_error(self._make_error(401, "unauthorized"))
        assert isinstance(err, AuthenticationError)

    def test_404_model_not_found(self):
        err = _handle_error(self._make_error(404, "not found"))
        assert isinstance(err, ModelNotFoundError)

    def test_429_rate_limit(self):
        err = _handle_error(self._make_error(429, "slow down"))
        assert isinstance(err, RateLimitError)

    def test_400_bad_request(self):
        err = _handle_error(self._make_error(400, "bad input"))
        assert isinstance(err, BadRequestError)

    def test_500_server_error(self):
        err = _handle_error(self._make_error(500, "internal"))
        assert isinstance(err, ServerError)

    def test_503_load_timeout(self):
        err = _handle_error(self._make_error(503, "model load timeout"))
        assert isinstance(err, ModelLoadTimeoutError)

    def test_generic_error(self):
        err = _handle_error(self._make_error(418, "teapot"))
        assert isinstance(err, NovaMLXError)


class TestClientInit:
    def test_explicit_params(self):
        client = Client(base_url="http://localhost:9999/v1", api_key="sk-test", timeout=30.0)
        assert client._base_url == "http://localhost:9999/v1"
        assert client._api_key == "sk-test"
        assert client._timeout == 30.0
        client.close()

    @patch("novamlx._client.discover_base_url", return_value="http://discovered:6590/v1")
    @patch("novamlx._client.discover_api_key", return_value=None)
    def test_auto_discover(self, mock_key, mock_url):
        client = Client()
        assert client._base_url == "http://discovered:6590/v1"
        assert client._api_key is None
        client.close()
