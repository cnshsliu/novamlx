"""SSE stream parser for NovaMLX streaming responses."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator, Type, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_sse_stream(response: httpx.Response, model_type: Type[T]) -> Iterator[T]:
    """Parse SSE stream from NovaMLX into typed chunks (sync)."""
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            yield model_type.model_validate_json(data)
        elif line.startswith("event:"):
            continue
        elif line.startswith(":"):
            continue


def parse_sse_raw(response: httpx.Response) -> Iterator[dict[str, Any]]:
    """Parse SSE stream into raw dicts (sync)."""
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


async def async_parse_sse_stream(
    response: httpx.Response, model_type: Type[T]
) -> AsyncIterator[T]:
    """Parse SSE stream from NovaMLX into typed chunks (async)."""
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            yield model_type.model_validate_json(data)
        elif line.startswith("event:"):
            continue
        elif line.startswith(":"):
            continue


async def async_parse_sse_raw(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
    """Parse SSE stream into raw dicts (async)."""
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Anthropic SSE parser — events have "event:" + "data:" lines
# ---------------------------------------------------------------------------

def parse_anthropic_sse(
    response: httpx.Response,
) -> Iterator[dict[str, Any]]:
    """Parse Anthropic-format SSE stream (sync)."""
    event_type: str | None = None
    for line in response.iter_lines():
        if not line:
            event_type = None
            continue
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            data = line[6:]
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event_type:
                payload["_event_type"] = event_type
            yield payload
            event_type = None


async def async_parse_anthropic_sse(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
    """Parse Anthropic-format SSE stream (async)."""
    event_type: str | None = None
    async for line in response.aiter_lines():
        if not line:
            event_type = None
            continue
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            data = line[6:]
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event_type:
                payload["_event_type"] = event_type
            yield payload
            event_type = None
