# SPDX-License-Identifier: Apache-2.0
"""
API Adapters for oMLX.

This package provides adapters for different API formats (OpenAI, Anthropic),
enabling clean separation between API-specific logic and core inference.
"""

from .base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .sse_formatter import (
    SSEFormatter,
    OpenAISSEFormatter,
    AnthropicSSEFormatter,
)

__all__ = [
    # Base classes and types
    "BaseAdapter",
    "InternalMessage",
    "InternalRequest",
    "InternalResponse",
    "StreamChunk",
    # Adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    # SSE Formatters
    "SSEFormatter",
    "OpenAISSEFormatter",
    "AnthropicSSEFormatter",
]
