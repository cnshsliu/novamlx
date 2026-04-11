# SPDX-License-Identifier: Apache-2.0
"""
SSE (Server-Sent Events) formatting utilities.

This module provides abstract and concrete implementations for formatting
SSE events in different API formats (OpenAI, Anthropic).
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class SSEFormatter(ABC):
    """Abstract base class for SSE event formatting."""

    @abstractmethod
    def format_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Format an SSE event.

        Args:
            event_type: Type of event (e.g., "content_block_delta").
            data: Event data to serialize.

        Returns:
            Formatted SSE event string.
        """
        pass

    @abstractmethod
    def format_end(self) -> str:
        """
        Format the stream end marker.

        Returns:
            Formatted SSE end marker string.
        """
        pass


class OpenAISSEFormatter(SSEFormatter):
    """
    SSE formatter for OpenAI API format.

    OpenAI uses a simple format: `data: {json}\n\n`
    The stream ends with `data: [DONE]\n\n`
    """

    def format_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Format an SSE event in OpenAI format.

        Note: OpenAI doesn't use event types in SSE, only data lines.

        Args:
            event_type: Ignored in OpenAI format.
            data: Event data to serialize.

        Returns:
            Formatted SSE event string.
        """
        return f"data: {json.dumps(data)}\n\n"

    def format_end(self) -> str:
        """
        Format the stream end marker for OpenAI format.

        Returns:
            The [DONE] marker.
        """
        return "data: [DONE]\n\n"


class AnthropicSSEFormatter(SSEFormatter):
    """
    SSE formatter for Anthropic API format.

    Anthropic uses event types: `event: {type}\ndata: {json}\n\n`
    The stream ends with a message_stop event (no separate [DONE] marker).
    """

    def format_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Format an SSE event in Anthropic format.

        Args:
            event_type: Type of event (e.g., "message_start", "content_block_delta").
            data: Event data to serialize.

        Returns:
            Formatted SSE event string with event type.
        """
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def format_end(self) -> str:
        """
        Format the stream end marker for Anthropic format.

        Anthropic doesn't use a separate end marker - the message_stop
        event is the end of the stream.

        Returns:
            Empty string.
        """
        return ""
