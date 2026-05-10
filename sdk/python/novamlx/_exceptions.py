"""NovaMLX error hierarchy."""

from __future__ import annotations


class NovaMLXError(Exception):
    """Base exception for all NovaMLX errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ConnectionError(NovaMLXError):
    """Can't reach the NovaMLX server."""


class AuthenticationError(NovaMLXError):
    """Invalid or missing API key."""


class ModelNotFoundError(NovaMLXError):
    """Requested model not found. Check model ID or load it via admin API."""


class ModelLoadTimeoutError(NovaMLXError):
    """Model cold-load exceeded timeout. Try increasing client.timeout."""


class RateLimitError(NovaMLXError):
    """Request rate limited."""


class ServerError(NovaMLXError):
    """Internal server error (5xx)."""


class BadRequestError(NovaMLXError):
    """Invalid request parameters (400)."""
