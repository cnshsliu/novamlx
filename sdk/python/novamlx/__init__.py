"""NovaMLX Python SDK — zero-config local LLM client.

Usage:
    import novamlx

    # Zero-config: auto-discovers from NOVA_MLX_URL or ~/.nova/config.json
    response = novamlx.chat.completions.create(
        model="Qwen3-8B-MLX-4bit",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)
"""

from ._client import AsyncClient, Client
from ._admin import AdminClient, AsyncAdminClient
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
from ._types import (
    AnthropicResponse,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatMessageContentPart,
    DeltaMessage,
    EmbeddingResponse,
    HealthResponse,
    ModelCapabilities,
    ModelInfo,
    ModelListResponse,
    NovaExtension,
    RerankResponse,
    ToolCall,
    ToolCallFunction,
    Usage,
)

__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    "AdminClient",
    "AsyncAdminClient",
    # Exceptions
    "NovaMLXError",
    "ConnectionError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ModelLoadTimeoutError",
    "RateLimitError",
    "ServerError",
    "BadRequestError",
    # Types
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatMessageContentPart",
    "DeltaMessage",
    "ToolCall",
    "ToolCallFunction",
    "Usage",
    "ModelInfo",
    "ModelListResponse",
    "ModelCapabilities",
    "NovaExtension",
    "HealthResponse",
    "EmbeddingResponse",
    "RerankResponse",
    "AnthropicResponse",
]

# Module-level singleton for zero-config usage
_client = Client()


def _get_client() -> Client:
    """Get or create the module-level client."""
    global _client
    if _client._http.is_closed:
        _client = Client()
    return _client


def configure(**kwargs: object) -> None:
    """Reconfigure the module-level singleton client.

    Accepts same args as Client: base_url, api_key, timeout.
    """
    global _client
    _client.close()
    _client = Client(**kwargs)  # type: ignore[arg-type]


class _ModuleChat:
    """Proxy that delegates to the module-level client's chat resource."""

    @property
    def completions(self):
        return _get_client().chat.completions


class _ModuleMessages:
    """Proxy that delegates to the module-level client's messages resource."""

    def create(self, **kwargs: object):
        return _get_client().messages.create(**kwargs)  # type: ignore[arg-type]


class _ModuleModels:
    """Proxy that delegates to the module-level client's models resource."""

    def list(self):
        return _get_client().models.list()


# Module-level API surface
chat = _ModuleChat()
models = _ModuleModels()
messages = _ModuleMessages()
