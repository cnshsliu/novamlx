"""Auto-discover NovaMLX configuration from env vars and config file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_config() -> dict[str, Any]:
    """Read raw config dict from ~/.nova/config.json, or return empty."""
    config_path = Path.home() / ".nova" / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def discover_base_url() -> str:
    """Auto-discover NovaMLX inference API base URL.

    Priority:
    1. NOVA_MLX_URL env var
    2. ~/.nova/config.json -> server.host + server.port
    3. Default http://127.0.0.1:6590/v1
    """
    if url := os.environ.get("NOVA_MLX_URL"):
        return url

    config = _read_config()
    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 6590)
    return f"http://{host}:{port}/v1"


def discover_admin_base_url() -> str:
    """Auto-discover NovaMLX admin API base URL."""
    if url := os.environ.get("NOVA_MLX_ADMIN_URL"):
        return url

    config = _read_config()
    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("adminPort", 6591)
    return f"http://{host}:{port}"


def discover_api_key() -> str | None:
    """Auto-discover API key.

    Priority:
    1. NOVA_API_KEY env var
    2. ~/.nova/config.json -> server.apiKeys[0]
    """
    if key := os.environ.get("NOVA_API_KEY"):
        return key

    config = _read_config()
    server = config.get("server", {})
    keys = server.get("apiKeys", [])
    return keys[0] if keys else None
