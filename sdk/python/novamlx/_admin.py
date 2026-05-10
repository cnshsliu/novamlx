"""NovaMLX Admin API client — model management, sessions, cache, benchmarks."""

from __future__ import annotations

from typing import Any

import httpx

from ._client import _build_headers, _handle_error
from ._config import discover_admin_base_url, discover_api_key
from ._exceptions import ConnectionError, ModelLoadTimeoutError
from ._types import (
    AdminModelListResponse,
    BenchmarkRequest,
    BenchmarkStatus,
    CacheStats,
    DeviceInfo,
    ModelSettings,
    SessionInfo,
    SessionListResponse,
)

_DEFAULT_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Nested admin resource classes
# ---------------------------------------------------------------------------

class _AdminModels:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def list(self) -> AdminModelListResponse:
        data = self._request_json("GET", "/admin/models")
        if isinstance(data, list):
            return AdminModelListResponse(models=data)
        return AdminModelListResponse.model_validate(data)

    def load(self, model_id: str) -> dict[str, Any]:
        return self._request_json("POST", "/admin/models/load", json={"model_id": model_id})

    def unload(self, model_id: str) -> dict[str, Any]:
        return self._request_json("POST", "/admin/models/unload", json={"model_id": model_id})

    def download(self, model_id: str) -> dict[str, Any]:
        return self._request_json("POST", "/admin/models/download", json={"model_id": model_id})

    def download_status(self, model_id: str) -> dict[str, Any]:
        return self._request_json("POST", "/admin/models/status", json={"model_id": model_id})

    def discover(self) -> dict[str, Any]:
        return self._request_json("POST", "/admin/models/discover")

    def delete(self, model_id: str) -> dict[str, Any]:
        resp = self._request_raw("DELETE", f"/admin/models/{model_id}")
        return resp.json()

    def get_settings(self, model_id: str) -> ModelSettings:
        data = self._request_json("GET", f"/admin/models/{model_id}/settings")
        return ModelSettings.model_validate(data)

    def update_settings(self, model_id: str, settings: dict[str, Any]) -> ModelSettings:
        data = self._request_json("PUT", f"/admin/models/{model_id}/settings", json=settings)
        return ModelSettings.model_validate(data)

    def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, path, **kwargs)
        return response.json()


class _AdminSessions:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def list(self) -> SessionListResponse:
        data = self._request_json("GET", "/admin/sessions")
        if isinstance(data, list):
            return SessionListResponse(sessions=data)
        return SessionListResponse.model_validate(data)

    def delete(self, session_id: str) -> dict[str, Any]:
        resp = self._request_raw("DELETE", f"/admin/sessions/{session_id}")
        return resp.json()

    def delete_all(self) -> dict[str, Any]:
        resp = self._request_raw("DELETE", "/admin/sessions")
        return resp.json()

    def save(self, session_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/admin/sessions/{session_id}/save")

    def fork(
        self, source_id: str, target_id: str, model_id: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if model_id:
            body["model_id"] = model_id
        return self._request_json("POST", "/admin/sessions/fork", json=body)

    def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, path, **kwargs)
        return response.json()


class _AdminCache:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def stats(self, model_id: str) -> CacheStats:
        data = self._request_json("GET", f"/admin/cache/{model_id}/stats")
        return CacheStats.model_validate(data)

    def clear(self, model_id: str) -> dict[str, Any]:
        resp = self._request_raw("DELETE", f"/admin/cache/{model_id}")
        return resp.json()

    def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, path, **kwargs)
        return response.json()


class _AdminBenchmark:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def start(
        self,
        model_id: str,
        prompt_lengths: list[int] | None = None,
        generation_length: int = 128,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_id": model_id,
            "prompt_lengths": prompt_lengths or [1024, 4096],
            "generation_length": generation_length,
        }
        return self._request_json("POST", "/admin/api/bench/start", json=body)

    def status(self) -> BenchmarkStatus:
        data = self._request_json("GET", "/admin/api/bench/status")
        return BenchmarkStatus.model_validate(data)

    def cancel(self) -> dict[str, Any]:
        return self._request_json("POST", "/admin/api/bench/cancel")

    def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, path, **kwargs)
        return response.json()


# ---------------------------------------------------------------------------
# AdminClient (sync)
# ---------------------------------------------------------------------------

class AdminClient:
    """Synchronous NovaMLX Admin API client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (base_url or discover_admin_base_url()).rstrip("/")
        self._api_key = api_key if api_key is not None else discover_api_key()
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=_build_headers(self._api_key),
            timeout=httpx.Timeout(timeout, read=timeout),
        )
        self.models = _AdminModels(self._http)
        self.sessions = _AdminSessions(self._http)
        self.cache = _AdminCache(self._http)
        self.benchmark = _AdminBenchmark(self._http)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AdminClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def device_info(self) -> DeviceInfo:
        data = self._request_json("GET", "/admin/api/device-info")
        return DeviceInfo.model_validate(data)

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

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
            raise ModelLoadTimeoutError("Admin request timed out") from e

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, path, **kwargs)
        return response.json()


# ---------------------------------------------------------------------------
# AsyncAdminClient
# ---------------------------------------------------------------------------

class _AsyncAdminModels:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def list(self) -> AdminModelListResponse:
        data = await self._request_json("GET", "/admin/models")
        if isinstance(data, list):
            return AdminModelListResponse(models=data)
        return AdminModelListResponse.model_validate(data)

    async def load(self, model_id: str) -> dict[str, Any]:
        return await self._request_json("POST", "/admin/models/load", json={"model_id": model_id})

    async def unload(self, model_id: str) -> dict[str, Any]:
        return await self._request_json("POST", "/admin/models/unload", json={"model_id": model_id})

    async def download(self, model_id: str) -> dict[str, Any]:
        return await self._request_json("POST", "/admin/models/download", json={"model_id": model_id})

    async def download_status(self, model_id: str) -> dict[str, Any]:
        return await self._request_json("POST", "/admin/models/status", json={"model_id": model_id})

    async def discover(self) -> dict[str, Any]:
        return await self._request_json("POST", "/admin/models/discover")

    async def delete(self, model_id: str) -> dict[str, Any]:
        resp = await self._request_raw("DELETE", f"/admin/models/{model_id}")
        return resp.json()

    async def get_settings(self, model_id: str) -> ModelSettings:
        data = await self._request_json("GET", f"/admin/models/{model_id}/settings")
        return ModelSettings.model_validate(data)

    async def update_settings(self, model_id: str, settings: dict[str, Any]) -> ModelSettings:
        data = await self._request_json("PUT", f"/admin/models/{model_id}/settings", json=settings)
        return ModelSettings.model_validate(data)

    async def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request_raw(method, path, **kwargs)
        return response.json()


class _AsyncAdminSessions:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def list(self) -> SessionListResponse:
        data = await self._request_json("GET", "/admin/sessions")
        if isinstance(data, list):
            return SessionListResponse(sessions=data)
        return SessionListResponse.model_validate(data)

    async def delete(self, session_id: str) -> dict[str, Any]:
        resp = await self._request_raw("DELETE", f"/admin/sessions/{session_id}")
        return resp.json()

    async def delete_all(self) -> dict[str, Any]:
        resp = await self._request_raw("DELETE", "/admin/sessions")
        return resp.json()

    async def save(self, session_id: str) -> dict[str, Any]:
        return await self._request_json("POST", f"/admin/sessions/{session_id}/save")

    async def fork(
        self, source_id: str, target_id: str, model_id: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"source_id": source_id, "target_id": target_id}
        if model_id:
            body["model_id"] = model_id
        return await self._request_json("POST", "/admin/sessions/fork", json=body)

    async def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request_raw(method, path, **kwargs)
        return response.json()


class _AsyncAdminCache:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def stats(self, model_id: str) -> CacheStats:
        data = await self._request_json("GET", f"/admin/cache/{model_id}/stats")
        return CacheStats.model_validate(data)

    async def clear(self, model_id: str) -> dict[str, Any]:
        resp = await self._request_raw("DELETE", f"/admin/cache/{model_id}")
        return resp.json()

    async def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request_raw(method, path, **kwargs)
        return response.json()


class _AsyncAdminBenchmark:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def start(
        self,
        model_id: str,
        prompt_lengths: list[int] | None = None,
        generation_length: int = 128,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_id": model_id,
            "prompt_lengths": prompt_lengths or [1024, 4096],
            "generation_length": generation_length,
        }
        return await self._request_json("POST", "/admin/api/bench/start", json=body)

    async def status(self) -> BenchmarkStatus:
        data = await self._request_json("GET", "/admin/api/bench/status")
        return BenchmarkStatus.model_validate(data)

    async def cancel(self) -> dict[str, Any]:
        return await self._request_json("POST", "/admin/api/bench/cancel")

    async def _request_raw(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _handle_error(e) from e
        except httpx.ConnectError as e:
            raise ConnectionError("Cannot connect to admin API") from e
        except httpx.TimeoutException as e:
            raise ModelLoadTimeoutError("Admin request timed out") from e

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request_raw(method, path, **kwargs)
        return response.json()


class AsyncAdminClient:
    """Asynchronous NovaMLX Admin API client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (base_url or discover_admin_base_url()).rstrip("/")
        self._api_key = api_key if api_key is not None else discover_api_key()
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=_build_headers(self._api_key),
            timeout=httpx.Timeout(timeout, read=timeout),
        )
        self.models = _AsyncAdminModels(self._http)
        self.sessions = _AsyncAdminSessions(self._http)
        self.cache = _AsyncAdminCache(self._http)
        self.benchmark = _AsyncAdminBenchmark(self._http)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncAdminClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def device_info(self) -> DeviceInfo:
        data = await self._request_json("GET", "/admin/api/device-info")
        return DeviceInfo.model_validate(data)

    async def health(self) -> dict[str, Any]:
        return await self._request_json("GET", "/health")

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
            raise ModelLoadTimeoutError("Admin request timed out") from e

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request_raw(method, path, **kwargs)
        return response.json()
