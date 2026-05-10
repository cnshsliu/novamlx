"""Tests for Admin API client."""

import pytest
import httpx
from unittest.mock import MagicMock, patch

from novamlx._admin import AdminClient
from novamlx._types import (
    AdminModelListResponse,
    CacheStats,
    DeviceInfo,
    ModelSettings,
    SessionListResponse,
    BenchmarkStatus,
)


class TestAdminClientInit:
    @patch("novamlx._admin.discover_admin_base_url", return_value="http://localhost:6591")
    @patch("novamlx._admin.discover_api_key", return_value="sk-test")
    def test_auto_discover(self, mock_key, mock_url):
        admin = AdminClient()
        assert admin._base_url == "http://localhost:6591"
        assert admin._api_key == "sk-test"
        admin.close()

    def test_explicit_params(self):
        admin = AdminClient(base_url="http://myhost:9999", api_key="sk-123")
        assert admin._base_url == "http://myhost:9999"
        assert admin._api_key == "sk-123"
        admin.close()


class TestAdminModelsMocked:
    @patch("novamlx._admin.discover_admin_base_url", return_value="http://localhost:6591")
    @patch("novamlx._admin.discover_api_key", return_value="sk-test")
    def test_list_models(self, mock_key, mock_url):
        admin = AdminClient()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"model_id": "model-1", "status": "loaded", "loaded": True, "downloaded": True},
            {"model_id": "model-2", "status": "unloaded", "loaded": False, "downloaded": True},
        ]
        mock_resp.raise_for_status.return_value = None

        with patch.object(admin._http, "request", return_value=mock_resp):
            result = admin.models.list()
            assert isinstance(result, AdminModelListResponse)
            assert len(result.models) == 2
            assert result.models[0].model_id == "model-1"
            assert result.models[0].loaded is True
        admin.close()


class TestAdminCacheMocked:
    @patch("novamlx._admin.discover_admin_base_url", return_value="http://localhost:6591")
    @patch("novamlx._admin.discover_api_key", return_value="sk-test")
    def test_cache_stats(self, mock_key, mock_url):
        admin = AdminClient()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"hits": 100, "misses": 10, "tokens_saved": 5000, "blocks": 50}
        mock_resp.raise_for_status.return_value = None

        with patch.object(admin._http, "request", return_value=mock_resp):
            stats = admin.cache.stats("test-model")
            assert isinstance(stats, CacheStats)
            assert stats.hits == 100
            assert stats.tokens_saved == 5000
        admin.close()


class TestAdminBenchmarkMocked:
    @patch("novamlx._admin.discover_admin_base_url", return_value="http://localhost:6591")
    @patch("novamlx._admin.discover_api_key", return_value="sk-test")
    def test_benchmark_status(self, mock_key, mock_url):
        admin = AdminClient()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"running": False, "model_id": None, "progress": 0.0, "results": None}
        mock_resp.raise_for_status.return_value = None

        with patch.object(admin._http, "request", return_value=mock_resp):
            status = admin.benchmark.status()
            assert isinstance(status, BenchmarkStatus)
            assert status.running is False
        admin.close()
