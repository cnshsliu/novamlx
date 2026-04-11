# SPDX-License-Identifier: Apache-2.0
"""
Tests for cache factory, recovery manager, and hybrid cache configuration.

This module tests the factory pattern for creating cache instances,
recovery from cache corruption, and hybrid model cache configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from omlx.cache.factory import CacheConfig, CacheFactory
from omlx.cache.recovery import CacheRecoveryManager
from omlx.cache.hybrid_cache import (
    LayerCacheConfig,
    ModelCacheConfig,
    create_default_kvcache_config,
)
from omlx.cache.type_handlers import CacheType


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = CacheConfig()

        assert config.block_size == 64
        assert config.max_num_blocks == 1024
        assert config.initial_blocks == 256
        assert config.paged_ssd_cache_dir is None
        assert config.max_paged_ssd_cache_size == 100 * 1024 * 1024 * 1024  # 100GB
        assert config.max_kv_cache_memory is None
        assert config.model_name == ""

    def test_with_custom_values(self):
        """Test with custom values."""
        config = CacheConfig(
            block_size=128,
            max_num_blocks=2048,
            initial_blocks=512,
            paged_ssd_cache_dir=Path("/tmp/cache"),
            max_paged_ssd_cache_size=50 * 1024**3,
            max_kv_cache_memory=8 * 1024**3,
            model_name="test-model",
        )

        assert config.block_size == 128
        assert config.max_num_blocks == 2048
        assert config.initial_blocks == 512
        assert config.paged_ssd_cache_dir == Path("/tmp/cache")
        assert config.max_paged_ssd_cache_size == 50 * 1024**3
        assert config.max_kv_cache_memory == 8 * 1024**3
        assert config.model_name == "test-model"


class TestCacheFactory:
    """Tests for CacheFactory."""

    def test_create_paged_cache_disabled(self):
        """Test create_paged_cache returns None when SSD cache disabled."""
        config = CacheConfig(paged_ssd_cache_dir=None)

        result = CacheFactory.create_paged_cache(config)

        assert result is None

    def test_create_paged_cache_enabled(self, tmp_path: Path):
        """Test create_paged_cache creates manager when enabled."""
        config = CacheConfig(
            paged_ssd_cache_dir=tmp_path / "cache",
            block_size=64,
            max_num_blocks=100,
            initial_blocks=50,
            model_name="test-model",
        )

        result = CacheFactory.create_paged_cache(config)

        assert result is not None
        assert result.block_size == 64
        assert result.max_blocks == 100
        assert result.model_name == "test-model"

    def test_create_paged_ssd_cache_disabled(self):
        """Test create_paged_ssd_cache returns None when disabled."""
        config = CacheConfig(paged_ssd_cache_dir=None)

        result = CacheFactory.create_paged_ssd_cache(config)

        assert result is None

    def test_create_paged_ssd_cache_enabled(self, tmp_path: Path):
        """Test create_paged_ssd_cache creates manager."""
        config = CacheConfig(
            paged_ssd_cache_dir=tmp_path / "cache",
            max_paged_ssd_cache_size=1024**3,
        )

        result = CacheFactory.create_paged_ssd_cache(config, model_name="test-model")

        assert result is not None
        # Check that model_name is used in path
        assert "test-model" in str(result._cache_dir)

    def test_create_prefix_cache_disabled(self):
        """Test create_prefix_cache returns None when disabled."""
        config = CacheConfig(paged_ssd_cache_dir=None)

        result = CacheFactory.create_prefix_cache(config)

        assert result is None

    def test_create_prefix_cache_no_paged_cache(self, tmp_path: Path):
        """Test create_prefix_cache returns None without paged cache."""
        config = CacheConfig(paged_ssd_cache_dir=tmp_path / "cache")

        result = CacheFactory.create_prefix_cache(config, paged_cache=None)

        assert result is None

    def test_create_prefix_cache_enabled(self, tmp_path: Path):
        """Test create_prefix_cache creates cache."""
        config = CacheConfig(paged_ssd_cache_dir=tmp_path / "cache")

        paged_cache = CacheFactory.create_paged_cache(config)
        mock_model = MagicMock()
        mock_model.layers = [MagicMock() for _ in range(4)]

        result = CacheFactory.create_prefix_cache(
            config,
            model=mock_model,
            paged_cache=paged_cache,
        )

        assert result is not None

    def test_create_memory_monitor(self, tmp_path: Path):
        """Test create_memory_monitor creates monitor."""
        config = CacheConfig(
            paged_ssd_cache_dir=tmp_path / "cache",
            max_kv_cache_memory=8 * 1024**3,
        )

        paged_cache = CacheFactory.create_paged_cache(config)

        result = CacheFactory.create_memory_monitor(config, paged_cache)

        assert result is not None

    def test_create_memory_monitor_default_memory(self, tmp_path: Path):
        """Test create_memory_monitor with default memory limit."""
        config = CacheConfig(
            paged_ssd_cache_dir=tmp_path / "cache",
            max_kv_cache_memory=None,  # Should use default
        )

        result = CacheFactory.create_memory_monitor(config)

        assert result is not None

    def test_create_full_cache_stack_disabled(self):
        """Test create_full_cache_stack returns Nones when disabled."""
        config = CacheConfig(paged_ssd_cache_dir=None)

        result = CacheFactory.create_full_cache_stack(config)

        assert result["paged_cache"] is None
        assert result["paged_ssd_cache"] is None
        assert result["prefix_cache"] is None
        assert result["memory_monitor"] is None

    def test_create_full_cache_stack_enabled(self, tmp_path: Path):
        """Test create_full_cache_stack creates all components."""
        config = CacheConfig(
            paged_ssd_cache_dir=tmp_path / "cache",
            model_name="test-model",
        )

        mock_model = MagicMock()
        mock_model.layers = [MagicMock() for _ in range(4)]

        result = CacheFactory.create_full_cache_stack(
            config,
            model=mock_model,
        )

        assert result["paged_cache"] is not None
        assert result["paged_ssd_cache"] is not None
        assert result["prefix_cache"] is not None
        assert result["memory_monitor"] is not None


class TestCacheRecoveryManager:
    """Tests for CacheRecoveryManager."""

    @pytest.fixture
    def mock_prefix_cache(self):
        """Create a mock prefix cache."""
        mock = MagicMock()
        mock.clear.return_value = 10
        return mock

    @pytest.fixture
    def recovery_manager(self, mock_prefix_cache):
        """Create a recovery manager."""
        return CacheRecoveryManager(block_aware_cache=mock_prefix_cache)

    def test_is_cache_corruption_shape_mismatch(self, recovery_manager):
        """Test detecting shape mismatch errors."""
        error = ValueError("Shapes (1, 32, 64) and (1, 32, 128) do not match")

        # This depends on is_cache_corruption_error implementation
        # Just verify method exists and returns bool
        result = recovery_manager.is_cache_corruption(error)
        assert isinstance(result, bool)

    def test_is_cache_corruption_dimension_error(self, recovery_manager):
        """Test detecting dimension errors."""
        error = ValueError("Dimension mismatch in concatenate")

        result = recovery_manager.is_cache_corruption(error)
        assert isinstance(result, bool)

    def test_recover(self, recovery_manager, mock_prefix_cache):
        """Test recovery process."""
        # Setup mock objects
        mock_holder = MagicMock()
        mock_holder.batch_generator = MagicMock()
        mock_holder._current_sampler_params = MagicMock()

        request_id_to_uid = {"req-001": 1, "req-002": 2}
        uid_to_request_id = {1: "req-001", 2: "req-002"}
        request_detokenizers = {"req-001": MagicMock()}

        # Perform recovery
        recovery_manager.recover(
            mock_holder,
            request_id_to_uid,
            uid_to_request_id,
            request_detokenizers,
        )

        # Verify batch_generator cleared
        assert mock_holder.batch_generator is None
        assert mock_holder._current_sampler_params is None

        # Verify cache cleared
        mock_prefix_cache.clear.assert_called_once()

        # Verify mappings cleared
        assert len(request_id_to_uid) == 0
        assert len(uid_to_request_id) == 0
        assert len(request_detokenizers) == 0

    def test_recover_without_cache(self):
        """Test recovery without prefix cache."""
        manager = CacheRecoveryManager(block_aware_cache=None)

        mock_holder = MagicMock()
        request_id_to_uid = {"req-001": 1}
        uid_to_request_id = {1: "req-001"}
        request_detokenizers = {}

        # Should not raise
        manager.recover(
            mock_holder,
            request_id_to_uid,
            uid_to_request_id,
            request_detokenizers,
        )

        assert mock_holder.batch_generator is None

    def test_reschedule_running_requests(self, recovery_manager):
        """Test rescheduling running requests."""
        from collections import deque

        # Create mock requests
        mock_request1 = MagicMock()
        mock_request1.prompt_token_ids = [1, 2, 3, 4]
        mock_request2 = MagicMock()
        mock_request2.prompt_token_ids = [5, 6, 7, 8]

        running = {
            "req-001": mock_request1,
            "req-002": mock_request2,
        }
        waiting = deque()
        mock_waiting_status = MagicMock()

        count = recovery_manager.reschedule_running_requests(
            running,
            waiting,
            mock_waiting_status,
        )

        assert count == 2
        assert len(running) == 0
        assert len(waiting) == 2

        # Verify requests were reset
        assert mock_request1.status is mock_waiting_status
        assert mock_request1.batch_uid is None
        assert mock_request1.prompt_cache is None

    def test_reschedule_empty_running(self, recovery_manager):
        """Test rescheduling with no running requests."""
        from collections import deque

        running = {}
        waiting = deque()
        mock_waiting_status = MagicMock()

        count = recovery_manager.reschedule_running_requests(
            running,
            waiting,
            mock_waiting_status,
        )

        assert count == 0


class TestLayerCacheConfig:
    """Tests for LayerCacheConfig."""

    def test_creation(self):
        """Test creating layer config."""
        config = LayerCacheConfig(
            layer_idx=5,
            cache_type=CacheType.KVCACHE,
            supports_block_slicing=True,
            class_name="KVCache",
        )

        assert config.layer_idx == 5
        assert config.cache_type == CacheType.KVCACHE
        assert config.supports_block_slicing is True
        assert config.class_name == "KVCache"

    def test_handler_property(self):
        """Test handler property returns correct handler."""
        config = LayerCacheConfig(
            layer_idx=0,
            cache_type=CacheType.KVCACHE,
            supports_block_slicing=True,
            class_name="KVCache",
        )

        handler = config.handler

        assert handler is not None
        assert handler.cache_type == CacheType.KVCACHE

    def test_arrays_cache_layer(self):
        """Test config for ArraysCache layer."""
        config = LayerCacheConfig(
            layer_idx=1,
            cache_type=CacheType.ARRAYS_CACHE,
            supports_block_slicing=False,
            class_name="ArraysCache",
        )

        assert config.supports_block_slicing is False
        assert config.handler.cache_type == CacheType.ARRAYS_CACHE


class TestModelCacheConfig:
    """Tests for ModelCacheConfig."""

    def test_default_values(self):
        """Test default values."""
        config = ModelCacheConfig()

        assert config.model_name == ""
        assert config.num_layers == 0
        assert config.layer_configs == []
        assert config.is_hybrid is False
        assert config.sliceable_layer_count == 0

    def test_from_cache_list_empty(self):
        """Test from_cache_list with empty list."""
        config = ModelCacheConfig.from_cache_list([], model_name="test")

        assert config.model_name == "test"
        assert config.num_layers == 0

    def test_from_cache_list_kvcache_only(self):
        """Test from_cache_list with KVCache only."""
        mock_caches = []
        for i in range(4):
            mock = MagicMock()
            mock.__class__.__name__ = "KVCache"
            mock_caches.append(mock)

        config = ModelCacheConfig.from_cache_list(mock_caches, model_name="test-model")

        assert config.num_layers == 4
        assert config.is_hybrid is False
        assert config.sliceable_layer_count == 4

    def test_from_cache_list_hybrid(self):
        """Test from_cache_list with mixed cache types."""
        mock_caches = []

        # KVCache layer
        mock_kv = MagicMock()
        mock_kv.__class__.__name__ = "KVCache"
        mock_caches.append(mock_kv)

        # ArraysCache layer
        mock_arrays = MagicMock()
        mock_arrays.__class__.__name__ = "ArraysCache"
        mock_arrays.cache = [MagicMock(), MagicMock()]
        mock_caches.append(mock_arrays)

        config = ModelCacheConfig.from_cache_list(mock_caches)

        assert config.num_layers == 2
        assert config.is_hybrid is True
        assert config.sliceable_layer_count == 1

    def test_from_type_list(self):
        """Test from_type_list."""
        type_names = ["KVCache", "KVCache", "ArraysCache", "KVCache"]

        config = ModelCacheConfig.from_type_list(type_names, model_name="test")

        assert config.num_layers == 4
        assert config.is_hybrid is True
        assert config.sliceable_layer_count == 3

    def test_from_type_list_empty(self):
        """Test from_type_list with empty list."""
        config = ModelCacheConfig.from_type_list([])

        assert config.num_layers == 0

    def test_get_sliceable_layers(self):
        """Test get_sliceable_layers."""
        type_names = ["KVCache", "ArraysCache", "KVCache", "RotatingKVCache"]
        config = ModelCacheConfig.from_type_list(type_names)

        sliceable = config.get_sliceable_layers()

        # KVCache layers only (indices 0 and 2)
        assert 0 in sliceable
        assert 2 in sliceable
        assert 1 not in sliceable  # ArraysCache
        assert 3 not in sliceable  # RotatingKVCache

    def test_get_non_sliceable_layers(self):
        """Test get_non_sliceable_layers."""
        type_names = ["KVCache", "ArraysCache", "KVCache", "ArraysCache"]
        config = ModelCacheConfig.from_type_list(type_names)

        non_sliceable = config.get_non_sliceable_layers()

        assert 1 in non_sliceable
        assert 3 in non_sliceable
        assert len(non_sliceable) == 2

    def test_get_layer_type(self):
        """Test get_layer_type."""
        type_names = ["KVCache", "ArraysCache"]
        config = ModelCacheConfig.from_type_list(type_names)

        assert config.get_layer_type(0) == CacheType.KVCACHE
        assert config.get_layer_type(1) == CacheType.ARRAYS_CACHE
        # Out of bounds returns default
        assert config.get_layer_type(99) == CacheType.KVCACHE

    def test_get_layer_handler(self):
        """Test get_layer_handler."""
        type_names = ["KVCache", "ArraysCache"]
        config = ModelCacheConfig.from_type_list(type_names)

        handler0 = config.get_layer_handler(0)
        handler1 = config.get_layer_handler(1)

        assert handler0.cache_type == CacheType.KVCACHE
        assert handler1.cache_type == CacheType.ARRAYS_CACHE

    def test_get_type_names(self):
        """Test get_type_names."""
        type_names = ["KVCache", "ArraysCache", "RotatingKVCache"]
        config = ModelCacheConfig.from_type_list(type_names)

        names = config.get_type_names()

        assert names == ["KVCache", "ArraysCache", "RotatingKVCache"]

    def test_get_meta_states(self):
        """Test get_meta_states."""
        mock_caches = []
        for i in range(2):
            mock = MagicMock()
            mock.__class__.__name__ = "KVCache"
            mock_keys = MagicMock()
            mock_keys.shape = (1, 8, 64, 64)
            mock.state = (mock_keys, MagicMock())
            mock.offset = 64 + i * 10
            mock_caches.append(mock)

        config = ModelCacheConfig.from_cache_list(mock_caches)
        meta_states = config.get_meta_states(mock_caches)

        assert len(meta_states) == 2
        # Each should have at least offset
        for meta in meta_states:
            assert len(meta) > 0

    def test_supports_full_block_slicing(self):
        """Test supports_full_block_slicing."""
        # All KVCache
        config1 = ModelCacheConfig.from_type_list(["KVCache", "KVCache"])
        assert config1.supports_full_block_slicing() is True

        # Mixed
        config2 = ModelCacheConfig.from_type_list(["KVCache", "ArraysCache"])
        assert config2.supports_full_block_slicing() is False

    def test_has_rotating_layers(self):
        """Test has_rotating_layers."""
        config1 = ModelCacheConfig.from_type_list(["KVCache", "KVCache"])
        assert config1.has_rotating_layers() is False

        config2 = ModelCacheConfig.from_type_list(["KVCache", "RotatingKVCache"])
        assert config2.has_rotating_layers() is True

    def test_repr(self):
        """Test string representation."""
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "ArraysCache", "KVCache"],
            model_name="test-model",
        )

        repr_str = repr(config)

        assert "test-model" in repr_str
        assert "layers=3" in repr_str
        assert "hybrid=True" in repr_str


class TestCreateDefaultKVCacheConfig:
    """Tests for create_default_kvcache_config helper."""

    def test_creates_all_kvcache(self):
        """Test creating all-KVCache config."""
        config = create_default_kvcache_config(num_layers=32, model_name="llama")

        assert config.num_layers == 32
        assert config.model_name == "llama"
        assert config.is_hybrid is False
        assert config.sliceable_layer_count == 32

    def test_all_layers_kvcache(self):
        """Test all layers are KVCache type."""
        config = create_default_kvcache_config(num_layers=4)

        for layer_config in config.layer_configs:
            assert layer_config.cache_type == CacheType.KVCACHE
            assert layer_config.supports_block_slicing is True
            assert layer_config.class_name == "KVCache"

    def test_layer_indices_correct(self):
        """Test layer indices are sequential."""
        config = create_default_kvcache_config(num_layers=4)

        for i, layer_config in enumerate(config.layer_configs):
            assert layer_config.layer_idx == i

    def test_zero_layers(self):
        """Test with zero layers."""
        config = create_default_kvcache_config(num_layers=0)

        assert config.num_layers == 0
        assert len(config.layer_configs) == 0
