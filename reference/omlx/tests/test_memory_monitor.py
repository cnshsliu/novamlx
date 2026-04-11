# SPDX-License-Identifier: Apache-2.0
"""Tests for memory_monitor module (SSD-only mode)."""

import pytest
from unittest.mock import patch, MagicMock

from omlx.memory_monitor import MemoryMonitor, MemoryInfo
from omlx.utils.hardware import format_bytes


class TestMemoryInfo:
    """Tests for MemoryInfo dataclass."""

    def test_create_memory_info(self):
        """Test creating MemoryInfo."""
        info = MemoryInfo(
            total_bytes=16 * 1024**3,
            used_bytes=8 * 1024**3,
            available_bytes=8 * 1024**3,
            utilization=0.5,
        )
        assert info.total_bytes == 16 * 1024**3
        assert info.used_bytes == 8 * 1024**3
        assert info.available_bytes == 8 * 1024**3
        assert info.utilization == 0.5

    def test_memory_info_zero_usage(self):
        """Test MemoryInfo with zero usage."""
        info = MemoryInfo(
            total_bytes=16 * 1024**3,
            used_bytes=0,
            available_bytes=16 * 1024**3,
            utilization=0.0,
        )
        assert info.used_bytes == 0
        assert info.utilization == 0.0


class TestMemoryMonitor:
    """Test MemoryMonitor class for SSD-only mode."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        max_kv_cache = 2 * 1024**3  # 2GB
        monitor = MemoryMonitor(max_kv_cache_memory=max_kv_cache)
        assert monitor.max_kv_cache_memory == max_kv_cache

    def test_init_invalid_max_kv_cache_memory_zero(self):
        """Test initialization with zero max_kv_cache_memory."""
        with pytest.raises(ValueError, match="max_kv_cache_memory"):
            MemoryMonitor(max_kv_cache_memory=0)

    def test_init_invalid_max_kv_cache_memory_negative(self):
        """Test initialization with negative max_kv_cache_memory."""
        with pytest.raises(ValueError, match="max_kv_cache_memory"):
            MemoryMonitor(max_kv_cache_memory=-1)

    def test_get_memory_info(self):
        """Test get_memory_info returns valid data."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        info = monitor.get_memory_info()

        assert isinstance(info, MemoryInfo)
        assert info.total_bytes == monitor.max_memory
        # In SSD-only mode, used_bytes is always 0
        assert info.used_bytes == 0
        assert info.available_bytes == monitor.max_memory
        assert info.utilization == 0.0

    def test_get_memory_info_throttling(self):
        """Test that memory info checks are throttled."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3, check_interval=10.0)

        # First call
        info1 = monitor.get_memory_info()
        # Second call within interval should return cached value
        info2 = monitor.get_memory_info()

        # Should be the same object (cached)
        assert info1 is info2

    def test_is_under_pressure_always_false(self):
        """Test is_under_pressure always returns False in SSD-only mode."""
        monitor = MemoryMonitor(max_kv_cache_memory=10000)
        # In SSD-only mode, always returns False
        assert not monitor.is_under_pressure()

    def test_bytes_to_free_always_zero(self):
        """Test bytes_to_free always returns 0 in SSD-only mode."""
        monitor = MemoryMonitor(max_kv_cache_memory=10000)
        # In SSD-only mode, always returns 0
        assert monitor.bytes_to_free() == 0

    def test_set_model_info(self):
        """Test setting model information."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        # Internal state should be set
        assert monitor._num_layers == 32
        assert monitor._num_kv_heads == 8
        assert monitor._head_dim == 128
        assert monitor._dtype_size == 2

    def test_estimate_block_memory(self):
        """Test block memory estimation."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        # Set model info
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        # Estimate for 64 tokens
        estimate = monitor.estimate_block_memory(64)
        # Expected: 64 * 8 * 128 * 2 * 2 (keys+values) * 32 layers
        expected = 64 * 8 * 128 * 2 * 2 * 32
        assert estimate == expected

    def test_estimate_block_memory_default_values(self):
        """Test block memory estimation with default values."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        # Without setting model info, should use defaults
        estimate = monitor.estimate_block_memory(64)
        # Default: 32 layers, 8 kv_heads, 128 head_dim, 2 dtype_size
        expected = 64 * 8 * 128 * 2 * 2 * 32
        assert estimate == expected

    def test_estimate_block_memory_with_overrides(self):
        """Test block memory estimation with parameter overrides."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        # Override some parameters
        estimate = monitor.estimate_block_memory(
            block_size=32,
            num_layers=16,  # Override
            dtype_size=4,  # Override
        )
        expected = 32 * 8 * 128 * 4 * 2 * 16
        assert estimate == expected

    def test_estimate_blocks_to_free(self):
        """Test estimation of blocks to free."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        block_size = 64
        block_mem = monitor.estimate_block_memory(block_size)

        # Need to free 10 blocks worth
        bytes_to_free = block_mem * 10
        num_blocks = monitor.estimate_blocks_to_free(bytes_to_free, block_size)
        assert num_blocks == 10

    def test_estimate_blocks_to_free_rounds_up(self):
        """Test that blocks to free rounds up."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        block_size = 64
        block_mem = monitor.estimate_block_memory(block_size)

        # Need to free slightly more than 9 blocks
        bytes_to_free = block_mem * 9 + 1
        num_blocks = monitor.estimate_blocks_to_free(bytes_to_free, block_size)
        assert num_blocks == 10  # Should round up

    def test_get_stats(self):
        """Test get_stats returns dict with expected keys."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        stats = monitor.get_stats()

        assert "total_bytes" in stats
        assert "used_bytes" in stats
        assert "available_bytes" in stats
        assert "utilization" in stats
        assert "max_kv_cache_memory" in stats
        assert "total_formatted" in stats
        assert "used_formatted" in stats
        assert "available_formatted" in stats
        # In SSD-only mode, used_bytes should be 0
        assert stats["used_bytes"] == 0

    def test_format_bytes(self):
        """Test format_bytes utility function."""
        assert "1.00 KB" == format_bytes(1024)
        assert "1.00 MB" == format_bytes(1024 * 1024)
        assert "1.00 GB" == format_bytes(1024 * 1024 * 1024)
        assert "512 B" == format_bytes(512)

    def test_repr(self):
        """Test string representation."""
        monitor = MemoryMonitor(max_kv_cache_memory=2 * 1024**3)
        repr_str = repr(monitor)
        assert "MemoryMonitor" in repr_str
        assert "max_kv_cache" in repr_str
        assert "used" in repr_str

    def test_properties(self):
        """Test property accessors."""
        max_kv_cache = 2 * 1024**3
        monitor = MemoryMonitor(max_kv_cache_memory=max_kv_cache)

        assert monitor.max_kv_cache_memory == max_kv_cache
        assert monitor.max_memory > 0

    def test_set_paged_cache_manager(self):
        """Test setting paged cache manager."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        mock_manager = MagicMock()
        monitor.set_paged_cache_manager(mock_manager, block_size=128)

        assert monitor._paged_cache_manager is mock_manager
        assert monitor._block_size == 128

    def test_set_baseline_memory(self):
        """Test setting baseline memory."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        # This should not raise (uses MLX if available, otherwise sets to 0)
        monitor.set_baseline_memory()

    def test_set_request_stats(self):
        """Test setting request stats."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        monitor.set_request_stats(running=5, waiting=10)

        assert monitor._running_requests == 5
        assert monitor._waiting_requests == 10

    def test_check_interval_parameter(self):
        """Test check_interval parameter."""
        monitor = MemoryMonitor(
            max_kv_cache_memory=1024**3,
            check_interval=5.0,
        )

        assert monitor._check_interval == 5.0
