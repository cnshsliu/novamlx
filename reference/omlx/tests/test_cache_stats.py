# SPDX-License-Identifier: Apache-2.0
"""
Tests for cache statistics classes and interface.

This module tests the unified cache statistics for oMLX, including
base classes and implementation-specific metrics.
"""

from typing import Any, Dict

import pytest

from omlx.cache.stats import (
    BaseCacheStats,
    PagedCacheStats,
    PagedSSDCacheStats,
    PrefixCacheStats,
    VLMCacheStats,
)
from omlx.cache.interface import CacheManager


class TestBaseCacheStats:
    """Tests for BaseCacheStats base class."""

    def test_default_values(self):
        """Test default values on initialization."""
        stats = BaseCacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_with_values(self):
        """Test initialization with values."""
        stats = BaseCacheStats(hits=10, misses=5, evictions=2)

        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2

    def test_total_queries(self):
        """Test total_queries property."""
        stats = BaseCacheStats(hits=10, misses=5)

        assert stats.total_queries == 15

    def test_hit_rate_with_queries(self):
        """Test hit_rate calculation."""
        stats = BaseCacheStats(hits=75, misses=25)

        assert stats.hit_rate == pytest.approx(0.75)

    def test_hit_rate_zero_queries(self):
        """Test hit_rate with no queries."""
        stats = BaseCacheStats()

        assert stats.hit_rate == 0.0

    def test_record_hit(self):
        """Test recording a cache hit."""
        stats = BaseCacheStats()

        stats.record_hit()

        assert stats.hits == 1

        stats.record_hit()
        stats.record_hit()

        assert stats.hits == 3

    def test_record_miss(self):
        """Test recording a cache miss."""
        stats = BaseCacheStats()

        stats.record_miss()

        assert stats.misses == 1

    def test_record_eviction(self):
        """Test recording an eviction."""
        stats = BaseCacheStats()

        stats.record_eviction()

        assert stats.evictions == 1

    def test_reset(self):
        """Test resetting statistics."""
        stats = BaseCacheStats(hits=100, misses=50, evictions=10)

        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = BaseCacheStats(hits=10, misses=5, evictions=2)

        d = stats.to_dict()

        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["total_queries"] == 15
        assert d["hit_rate"] == pytest.approx(10 / 15)


class TestPrefixCacheStats:
    """Tests for PrefixCacheStats."""

    def test_default_values(self):
        """Test default values."""
        stats = PrefixCacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0

    def test_tokens_saved(self):
        """Test tokens_saved field."""
        stats = PrefixCacheStats(tokens_saved=1000)

        assert stats.tokens_saved == 1000

    def test_total_queries_property(self):
        """Test total_queries property (inherits from base)."""
        stats = PrefixCacheStats(hits=10, misses=5)

        assert stats.total_queries == 15

    def test_total_queries_explicit(self):
        """Test setting total_queries explicitly."""
        stats = PrefixCacheStats(hits=10, misses=5)
        stats.total_queries = 20

        assert stats.total_queries == 20

    def test_reset(self):
        """Test resetting prefix cache stats."""
        stats = PrefixCacheStats(hits=100, misses=50, tokens_saved=5000)
        stats.total_queries = 200

        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats._total_queries == 0


class TestPagedCacheStats:
    """Tests for PagedCacheStats."""

    def test_default_values(self):
        """Test default values."""
        stats = PagedCacheStats()

        assert stats.total_blocks == 0
        assert stats.allocated_blocks == 0
        assert stats.free_blocks == 0
        assert stats.shared_blocks == 0
        assert stats.total_tokens_cached == 0
        assert stats.cow_copies == 0

    def test_with_values(self):
        """Test initialization with values."""
        stats = PagedCacheStats(
            total_blocks=1000,
            allocated_blocks=500,
            free_blocks=500,
            shared_blocks=50,
            total_tokens_cached=32000,
            cow_copies=10,
        )

        assert stats.total_blocks == 1000
        assert stats.allocated_blocks == 500
        assert stats.free_blocks == 500
        assert stats.shared_blocks == 50
        assert stats.total_tokens_cached == 32000
        assert stats.cow_copies == 10

    def test_utilization(self):
        """Test utilization property."""
        stats = PagedCacheStats(total_blocks=100, allocated_blocks=75)

        assert stats.utilization == pytest.approx(0.75)

    def test_utilization_zero_blocks(self):
        """Test utilization with zero total blocks."""
        stats = PagedCacheStats(total_blocks=0)

        assert stats.utilization == 0.0

    def test_reset(self):
        """Test resetting (only runtime stats, not capacity)."""
        stats = PagedCacheStats(
            hits=100,
            misses=50,
            evictions=10,
            total_blocks=1000,
            allocated_blocks=500,
            cow_copies=20,
        )

        stats.reset()

        # Runtime stats reset
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.cow_copies == 0

        # Capacity metrics preserved
        assert stats.total_blocks == 1000
        assert stats.allocated_blocks == 500

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = PagedCacheStats(
            total_blocks=100,
            allocated_blocks=75,
            hits=50,
            misses=25,
        )

        d = stats.to_dict()

        assert d["total_blocks"] == 100
        assert d["allocated_blocks"] == 75
        assert d["utilization"] == pytest.approx(0.75)
        assert d["hit_rate"] == pytest.approx(50 / 75)


class TestVLMCacheStats:
    """Tests for VLMCacheStats."""

    def test_default_values(self):
        """Test default values."""
        stats = VLMCacheStats()

        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0

    def test_record_image_hit(self):
        """Test recording an image cache hit."""
        stats = VLMCacheStats()

        stats.record_image_hit()

        assert stats.image_cache_hits == 1

        stats.record_image_hit()
        stats.record_image_hit()

        assert stats.image_cache_hits == 3

    def test_reset(self):
        """Test resetting VLM stats."""
        stats = VLMCacheStats(
            hits=100,
            tokens_saved=5000,
            image_cache_hits=25,
        )

        stats.reset()

        assert stats.hits == 0
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0


class TestPagedSSDCacheStats:
    """Tests for PagedSSDCacheStats."""

    def test_default_values(self):
        """Test default values."""
        stats = PagedSSDCacheStats()

        assert stats.saves == 0
        assert stats.loads == 0
        assert stats.errors == 0
        assert stats.total_size_bytes == 0
        assert stats.num_files == 0

    def test_with_values(self):
        """Test initialization with values."""
        stats = PagedSSDCacheStats(
            saves=100,
            loads=50,
            errors=5,
            total_size_bytes=1024 * 1024 * 100,  # 100MB
            num_files=150,
        )

        assert stats.saves == 100
        assert stats.loads == 50
        assert stats.errors == 5
        assert stats.total_size_bytes == 1024 * 1024 * 100
        assert stats.num_files == 150

    def test_save_rate(self):
        """Test save_rate calculation."""
        stats = PagedSSDCacheStats(saves=90, errors=10)

        assert stats.save_rate == pytest.approx(0.9)

    def test_save_rate_no_operations(self):
        """Test save_rate with no operations."""
        stats = PagedSSDCacheStats()

        assert stats.save_rate == 0.0

    def test_record_save(self):
        """Test recording a save operation."""
        stats = PagedSSDCacheStats()

        stats.record_save()

        assert stats.saves == 1

    def test_record_load(self):
        """Test recording a load operation."""
        stats = PagedSSDCacheStats()

        stats.record_load()

        assert stats.loads == 1
        assert stats.hits == 1  # Also counts as hit

    def test_record_error(self):
        """Test recording an error."""
        stats = PagedSSDCacheStats()

        stats.record_error()

        assert stats.errors == 1

    def test_reset(self):
        """Test resetting runtime stats."""
        stats = PagedSSDCacheStats(
            hits=100,
            misses=50,
            saves=80,
            loads=70,
            errors=5,
            total_size_bytes=1024 * 1024,
            num_files=100,
        )

        stats.reset()

        # Runtime stats reset
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.saves == 0
        assert stats.loads == 0
        assert stats.errors == 0

        # Size metrics preserved
        assert stats.total_size_bytes == 1024 * 1024
        assert stats.num_files == 100

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = PagedSSDCacheStats(
            saves=90,
            errors=10,
            total_size_bytes=1024,
        )

        d = stats.to_dict()

        assert d["saves"] == 90
        assert d["errors"] == 10
        assert d["save_rate"] == pytest.approx(0.9)
        assert d["total_size_bytes"] == 1024


class TestCacheManagerInterface:
    """Tests for CacheManager ABC interface."""

    def test_interface_methods_are_abstract(self):
        """Test that interface methods are abstract."""
        # CacheManager is ABC, cannot be instantiated directly
        with pytest.raises(TypeError):
            CacheManager()

    def test_utilization_property(self):
        """Test utilization property implementation."""

        class MockCacheManager(CacheManager):
            """Mock implementation for testing."""

            def __init__(self, size: int, max_size: int):
                self._size = size
                self._max_size = max_size

            def fetch(self, key):
                return None, False

            def store(self, key, value):
                return True

            def evict(self, key):
                return True

            def clear(self):
                return 0

            def get_stats(self):
                return BaseCacheStats()

            @property
            def size(self):
                return self._size

            @property
            def max_size(self):
                return self._max_size

        # Test utilization calculation
        cache = MockCacheManager(size=75, max_size=100)
        assert cache.utilization == pytest.approx(0.75)

        # Test with zero max_size
        cache_zero = MockCacheManager(size=0, max_size=0)
        assert cache_zero.utilization == 0.0


class TestStatsIntegration:
    """Integration tests for stats classes."""

    def test_hit_rate_tracking(self):
        """Test tracking hit rate over multiple operations."""
        stats = BaseCacheStats()

        # Simulate 70% hit rate
        for _ in range(70):
            stats.record_hit()
        for _ in range(30):
            stats.record_miss()

        assert stats.hits == 70
        assert stats.misses == 30
        assert stats.hit_rate == pytest.approx(0.70)

    def test_prefix_cache_tokens_efficiency(self):
        """Test tracking token savings."""
        stats = PrefixCacheStats()

        # Simulate caching scenarios
        stats.record_hit()
        stats.tokens_saved += 1024  # Saved 1024 tokens

        stats.record_hit()
        stats.tokens_saved += 512  # Saved 512 more

        stats.record_miss()  # Miss

        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.tokens_saved == 1536
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_paged_cache_block_tracking(self):
        """Test tracking paged cache blocks."""
        stats = PagedCacheStats(
            total_blocks=1000,
            allocated_blocks=100,
            free_blocks=900,
        )

        # Simulate allocation
        stats.allocated_blocks += 50
        stats.free_blocks -= 50

        assert stats.allocated_blocks == 150
        assert stats.free_blocks == 850
        assert stats.utilization == pytest.approx(0.15)

        # Simulate sharing (COW)
        stats.shared_blocks += 10
        stats.cow_copies += 10

        assert stats.shared_blocks == 10
        assert stats.cow_copies == 10

    def test_paged_ssd_cache_io_tracking(self):
        """Test tracking SSD I/O operations."""
        stats = PagedSSDCacheStats(
            total_size_bytes=1024 * 1024 * 1024,  # 1GB
            num_files=0,
        )

        # Simulate save operations
        for _ in range(100):
            stats.record_save()
            stats.num_files += 1

        # Simulate some errors
        for _ in range(5):
            stats.record_error()

        # Simulate loads (cache hits)
        for _ in range(80):
            stats.record_load()

        assert stats.saves == 100
        assert stats.errors == 5
        assert stats.loads == 80
        assert stats.hits == 80  # Loads count as hits
        assert stats.num_files == 100
        assert stats.save_rate == pytest.approx(100 / 105)  # 100/(100+5)
