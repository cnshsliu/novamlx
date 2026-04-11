# SPDX-License-Identifier: Apache-2.0
"""
Cache manager interface for oMLX.

This module defines the abstract interface that all cache implementations
should follow for consistency across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .stats import BaseCacheStats


class CacheManager(ABC):
    """
    Abstract interface for all cache implementations.

    This interface provides a consistent API for:
    - Prefix cache (trie-based LRU)
    - Paged cache (block-based KV cache)
    - VLM cache (vision-language model cache)
    - Paged SSD cache (disk-based persistence)
    """

    @abstractmethod
    def fetch(self, key: Any) -> Tuple[Optional[Any], bool]:
        """
        Fetch a value from the cache.

        Args:
            key: The cache key (varies by implementation).

        Returns:
            Tuple of (value, hit) where hit is True if found.
        """
        pass

    @abstractmethod
    def store(self, key: Any, value: Any) -> bool:
        """
        Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.

        Returns:
            True if stored successfully.
        """
        pass

    @abstractmethod
    def evict(self, key: Any) -> bool:
        """
        Evict a specific entry from the cache.

        Args:
            key: The cache key to evict.

        Returns:
            True if evicted, False if not found.
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all entries from the cache.

        Returns:
            Number of entries cleared.
        """
        pass

    @abstractmethod
    def get_stats(self) -> BaseCacheStats:
        """
        Get cache statistics.

        Returns:
            BaseCacheStats or subclass with cache metrics.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the current number of entries in the cache.

        Returns:
            Number of cached entries.
        """
        pass

    @property
    @abstractmethod
    def max_size(self) -> int:
        """
        Get the maximum capacity of the cache.

        Returns:
            Maximum number of entries.
        """
        pass

    @property
    def utilization(self) -> float:
        """
        Get cache utilization as a fraction.

        Returns:
            Utilization between 0.0 and 1.0.
        """
        if self.max_size == 0:
            return 0.0
        return self.size / self.max_size
