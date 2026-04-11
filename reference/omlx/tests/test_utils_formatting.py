# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.formatting module."""

import pytest

from omlx.utils.formatting import format_bytes


class TestFormatBytes:
    """Test cases for format_bytes function."""

    def test_format_gigabytes(self):
        """Test formatting values in GB range."""
        # Exact GB values
        assert format_bytes(1024**3) == "1.00 GB"
        assert format_bytes(2 * 1024**3) == "2.00 GB"
        assert format_bytes(16 * 1024**3) == "16.00 GB"

    def test_format_gigabytes_with_decimals(self):
        """Test formatting GB values with decimal places."""
        # 1.5 GB
        assert format_bytes(int(1.5 * 1024**3)) == "1.50 GB"
        # 2.75 GB
        assert format_bytes(int(2.75 * 1024**3)) == "2.75 GB"

    def test_format_megabytes(self):
        """Test formatting values in MB range."""
        # Exact MB values
        assert format_bytes(1024**2) == "1.00 MB"
        assert format_bytes(256 * 1024**2) == "256.00 MB"
        assert format_bytes(512 * 1024**2) == "512.00 MB"

    def test_format_megabytes_with_decimals(self):
        """Test formatting MB values with decimal places."""
        # 1.5 MB
        assert format_bytes(int(1.5 * 1024**2)) == "1.50 MB"
        # 100.25 MB
        assert format_bytes(int(100.25 * 1024**2)) == "100.25 MB"

    def test_format_kilobytes(self):
        """Test formatting values in KB range."""
        # Exact KB values
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(512 * 1024) == "512.00 KB"

    def test_format_kilobytes_with_decimals(self):
        """Test formatting KB values with decimal places."""
        # 1.5 KB
        assert format_bytes(int(1.5 * 1024)) == "1.50 KB"

    def test_format_bytes_small(self):
        """Test formatting values in bytes range (< 1 KB)."""
        assert format_bytes(0) == "0 B"
        assert format_bytes(1) == "1 B"
        assert format_bytes(512) == "512 B"
        assert format_bytes(1023) == "1023 B"

    def test_boundary_values(self):
        """Test boundary values between units."""
        # Just under 1 KB
        assert format_bytes(1023) == "1023 B"
        # Exactly 1 KB
        assert format_bytes(1024) == "1.00 KB"

        # Just under 1 MB
        assert format_bytes(1024**2 - 1) == "1024.00 KB"
        # Exactly 1 MB
        assert format_bytes(1024**2) == "1.00 MB"

        # Just under 1 GB
        assert format_bytes(1024**3 - 1) == "1024.00 MB"
        # Exactly 1 GB
        assert format_bytes(1024**3) == "1.00 GB"

    def test_large_values(self):
        """Test very large values (TB+ range)."""
        # 1 TB in bytes - should still show as GB
        assert format_bytes(1024**4) == "1024.00 GB"
        # 2 TB
        assert format_bytes(2 * 1024**4) == "2048.00 GB"

    def test_realistic_memory_sizes(self):
        """Test realistic Apple Silicon memory sizes."""
        # Common Mac memory configurations
        assert format_bytes(8 * 1024**3) == "8.00 GB"   # M1 base
        assert format_bytes(16 * 1024**3) == "16.00 GB"  # M1 Pro
        assert format_bytes(32 * 1024**3) == "32.00 GB"  # M1 Max
        assert format_bytes(64 * 1024**3) == "64.00 GB"  # M1 Ultra
        assert format_bytes(128 * 1024**3) == "128.00 GB"  # M2 Ultra
        assert format_bytes(192 * 1024**3) == "192.00 GB"  # M4 Ultra max
