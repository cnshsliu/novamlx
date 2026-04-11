# SPDX-License-Identifier: Apache-2.0
"""
Tests for _run_with_disconnect_guard in server module.

Tests cover:
- Normal completion returns result
- Client disconnect cancels task
- Fast completion has no overhead from polling
"""

import asyncio
from unittest.mock import AsyncMock

import pytest


class TestDisconnectGuard:
    """Tests for _run_with_disconnect_guard."""

    @pytest.fixture
    def mock_request_connected(self):
        """Mock HTTP request that stays connected."""
        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.fixture
    def mock_request_disconnects(self):
        """Mock HTTP request that disconnects after first check."""
        request = AsyncMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        return request

    @pytest.mark.asyncio
    async def test_normal_completion(self, mock_request_connected):
        """Test that normal completion returns result."""
        from omlx.server import _run_with_disconnect_guard

        async def fake_generate():
            return "result"

        result = await _run_with_disconnect_guard(
            mock_request_connected, fake_generate(), poll_interval=0.1
        )
        assert result == "result"

    @pytest.mark.asyncio
    async def test_disconnect_cancels_task(self, mock_request_disconnects):
        """Test that disconnect cancels the running task."""
        from omlx.server import _run_with_disconnect_guard

        cancel_detected = False

        async def slow_generate():
            nonlocal cancel_detected
            try:
                await asyncio.sleep(10)
                return "should not reach"
            except asyncio.CancelledError:
                cancel_detected = True
                raise

        result = await _run_with_disconnect_guard(
            mock_request_disconnects, slow_generate(), poll_interval=0.1
        )

        assert result is None  # Client disconnected
        assert cancel_detected  # Task was actually cancelled

    @pytest.mark.asyncio
    async def test_fast_completion_no_disconnect_check(self, mock_request_connected):
        """Test that fast completions finish without disconnect check."""
        from omlx.server import _run_with_disconnect_guard

        async def fast_generate():
            return "fast_result"

        result = await _run_with_disconnect_guard(
            mock_request_connected, fast_generate(), poll_interval=1.0
        )
        assert result == "fast_result"
        # Task completed before poll interval, so is_disconnected should not be called
        mock_request_connected.is_disconnected.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_during_long_generation(self):
        """Test disconnect detection during a long-running generation."""
        from omlx.server import _run_with_disconnect_guard

        call_count = 0

        async def delayed_disconnect():
            nonlocal call_count
            call_count += 1
            # Stay connected for 2 checks, then disconnect
            return call_count > 2

        mock_request = AsyncMock()
        mock_request.is_disconnected = delayed_disconnect

        async def slow_generate():
            await asyncio.sleep(10)
            return "should not reach"

        result = await _run_with_disconnect_guard(
            mock_request, slow_generate(), poll_interval=0.05
        )

        assert result is None
        assert call_count == 3  # Connected, connected, disconnected

    @pytest.mark.asyncio
    async def test_task_exception_propagates(self, mock_request_connected):
        """Test that task exceptions propagate correctly."""
        from omlx.server import _run_with_disconnect_guard

        async def failing_generate():
            raise ValueError("generation failed")

        with pytest.raises(ValueError, match="generation failed"):
            await _run_with_disconnect_guard(
                mock_request_connected, failing_generate(), poll_interval=0.1
            )
