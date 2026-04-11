# SPDX-License-Identifier: Apache-2.0
"""Tests for Anthropic context scaling functionality."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from omlx.settings import ClaudeCodeSettings, GlobalSettings


@dataclass
class MockServerState:
    """Mock server state for testing."""

    global_settings: GlobalSettings | None = None
    sampling: MagicMock = field(default_factory=lambda: MagicMock(max_context_window=32768))
    settings_manager: object | None = None


class TestScaleAnthropicTokens:
    """Tests for scale_anthropic_tokens function."""

    def _make_server_state(
        self,
        enabled: bool = False,
        target: int = 200000,
        max_context: int = 32768,
    ) -> MockServerState:
        """Create a mock server state with given settings."""
        gs = GlobalSettings.__new__(GlobalSettings)
        gs.claude_code = ClaudeCodeSettings(
            context_scaling_enabled=enabled,
            target_context_size=target,
        )
        state = MockServerState(global_settings=gs)
        state.sampling = MagicMock(max_context_window=max_context)
        state.settings_manager = None
        return state

    def test_scaling_disabled_returns_original(self):
        """When scaling is disabled, return original token count."""
        state = self._make_server_state(enabled=False, target=200000, max_context=32768)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=32768),
        ):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(1000)
            assert result == 1000

    def test_scaling_enabled_basic(self):
        """Test basic scaling: 32k model, 200k target."""
        state = self._make_server_state(enabled=True, target=200000, max_context=100000)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=100000),
        ):
            from omlx.server import scale_anthropic_tokens

            # 50000 * (200000 / 100000) = 100000
            result = scale_anthropic_tokens(50000, "test-model")
            assert result == 100000

    def test_scaling_preserves_ratio(self):
        """Test that scaling preserves usage ratio."""
        state = self._make_server_state(enabled=True, target=200000, max_context=131072)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=131072),
        ):
            from omlx.server import scale_anthropic_tokens

            # 96k used on 128k model = 75%
            # 96k used on 128k model = 73.2%
            # scaled: 96000 * (200000/131072) = 146484
            result = scale_anthropic_tokens(96000, "test-model")
            expected = int(96000 * 200000 / 131072)
            assert result == expected

    def test_no_scaling_when_actual_ge_target(self):
        """When actual context >= target, no scaling needed."""
        state = self._make_server_state(enabled=True, target=100000, max_context=200000)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=200000),
        ):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(50000, "test-model")
            assert result == 50000

    def test_no_scaling_when_actual_equals_target(self):
        """When actual context == target, no scaling."""
        state = self._make_server_state(enabled=True, target=200000, max_context=250000)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=250000),
        ):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(100000, "test-model")
            assert result == 100000

    def test_global_settings_none_returns_original(self):
        """When global_settings is None, return original."""
        state = MockServerState(global_settings=None)
        with patch("omlx.server._server_state", state):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(5000, "test-model")
            assert result == 5000

    def test_actual_context_none_returns_original(self):
        """When get_max_context_window returns None, return original."""
        state = self._make_server_state(enabled=True, target=200000, max_context=32768)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=None),
        ):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(5000, "test-model")
            assert result == 5000

    def test_zero_tokens(self):
        """Test with zero token count."""
        state = self._make_server_state(enabled=True, target=200000, max_context=32768)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=32768),
        ):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(0, "test-model")
            assert result == 0

    def test_scaling_returns_int(self):
        """Test that scaling always returns integer."""
        state = self._make_server_state(enabled=True, target=200000, max_context=131072)
        with (
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_max_context_window", return_value=131072),
        ):
            from omlx.server import scale_anthropic_tokens

            result = scale_anthropic_tokens(12345, "test-model")
            assert isinstance(result, int)
