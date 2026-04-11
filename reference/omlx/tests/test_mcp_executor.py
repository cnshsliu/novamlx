# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP tool executor (omlx/mcp/executor.py).
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.mcp.executor import ToolExecutor, execute_single_tool
from omlx.mcp.manager import MCPClientManager
from omlx.mcp.types import (
    MCPConfig,
    MCPServerConfig,
    MCPServerState,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)


class TestToolExecutorInit:
    """Tests for ToolExecutor initialization."""

    def test_init_defaults(self):
        """Test executor initialization with defaults."""
        config = MCPConfig(default_timeout=30.0)
        manager = MCPClientManager(config)
        executor = ToolExecutor(manager)

        assert executor.manager is manager
        assert executor.max_parallel == 5
        assert executor.default_timeout == 30.0

    def test_init_custom_values(self):
        """Test executor initialization with custom values."""
        config = MCPConfig(default_timeout=60.0)
        manager = MCPClientManager(config)
        executor = ToolExecutor(manager, max_parallel=10, default_timeout=120.0)

        assert executor.max_parallel == 10
        assert executor.default_timeout == 120.0


class TestToolExecutorExecuteToolCalls:
    """Tests for ToolExecutor.execute_tool_calls()."""

    @pytest.fixture
    def executor_with_manager(self) -> ToolExecutor:
        """Create executor with mock manager."""
        config = MCPConfig.from_dict({
            "servers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                },
            },
            "default_timeout": 10.0,
        })
        manager = MCPClientManager(config)
        manager._clients["test"]._state = MCPServerState.CONNECTED
        manager._clients["test"]._tools = [
            MCPTool(server_name="test", name="tool1", description=""),
            MCPTool(server_name="test", name="tool2", description=""),
        ]
        return ToolExecutor(manager)

    @pytest.mark.asyncio
    async def test_execute_empty_list(self, executor_with_manager: ToolExecutor):
        """Test executing empty tool calls list."""
        results = await executor_with_manager.execute_tool_calls([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_parallel(self, executor_with_manager: ToolExecutor):
        """Test parallel execution of tool calls."""
        executor_with_manager.manager.execute_tool_call = AsyncMock(
            side_effect=[
                MCPToolResult(tool_name="tool1", content="Result 1"),
                MCPToolResult(tool_name="tool2", content="Result 2"),
            ]
        )

        tool_calls = [
            {"id": "call_1", "function": {"name": "test__tool1", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "test__tool2", "arguments": "{}"}},
        ]

        results = await executor_with_manager.execute_tool_calls(tool_calls, parallel=True)

        assert len(results) == 2
        assert results[0][0].content == "Result 1"
        assert results[0][1] == "call_1"
        assert results[1][0].content == "Result 2"
        assert results[1][1] == "call_2"

    @pytest.mark.asyncio
    async def test_execute_sequential(self, executor_with_manager: ToolExecutor):
        """Test sequential execution of tool calls."""
        call_order = []

        async def track_call(tc, **kwargs):
            name = tc["function"]["name"]
            call_order.append(name)
            return MCPToolResult(tool_name=name, content=f"Result for {name}")

        executor_with_manager.manager.execute_tool_call = AsyncMock(side_effect=track_call)

        tool_calls = [
            {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}},
        ]

        results = await executor_with_manager.execute_tool_calls(tool_calls, parallel=False)

        assert len(results) == 2
        # Verify sequential order
        assert call_order == ["tool1", "tool2"]

    @pytest.mark.asyncio
    async def test_execute_parallel_with_exception(self, executor_with_manager: ToolExecutor):
        """Test parallel execution handles exceptions."""
        executor_with_manager.manager.execute_tool_call = AsyncMock(
            side_effect=[
                MCPToolResult(tool_name="tool1", content="Success"),
                RuntimeError("Tool failed"),
            ]
        )

        tool_calls = [
            {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}},
        ]

        results = await executor_with_manager.execute_tool_calls(tool_calls, parallel=True)

        assert len(results) == 2
        assert results[0][0].is_error is False
        assert results[1][0].is_error is True
        assert "Tool failed" in results[1][0].error_message

    @pytest.mark.asyncio
    async def test_execute_sequential_with_exception(self, executor_with_manager: ToolExecutor):
        """Test sequential execution handles exceptions."""
        executor_with_manager.manager.execute_tool_call = AsyncMock(
            side_effect=[
                RuntimeError("First failed"),
                MCPToolResult(tool_name="tool2", content="Success"),
            ]
        )

        tool_calls = [
            {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}},
        ]

        results = await executor_with_manager.execute_tool_calls(tool_calls, parallel=False)

        assert len(results) == 2
        assert results[0][0].is_error is True
        assert results[1][0].is_error is False

    @pytest.mark.asyncio
    async def test_execute_parallel_respects_semaphore(self, executor_with_manager: ToolExecutor):
        """Test parallel execution respects max_parallel limit."""
        executor_with_manager.max_parallel = 2
        concurrent_count = []
        current_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrency(tc, **kwargs):
            nonlocal current_concurrent
            async with lock:
                current_concurrent += 1
                concurrent_count.append(current_concurrent)
            await asyncio.sleep(0.05)  # Simulate work
            async with lock:
                current_concurrent -= 1
            return MCPToolResult(tool_name="tool", content="ok")

        executor_with_manager.manager.execute_tool_call = AsyncMock(side_effect=track_concurrency)

        tool_calls = [
            {"id": f"call_{i}", "function": {"name": f"tool{i}", "arguments": "{}"}}
            for i in range(5)
        ]

        await executor_with_manager.execute_tool_calls(tool_calls, parallel=True)

        # Max concurrent should never exceed 2
        assert max(concurrent_count) <= 2


class TestToolExecutorExecuteAndFormat:
    """Tests for ToolExecutor.execute_and_format()."""

    @pytest.fixture
    def executor(self) -> ToolExecutor:
        """Create executor with mock manager."""
        config = MCPConfig(default_timeout=10.0)
        manager = MCPClientManager(config)
        return ToolExecutor(manager)

    @pytest.mark.asyncio
    async def test_execute_and_format(self, executor: ToolExecutor):
        """Test execute_and_format returns formatted messages."""
        executor.manager.execute_tool_call = AsyncMock(
            return_value=MCPToolResult(tool_name="test", content="Result data")
        )

        tool_calls = [
            {"id": "call_123", "function": {"name": "test__tool", "arguments": "{}"}},
        ]

        messages = await executor.execute_and_format(tool_calls)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_123"
        assert messages[0]["content"] == "Result data"

    @pytest.mark.asyncio
    async def test_execute_and_format_error(self, executor: ToolExecutor):
        """Test execute_and_format formats error messages."""
        executor.manager.execute_tool_call = AsyncMock(
            return_value=MCPToolResult(
                tool_name="test",
                content=None,
                is_error=True,
                error_message="Tool failed",
            )
        )

        tool_calls = [
            {"id": "call_456", "function": {"name": "test__tool", "arguments": "{}"}},
        ]

        messages = await executor.execute_and_format(tool_calls)

        assert len(messages) == 1
        assert "Error: Tool failed" in messages[0]["content"]


class TestToolExecutorExtractAndValidate:
    """Tests for ToolExecutor.extract_and_validate()."""

    @pytest.fixture
    def executor_with_tools(self) -> ToolExecutor:
        """Create executor with tools."""
        config = MCPConfig.from_dict({
            "servers": {
                "test": {"transport": "stdio", "command": "python"},
            }
        })
        manager = MCPClientManager(config)
        manager._clients["test"]._state = MCPServerState.CONNECTED
        manager._clients["test"]._tools = [
            MCPTool(server_name="test", name="valid_tool", description=""),
        ]
        return ToolExecutor(manager)

    def test_extract_valid_tools(self, executor_with_tools: ToolExecutor):
        """Test extracting and validating valid tool calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "test__valid_tool",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }

        calls, all_valid = executor_with_tools.extract_and_validate(response)

        assert len(calls) == 1
        assert all_valid is True

    def test_extract_invalid_tool(self, executor_with_tools: ToolExecutor):
        """Test extracting and validating invalid tool."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "nonexistent_tool",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }

        calls, all_valid = executor_with_tools.extract_and_validate(response)

        assert len(calls) == 1
        assert all_valid is False

    def test_extract_no_tool_calls(self, executor_with_tools: ToolExecutor):
        """Test extracting from response without tool calls."""
        response = {
            "choices": [{"message": {"content": "No tools"}}]
        }

        calls, all_valid = executor_with_tools.extract_and_validate(response)

        assert calls == []
        assert all_valid is True

    def test_extract_mixed_valid_invalid(self, executor_with_tools: ToolExecutor):
        """Test extracting mix of valid and invalid tools."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "test__valid_tool", "arguments": "{}"},
                            },
                            {
                                "id": "call_2",
                                "function": {"name": "invalid_tool", "arguments": "{}"},
                            },
                        ]
                    }
                }
            ]
        }

        calls, all_valid = executor_with_tools.extract_and_validate(response)

        assert len(calls) == 2
        assert all_valid is False


class TestToolExecutorToolExists:
    """Tests for ToolExecutor._tool_exists()."""

    @pytest.fixture
    def executor(self) -> ToolExecutor:
        """Create executor with tools."""
        config = MCPConfig.from_dict({
            "servers": {
                "server1": {"transport": "stdio", "command": "python"},
                "server2": {"transport": "stdio", "command": "node"},
            }
        })
        manager = MCPClientManager(config)
        manager._clients["server1"]._state = MCPServerState.CONNECTED
        manager._clients["server1"]._tools = [
            MCPTool(server_name="server1", name="tool_a", description=""),
        ]
        manager._clients["server2"]._state = MCPServerState.CONNECTED
        manager._clients["server2"]._tools = [
            MCPTool(server_name="server2", name="tool_b", description=""),
        ]
        return ToolExecutor(manager)

    def test_tool_exists_full_name(self, executor: ToolExecutor):
        """Test finding tool by full name."""
        assert executor._tool_exists("server1__tool_a") is True
        assert executor._tool_exists("server2__tool_b") is True
        assert executor._tool_exists("server1__nonexistent") is False

    def test_tool_exists_short_name(self, executor: ToolExecutor):
        """Test finding tool by short name."""
        assert executor._tool_exists("tool_a") is True
        assert executor._tool_exists("tool_b") is True
        assert executor._tool_exists("nonexistent") is False


class TestExecuteSingleTool:
    """Tests for execute_single_tool convenience function."""

    @pytest.mark.asyncio
    async def test_execute_single_tool(self):
        """Test execute_single_tool calls manager."""
        config = MCPConfig()
        manager = MCPClientManager(config)
        manager.execute_tool = AsyncMock(
            return_value=MCPToolResult(tool_name="tool", content="Result")
        )

        result = await execute_single_tool(
            manager,
            "server__tool",
            {"arg": "value"},
            timeout=30.0,
        )

        assert result.content == "Result"
        manager.execute_tool.assert_called_once_with(
            "server__tool", {"arg": "value"}, 30.0
        )

    @pytest.mark.asyncio
    async def test_execute_single_tool_default_timeout(self):
        """Test execute_single_tool with default timeout."""
        config = MCPConfig()
        manager = MCPClientManager(config)
        manager.execute_tool = AsyncMock(
            return_value=MCPToolResult(tool_name="tool", content="Result")
        )

        await execute_single_tool(manager, "tool", {})

        manager.execute_tool.assert_called_once_with("tool", {}, None)
