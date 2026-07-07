"""Tests for the MCP tool adapter. Skips if mcp not installed."""

import importlib.util

import pytest

mcp_available = importlib.util.find_spec("mcp") is not None
pytestmark = pytest.mark.skipif(not mcp_available, reason="mcp not installed")


def test_mcp_tool_adapter_from_definition():
    from python_ai_agents import ToolEffect
    from python_ai_agents.adapters.mcp import McpToolAdapter

    tool_def = {
        "name": "search",
        "description": "Search the web",
        "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
    adapter = McpToolAdapter.from_mcp_tool(tool_def, transport=None)
    assert adapter.spec.name == "search"
    assert adapter.spec.description == "Search the web"
    assert adapter.spec.effect == ToolEffect.READ_ONLY


def test_mcp_tool_adapter_invoke_without_transport():
    import anyio

    from python_ai_agents import RequestContext
    from python_ai_agents.adapters.mcp import McpToolAdapter

    adapter = McpToolAdapter.from_mcp_tool(
        {"name": "test", "description": "test", "inputSchema": {}},
        transport=None,
    )

    async def run():
        result = await adapter.invoke({}, RequestContext.ephemeral())
        assert result.error
        assert "no MCP transport" in result.content

    anyio.run(run)
