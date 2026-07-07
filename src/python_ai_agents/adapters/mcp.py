"""MCP (Model Context Protocol) adapter.

Wraps the official ``mcp`` Python SDK to expose MCP tools as our ``Tool``
protocol. Install::

    pip install python-ai-agents[mcp]

Usage::

    from python_ai_agents.adapters.mcp import McpToolAdapter

    tool = McpToolAdapter.from_mcp_tool(mcp_tool, transport=transport)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.tool import ToolEffect, ToolResult, ToolSpec

__all__ = ["McpToolAdapter"]


@dataclass(slots=True)
class McpToolAdapter:
    """Adapts an MCP tool to our ``Tool`` protocol.

    Wraps a tool discovered via the MCP SDK, mapping its schema and invocation
    to our ``ToolSpec`` / ``ToolResult`` types.
    """

    _spec: ToolSpec
    _transport: Any = None
    _tool_name: str = ""

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, arguments: dict[str, Any], context: RequestContext) -> ToolResult:
        if self._transport is None:
            return ToolResult.failed("no MCP transport configured")
        try:
            result = await self._transport.call_tool(self._tool_name, arguments)
            content = _extract_content(result)
            return ToolResult.ok(content)
        except Exception as exc:
            return ToolResult.failed(f"MCP tool '{self._tool_name}' failed: {exc}")

    @classmethod
    def from_mcp_tool(
        cls,
        tool_def: dict[str, Any],
        transport: Any,
        *,
        effect: ToolEffect = ToolEffect.READ_ONLY,
    ) -> McpToolAdapter:
        """Build a ``McpToolAdapter`` from an MCP tool definition dict."""
        name = tool_def.get("name", "")
        description = tool_def.get("description", "")
        schema = tool_def.get("inputSchema", {"type": "object"})
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=schema,
            effect=effect,
        )
        return cls(_spec=spec, _transport=transport, _tool_name=name)


def _extract_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content", [])
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(result)
    return str(result)
