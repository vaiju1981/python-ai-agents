"""Agents-as-tools: wrap an ``Agent`` as a ``Tool`` so one agent can call another.

The calling agent's model decides, per turn, which specialist(s) to invoke and
with what request. The caller's identity, tenant, trace, and deadline flow
into the specialist (in a fresh child session), and the specialist's own
guardrails and tool authorization still apply.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.tool import Tool, ToolEffect, ToolResult, ToolSpec

__all__ = ["agent_as_tool"]

_REQUEST_PARAM = "request"


def agent_as_tool(
    name: str,
    description: str,
    agent: Agent,
    *,
    effect: ToolEffect = ToolEffect.EFFECTFUL,
) -> Tool:
    """Wrap ``agent`` as a tool that the calling model can invoke.

    The tool has one parameter (``request``) — the string the calling model
    fills with the specialist's input. Effectful by default (the safe choice
    — a specialist can do whatever its own tools allow).
    """
    schema = {
        "type": "object",
        "properties": {
            _REQUEST_PARAM: {
                "type": "string",
                "description": f"the request to send to the {name} agent",
            }
        },
        "required": [_REQUEST_PARAM],
    }
    spec = ToolSpec(name=name, description=description, input_schema=schema, effect=effect)

    class _AgentTool:
        def __init__(self):
            self._spec = spec
            self._agent = agent

        @property
        def spec(self) -> ToolSpec:
            return self._spec

        async def invoke(self, arguments: dict[str, Any], context: RequestContext) -> ToolResult:
            text = arguments.get(_REQUEST_PARAM, "")
            if not isinstance(text, str) or not text.strip():
                text = json.dumps(arguments)
            child = RequestContext(
                session_id=str(uuid4()),
                principal=context.principal,
                tenant=context.tenant,
                trace_id=context.trace_id,
                deadline=context.deadline,
                attributes=dict(context.attributes),
            )
            response = await self._agent.run(AgentRequest(input=text, context=child))
            if response.blocked:
                return ToolResult.failed(response.output)
            return ToolResult.ok(response.output)

    return _AgentTool()
