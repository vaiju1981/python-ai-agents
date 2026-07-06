"""A2A (Agent-to-Agent) protocol adapter.

Wraps the A2A SDK to expose remote agents as our ``Agent`` protocol and
serve local agents over A2A. Install::

    pip install python-ai-agents[a2a]

Usage::

    from python_ai_agents.adapters.a2a import RemoteA2aAgent

    agent = RemoteA2aAgent(agent_url="http://localhost:8001")
    response = await agent.run(AgentRequest.ephemeral("hello"))
"""

from __future__ import annotations

from dataclasses import dataclass

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse

__all__ = ["RemoteA2aAgent"]


@dataclass(slots=True)
class RemoteA2aAgent:
    """Wraps a remote A2A agent as our ``Agent`` protocol.

    Uses the A2A SDK's client to send requests to a remote agent server.
    """

    agent_url: str
    _client: object | None = None

    def __post_init__(self) -> None:
        try:
            from a2a.client import A2AClient

            self._client = A2AClient(self.agent_url)
        except ImportError:
            pass

    async def run(self, request: AgentRequest) -> AgentResponse:
        if self._client is None:
            return AgentResponse.stopped(
                "A2A SDK not installed", "model_error"
            )
        try:
            result = await self._client.send_message(request.input)
            return AgentResponse.completed(_extract_text(result))
        except Exception as exc:
            return AgentResponse.stopped(
                f"remote agent error: {exc}", "model_error"
            )


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        parts = result.get("parts", [])
        if isinstance(parts, list):
            return " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in parts
            )
    return str(result)
