from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from python_ai_agents.core.context import RequestContext


@dataclass(frozen=True, slots=True)
class AgentRequest:
    input: str
    context: RequestContext

    @classmethod
    def ephemeral(cls, input: str) -> AgentRequest:
        return cls(input=input, context=RequestContext.ephemeral())


@dataclass(frozen=True, slots=True)
class AgentResponse:
    output: str
    blocked: bool = False
    stop_reason: str = "completed"

    @classmethod
    def completed(cls, output: str) -> AgentResponse:
        return cls(output=output)

    @classmethod
    def blocked_response(cls, output: str, reason: str) -> AgentResponse:
        return cls(output=output, blocked=True, stop_reason=reason)

    @classmethod
    def stopped(cls, output: str, reason: str) -> AgentResponse:
        return cls(output=output, blocked=False, stop_reason=reason)


class Agent(Protocol):
    """Universal agent seam.

    Adapters for OpenAI Agents SDK, Pydantic AI, LangGraph, or custom runtimes should
    implement this protocol.
    """

    async def run(self, request: AgentRequest) -> AgentResponse:
        ...

