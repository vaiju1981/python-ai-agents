from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from python_ai_agents.core.context import RequestContext


class StopCategory(str, Enum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"
    TIMEOUT = "timeout"
    ERROR = "error"


class StopReason(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    MAX_STEPS = "max_steps"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    MODEL_ERROR = "model_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNKNOWN = "unknown"

    @property
    def category(self) -> StopCategory:
        return {
            StopReason.COMPLETED: StopCategory.SUCCESS,
            StopReason.BLOCKED: StopCategory.BLOCKED,
            StopReason.MAX_STEPS: StopCategory.INCOMPLETE,
            StopReason.DEADLINE_EXCEEDED: StopCategory.TIMEOUT,
            StopReason.MODEL_ERROR: StopCategory.ERROR,
            StopReason.BUDGET_EXCEEDED: StopCategory.ERROR,
            StopReason.UNKNOWN: StopCategory.ERROR,
        }[self]

    @property
    def retryable(self) -> bool:
        return self in {StopReason.DEADLINE_EXCEEDED, StopReason.MODEL_ERROR}

    @classmethod
    def classify(cls, *, blocked: bool, stop_reason: str) -> StopReason:
        if blocked:
            return cls.BLOCKED
        try:
            return cls(stop_reason)
        except ValueError:
            return cls.UNKNOWN


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

    @property
    def is_completed(self) -> bool:
        return not self.blocked and self.stop_reason == StopReason.COMPLETED.value

    @property
    def reason(self) -> StopReason:
        return StopReason.classify(blocked=self.blocked, stop_reason=self.stop_reason)

    @property
    def retryable(self) -> bool:
        return self.reason.retryable


class Agent(Protocol):
    """Universal agent seam.

    Adapters for OpenAI Agents SDK, Pydantic AI, LangGraph, or custom runtimes should
    implement this protocol.
    """

    async def run(self, request: AgentRequest) -> AgentResponse:
        ...
