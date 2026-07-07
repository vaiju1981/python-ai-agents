from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from python_ai_agents.core.context import RequestContext


class GuardrailStage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    content: str
    blocked: bool = False
    reason: str = ""

    @classmethod
    def allow(cls, content: str) -> GuardrailDecision:
        return cls(content=content)

    @classmethod
    def block(cls, content: str, reason: str) -> GuardrailDecision:
        return cls(content=content, blocked=True, reason=reason)


class Guardrail(Protocol):
    async def check(
        self,
        stage: GuardrailStage,
        content: str,
        context: RequestContext,
    ) -> GuardrailDecision: ...
