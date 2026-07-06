"""Learning and reflection: self-critique with episodic memory.

``ReflectiveAgent`` recalls lessons from similar past episodes, injects them,
answers, self-critiques, and on a poor answer records the lesson and retries
(up to a budget) with that lesson applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from pydantic import BaseModel

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.episodic import Episode, EpisodicStore, InMemoryEpisodicStore
from python_ai_agents.core.model import Message, ModelPort, ModelRequest
from python_ai_agents.core.structured import extract_structured
from python_ai_agents.core.context import RequestContext

__all__ = [
    "LlmReflector",
    "Reflection",
    "Reflector",
    "ReflectiveAgent",
]


@dataclass(frozen=True, slots=True)
class Reflection:
    """A self-critique verdict: was the answer good enough, and if not, the lesson."""

    satisfactory: bool
    lesson: str

    @classmethod
    def ok(cls) -> Reflection:
        return cls(satisfactory=True, lesson="")

    @classmethod
    def issue(cls, lesson: str) -> Reflection:
        return cls(satisfactory=False, lesson=lesson)


class Reflector(Protocol):
    """Judges whether an answer addresses a task, producing a lesson when it doesn't."""

    async def reflect(self, task: str, answer: str) -> Reflection:
        ...


class _VerdictSchema(BaseModel):
    satisfactory: bool = True
    lesson: str = ""


@dataclass(slots=True)
class LlmReflector:
    """A ``Reflector`` backed by a model via ``extract_structured`` (Pydantic JSON)."""

    model: ModelPort

    async def reflect(self, task: str, answer: str) -> Reflection:
        prompt = (
            "Decide whether the ANSWER correctly and completely addresses the TASK. "
            "If it does not, give a one-sentence lesson to fix it.\n\n"
            f"TASK: {task}\n\nANSWER: {answer}"
        )
        result = await extract_structured(
            self.model, _VerdictSchema, prompt, RequestContext.ephemeral()
        )
        if result.value is None:
            return Reflection.ok()
        if result.value.satisfactory:
            return Reflection.ok()
        return Reflection.issue(result.value.lesson)


@dataclass(slots=True)
class ReflectiveAgent:
    """An agent that learns from its mistakes.

    Before answering, recalls lessons from similar past episodes and injects
    them. After answering, self-critiques; on a poor answer it records the
    lesson and retries (up to ``max_attempts``) with that lesson applied.
    """

    worker_factory: Callable[[], Agent]
    reflector: Reflector
    memory: EpisodicStore = field(default_factory=InMemoryEpisodicStore)
    max_attempts: int = 3
    recall_limit: int = 5

    async def run(self, request: AgentRequest) -> AgentResponse:
        task = request.input
        tenant = request.context.tenant
        past = await self.memory.recall(tenant, task, self.recall_limit)
        lessons = _format_lessons(past)

        last: AgentResponse | None = None
        for attempt in range(1, self.max_attempts + 1):
            inp = task
            if lessons:
                inp = f"{task}\n\nLessons to apply (from earlier attempts):\n{lessons}"
            last = await self.worker_factory().run(
                AgentRequest(input=inp, context=request.context.child_session())
            )

            reflection = await self.reflector.reflect(task, last.output)
            if reflection.satisfactory:
                return last

            await self.memory.record(
                Episode(tenant=tenant, task=task, outcome=last.output, success=False,
                        lesson=reflection.lesson)
            )
            lessons = (lessons + f"- {reflection.lesson}\n") if lessons else f"- {reflection.lesson}\n"

        return last  # type: ignore[return-value]


def _format_lessons(episodes: list[Episode]) -> str:
    if not episodes:
        return ""
    return "".join(f"- {ep.lesson}\n" for ep in episodes if ep.lesson)
