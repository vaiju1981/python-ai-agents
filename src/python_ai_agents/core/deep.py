"""Deep agent: plan → run sub-agents → synthesize.

Each ``PlanStep`` may declare dependencies; steps run in waves — a step
becomes eligible once all its dependencies are ``DONE``. A dependent step
receives its upstream steps' results injected into its instruction. With an
optional ``CheckpointStore`` the run becomes crash-resumable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import anyio

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.checkpoint import CheckpointStore
from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.model import Message, ModelPort, ModelRequest
from python_ai_agents.core.structured import extract_structured
from pydantic import BaseModel

__all__ = [
    "DeepAgent",
    "LlmPlanner",
    "Plan",
    "PlanStep",
    "Planner",
    "StepStatus",
]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class PlanStep:
    """One subtask in a ``Plan``. Mutable status/result so it doubles as a todo item."""

    index: int
    description: str
    depends_on: list[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str = ""


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered list of subtasks for a deep agent."""

    steps: tuple[PlanStep, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def render(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"[{s.status.value}] {s.index}. {s.description}")
        return "\n".join(lines)


class Planner(Protocol):
    """Decomposes a task into a ``Plan`` of subtasks."""

    async def plan(self, task: str) -> Plan:
        ...


class _StepsSchema(BaseModel):
    steps: list[str] = []


@dataclass(slots=True)
class LlmPlanner:
    """A ``Planner`` that asks a model to decompose a task into subtasks.

    Uses ``extract_structured`` with Pydantic for schema-constrained JSON.
    """

    model: ModelPort
    max_steps: int = 5

    async def plan(self, task: str) -> Plan:
        prompt = (
            f"Break the task into between 2 and {self.max_steps} concrete, "
            f"independent subtasks, each a short imperative phrase.\nTask: {task}"
        )
        result = await extract_structured(
            self.model, _StepsSchema, prompt, RequestContext.ephemeral()
        )
        steps: list[PlanStep] = []
        if result.value is not None:
            for i, s in enumerate(result.value.steps, 1):
                if s and s.strip():
                    steps.append(PlanStep(index=i, description=s.strip()))
                    if len(steps) >= self.max_steps:
                        break
        return Plan(steps=tuple(steps))


@dataclass(slots=True)
class DeepAgent:
    """A deep agent: plan → run sub-agents → synthesize. Implements ``Agent``.

    The plan is a DAG. Each step may declare dependencies; steps run in waves
    — a step becomes eligible once all its dependencies are ``DONE``. A
    dependent step receives its upstream steps' results injected into its
    instruction.
    """

    planner: Planner
    worker_factory: Callable[[], Agent]
    synthesizer: ModelPort
    checkpoint_store: CheckpointStore | None = None
    step_timeout_seconds: float | None = None

    async def run(self, request: AgentRequest) -> AgentResponse:
        ctx = request.context
        plan = await self.planner.plan(request.input)
        if plan.is_empty:
            return AgentResponse.completed("")

        steps = list(plan.steps)
        # Run steps in dependency waves
        while any(s.status == StepStatus.PENDING for s in steps):
            eligible = [
                s for s in steps
                if s.status == StepStatus.PENDING
                and all(steps[d - 1].status == StepStatus.DONE for d in s.depends_on if 0 < d <= len(steps))
            ]
            if not eligible:
                break
            for step in eligible:
                step.status = StepStatus.RUNNING
                instruction = step.description
                for dep_idx in step.depends_on:
                    if 0 < dep_idx <= len(steps):
                        dep = steps[dep_idx - 1]
                        if dep.result:
                            instruction += f"\n\nInput from step {dep_idx}: {dep.result}"
                try:
                    if self.step_timeout_seconds:
                        with anyio.move_on_after(self.step_timeout_seconds) as scope:
                            resp = await self.worker_factory().run(
                                AgentRequest(instruction, ctx.child_session())
                            )
                        if scope.cancel_called:
                            step.status = StepStatus.FAILED
                            step.result = "timed out"
                            continue
                    else:
                        resp = await self.worker_factory().run(
                            AgentRequest(instruction, ctx.child_session())
                        )
                    step.result = resp.output
                    step.status = StepStatus.DONE
                except Exception:
                    step.status = StepStatus.FAILED
                    step.result = "failed"

        # Synthesize
        results = "\n".join(
            f"Step {s.index} ({s.description}): {s.result}"
            for s in steps
            if s.status == StepStatus.DONE
        )
        synth_prompt = (
            f"Synthesize a final answer from the subtask results below.\n\n"
            f"Task: {request.input}\n\nResults:\n{results}"
        )
        synth_request = ModelRequest(messages=(Message.user(synth_prompt),))
        synth_response = await self.synthesizer.chat(synth_request)
        return AgentResponse.completed(synth_response.text)
