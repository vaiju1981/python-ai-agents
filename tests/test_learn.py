"""Tests for reflective learning (retry on self-critique)."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    Reflection,
    ReflectiveAgent,
)


def test_reflection_ok_and_issue() -> None:
    ok = Reflection.ok()
    assert ok.satisfactory
    assert ok.lesson == ""

    issue = Reflection.issue("fix X")
    assert not issue.satisfactory
    assert issue.lesson == "fix X"


def test_reflective_agent_retries_on_failure() -> None:
    async def run() -> None:
        class AlwaysFailReflector:
            async def reflect(self, task: str, answer: str) -> Reflection:
                return Reflection.issue("not good enough")

        class StubAgent:
            def __init__(self):
                self.calls = 0

            async def run(self, request: AgentRequest) -> AgentResponse:
                self.calls += 1
                return AgentResponse.completed(f"attempt {self.calls}")

        worker = StubAgent()
        agent = ReflectiveAgent(
            worker_factory=lambda: worker,
            reflector=AlwaysFailReflector(),
            max_attempts=2,
        )
        response = await agent.run(AgentRequest.ephemeral("do X"))
        assert "attempt 2" in response.output  # retried

    anyio.run(run)


def test_reflective_agent_succeeds_first_try() -> None:
    async def run() -> None:
        class AlwaysOkReflector:
            async def reflect(self, task: str, answer: str) -> Reflection:
                return Reflection.ok()

        class StubAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed("perfect answer")

        agent = ReflectiveAgent(
            worker_factory=lambda: StubAgent(),
            reflector=AlwaysOkReflector(),
            max_attempts=3,
        )
        response = await agent.run(AgentRequest.ephemeral("do X"))
        assert response.output == "perfect answer"

    anyio.run(run)
