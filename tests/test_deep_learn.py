"""Tests for deep agent planning and reflective learning."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    DefaultAgent,
    DeepAgent,
    Episode,
    InMemoryEpisodicStore,
    LlmReflector,
    LlmPlanner,
    ModelRequest,
    ModelResponse,
    Plan,
    PlanStep,
    ReflectiveAgent,
    Reflection,
)


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.responses = list(responses or [])

    async def chat(self, request: ModelRequest) -> ModelResponse:
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse.text_response("ok")


def test_plan_render() -> None:
    plan = Plan(steps=(
        PlanStep(index=1, description="do A"),
        PlanStep(index=2, description="do B", depends_on=[1]),
    ))
    rendered = plan.render()
    assert "do A" in rendered
    assert "do B" in rendered
    assert plan.is_empty is False


def test_plan_empty() -> None:
    plan = Plan(steps=())
    assert plan.is_empty


def test_deep_agent_runs_steps() -> None:
    async def run() -> None:
        class StubPlanner:
            async def plan(self, task: str) -> Plan:
                return Plan(steps=(
                    PlanStep(index=1, description="step A"),
                    PlanStep(index=2, description="step B"),
                ))

        class StubAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed(f"result for: {request.input}")

        model = ScriptedModel(responses=[ModelResponse.text_response("final synthesis")])
        agent = DeepAgent(
            planner=StubPlanner(),
            worker_factory=lambda: StubAgent(),
            synthesizer=model,
        )
        response = await agent.run(AgentRequest.ephemeral("do something"))
        assert "final synthesis" in response.output

    anyio.run(run)


def test_deep_agent_respects_dependencies() -> None:
    async def run() -> None:
        class StubPlanner:
            async def plan(self, task: str) -> Plan:
                return Plan(steps=(
                    PlanStep(index=1, description="first"),
                    PlanStep(index=2, description="second", depends_on=[1]),
                ))

        call_order: list[str] = []

        class OrderingAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                call_order.append(request.input)
                return AgentResponse.completed(f"done: {request.input}")

        model = ScriptedModel(responses=[ModelResponse.text_response("synthesis")])
        agent = DeepAgent(
            planner=StubPlanner(),
            worker_factory=lambda: OrderingAgent(),
            synthesizer=model,
        )
        await agent.run(AgentRequest.ephemeral("task"))
        # Step 1 should run before step 2
        assert "first" in call_order[0]
        assert "second" in call_order[1]

    anyio.run(run)


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
