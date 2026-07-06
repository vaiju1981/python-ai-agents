"""Tests for token-budget enforcement."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    BudgetAgent,
    BudgetObserver,
    DefaultAgent,
    ModelRequest,
    ModelResponse,
    StopReason,
    TokenBudget,
    Usage,
)


class ScriptedModel:
    def __init__(self, usage_per_call: Usage | None = None) -> None:
        self.usage_per_call = usage_per_call or Usage(input_tokens=50, output_tokens=50)
        self.calls = 0

    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse.text_response(f"response-{self.calls}", self.usage_per_call)


def test_token_budget_tracks_usage() -> None:
    budget = TokenBudget(limit=1000)
    budget.consume(Usage(input_tokens=100, output_tokens=200))

    assert budget.input_tokens == 100
    assert budget.output_tokens == 200
    assert budget.total_tokens == 300
    assert budget.remaining == 700
    assert not budget.exhausted


def test_token_budget_exhausted() -> None:
    budget = TokenBudget(limit=100)
    budget.consume(Usage(input_tokens=60, output_tokens=50))

    assert budget.total_tokens == 110
    assert budget.exhausted
    assert budget.remaining == 0


def test_token_budget_reset() -> None:
    budget = TokenBudget(limit=100)
    budget.consume(Usage(input_tokens=50, output_tokens=30))
    budget.reset()

    assert budget.total_tokens == 0
    assert not budget.exhausted


def test_budget_observer_consumes_usage() -> None:
    async def run() -> None:
        budget = TokenBudget(limit=200)
        observer = BudgetObserver(budget)
        model = ScriptedModel(Usage(input_tokens=50, output_tokens=50))

        await DefaultAgent(model, observers=[observer]).run(AgentRequest.ephemeral("hi"))

        assert budget.total_tokens == 100
        assert not budget.exhausted

    anyio.run(run)


def test_budget_agent_short_circuits_when_exhausted() -> None:
    async def run() -> None:
        budget = TokenBudget(limit=100)
        observer = BudgetObserver(budget)
        model = ScriptedModel(Usage(input_tokens=60, output_tokens=60))
        inner = DefaultAgent(model, observers=[observer])
        agent = BudgetAgent(inner, budget)

        # First call consumes 120 tokens, exhausting the 100-token budget
        r1 = await agent.run(AgentRequest.ephemeral("first"))
        assert r1.stop_reason == "budget_exceeded"
        assert r1.reason == StopReason.BUDGET_EXCEEDED

    anyio.run(run)


def test_budget_agent_blocks_before_run_when_exhausted() -> None:
    async def run() -> None:
        budget = TokenBudget(limit=50)
        budget.consume(Usage(input_tokens=30, output_tokens=30))  # pre-exhaust
        model = ScriptedModel()
        inner = DefaultAgent(model)
        agent = BudgetAgent(inner, budget)

        r = await agent.run(AgentRequest.ephemeral("hi"))
        assert r.stop_reason == "budget_exceeded"
        assert model.calls == 0  # model never called

    anyio.run(run)


def test_budget_agent_allows_when_under_budget() -> None:
    async def run() -> None:
        budget = TokenBudget(limit=1000)
        observer = BudgetObserver(budget)
        model = ScriptedModel(Usage(input_tokens=10, output_tokens=10))
        inner = DefaultAgent(model, observers=[observer])
        agent = BudgetAgent(inner, budget)

        r = await agent.run(AgentRequest.ephemeral("hello"))
        assert r.is_completed
        assert r.output == "response-1"

    anyio.run(run)
