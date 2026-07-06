"""Token-budget enforcement over ``Usage`` events.

Provides a ``TokenBudget`` tracker, a ``BudgetObserver`` that consumes usage
into the budget, and a ``BudgetAgent`` wrapper that short-circuits when the
budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.model import Usage
from python_ai_agents.core.observe import NoopAgentObserver

__all__ = [
    "BudgetAgent",
    "BudgetObserver",
    "TokenBudget",
]


@dataclass(slots=True)
class TokenBudget:
    """Mutable token-budget tracker.

    Consume usage via ``consume`` and check ``exhausted`` / ``remaining``.
    """

    limit: int
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.total_tokens)

    @property
    def exhausted(self) -> bool:
        return self.total_tokens >= self.limit

    def consume(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


@dataclass(slots=True)
class BudgetObserver(NoopAgentObserver):
    """Observer that consumes ``Usage`` events into a ``TokenBudget``.

    Attach this to a ``DefaultAgent`` (or any agent that emits ``on_usage``)
    alongside a ``BudgetAgent`` wrapper to enforce the budget.
    """

    budget: TokenBudget = field(default_factory=lambda: TokenBudget(limit=10_000))

    async def on_usage(self, model: str, usage: Usage) -> None:
        self.budget.consume(usage)


@dataclass(slots=True)
class BudgetAgent:
    """Wraps an ``Agent`` and short-circuits when the token budget is exhausted.

    The wrapped delegate should have a ``BudgetObserver`` in its observers so
    that usage is tracked in the same ``TokenBudget`` instance.
    """

    delegate: Agent
    budget: TokenBudget
    exhausted_message: str = "I've reached my token budget for this session."

    async def run(self, request: AgentRequest) -> AgentResponse:
        if self.budget.exhausted:
            return AgentResponse.stopped(self.exhausted_message, "budget_exceeded")
        response = await self.delegate.run(request)
        if self.budget.exhausted and response.is_completed:
            return AgentResponse.stopped(response.output, "budget_exceeded")
        return response
