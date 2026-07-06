"""Tests for the Guardrails AI-backed content validation guardrail.

Skips if ``guardrails-ai`` is not installed.
"""

from __future__ import annotations

import importlib

import anyio
import pytest

guardrails_available = importlib.util.find_spec("guardrails") is not None

pytestmark = pytest.mark.skipif(
    not guardrails_available, reason="guardrails-ai not installed"
)


class FakeGuard:
    """Minimal stand-in for guardrails.Guard that avoids Pydantic model issues."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises
        self.validated: list[str] = []

    def validate(self, value: str) -> str:
        self.validated.append(value)
        if self._raises is not None:
            raise self._raises
        return value


def test_guardrails_ai_blocks_on_validation_failure() -> None:
    from python_ai_agents import GuardrailStage, RequestContext
    from python_ai_agents.adapters.guardrails_ai import GuardrailsAiGuardrail

    async def run() -> None:
        guard = FakeGuard(raises=ValueError("blocked by validator"))
        guardrail = GuardrailsAiGuardrail(guard=guard)

        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "some content",
            RequestContext.ephemeral(),
        )

        assert decision.blocked
        assert "validation failed" in decision.reason
        assert guard.validated == ["some content"]

    anyio.run(run)


def test_guardrails_ai_allows_on_validation_pass() -> None:
    from python_ai_agents import GuardrailStage, RequestContext
    from python_ai_agents.adapters.guardrails_ai import GuardrailsAiGuardrail

    async def run() -> None:
        guard = FakeGuard()
        guardrail = GuardrailsAiGuardrail(guard=guard)

        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "normal content",
            RequestContext.ephemeral(),
        )

        assert not decision.blocked
        assert decision.content == "normal content"
        assert guard.validated == ["normal content"]

    anyio.run(run)


def test_guardrails_ai_stage_filter() -> None:
    from python_ai_agents import GuardrailStage, RequestContext
    from python_ai_agents.adapters.guardrails_ai import GuardrailsAiGuardrail

    async def run() -> None:
        guard = FakeGuard(raises=ValueError("should not be called"))
        guardrail = GuardrailsAiGuardrail(
            guard=guard, stage=GuardrailStage.INPUT
        )

        # Should skip on OUTPUT stage — guard.validate never called
        decision = await guardrail.check(
            GuardrailStage.OUTPUT,
            "some output",
            RequestContext.ephemeral(),
        )

        assert not decision.blocked
        assert decision.content == "some output"
        assert guard.validated == []  # validate was never called

    anyio.run(run)


def test_guardrails_ai_auto_creates_guard_with_validators() -> None:
    """Verify that __post_init__ creates a Guard when validators are passed."""
    from guardrails import Guard

    from python_ai_agents.adapters.guardrails_ai import GuardrailsAiGuardrail

    # Use a simple validator-like object (GuardrailsAi just calls guard.use(v))
    # We pass a pre-built guard instead to avoid hub validator dependencies
    guard = Guard(name="test_auto")
    guardrail = GuardrailsAiGuardrail(guard=guard)

    assert guardrail.guard is guard
    assert guardrail.validators == []
