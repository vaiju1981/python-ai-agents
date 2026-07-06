"""Tests for the Presidio-backed PII scrubbing guardrail.

Skips if ``presidio-analyzer`` / ``presidio-anonymizer`` are not installed.
"""

from __future__ import annotations

import importlib

import anyio
import pytest

presidio_available = (
    importlib.util.find_spec("presidio_analyzer") is not None
    and importlib.util.find_spec("presidio_anonymizer") is not None
)

pytestmark = pytest.mark.skipif(not presidio_available, reason="presidio not installed")


@pytest.fixture
def guardrail():
    from python_ai_agents.adapters.presidio_guardrails import PresidioScrubGuardrail

    return PresidioScrubGuardrail()


def test_presidio_scrubs_email(guardrail) -> None:
    from python_ai_agents import GuardrailStage, RequestContext

    async def run() -> None:
        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "Contact me at john.doe@example.com please",
            RequestContext.ephemeral(),
        )
        assert not decision.blocked
        assert "john.doe@example.com" not in decision.content
        assert "[REDACTED]" in decision.content

    anyio.run(run)


def test_presidio_scrubs_phone(guardrail) -> None:
    from python_ai_agents import GuardrailStage, RequestContext

    async def run() -> None:
        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "Call (555) 123-4567",
            RequestContext.ephemeral(),
        )
        assert "555" not in decision.content

    anyio.run(run)


def test_presidio_preserves_non_pii(guardrail) -> None:
    from python_ai_agents import GuardrailStage, RequestContext

    async def run() -> None:
        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "Hello world, this is a normal message.",
            RequestContext.ephemeral(),
        )
        assert decision.content == "Hello world, this is a normal message."

    anyio.run(run)


def test_presidio_custom_replacement() -> None:
    from python_ai_agents import GuardrailStage, RequestContext
    from python_ai_agents.adapters.presidio_guardrails import PresidioScrubGuardrail

    async def run() -> None:
        guardrail = PresidioScrubGuardrail(replacement="[PII]")
        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "Email: test@test.com",
            RequestContext.ephemeral(),
        )
        assert "[PII]" in decision.content

    anyio.run(run)


def test_presidio_entity_filter() -> None:
    from python_ai_agents import GuardrailStage, RequestContext
    from python_ai_agents.adapters.presidio_guardrails import PresidioScrubGuardrail

    async def run() -> None:
        guardrail = PresidioScrubGuardrail(entities=["EMAIL_ADDRESS"])
        # Phone should NOT be scrubbed when only EMAIL_ADDRESS is configured
        decision = await guardrail.check(
            GuardrailStage.INPUT,
            "Call (555) 123-4567",
            RequestContext.ephemeral(),
        )
        assert "555" in decision.content  # phone preserved

    anyio.run(run)


def test_presidio_through_governed_agent(guardrail) -> None:
    from python_ai_agents import AgentRequest, AgentResponse, Trust

    class EchoAgent:
        async def run(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse.completed(request.input)

    async def run() -> None:
        agent = Trust.govern(EchoAgent(), guardrails=[guardrail])
        response = await agent.run(AgentRequest.ephemeral("my email is alice@test.com"))
        assert "alice@test.com" not in response.output
        assert "[REDACTED]" in response.output

    anyio.run(run)
