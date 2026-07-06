"""Guardrails AI-backed content validation guardrail.

Wraps the Guardrails AI library (``guardrails-ai``) to validate agent input
and output against hub validators.  For prompt-injection detection, install
a relevant hub validator and pass it to this adapter.

Install::

    pip install python-ai-agents[guardrails-ai]
    guardrails hub install hub://guardrails/regex_match  # example

Usage::

    from guardrails.hub import RegexMatch
    from python_ai_agents.adapters.guardrails_ai import GuardrailsAiGuardrail

    injection_validator = RegexMatch(
        regex=r"ignore\\s+(?:previous|prior)\\s+instructions",
        on_fail="exception",
    )
    guardrail = GuardrailsAiGuardrail(validators=[injection_validator])
    agent = Trust.govern(my_agent, guardrails=[guardrail])

This is the production-grade alternative to the zero-dependency
``InjectionHeuristicGuardrail`` in ``core.guardrails``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.guardrail import GuardrailDecision, GuardrailStage

__all__ = ["GuardrailsAiGuardrail"]


@dataclass(slots=True)
class GuardrailsAiGuardrail:
    """Content-validation guardrail backed by Guardrails AI.

    Wraps a ``guardrails.Guard`` with one or more hub validators.  When
    validation fails, the guardrail blocks the content.  When validation
    passes, the content is allowed through unchanged.

    Attributes:
        validators: List of guardrails-ai validator instances.  Each
            validator should be configured with ``on_fail="exception"``
            so that failures raise rather than silently pass.
        stage: Which guardrail stage(s) to apply on (``None`` = both).
        guard: Optional pre-configured ``Guard`` instance.  If provided,
            ``validators`` is ignored.
    """

    validators: list[Any] = field(default_factory=list)
    stage: GuardrailStage | None = None
    guard: Any = field(default=None)

    def __post_init__(self) -> None:
        if self.guard is None:
            from guardrails import Guard

            guard = Guard(name="content_validation")
            for validator in self.validators:
                guard = guard.use(validator)
            self.guard = guard

    async def check(
        self,
        stage: GuardrailStage,
        content: str,
        context: RequestContext,
    ) -> GuardrailDecision:
        if self.stage is not None and stage != self.stage:
            return GuardrailDecision.allow(content)

        try:
            self.guard.validate(content)
        except Exception as exc:
            return GuardrailDecision.block(
                content,
                f"guardrails-ai validation failed: {exc.__class__.__name__}",
            )
        return GuardrailDecision.allow(content)
