"""Presidio-backed PII scrubbing guardrail.

Wraps Microsoft Presidio (``presidio-analyzer`` + ``presidio-anonymizer``) to
detect and redact PII using NER-based recognisers rather than hand-rolled regex.

Install::

    pip install python-ai-agents[presidio]

Usage::

    from python_ai_agents.adapters.presidio_guardrails import PresidioScrubGuardrail

    guardrail = PresidioScrubGuardrail()
    agent = Trust.govern(my_agent, guardrails=[guardrail])

This is the production-grade alternative to the zero-dependency
``PiiScrubGuardrail`` in ``core.guardrails``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.guardrail import GuardrailDecision, GuardrailStage

__all__ = ["PresidioScrubGuardrail"]


@dataclass(slots=True)
class PresidioScrubGuardrail:
    """PII scrubbing guardrail backed by Presidio.

    Uses ``AnalyzerEngine`` for detection and ``AnonymizerEngine`` for
    redaction.  Configurable entity list, language, and replacement text.

    Attributes:
        language: Language for the analyzer (default ``"en"``).
        entities: Optional list of Presidio entity types to detect
            (e.g. ``["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"]``).
            ``None`` means all configured recognisers.
        replacement: Text to replace detected PII with.
        stage: Which guardrail stage(s) to apply on (``None`` = both).
        analyzer: Optional pre-configured ``AnalyzerEngine``.
        anonymizer: Optional pre-configured ``AnonymizerEngine``.
    """

    language: str = "en"
    entities: list[str] | None = None
    replacement: str = "[REDACTED]"
    stage: GuardrailStage | None = None
    analyzer: Any = field(default=None)
    anonymizer: Any = field(default=None)

    def __post_init__(self) -> None:
        if self.analyzer is None:
            from presidio_analyzer import AnalyzerEngine

            self.analyzer = AnalyzerEngine()
        if self.anonymizer is None:
            from presidio_anonymizer import AnonymizerEngine

            self.anonymizer = AnonymizerEngine()  # type: ignore[no-untyped-call]

    async def check(
        self,
        stage: GuardrailStage,
        content: str,
        context: RequestContext,
    ) -> GuardrailDecision:
        if self.stage is not None and stage != self.stage:
            return GuardrailDecision.allow(content)

        results = self.analyzer.analyze(
            text=content,
            language=self.language,
            entities=self.entities,
        )
        if not results:
            return GuardrailDecision.allow(content)

        from presidio_anonymizer.entities import OperatorConfig

        anonymized = self.anonymizer.anonymize(
            text=content,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": self.replacement}),
            },
        )
        return GuardrailDecision.allow(anonymized.text)
