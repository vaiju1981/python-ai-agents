"""Built-in guardrails: keyword blocklist, PII scrub, and injection heuristic.

These implement the ``Guardrail`` protocol from ``core.guardrail`` and can be
passed to ``Trust.govern(agent, guardrails=[...])``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.guardrail import GuardrailDecision, GuardrailStage

__all__ = [
    "InjectionHeuristicGuardrail",
    "KeywordBlocklistGuardrail",
    "PiiScrubGuardrail",
]


# ---------------------------------------------------------------------------
# Keyword blocklist
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeywordBlocklistGuardrail:
    """Blocks content containing any keyword from a blocklist."""

    keywords: set[str] = field(default_factory=set)
    case_sensitive: bool = False
    stage: GuardrailStage | None = None  # None = check both stages

    async def check(
        self,
        stage: GuardrailStage,
        content: str,
        context: RequestContext,
    ) -> GuardrailDecision:
        if self.stage is not None and stage != self.stage:
            return GuardrailDecision.allow(content)
        if not self.keywords:
            return GuardrailDecision.allow(content)
        haystack = content if self.case_sensitive else content.lower()
        for keyword in self.keywords:
            needle = keyword if self.case_sensitive else keyword.lower()
            if needle in haystack:
                return GuardrailDecision.block(content, f"blocked keyword: {keyword}")
        return GuardrailDecision.allow(content)


# ---------------------------------------------------------------------------
# PII scrub
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass(frozen=True, slots=True)
class PiiScrubGuardrail:
    """Scrubs PII patterns from content using hand-rolled regex.

    This is a zero-dependency fallback.  For production use, prefer
    ``PresidioScrubGuardrail`` from ``adapters.presidio_guardrails``, which
    uses NER-based detection via Microsoft Presidio.

    Unlike blocklist guardrails, this guardrail does not block — it transforms
    the content by removing sensitive data.
    """

    scrub_email: bool = True
    scrub_phone: bool = True
    scrub_ssn: bool = True
    scrub_credit_card: bool = True
    scrub_ip: bool = False
    replacement: str = "[REDACTED]"
    stage: GuardrailStage | None = None

    async def check(
        self,
        stage: GuardrailStage,
        content: str,
        context: RequestContext,
    ) -> GuardrailDecision:
        if self.stage is not None and stage != self.stage:
            return GuardrailDecision.allow(content)
        scrubbed = content
        if self.scrub_email:
            scrubbed = _EMAIL_RE.sub(self.replacement, scrubbed)
        if self.scrub_phone:
            scrubbed = _PHONE_RE.sub(self.replacement, scrubbed)
        if self.scrub_ssn:
            scrubbed = _SSN_RE.sub(self.replacement, scrubbed)
        if self.scrub_credit_card:
            scrubbed = _CREDIT_CARD_RE.sub(self.replacement, scrubbed)
        if self.scrub_ip:
            scrubbed = _IP_RE.sub(self.replacement, scrubbed)
        return GuardrailDecision.allow(scrubbed)


# ---------------------------------------------------------------------------
# Prompt-injection heuristic
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|prompts?)",
        re.IGNORECASE,
    ),
    re.compile(r"disregard\s+(?:all\s+)?(?:the\s+)?(?:above|previous|prior)", re.IGNORECASE),
    re.compile(r"forget\s+(?:everything|all\s+(?:previous|prior))", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an)\s+", re.IGNORECASE),
    re.compile(
        r"(?:show|reveal|print|output)\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?prompt",
        re.IGNORECASE,
    ),
    re.compile(r"(?:new\s+instructions?|override)\s*:", re.IGNORECASE),
    re.compile(r"</\s*(?:system|instructions?|prompt)\s*>", re.IGNORECASE),
    re.compile(r"<\s*(?:system|instructions?|prompt)\s*>", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class InjectionHeuristicGuardrail:
    """Heuristic prompt-injection detector using regex patterns.

    This is a zero-dependency fallback.  For production use, prefer
    ``GuardrailsAiGuardrail`` from ``adapters.guardrails_ai``, which wraps
    the Guardrails AI hub validators for ML-based injection detection.

    Flags common injection patterns such as "ignore previous instructions",
    role-hijacking ("you are now..."), and system-prompt extraction attempts.
    Uses a simple pattern-match count threshold to reduce false positives.
    """

    stage: GuardrailStage = GuardrailStage.INPUT
    threshold: int = 1

    async def check(
        self,
        stage: GuardrailStage,
        content: str,
        context: RequestContext,
    ) -> GuardrailDecision:
        if stage != self.stage:
            return GuardrailDecision.allow(content)
        matches = sum(1 for pattern in _INJECTION_PATTERNS if pattern.search(content))
        if matches >= self.threshold:
            return GuardrailDecision.block(
                content,
                f"potential prompt injection detected ({matches} pattern match(es))",
            )
        return GuardrailDecision.allow(content)
