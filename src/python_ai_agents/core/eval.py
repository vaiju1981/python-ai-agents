"""Lightweight eval harness for agent responses.

Provides ``EvalCase``, ``EvalResult``, pluggable ``Scorer`` protocols, and an
``EvalRunner`` that exercises an ``Agent`` against a list of cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import anyio

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.context import RequestContext

__all__ = [
    "ContainsScorer",
    "EvalCase",
    "EvalResult",
    "EvalRunner",
    "ExactMatchScorer",
    "Scorer",
]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A single evaluation case."""

    input: str
    expected: str | None = None
    criteria: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result of running one eval case."""

    case: EvalCase
    passed: bool
    output: str
    detail: str = ""


class Scorer(Protocol):
    """Scores an agent response against an eval case."""

    def score(self, case: EvalCase, response: AgentResponse) -> tuple[bool, str]: ...


@dataclass(frozen=True, slots=True)
class ExactMatchScorer:
    """Passes when the response output exactly matches the expected string.

    Zero-dependency fallback scorer.  For LLM-as-judge evaluation, use
    ``DeepEvalScorer`` from ``adapters.deepeval``.
    """

    case_sensitive: bool = False

    def score(self, case: EvalCase, response: AgentResponse) -> tuple[bool, str]:
        if case.expected is None:
            return True, "no expected output specified"
        expected = case.expected if self.case_sensitive else case.expected.strip().lower()
        actual = response.output if self.case_sensitive else response.output.strip().lower()
        if expected == actual:
            return True, "exact match"
        return False, f"expected {expected!r}, got {actual!r}"


@dataclass(frozen=True, slots=True)
class ContainsScorer:
    """Passes when the expected string appears anywhere in the response."""

    case_sensitive: bool = True

    def score(self, case: EvalCase, response: AgentResponse) -> tuple[bool, str]:
        if case.expected is None:
            return True, "no expected output specified"
        haystack = response.output if self.case_sensitive else response.output.lower()
        needle = case.expected if self.case_sensitive else case.expected.lower()
        if needle in haystack:
            return True, "contains expected substring"
        return False, f"response does not contain {needle!r}"


@dataclass(slots=True)
class EvalRunner:
    """Runs an ``Agent`` against a list of ``EvalCase``s and collects results."""

    agent: Agent
    scorer: Scorer = field(default_factory=ExactMatchScorer)

    async def run(
        self,
        cases: list[EvalCase],
        context: RequestContext | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        for case in cases:
            ctx = context or RequestContext.ephemeral()
            response = await self.agent.run(AgentRequest(input=case.input, context=ctx))
            # Offload scoring: a real scorer (e.g. DeepEval LLM-as-judge) does blocking
            # I/O and would otherwise stall the event loop.
            passed, detail = await anyio.to_thread.run_sync(self.scorer.score, case, response)
            results.append(
                EvalResult(
                    case=case,
                    passed=passed,
                    output=response.output,
                    detail=detail,
                )
            )
        return results

    @staticmethod
    def summarize(results: list[EvalResult]) -> dict[str, float | int]:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0.0,
        }
