"""Tests for the eval harness."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    ContainsScorer,
    EvalCase,
    EvalRunner,
    ExactMatchScorer,
    RequestContext,
)


class ScriptedAgent:
    """Returns canned responses indexed by call order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.calls.append(request.input)
        text = self.responses.pop(0) if self.responses else "fallback"
        return AgentResponse.completed(text)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def test_exact_match_scorer_pass() -> None:
    scorer = ExactMatchScorer()
    case = EvalCase(input="hi", expected="hello")
    passed, detail = scorer.score(case, AgentResponse.completed("hello"))

    assert passed
    assert "exact match" in detail


def test_exact_match_scorer_fail() -> None:
    scorer = ExactMatchScorer()
    case = EvalCase(input="hi", expected="hello")
    passed, detail = scorer.score(case, AgentResponse.completed("bye"))

    assert not passed
    assert "mismatch" in detail.lower() or "expected" in detail.lower()


def test_exact_match_scorer_case_insensitive() -> None:
    scorer = ExactMatchScorer(case_sensitive=False)
    case = EvalCase(input="hi", expected="Hello")
    passed, _ = scorer.score(case, AgentResponse.completed("hello"))

    assert passed


def test_exact_match_scorer_no_expected() -> None:
    scorer = ExactMatchScorer()
    case = EvalCase(input="hi")
    passed, _ = scorer.score(case, AgentResponse.completed("anything"))

    assert passed


def test_contains_scorer_pass() -> None:
    scorer = ContainsScorer()
    case = EvalCase(input="hi", expected="world")
    passed, _ = scorer.score(case, AgentResponse.completed("hello world bye"))

    assert passed


def test_contains_scorer_fail() -> None:
    scorer = ContainsScorer()
    case = EvalCase(input="hi", expected="world")
    passed, _ = scorer.score(case, AgentResponse.completed("hello bye"))

    assert not passed


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


def test_eval_runner_runs_cases() -> None:
    async def run() -> None:
        agent = ScriptedAgent(["hello", "world"])
        runner = EvalRunner(agent, scorer=ExactMatchScorer())
        cases = [
            EvalCase(input="greet", expected="hello"),
            EvalCase(input="topic", expected="world"),
        ]

        results = await runner.run(cases)

        assert len(results) == 2
        assert results[0].passed
        assert results[0].output == "hello"
        assert results[1].passed
        assert results[1].output == "world"
        assert agent.calls == ["greet", "topic"]

    anyio.run(run)


def test_eval_runner_detects_failures() -> None:
    async def run() -> None:
        agent = ScriptedAgent(["wrong", "correct"])
        runner = EvalRunner(agent, scorer=ExactMatchScorer())
        cases = [
            EvalCase(input="q1", expected="right"),
            EvalCase(input="q2", expected="correct"),
        ]

        results = await runner.run(cases)

        assert not results[0].passed
        assert results[1].passed

    anyio.run(run)


def test_eval_runner_with_contains_scorer() -> None:
    async def run() -> None:
        agent = ScriptedAgent(["The answer is 42 according to the guide"])
        runner = EvalRunner(agent, scorer=ContainsScorer())
        cases = [EvalCase(input="what", expected="42")]

        results = await runner.run(cases)

        assert results[0].passed

    anyio.run(run)


def test_eval_runner_summarize() -> None:
    async def run() -> None:
        agent = ScriptedAgent(["ok", "wrong", "ok"])
        runner = EvalRunner(agent, scorer=ExactMatchScorer())
        cases = [
            EvalCase(input="1", expected="ok"),
            EvalCase(input="2", expected="ok"),
            EvalCase(input="3", expected="ok"),
        ]

        results = await runner.run(cases)
        summary = EvalRunner.summarize(results)

        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 2 / 3

    anyio.run(run)


def test_eval_runner_with_custom_context() -> None:
    async def run() -> None:
        agent = ScriptedAgent(["response"])
        ctx = RequestContext(session_id="eval-session", principal="evaluator")
        runner = EvalRunner(agent, scorer=ExactMatchScorer())
        cases = [EvalCase(input="test", expected="response")]

        results = await runner.run(cases, context=ctx)

        assert results[0].passed

    anyio.run(run)


def test_eval_case_metadata() -> None:
    case = EvalCase(input="test", expected="out", metadata={"category": "math"})

    assert case.metadata["category"] == "math"
    assert case.input == "test"
    assert case.expected == "out"
