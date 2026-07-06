"""Tests for the DeepEval-backed eval scoring adapter.

Skips if ``deepeval`` is not installed.
"""

from __future__ import annotations

import importlib

import pytest

deepeval_available = importlib.util.find_spec("deepeval") is not None

pytestmark = pytest.mark.skipif(not deepeval_available, reason="deepeval not installed")


def test_deepeval_scorer_scores_response() -> None:
    from python_ai_agents import AgentResponse, EvalCase
    from python_ai_agents.adapters.deepeval import DeepEvalScorer

    class StubMetric:
        def __init__(self) -> None:
            self.score = 0.8
            self.reason = "outputs match"

        def measure(self, test_case) -> None:
            pass

    scorer = DeepEvalScorer(metric=StubMetric(), threshold=0.5)
    case = EvalCase(input="What is 2+2?", expected="4")
    response = AgentResponse.completed("4")

    passed, detail = scorer.score(case, response)

    assert passed
    assert "score=0.80" in detail


def test_deepeval_scorer_fails_below_threshold() -> None:
    from python_ai_agents import AgentResponse, EvalCase
    from python_ai_agents.adapters.deepeval import DeepEvalScorer

    class StubMetric:
        def __init__(self) -> None:
            self.score = 0.3
            self.reason = "outputs don't match"

        def measure(self, test_case) -> None:
            pass

    scorer = DeepEvalScorer(metric=StubMetric(), threshold=0.5)
    case = EvalCase(input="What is 2+2?", expected="4")
    response = AgentResponse.completed("5")

    passed, detail = scorer.score(case, response)

    assert not passed
    assert "score=0.30" in detail


def test_deepeval_scorer_detail_includes_reason() -> None:
    from python_ai_agents import AgentResponse, EvalCase
    from python_ai_agents.adapters.deepeval import DeepEvalScorer

    class StubMetric:
        def __init__(self) -> None:
            self.score = 0.9
            self.reason = "perfect match"

        def measure(self, test_case) -> None:
            pass

    scorer = DeepEvalScorer(metric=StubMetric(), threshold=0.5)
    case = EvalCase(input="test", expected="result")
    response = AgentResponse.completed("result")

    _, detail = scorer.score(case, response)

    assert "reason=perfect match" in detail


def test_deepeval_scorer_with_eval_runner() -> None:
    import anyio

    from python_ai_agents import AgentRequest, AgentResponse, EvalCase, EvalRunner
    from python_ai_agents.adapters.deepeval import DeepEvalScorer

    class StubMetric:
        def __init__(self) -> None:
            self.score = 0.85
            self.reason = "good"

        def measure(self, test_case) -> None:
            pass

    class StubAgent:
        async def run(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse.completed("the answer is 42")

    async def run() -> None:
        scorer = DeepEvalScorer(metric=StubMetric(), threshold=0.5)
        runner = EvalRunner(StubAgent(), scorer=scorer)
        cases = [EvalCase(input="what", expected="42")]
        results = await runner.run(cases)

        assert len(results) == 1
        assert results[0].passed

    anyio.run(run)
