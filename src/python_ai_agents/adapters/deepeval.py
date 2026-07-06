"""DeepEval-backed eval scoring adapter.

Wraps the DeepEval library (``deepeval``) to score agent responses using
LLM-as-judge metrics, semantic similarity, and other evaluation strategies
that go beyond simple string matching.

Install::

    pip install python-ai-agents[deepeval]

Usage::

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    from python_ai_agents.adapters.deepeval import DeepEvalScorer

    metric = GEval(
        name="correctness",
        criteria="determine whether the actual output matches the expected output",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
    )
    scorer = DeepEvalScorer(metric=metric, threshold=0.5)
    runner = EvalRunner(my_agent, scorer=scorer)
    results = await runner.run(cases)

This is the production-grade alternative to the zero-dependency
``ExactMatchScorer`` and ``ContainsScorer`` in ``core.eval``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from python_ai_agents.core.agent import AgentResponse
from python_ai_agents.core.eval import EvalCase

__all__ = ["DeepEvalScorer"]


@dataclass(slots=True)
class DeepEvalScorer:
    """Eval scorer backed by DeepEval metrics.

    Wraps a DeepEval metric (e.g. ``GEval``, ``AnswerRelevancyMetric``) in
    our ``Scorer`` protocol.  Each eval case is converted to a
    ``LLMTestCase``, measured by the metric, and scored against a
    configurable threshold.

    Attributes:
        metric: A DeepEval metric instance.  If ``None``, a default
            ``GEval`` correctness metric is created (requires an LLM
            endpoint configured for DeepEval).
        threshold: Minimum score (0–1) to pass.  Defaults to ``0.5``.
    """

    metric: Any = field(default=None)
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.metric is None:
            self.metric = self._default_metric()

    def score(self, case: EvalCase, response: AgentResponse) -> tuple[bool, str]:
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=case.input,
            actual_output=response.output,
            expected_output=case.expected or "",
        )
        self.metric.measure(test_case)
        raw_score = float(self.metric.score)
        passed = raw_score >= self.threshold
        reason = getattr(self.metric, "reason", "") or ""
        detail = f"deepeval score={raw_score:.2f} threshold={self.threshold}"
        if reason:
            detail += f" reason={reason}"
        return passed, detail

    @staticmethod
    def _default_metric() -> Any:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        return GEval(
            name="correctness",
            criteria=(
                "determine whether the actual output is factually correct "
                "and matches the expected output"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )
