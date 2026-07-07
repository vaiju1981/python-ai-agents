"""Gated NL->tool accuracy eval for the analytics agent (needs a live Ollama model).

Wires the core ``EvalRunner`` over the demo agent so answer quality has a
regression guard. Opt in with ``PAA_RUN_OLLAMA_TESTS=1`` and a running Ollama.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("duckdb")
if not os.environ.get("PAA_RUN_OLLAMA_TESTS"):
    pytest.skip("set PAA_RUN_OLLAMA_TESTS=1 for the live analytics eval", allow_module_level=True)

import anyio

from demos.analytics.src.analytics.agent import create_agent
from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.models import from_env as model_from_env
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from python_ai_agents import ContainsScorer, EvalCase, EvalRunner


def test_analytics_nl_to_tool_eval(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "date,region,amount\n"
        "2024-01-01,North,100\n"
        "2024-01-02,South,90\n"
        "2024-01-03,North,120\n"
        "2024-01-04,South,80\n"
    )
    source = CsvSource(named_csvs={"sales": csv})
    semantic = SemanticModel.from_profile(profile_dataset(source))
    agent = create_agent(source, model_from_env(), semantic)

    cases = [
        EvalCase(input="What is the total amount across all rows?", expected="390"),
        EvalCase(input="Which region has the higher total amount?", expected="North"),
    ]
    results = anyio.run(EvalRunner(agent, scorer=ContainsScorer()).run, cases)
    source.close()

    summary = EvalRunner.summarize(results)
    assert summary["pass_rate"] >= 0.5, [(r.output, r.detail) for r in results]
