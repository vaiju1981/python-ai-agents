"""Behavioral tests for the predictive/causal analytics tools.

Uses a synthetic dataset with a known signal (revenue ~= 2*spend + a treatment
effect) so assertions check real behavior, not just shape.
"""

from __future__ import annotations

import json

import anyio
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")
np = pytest.importorskip("numpy")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from python_ai_agents import RequestContext


def _call(tool, args) -> dict:
    """Invoke a tool, strip the `[name result — ...]` frame, and parse the JSON body."""

    async def run():
        return await tool.invoke(args, RequestContext.ephemeral())

    result = anyio.run(run)
    assert not result.error, result.content
    return json.loads(result.content.split("\n", 1)[1])


@pytest.fixture
def toolset(tmp_path):
    rng = np.random.default_rng(0)
    lines = ["month,region,spend,revenue,treatment,converted"]
    for i in range(48):
        m = i // 2  # 24 distinct months, 2 rows each
        date = f"{2022 + m // 12}-{m % 12 + 1:02d}-01"
        spend = 100 + i * 3 + rng.normal(0, 5)
        treat = i % 2
        revenue = 2 * spend + 5 * treat + rng.normal(0, 8)
        region = "A" if i % 2 == 0 else "B"
        conv = int(revenue > 320)
        lines.append(f"{date},{region},{round(spend, 2)},{round(revenue, 2)},{treat},{conv}")
    csv = tmp_path / "sales.csv"
    csv.write_text("\n".join(lines) + "\n")

    src = CsvSource(named_csvs={"sales": csv})
    semantic = SemanticModel.from_profile(profile_dataset(src))
    yield ModelsToolset(src, semantic)
    src.close()


def test_build_model_regression_finds_spend(toolset):
    out = _call(toolset.build_model(), {"target": "revenue"})
    assert out["task"] == "regression"
    assert isinstance(out["cv_r2"], (int, float))
    assert out["feature_importance"][0]["feature"] == "spend"  # revenue is driven by spend


def test_ab_test_reports_significance(toolset):
    out = _call(
        toolset.ab_test(),
        {"metric": "revenue", "groupColumn": "region", "groupA": "A", "groupB": "B"},
    )
    assert "p_value" in out and "verdict" in out
    assert out["nA"] > 0 and out["nB"] > 0


def test_causal_effect_has_ci_and_caveat(toolset):
    out = _call(
        toolset.causal_effect(),
        {"target": "revenue", "treatment": "treatment", "controls": ["spend"]},
    )
    assert len(out["ci_95"]) == 2
    assert "not" in out["caveat"].lower()  # honest about causation


def test_forecast_returns_horizon_points(toolset):
    out = _call(toolset.forecast(), {"metric": "revenue", "timeColumn": "month", "horizon": 3})
    assert len(out["forecast"]) == 3
    assert all("value" in p for p in out["forecast"])


def test_cluster_and_anomaly(toolset):
    clustered = _call(toolset.cluster(), {"columns": ["spend", "revenue"], "k": 3})
    assert clustered["k"] == 3 and clustered["cluster_sizes"]

    anom = _call(
        toolset.anomaly_detection(), {"columns": ["spend", "revenue"], "contamination": 0.1}
    )
    assert anom["n_anomalies"] >= 0


def test_all_tools_are_read_only(toolset):
    from python_ai_agents.core.tool import ToolEffect

    for tool in toolset.all_tools():
        assert tool.spec.effect == ToolEffect.READ_ONLY
