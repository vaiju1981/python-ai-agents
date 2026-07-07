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
from demos.analytics.src.analytics.model_store import InMemoryModelStore
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


def test_build_model_caches_then_retrains_on_request(tmp_path):
    csv = tmp_path / "s.csv"
    lines = ["spend,revenue"]
    rng = np.random.default_rng(1)
    for i in range(40):
        spend = 100 + i * 2 + rng.normal(0, 3)
        lines.append(f"{round(spend, 2)},{round(2 * spend + rng.normal(0, 5), 2)}")
    csv.write_text("\n".join(lines) + "\n")

    src = CsvSource(named_csvs={"s": csv})
    semantic = SemanticModel.from_profile(profile_dataset(src))
    tools = ModelsToolset(src, semantic, store=InMemoryModelStore(), dataset_sig="sig-1")

    first = _call(tools.build_model(), {"target": "revenue"})
    second = _call(tools.build_model(), {"target": "revenue"})
    forced = _call(tools.build_model(), {"target": "revenue", "retrain": True})
    src.close()

    assert first["cached"] is False  # trained
    assert second["cached"] is True  # served from the store, not retrained
    assert forced["cached"] is False  # explicit retrain bypasses the cache


def test_predict_serves_stored_model_without_retraining(tmp_path):
    csv = tmp_path / "s.csv"
    rng = np.random.default_rng(3)
    lines = ["spend,revenue"]
    for i in range(60):
        spend = 100 + i * 2 + rng.normal(0, 3)
        lines.append(f"{round(spend, 2)},{round(2 * spend + rng.normal(0, 5), 2)}")
    csv.write_text("\n".join(lines) + "\n")

    src = CsvSource(named_csvs={"s": csv})
    semantic = SemanticModel.from_profile(profile_dataset(src))
    tools = ModelsToolset(src, semantic, store=InMemoryModelStore(), dataset_sig="sig-p")

    first = _call(tools.predict(), {"target": "revenue"})
    second = _call(tools.predict(), {"target": "revenue"})
    src.close()

    assert first["model_cached"] is False  # trained once on first use
    assert second["model_cached"] is True  # served from the store afterwards
    assert first["task"] == "regression"
    assert first["n_scored"] == 60
    # predictions land in the plausible range of revenue ≈ 2*spend
    assert 150 < first["prediction"]["mean"] < 700
    assert first["drift"]["checked"] is True
    assert first["drift"]["detected"] is False  # scored on the training distribution


def test_predict_filters_rows_and_flags_drift(tmp_path):
    csv = tmp_path / "s.csv"
    rng = np.random.default_rng(4)
    lines = ["x,y"]
    for i in range(500):
        lines.append(f"{i},{round(2 * i + rng.normal(0, 1), 2)}")
    csv.write_text("\n".join(lines) + "\n")

    src = CsvSource(named_csvs={"s": csv})
    semantic = SemanticModel.from_profile(profile_dataset(src))
    tools = ModelsToolset(src, semantic, store=InMemoryModelStore(), dataset_sig="sig-d")

    out = _call(
        tools.predict(),
        {
            "target": "y",
            "predictors": ["x"],
            "filters": [{"column": "x", "op": ">", "value": "400"}],
        },
    )
    src.close()

    assert out["n_scored"] == 99  # filter applied: x in 401..499
    # trained on x∈[0,499] (mean≈250), scored on x>400 (mean≈450) → drift must flag
    assert out["drift"]["detected"] is True
    assert out["drift"]["worst_feature"] == "x"
    assert "retrain" in out["drift"]["recommendation"]


def test_predict_classification_returns_label_distribution(toolset):
    out = _call(toolset.predict(), {"target": "converted"})
    assert out["task"] == "classification"
    dist = out["prediction"]["class_distribution"]
    assert dist and set(dist) <= {"0", "1"}  # decoded labels, not raw codes


def test_training_uses_full_table_by_default_and_honors_user_cap(tmp_path):
    csv = tmp_path / "s.csv"
    rng = np.random.default_rng(2)
    lines = ["x,y"]
    for i in range(500):
        lines.append(f"{i},{round(2 * i + rng.normal(0, 1), 2)}")
    csv.write_text("\n".join(lines) + "\n")

    src = CsvSource(named_csvs={"s": csv})
    semantic = SemanticModel.from_profile(profile_dataset(src))

    full = _call(ModelsToolset(src, semantic).build_model(), {"target": "y", "predictors": ["x"]})
    capped = _call(
        ModelsToolset(src, semantic, max_train_rows=50).build_model(),
        {"target": "y", "predictors": ["x"]},
    )
    src.close()

    assert full["n_rows"] == 500  # default: the whole table, no artificial cap
    assert capped["n_rows"] == 50  # user's cap is honored, and reported
