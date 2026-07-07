"""Tests for ML and statistical analysis tools."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")
sklearn = pytest.importorskip("sklearn")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.ml_tools import MLToolset
from python_ai_agents import RequestContext, ToolEffect


@pytest.fixture
def ml_data(tmp_path):
    """Create a dataset suitable for ML: target, predictors, treatment, group."""
    import csv
    import random
    random.seed(42)
    csv_path = tmp_path / "ml_data.csv"
    rows = []
    for i in range(60):
        x1 = random.gauss(50, 15)
        x2 = random.gauss(30, 10)
        x3 = random.gauss(100, 25)
        treatment = 1 if i % 2 == 0 else 0
        group = "A" if x1 > 50 else "B"
        y = 10 + 0.5 * x1 + 0.3 * x2 + 5 * treatment + random.gauss(0, 2)
        rows.append({"y": round(y, 2), "x1": round(x1, 2), "x2": round(x2, 2),
                      "x3": round(x3, 2), "group": group, "treatment": treatment})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["y", "x1", "x2", "x3", "group", "treatment"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def ml_setup(ml_data):
    source = CsvSource(named_csvs={"ml_data": ml_data})
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    yield source, semantic
    source.close()


def test_regression_linear(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.regression().invoke(
            {"target": "ml_data.y", "method": "linear"}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "r_squared" in result.content

    anyio.run(run)


def test_regression_rf(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.regression().invoke(
            {"target": "ml_data.y", "method": "rf"}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "importance" in result.content

    anyio.run(run)


def test_classification(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.classification().invoke(
            {"target": "ml_data.group", "method": "rf"}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "accuracy" in result.content

    anyio.run(run)


def test_clustering(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.clustering().invoke(
            {"columns": ["x1", "x2"], "nClusters": 3}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "silhouette_score" in result.content

    anyio.run(run)


def test_causal_analysis(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.causal_analysis().invoke(
            {"target": "ml_data.y", "treatment": "ml_data.treatment",
             "method": "difference_in_means"}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "ate" in result.content

    anyio.run(run)


def test_uplift_modeling(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.uplift_modeling().invoke(
            {"target": "ml_data.y", "treatment": "ml_data.treatment",
             "method": "t_learner"}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "uplift_deciles" in result.content

    anyio.run(run)


def test_feature_importance(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.feature_importance().invoke(
            {"target": "ml_data.y", "method": "tree"}, RequestContext.ephemeral()
        )
        assert not result.error
        assert "features" in result.content

    anyio.run(run)


def test_statistical_test(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.statistical_test().invoke(
            {"test": "ttest", "column": "ml_data.y", "groupColumn": "ml_data.treatment"},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "p_value" in result.content

    anyio.run(run)


def test_anomaly_detection(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.anomaly_detection().invoke(
            {"columns": ["x1", "x2", "x3"], "method": "isolation_forest"},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "n_anomalies" in result.content

    anyio.run(run)


def test_cross_validate(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)

    async def run():
        result = await ml.cross_validate().invoke(
            {"target": "ml_data.y", "method": "linear", "cvFolds": 5},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "mean_score" in result.content

    anyio.run(run)


def test_all_ml_tools_are_read_only(ml_setup):
    source, semantic = ml_setup
    ml = MLToolset(source, semantic)
    for tool in ml.all_tools():
        assert tool.spec.effect == ToolEffect.READ_ONLY
        assert tool.spec.input_schema["type"] == "object"
        assert "properties" in tool.spec.input_schema


def test_all_ml_tool_names():
    """Verify we have 10 ML tools."""
    from demos.analytics.src.analytics.csv_source import CsvSource
    from demos.analytics.src.analytics.profiler import profile_dataset
    from demos.analytics.src.analytics.semantic_model import SemanticModel
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    csv = tmp / "t.csv"
    csv.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14\n15,16\n17,18\n19,20\n")
    source = CsvSource(named_csvs={"t": csv})
    semantic = SemanticModel.from_profile(profile_dataset(source))
    ml = MLToolset(source, semantic)
    tools = ml.all_tools()
    assert len(tools) == 10
    names = {t.spec.name for t in tools}
    assert names == {
        "ml_regression", "classification", "clustering", "forecast",
        "causal_analysis", "uplift_modeling", "feature_importance",
        "statistical_test", "anomaly_detection", "cross_validate",
    }
    source.close()
