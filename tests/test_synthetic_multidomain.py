"""Generic multi-domain test suite: proves the analytics engine is domain-agnostic.

Loads the synthetic data from each domain (casino, ecommerce, health, education,
real estate), discovers relationships with the *real* profiler, and verifies that:

  * run_query joins across tables using the discovered relationships (no domain
    knowledge required),
  * build_model/predict assemble multi-table features (predictors from related
    tables),
  * event_impact computes an event-anchored before/after comparison generically.

Because every generator shares the same engine contract (star/snowflake tables
linked by shared keys), the same assertions hold across all five domains.
"""

from __future__ import annotations

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from pathlib import Path

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.query_planner import QuerySpec, plan
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.synthetic_data import GENERATORS
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents import RequestContext

# (metric from one table, dimension from a related table, the join key that must appear)
JOIN_SPECS: dict[str, tuple[str, str, str]] = {
    "casino": ("sessions.coinIn", "players.tier", "playerId"),
    "ecommerce": ("orders.amount", "customers.segment", "customerId"),
    "health": ("visits.cost", "patients.region", "patientId"),
    "education": ("enrollments.grade", "students.school", "studentId"),
    "realestate": ("listings.sqft", "neighborhoods.city", "neighborhoodId"),
}

# (target, predictor from a *different* table) — exercises multi-table assembly
MODEL_SPECS: dict[str, tuple[str, str]] = {
    "casino": ("sessions.coinIn", "players.tenureDays"),
    "ecommerce": ("orders.amount", "products.price"),
    "health": ("visits.cost", "patients.age"),
    "education": ("enrollments.grade", "students.cohortYear"),
    "realestate": ("listings.price", "neighborhoods.medianIncome"),
}

DOMAINS = list(GENERATORS)


def _load_domain(domain: str, tmp_path: Path) -> tuple[CsvSource, SemanticModel]:
    """Generate one domain's CSVs, import them, and build a real semantic model."""
    out = tmp_path / domain
    paths = GENERATORS[domain](out)
    source = CsvSource(named_csvs={p.stem: p for p in paths.values()})
    model = SemanticModel.from_profile(profile_dataset(source))
    return source, model


def _has_relationship(
    model: SemanticModel, from_table: str, from_col: str, to_table: str, to_col: str
) -> bool:
    for r in model.relationships:
        if (
            r.from_table == from_table
            and r.from_columns
            and r.from_columns[0] == from_col
            and r.to_table == to_table
            and r.to_columns
            and r.to_columns[0] == to_col
        ):
            return True
        if (
            r.to_table == from_table
            and r.to_columns
            and r.to_columns[0] == from_col
            and r.from_table == to_table
            and r.from_columns
            and r.from_columns[0] == to_col
        ):
            return True
    return False


def _json(result: object) -> dict:
    """Extract the JSON payload from a framed ToolResult (modeling tools put it
    in ``content`` rather than ``data``)."""
    assert not result.error, result.content  # type: ignore[attr-defined]
    content = result.content  # type: ignore[attr-defined]
    body = content.split("\n", 1)[1] if "\n" in content else content
    import json as _json_mod

    return _json_mod.loads(body)


@pytest.mark.parametrize("domain", DOMAINS)
def test_relationships_discovered_across_tables(domain, tmp_path):
    source, model = _load_domain(domain, tmp_path)
    metric, dimension, key = JOIN_SPECS[domain]
    fact_table, fact_col = metric.split(".")
    dim_table = dimension.split(".")[0]

    # The join key that links the fact to the dimension must be discovered
    # automatically by the profiler — that is what makes run_query generic.
    try:
        assert _has_relationship(model, fact_table, key, dim_table, key), (
            f"expected {fact_table}.{key} ~ {dim_table}.{key} in {domain}"
        )
    finally:
        source.close()


@pytest.mark.parametrize("domain", DOMAINS)
def test_run_query_plans_cross_table_join(domain, tmp_path):
    source, model = _load_domain(domain, tmp_path)
    metric, dimension, key = JOIN_SPECS[domain]
    fact_table = metric.split(".")[0]
    dim_table = dimension.split(".")[0]

    try:
        sql = plan(
            model,
            QuerySpec(metrics=(metric,), dimensions=(dimension,), limit=20),
        )
        # Proof that run_query is relationship-aware: the planned SQL joins the
        # two tables on the discovered key rather than querying one table.
        assert "JOIN" in sql
        assert fact_table in sql and dim_table in sql
        assert key in sql
    finally:
        source.close()


@pytest.mark.parametrize("domain", DOMAINS)
def test_run_query_executes_cross_table_join(domain, tmp_path):
    source, model = _load_domain(domain, tmp_path)
    metric, dimension, _key = JOIN_SPECS[domain]
    dim_col = dimension.split(".")[-1]
    tools = AnalyticsToolset(source, model)

    async def run():
        result = await tools.run_query().invoke(
            {
                "metrics": [metric],
                "dimensions": [dimension],
                "limit": 50,
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        rows = result.data
        assert rows, "expected grouped rows from the cross-table join"
        # Each grouped row carries the dimension value.
        assert all(dim_col in r for r in rows)
        assert len({r.get(dim_col) for r in rows}) >= 1

    try:
        anyio.run(run)
    finally:
        source.close()


@pytest.mark.parametrize("domain", DOMAINS)
def test_build_model_assembles_multitable_features(domain, tmp_path):
    source, model = _load_domain(domain, tmp_path)
    target, predictor = MODEL_SPECS[domain]
    pred_table, pred_col = predictor.split(".")

    tools = ModelsToolset(source, model, dataset_sig=domain)

    async def run():
        result = await tools.build_model().invoke(
            {"target": target, "predictors": [predictor]},
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        meta = _json(result)
        assert meta["task"] in ("regression", "classification")
        # The predictor from the *related* table must be part of the model.
        features = {f["feature"] for f in meta.get("feature_importance", [])}
        assert pred_col in features, f"predictor {predictor} missing from {features}"

    try:
        anyio.run(run)
    finally:
        source.close()


@pytest.mark.parametrize("domain", DOMAINS)
def test_predict_serves_multitable_model(domain, tmp_path):
    source, model = _load_domain(domain, tmp_path)
    target, predictor = MODEL_SPECS[domain]

    tools = ModelsToolset(source, model, dataset_sig=domain)

    async def run():
        built = await tools.build_model().invoke(
            {"target": target, "predictors": [predictor]},
            RequestContext.ephemeral(),
        )
        assert not built.error, built.content
        # Serving reuses the trained model; no retraining per call.
        served = await tools.predict().invoke(
            {"target": target, "predictors": [predictor]},
            RequestContext.ephemeral(),
        )
        assert not served.error, served.content
        assert _json(served)["n_scored"] > 0

    try:
        anyio.run(run)
    finally:
        source.close()


def test_event_impact_before_after_casino(tmp_path):
    """Event-anchored before/after impact is generic: casino denom changes on
    assetDaily via the changeLog's assetId key."""
    source, model = _load_domain("casino", tmp_path)
    tools = AnalyticsToolset(source, model)

    async def run():
        result = await tools.event_impact().invoke(
            {
                "metric": "assetDaily.coinIn",
                "eventTable": "changeLog",
                "anchorKey": "assetId",
                "windowDays": 14,
                "eventFilter": "changeType='Denom Change'",
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        rows = result.data
        assert rows, "expected pre/post impact rows"
        row = rows[0]
        assert "pre" in row and "post" in row
        assert (row.get("n_pre") or 0) > 0
        assert (row.get("n_post") or 0) > 0

    try:
        anyio.run(run)
    finally:
        source.close()
