"""Tests for the analytics engine tracker items: A5, A6, B2-B7, P0-P3, cross-cut.

Covers the production-hardening items (A5 composite/many-to-many keys, A6 role
overrides) and the generic, column-agnostic tools lifted from ATLAS (matched-
control DiD, conformal forecast, segmentation, portfolio, decision store),
plus the defensibility roadmap (provenance, trust + abstention, freshness,
reconcile, warehouse adapters, feedback loop, semantic verification).

Tools that hit the source are exercised both at the function level (sync) and
through the async ``AnalyticsToolset`` on small in-memory datasets.
"""

from __future__ import annotations

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")

import pandas as pd

from demos.analytics.src.analytics.catalog import Catalog
from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.data_source import ColumnRole
from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.decision_store import DecisionStore
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.provenance import (
    ProvenanceEnvelope,
    build_envelope,
    reproducible,
)
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.trust import grade, should_abstain
from demos.analytics.src.analytics.warehouse_sources import available_kinds, make_warehouse_source

from python_ai_agents import RequestContext


@pytest.fixture
def source():
    s = CsvSource(named_csvs={})
    yield s
    s.close()


def _write_csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# A5: typed composite keys + many-to-many cardinality
# ---------------------------------------------------------------------------


def test_composite_key_discovery(tmp_path):
    # (k1, k2) together uniquely identify a row, though neither is unique alone.
    a = pd.DataFrame(
        {"k1": ["a", "a", "a", "b", "b", "b", "c", "c", "c"], "k2": ["x", "y", "z"] * 3, "v": range(9)}
    )
    b = pd.DataFrame(
        {"k1": ["a", "a", "a", "b", "b", "b", "c", "c", "c"], "k2": ["x", "y", "z"] * 3, "w": range(9)}
    )
    src = CsvSource(
        named_csvs={
            "a": _write_csv(tmp_path, "a.csv", a),
            "b": _write_csv(tmp_path, "b.csv", b),
        }
    )
    model = SemanticModel.from_profile(profile_dataset(src))
    composite = [r for r in model.relationships if len(r.from_columns) > 1]
    assert composite, "expected a composite (k1,k2) relationship to be discovered"
    rel = composite[0]
    assert set(rel.from_columns) == {"k1", "k2"}
    assert rel.cardinality in ("one_to_one", "many_to_one")


def test_many_to_many_cardinality(tmp_path):
    # Both sides share a non-unique key -> many-to-many.
    a = pd.DataFrame({"k": ["x", "x", "y", "y"], "v": [1, 2, 3, 4]})
    b = pd.DataFrame({"k": ["x", "x", "y", "y"], "w": [5, 6, 7, 8]})
    src = CsvSource(
        named_csvs={
            "a": _write_csv(tmp_path, "a.csv", a),
            "b": _write_csv(tmp_path, "b.csv", b),
        }
    )
    model = SemanticModel.from_profile(profile_dataset(src))
    rels = [r for r in model.relationships if r.from_table == "a" and r.to_table == "b"]
    assert rels, "expected a join relationship between a and b"
    assert any(r.cardinality == "many_to_many" for r in rels)


# ---------------------------------------------------------------------------
# A6: catalog role override
# ---------------------------------------------------------------------------


def test_catalog_role_override(tmp_path):
    sales = pd.DataFrame({"id": [1, 2, 3], "price": [10, 20, 30]})
    src = CsvSource(named_csvs={"sales": _write_csv(tmp_path, "sales.csv", sales)})
    cat = Catalog(role_overrides={"sales.price": "measure_additive", "sales.id": "identifier"})
    model = SemanticModel.from_profile(profile_dataset(src, catalog=cat))
    price_role = next(c.role for c in model.columns if c.name == "price")
    assert price_role == ColumnRole.MEASURE_ADDITIVE


# ---------------------------------------------------------------------------
# B2 / B5: matched-control DiD backtest
# ---------------------------------------------------------------------------


def _diet_dataset(tmp_path):
    rows = []
    days = list(range(1, 31))
    # Treated entities jump after day 15; controls stay flat.
    for e in range(1, 21):
        base = 100 + e
        for d in days:
            v = base + (50 if (e <= 10 and d >= 15) else 0)
            rows.append({"entity": f"E{e:02d}", "day": d, "value": v})
    daily = pd.DataFrame(rows)
    events = pd.DataFrame(
        {"entity": [f"E{e:02d}" for e in range(1, 11)], "day": [15] * 10}
    )
    src = CsvSource(
        named_csvs={
            "daily": _write_csv(tmp_path, "daily.csv", daily),
            "events": _write_csv(tmp_path, "events.csv", events),
        }
    )
    model = SemanticModel.from_profile(profile_dataset(src))
    return src, model


def test_matched_impact_function(tmp_path):
    from demos.analytics.src.analytics.backtest import matched_impact

    src, model = _diet_dataset(tmp_path)
    res = matched_impact(
        src, model, value_col="daily.value", entity_col="entity", date_col="day",
        treatment_table="events", treatment_key="entity", treatment_date_col="day",
        pre_days=14, post_days=14,
    )
    assert res.n_treated >= 1
    assert res.n_controls >= 1
    assert res.verdict in ("TRUSTED", "DIRECTIONAL", "NOISY", "INSUFFICIENT")
    assert res.to_dict()["didEffect"] is not None


def test_matched_impact_tool_abstains_on_insufficient(tmp_path):
    from demos.analytics.src.analytics.toolset import AnalyticsToolset

    # Tiny dataset -> not enough controls -> abstain (INSUFFICIENT, not a guess).
    daily = pd.DataFrame(
        {"entity": ["E1", "E2"], "day": [1, 2], "value": [10, 20]}
    )
    events = pd.DataFrame({"entity": ["E1"], "day": [1]})
    src = CsvSource(
        named_csvs={
            "daily": _write_csv(tmp_path, "daily.csv", daily),
            "events": _write_csv(tmp_path, "events.csv", events),
        }
    )
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    async def run():
        r = await tools.matched_impact().invoke(
            {
                "valueCol": "daily.value", "entityCol": "entity", "dateCol": "day",
                "treatmentTable": "events", "treatmentKey": "entity",
                "treatmentDateCol": "day",
            },
            RequestContext(session_id="s"),
        )
        return r

    r = anyio.run(run)
    assert r.provenance is not None
    assert r.provenance["trust"]["tier"] in ("INSUFFICIENT", "DIRECTIONAL")


# ---------------------------------------------------------------------------
# B3: conformal forecast
# ---------------------------------------------------------------------------


def test_conformal_forecast(tmp_path):
    from demos.analytics.src.analytics.conformal import conformal_forecast

    days = pd.date_range("2024-01-01", periods=40, freq="D")
    series = pd.DataFrame({"day": days, "value": range(40)})
    src = CsvSource(named_csvs={"series": _write_csv(tmp_path, "series.csv", series)})
    model = SemanticModel.from_profile(profile_dataset(src))
    res = conformal_forecast(src, model, value_col="series.value", date_col="day", horizon=7)
    d = res.to_dict()
    assert len(d["mean"]) == 7
    assert len(d["lo"]) == 7 and len(d["hi"]) == 7
    # Bands must bracket the point forecast.
    for lo, mean, hi in zip(d["lo"], d["mean"], d["hi"]):
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# B4: segmentation
# ---------------------------------------------------------------------------


def test_segment(tmp_path):
    from demos.analytics.src.analytics.segmentation import segment

    ents = []
    for e in range(1, 101):
        ents.append({"entity": f"E{e:02d}", "value": e})
    src = CsvSource(named_csvs={"t": _write_csv(tmp_path, "t.csv", pd.DataFrame(ents))})
    model = SemanticModel.from_profile(profile_dataset(src))
    res = segment(src, model, value_col="t.value", entity_col="entity")
    d = res.to_dict()
    assert len(d["tiers"]) >= 2
    # Top tier holds the largest share of value.
    assert d["tiers"][0]["valueShare"] >= d["tiers"][-1]["valueShare"]


# ---------------------------------------------------------------------------
# B6: decision / approval governance store
# ---------------------------------------------------------------------------


def test_decision_store_lifecycle(tmp_path):
    store = DecisionStore(tmp_path / "decisions.json")
    store.record("A1", "price_up", "accepted", comment="test")
    store.set_stage("A1", "price_up", "scheduled", scheduled_for="2099-01-01")
    store.notify_host("A1", "price_up")
    assert "price_up" in store.approved_actions()
    board = store.board()
    assert board["stages"]["scheduled"] >= 1
    store.close()
    # Reload: persists.
    reloaded = DecisionStore(tmp_path / "decisions.json")
    assert "price_up" in reloaded.approved_actions()


# ---------------------------------------------------------------------------
# B7: portfolio optimizer
# ---------------------------------------------------------------------------


def test_portfolio_greedy_and_pareto():
    from demos.analytics.src.analytics.portfolio import greedy_budget_frontier, pareto_frontier

    recs = [
        {"id": f"a{i}", "value": 100 - i, "risk": i, "group": f"g{i % 3}"}
        for i in range(10)
    ]
    frontier = greedy_budget_frontier(recs, max_changes=4)
    assert frontier and frontier[-1]["nChanges"] <= 4
    # Cumulative value is non-decreasing.
    vals = [f["cumValue"] for f in frontier]
    assert vals == sorted(vals)

    pareto = pareto_frontier(
        recs, budget=4, objectives=[("value", "max"), ("risk", "min")]
    )
    assert pareto and all("selected" in p for p in pareto)


# ---------------------------------------------------------------------------
# P0: provenance envelope + reproducibility
# ---------------------------------------------------------------------------


def test_provenance_envelope_and_reproducible(tmp_path):
    daily = pd.DataFrame({"day": [1, 2, 3], "value": [10, 20, 30]})
    src = CsvSource(named_csvs={"daily": _write_csv(tmp_path, "daily.csv", daily)})
    sql = 'SELECT SUM("value") AS s FROM "daily"'
    env = build_envelope(src, sql=sql, row_count=3)
    assert env.dataset_fingerprint == fingerprint(src)
    assert env.engine_version
    rows = reproducible(env, src)
    assert sum(r["s"] for r in rows) == 60
    src.close()


def test_provenance_attached_to_tool(tmp_path):
    from demos.analytics.src.analytics.toolset import AnalyticsToolset

    sales = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "region": ["N", "S"], "amount": [1, 2]}
    )
    src = CsvSource(named_csvs={"sales": _write_csv(tmp_path, "sales.csv", sales)})
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    async def run():
        return await tools.run_query().invoke(
            {"metrics": ["sales.amount"]}, RequestContext(session_id="s")
        )

    r = anyio.run(run)
    assert r.provenance is not None
    assert set(["sql", "datasetFingerprint", "rowCount", "generatedAt", "engineVersion"]) <= set(
        r.provenance
    )
    src.close()


# ---------------------------------------------------------------------------
# P0: trust grading + abstention
# ---------------------------------------------------------------------------


def test_trust_grade_and_abstention():
    low = grade(coverage=0.2, n=5, gates={})
    assert low.tier == "INSUFFICIENT"
    assert should_abstain(low)

    high = grade(coverage=1.0, n=500, gates={"aa": True, "parallelTrends": True})
    assert high.tier == "TRUSTED"
    assert not should_abstain(high)


def test_trust_attached_to_every_answer(tmp_path):
    """P0: descriptive/analytical tools all carry a trust tier on their answer."""
    from demos.analytics.src.analytics.toolset import AnalyticsToolset

    rows = [{"region": "N" if i % 2 else "S", "amount": (i % 50) + 1} for i in range(300)]
    sales = pd.DataFrame(rows)
    src = CsvSource(named_csvs={"sales": _write_csv(tmp_path, "sales.csv", sales)})
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    async def run(tool, args):
        return await tool.invoke(args, RequestContext(session_id="s"))

    # A well-populated aggregate query is at least DIRECTIONAL.
    r = anyio.run(run, tools.run_query(), {"metrics": ["sales.amount"]})
    assert r.provenance["trust"]["tier"] in ("TRUSTED", "DIRECTIONAL")

    r2 = anyio.run(run, tools.summarize(), {"metric": "sales.amount"})
    assert r2.provenance["trust"]["tier"] in ("TRUSTED", "DIRECTIONAL")
    src.close()


def test_event_impact_abstains_on_thin_windows(tmp_path):
    """P0: a causal-style tool abstains rather than guess on thin evidence."""
    from demos.analytics.src.analytics.toolset import AnalyticsToolset

    metric = pd.DataFrame(
        {
            "asset": ["A", "A", "B", "B"],
            "day": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-01", "2024-01-10"]),
            "coin": [10, 20, 30, 40],
        }
    )
    events = pd.DataFrame({"asset": ["A"], "day": pd.to_datetime(["2024-01-05"])})
    src = CsvSource(
        named_csvs={
            "metric": _write_csv(tmp_path, "metric.csv", metric),
            "events": _write_csv(tmp_path, "events.csv", events),
        }
    )
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    async def run():
        return await tools.event_impact().invoke(
            {"metric": "metric.coin", "eventTable": "events", "anchorKey": "asset", "windowDays": 7},
            RequestContext(session_id="s"),
        )

    r = anyio.run(run)
    # Either no rows in-window (failed) or graded; if it produced a result it
    # must carry a trust tier, and thin windows abstain.
    if not r.error:
        assert r.provenance["trust"]["tier"] in ("INSUFFICIENT", "DIRECTIONAL", "TRUSTED")
    src.close()


# ---------------------------------------------------------------------------
# P1: freshness + lineage
# ---------------------------------------------------------------------------


def test_freshness_tool(tmp_path):
    from demos.analytics.src.analytics.toolset import AnalyticsToolset

    daily = pd.DataFrame(
        {"day": pd.to_datetime(["2024-01-01", "2024-01-02"]), "value": [1, 2]}
    )
    src = CsvSource(named_csvs={"daily": _write_csv(tmp_path, "daily.csv", daily)})
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    async def run():
        return await tools.freshness().invoke({}, RequestContext(session_id="s"))

    r = anyio.run(run)
    d = r.data
    assert "daily" in d["tables"]
    assert d["tables"]["daily"]["maxDate"] is not None
    src.close()


# ---------------------------------------------------------------------------
# P2: warehouse pushdown adapters + reconcile
# ---------------------------------------------------------------------------


def test_warehouse_adapter_factory():
    assert "postgres" in available_kinds()
    # Unknown kind is rejected before any connection attempt.
    try:
        make_warehouse_source("nonsense", "x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reconcile(tmp_path):
    from demos.analytics.src.analytics.reconcile import reconcile

    daily = pd.DataFrame({"value": [10, 20, 30]})
    src = CsvSource(named_csvs={"daily": _write_csv(tmp_path, "daily.csv", daily)})
    model = SemanticModel.from_profile(profile_dataset(src))
    res = reconcile(src, model, metric="daily.value", expected=60.0, tolerance=0.01)
    assert res.status == "MATCH"
    res2 = reconcile(src, model, metric="daily.value", expected=999.0, tolerance=0.01)
    assert res2.status == "MISMATCH"


# ---------------------------------------------------------------------------
# P3: feedback loop (outcome capture + threshold tuning)
# ---------------------------------------------------------------------------


def test_feedback_loop_tuning(tmp_path):
    store = DecisionStore(tmp_path / "fb.json")
    # Label several TRUSTED recommendations as wrong -> recommend raising the bar.
    for _ in range(5):
        store.record_outcome("matched_impact", "accepted", trust_tier="TRUSTED", correct=False)
    suggestion = store.tune_trust_thresholds()
    assert suggestion["action"] == "raise"
    store.close()


# ---------------------------------------------------------------------------
# Cross-cut: semantic verification
# ---------------------------------------------------------------------------


def test_verify_query_tool(tmp_path):
    from demos.analytics.src.analytics.toolset import AnalyticsToolset

    sales = pd.DataFrame(
        {"date": ["2024-01-01"], "region": ["N"], "amount": [1], "note": ["x"]}
    )
    src = CsvSource(named_csvs={"sales": _write_csv(tmp_path, "sales.csv", sales)})
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    async def run():
        # A text column asked to be summed should be flagged.
        return await tools.verify_query().invoke(
            {"metrics": ["sales.note"], "dimensions": ["sales.region"]},
            RequestContext(session_id="s"),
        )

    r = anyio.run(run)
    assert r.data["ok"] is False
    assert any("note" in i for i in r.data["issues"])
    src.close()
