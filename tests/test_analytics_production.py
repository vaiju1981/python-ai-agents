"""Production-hardening + generic-tool tests for the analytics engine.

Covers:
  A1 dataset-fingerprint model invalidation
  A2 PSI/KS drift upgrade
  A3 sqlglot-based safe run_sql
  A4 bounded relationship discovery (candidate filter + time budget)
  B1 generic change_point detection
"""

from __future__ import annotations

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.models_tools import ModelsToolset, _drift_check, _psi
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.relationships import discover
from demos.analytics.src.analytics.safe_sql import safe_sql_error
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.synthetic_data import GENERATORS
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents import RequestContext


def _domain(domain: str, tmp_path):
    out = tmp_path / domain
    paths = GENERATORS[domain](out)
    source = CsvSource(named_csvs={p.stem: p for p in paths.values()})
    model = SemanticModel.from_profile(profile_dataset(source))
    return source, model


# --- A1: dataset fingerprint ------------------------------------------------


def test_fingerprint_changes_when_data_changes(tmp_path):
    s1, _ = _domain("health", tmp_path)
    sig1 = fingerprint(s1)
    s1.close()
    # A different domain is a different dataset → different signature.
    s2, _ = _domain("casino", tmp_path)
    sig2 = fingerprint(s2)
    s2.close()
    assert sig1 != sig2
    assert len(sig1) == 16


def test_models_toolset_default_sig_is_fingerprint(tmp_path):
    s, m = _domain("health", tmp_path)
    tools = ModelsToolset(s, m)  # no dataset_sig supplied
    try:
        # The model-cache signature is row-count-agnostic (PR-11): pure row
        # growth must not invalidate trained models, only schema/role changes do.
        assert tools.dataset_sig == fingerprint(s, row_count_aware=False)
    finally:
        s.close()


def test_fingerprint_invalidates_cached_model(tmp_path):
    # Two semantically different datasets must NOT share a cached model key path.
    s1, m1 = _domain("health", tmp_path)
    t1 = ModelsToolset(s1, m1)
    s2, m2 = _domain("casino", tmp_path)
    t2 = ModelsToolset(s2, m2)
    try:
        k1 = t1._prepare_model_spec({"target": "visits.cost", "predictors": ["patients.age"]})
        k2 = t2._prepare_model_spec(
            {"target": "sessions.coinIn", "predictors": ["players.tenureDays"]}
        )
        # dataset_sig is part of the key; different data → different key
        assert k1[5] != k2[5]
    finally:
        s1.close()
        s2.close()


# --- A2: PSI / KS drift ------------------------------------------------------


def test_psi_zero_for_identical_distribution():
    import numpy as np

    edges = [float(x) for x in np.quantile(np.arange(100), [i / 10 for i in range(11)])]
    a = np.arange(100).astype(float)
    assert _psi(edges, a) < 1e-6


def test_psi_large_for_shifted_distribution():
    import numpy as np

    edges = [float(x) for x in np.quantile(np.arange(100), [i / 10 for i in range(11)])]
    a = np.arange(100).astype(float)
    shifted = a + 50.0
    assert _psi(edges, shifted) > 0.1


def test_drift_check_reports_psi(tmp_path):
    import pandas as pd

    s, m = _domain("health", tmp_path)
    try:
        train_stats = {
            "age": {
                "mean": 50.0,
                "std": 15.0,
                "quantiles": [float(x) for x in range(18, 92, 7)],
                "sample": [20.0, 40.0, 60.0, 80.0],
            }
        }
        # Serving distribution far from training → drift detected via PSI.
        scored = pd.DataFrame({"age": [80.0, 82.0, 85.0, 88.0, 90.0] * 10})
        res = _drift_check(train_stats, scored, ["age"])
        assert res["checked"] is True
        assert res["detected"] is True
        assert "psi" in res["features"]["age"]
    finally:
        s.close()


# --- A3: safe run_sql --------------------------------------------------------


def test_safe_sql_blocks_writes():
    assert safe_sql_error("DELETE FROM sales") is not None
    assert safe_sql_error("DROP TABLE sales") is not None
    assert safe_sql_error("INSERT INTO sales VALUES (1)") is not None


def test_safe_sql_blocks_multiple_statements():
    assert safe_sql_error("SELECT 1; DROP TABLE sales") is not None


def test_safe_sql_allows_reads():
    assert safe_sql_error("SELECT region, COUNT(*) AS n FROM sales GROUP BY region") is None
    # Keyword-in-identifier is allowed by the AST check (no forbidden node).
    assert safe_sql_error("SELECT date AS created_at FROM sales LIMIT 1") is None


def test_safe_sql_blocks_file_reads():
    assert safe_sql_error("SELECT * FROM read_csv('/etc/passwd')") is not None


def test_run_sql_tool_uses_safe_guard(tmp_path):
    s, m = _domain("health", tmp_path)
    tools = AnalyticsToolset(s, m)

    async def run():
        bad = await tools.run_sql().invoke(
            {"sql": "DELETE FROM visits"}, RequestContext.ephemeral()
        )
        assert bad.error
        good = await tools.run_sql().invoke(
            {"sql": "SELECT patientId, cost FROM visits LIMIT 3"},
            RequestContext.ephemeral(),
        )
        assert not good.error

    try:
        anyio.run(run)
    finally:
        s.close()


# --- A4: bounded discovery ---------------------------------------------------


def test_discovery_budget_aborts_gracefully(tmp_path, monkeypatch):
    s, _ = _domain("health", tmp_path)
    prof = profile_dataset(s)
    roles = {t.name: {c.name: c.role for c in t.columns} for t in prof.tables}
    stats = {f"{c.table}.{c.name}": c for c in prof.columns}
    monkeypatch.setenv("ANALYTICS_DISCOVERY_BUDGET_SECONDS", "0")
    # Budget 0 → discovery returns immediately (empty) without raising.
    rels = discover(s, roles, stats)
    assert isinstance(rels, list)
    s.close()


def test_discovery_still_finds_real_keys(tmp_path):
    s, _ = _domain("casino", tmp_path)
    prof = profile_dataset(s)
    roles = {t.name: {c.name: c.role for c in t.columns} for t in prof.tables}
    stats = {f"{c.table}.{c.name}": c for c in prof.columns}
    rels = discover(s, roles, stats)
    pairs = {(r.from_table, r.from_columns[0], r.to_table, r.to_columns[0]) for r in rels}
    assert ("sessions", "playerId", "players", "playerId") in pairs
    s.close()


# --- B1: change_point -------------------------------------------------------


def test_change_point_detects_injected_break(tmp_path):
    import numpy as np

    csv = tmp_path / "series.csv"
    rng = np.random.default_rng(0)
    vals = list(rng.normal(100, 5, 30)) + list(rng.normal(200, 5, 30))
    lines = ["day,value"] + [
        f"2025-01-{i + 1:02d},{v:.2f}" if i < 30 else f"2025-02-{i - 29:02d},{v:.2f}"
        for i, v in enumerate(vals)
    ]
    csv.write_text("\n".join(lines) + "\n")
    s = CsvSource(named_csvs={"series": csv})
    model = SemanticModel.from_profile(profile_dataset(s))
    tools = AnalyticsToolset(s, model)

    async def run():
        result = await tools.change_point().invoke(
            {"metric": "series.value", "timeColumn": "series.day"},
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        assert result.data, "expected at least one break"
        # The injected break is around index 30.
        dates = [d["date"] for d in result.data]
        assert any("2025-01-31" <= d <= "2025-02-02" for d in dates)

    try:
        anyio.run(run)
    finally:
        s.close()
