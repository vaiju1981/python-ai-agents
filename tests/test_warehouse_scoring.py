"""PR-8 verification: warehouse-side model scoring (tracker §G G1).

Pushes row-level model scoring to the warehouse so serving no longer
materializes the scored frame into pandas. Linear models are expressed as exact
SQL arithmetic; tree/RF/k-means/isolation-forest fall back to local (documented
limit).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import Metric, SemanticModel
from demos.analytics.src.analytics.warehouse_sources import (
    make_warehouse_source,
    score_warehouse,
)

pytest.importorskip("duckdb")
pytest.importorskip("sklearn")


def _build_warehouse(tmp_path, n: int = 3000):
    """DuckDB file standing in for a warehouse, with a large `facts` table."""
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-10, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.7 + rng.normal(0, 0.1, n)
    wh = tmp_path / "wh.duckdb"
    con = duckdb.connect(str(wh))
    values = ", ".join("(" + ",".join(map(str, r)) + ")" for r in zip(x1, x2, y, strict=False))
    con.execute(
        "CREATE TABLE facts AS SELECT "
        "x1::DOUBLE AS x1, x2::DOUBLE AS x2, y::DOUBLE AS y FROM ("
        f"SELECT * FROM (VALUES {values}) AS t(x1, x2, y))"
    )
    con.close()
    return str(wh)


def test_score_warehouse_linear_matches_pandas_and_pulls_no_rows(tmp_path):
    wh_path = _build_warehouse(tmp_path, n=3000)
    src = make_warehouse_source("duckdb", wh_path, alias="wh")

    # Train on a bounded sample (the train/serve split).
    sample = src.native_query("SELECT x1, x2, y FROM wh.facts USING SAMPLE 400 ROWS")
    sdf = pd.DataFrame(sample)
    model = LinearRegression().fit(sdf[["x1", "x2"]], sdf["y"])

    frame_sql = "SELECT x1, x2 FROM wh.facts"

    # score_warehouse must NOT pull any rows — it only builds SQL.
    calls = {"n": 0}
    real_nq = src.native_query

    def spy(sql):
        calls["n"] += 1
        return real_nq(sql)

    src.native_query = spy
    score_sql = score_warehouse(src, frame_sql, model, ["x1", "x2"], task="regression")
    src.native_query = real_nq
    assert calls["n"] == 0, "score_warehouse must not materialize the frame"
    assert score_sql and "prediction" in score_sql

    # Run the in-warehouse scoring and compare to local pandas predictions.
    wh_preds = pd.DataFrame(real_nq(score_sql))["prediction"].to_numpy()
    full = pd.DataFrame(real_nq(frame_sql)).astype(float)
    local_preds = model.predict(full[["x1", "x2"]].to_numpy())
    assert wh_preds.shape == local_preds.shape
    np.testing.assert_allclose(np.sort(wh_preds), np.sort(local_preds), rtol=1e-6, atol=1e-6)
    src.close()


def test_score_warehouse_returns_none_for_tree_model(tmp_path):
    wh_path = _build_warehouse(tmp_path, n=50)
    src = make_warehouse_source("duckdb", wh_path, alias="wh")
    df = pd.DataFrame(src.native_query("SELECT x1, x2, y FROM wh.facts"))
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(df[["x1", "x2"]], df["y"])
    assert (
        score_warehouse(src, "SELECT x1, x2 FROM wh.facts", model, ["x1", "x2"], task="regression")
        is None
    )
    src.close()


def test_predict_scores_in_warehouse_for_linear_model(tmp_path):
    from demos.analytics.src.analytics.semantic_model import Metric

    wh_path = _build_warehouse(tmp_path, n=2000)
    src = make_warehouse_source("duckdb", wh_path, alias="wh")
    base = SemanticModel.from_profile(profile_dataset(src))
    # Force a single-table model so predictors resolve to `wh.facts`.
    model = SemanticModel(
        metrics=(
            Metric(table="wh.facts", column="y", aggregation="sum"),
            Metric(table="wh.facts", column="x1", aggregation="sum"),
            Metric(table="wh.facts", column="x2", aggregation="sum"),
        ),
        dimensions=(),
        entity_keys=(),
        time_columns=(),
        relationships=(),
        columns=base.columns,
    )

    tools = ModelsToolset(src, model)

    sample = src.native_query("SELECT x1, x2, y FROM wh.facts USING SAMPLE 400 ROWS")
    sdf = pd.DataFrame(sample)
    lin = LinearRegression().fit(sdf[["x1", "x2"]], sdf["y"])
    meta = {
        "task": "regression",
        "train_stats": {
            "x1": {"mean": float(sdf["x1"].mean()), "std": float(sdf["x1"].std(ddof=0))},
            "x2": {"mean": float(sdf["x2"].mean()), "std": float(sdf["x2"].std(ddof=0))},
        },
    }
    result = tools._predict_in_warehouse(
        "wh.facts",
        "y",
        [("wh.facts", "x1"), ("wh.facts", "x2")],
        lin,
        meta,
        [],
    )
    assert result is not None
    payload = result.data
    assert payload.get("scored_in_warehouse") is True
    assert "prediction" in payload
    assert payload["n_scored"] == 2000
    src.close()


def test_non_warehouse_source_keeps_local_predict(tmp_path):
    """A non-warehouse source (CSV) must keep its existing local predict path
    and must NOT be rewritten to report in-warehouse scoring (PR-8 gate)."""

    # Build a tiny CSV source with x1, x2, y.
    rng = np.random.default_rng(1)
    n = 60
    x1 = rng.uniform(-10, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.7 + rng.normal(0, 0.1, n)
    csv = tmp_path / "facts.csv"
    rows = "\n".join(f"{a},{b},{c}" for a, b, c in zip(x1, x2, y, strict=False))
    csv.write_text("x1,x2,y\n" + rows + "\n")

    src = CsvSource(named_csvs={"facts": csv})
    base = SemanticModel.from_profile(profile_dataset(src))
    model = SemanticModel(
        metrics=(
            Metric(table="facts", column="y", aggregation="sum"),
            Metric(table="facts", column="x1", aggregation="sum"),
            Metric(table="facts", column="x2", aggregation="sum"),
        ),
        dimensions=(),
        entity_keys=(),
        time_columns=(),
        relationships=(),
        columns=base.columns,
    )
    tools = ModelsToolset(src, model)

    lin = LinearRegression().fit(np.column_stack([x1, x2]), y)
    # The warehouse gate must return None for a non-warehouse source even when
    # the model is linear and expressible.
    wh = tools._predict_in_warehouse(
        "facts",
        "y",
        [("facts", "x1"), ("facts", "x2")],
        lin,
        {"task": "regression"},
        [],
    )
    assert wh is None
    assert getattr(src, "_is_warehouse", False) is False
    src.close()
