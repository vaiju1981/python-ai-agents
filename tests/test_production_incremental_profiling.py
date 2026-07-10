"""PR-11 verification: streaming / incremental profiling.

Covers the two E2E gates from docs/production_readiness.md:

1. Profile a dataset, append rows, run an incremental update; assert the column
   stats update AND that a *pure row-count growth* does NOT change the
   model-invalidation signature (while a schema change still does).
2. Incremental profiling of a growing stream stays within a bounded time budget.

The incremental update streams only the newly-arrived batch (O(batch)), never
re-reading the whole table, so per-update cost is independent of how much data
has already landed.
"""

from __future__ import annotations

import time

import duckdb
import pandas as pd
import pytest

from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.profiler import (
    ColumnProfile,
    merge_column_profiles,
    profile_dataset,
    profile_dataset_incremental,
)
from demos.analytics.src.analytics.warehouse_sources import make_warehouse_source

pytest.importorskip("duckdb")

ALIAS = "wh"


def _sales_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic sales rows drawn from a fixed distribution."""
    rows = []
    for i in range(n):
        j = i + seed
        rows.append(
            {
                "region": "N" if j % 3 == 0 else "S",
                "segment": "A" if j % 2 == 0 else "B",
                "amount": float((j % 50) + 1) * (1.5 if j % 3 == 0 else 1.0),
            }
        )
    return pd.DataFrame(rows)


def _build_warehouse(tmp_path, df: pd.DataFrame) -> str:
    wh = tmp_path / "wh.duckdb"
    con = duckdb.connect(str(wh))
    con.register("sales_df", df)
    con.execute("CREATE TABLE sales AS SELECT * FROM sales_df")
    con.close()
    return str(wh)


def _append(wh_path: str, df: pd.DataFrame) -> None:
    con = duckdb.connect(str(wh_path))
    con.register("add_df", df)
    con.execute("INSERT INTO sales SELECT * FROM add_df")
    con.close()


def _model_cache_sig(wh_path: str) -> str:
    """The row-count-agnostic signature used as the model-cache key (PR-11)."""
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    try:
        return fingerprint(src, row_count_aware=False)
    finally:
        src.close()


def test_incremental_update_refreshes_stats_and_keeps_sig_on_row_growth(tmp_path):
    initial = _sales_df(300)
    wh_path = _build_warehouse(tmp_path, initial)

    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    profile = profile_dataset(src)
    src.close()

    prev_col = next(
        c for c in profile.columns if c.table == f"{ALIAS}.sales" and c.name == "amount"
    )
    sig_before = _model_cache_sig(wh_path)

    # Append more rows drawn from the SAME distribution (pure growth).
    appended = _sales_df(300, seed=300)
    _append(wh_path, appended)
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    updated = profile_dataset_incremental(src, prev=profile, new_rows={f"{ALIAS}.sales": appended})
    src.close()

    new_col = next(c for c in updated.columns if c.table == f"{ALIAS}.sales" and c.name == "amount")
    # Stats advanced: row count grew, mean is preserved (same distribution).
    assert new_col.rows == prev_col.rows + len(appended)
    assert new_col.rows == 600
    assert prev_col.mean is not None and new_col.mean is not None
    assert abs(new_col.mean - prev_col.mean) < 0.5

    # Pure row growth must NOT invalidate the model cache.
    sig_after = _model_cache_sig(wh_path)
    assert sig_after == sig_before


def test_schema_change_still_invalidates_model_cache(tmp_path):
    initial = _sales_df(300)
    wh_path = _build_warehouse(tmp_path, initial)
    sig_before = _model_cache_sig(wh_path)

    # Same rows but add a new column -> schema change -> signature must change.
    wh2 = tmp_path / "wh2.duckdb"
    con = duckdb.connect(str(wh2))
    con.register("sales_df", initial.assign(promo=[0] * len(initial)))
    con.execute("CREATE TABLE sales AS SELECT * FROM sales_df")
    con.close()

    sig_after = _model_cache_sig(str(wh2))
    assert sig_after != sig_before


def test_merge_exactness_matches_full_profile(tmp_path):
    df = _sales_df(500)
    wh_path = _build_warehouse(tmp_path, df)
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)

    # Full profile of the whole table.
    full = profile_dataset(src)

    # Split into two batches and merge incrementally (no extra source reads).
    b1 = df.iloc[:200]
    b2 = df.iloc[200:]
    cp1 = ColumnProfile.from_batch(f"{ALIAS}.sales", full.columns[0], b1["region"].tolist())
    cp2 = ColumnProfile.from_batch(f"{ALIAS}.sales", full.columns[0], b2["region"].tolist())
    merged = merge_column_profiles(cp1, cp2)

    full_cp = next(c for c in full.columns if c.table == f"{ALIAS}.sales" and c.name == "region")
    assert merged.rows == full_cp.rows
    assert merged.distinct == full_cp.distinct
    assert merged.nulls == full_cp.nulls
    src.close()


def test_growing_stream_stays_within_bounded_time_budget(tmp_path):
    initial = _sales_df(200)
    wh_path = _build_warehouse(tmp_path, initial)
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    profile = profile_dataset(src)
    src.close()

    batch = _sales_df(200)
    # Warm-up iteration: absorbs connection/import cold-start so the measured
    # band reflects steady-state per-update cost, not one-time setup.
    _append(wh_path, batch)
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    profile = profile_dataset_incremental(src, prev=profile, new_rows={f"{ALIAS}.sales": batch})
    src.close()

    # Each step appends a constant-size batch; per-update cost must stay flat
    # regardless of total accumulated rows (O(batch), not O(total)).
    durations: list[float] = []
    for _ in range(12):
        _append(wh_path, batch)
        t0 = time.perf_counter()
        src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
        profile = profile_dataset_incremental(src, prev=profile, new_rows={f"{ALIAS}.sales": batch})
        src.close()
        durations.append(time.perf_counter() - t0)

    # Every update is fast AND cost is flat (no growth with accumulated data):
    # the slowest update is within a small multiple of the fastest.
    assert max(durations) < 1.0
    assert max(durations) <= 5 * min(durations) + 1e-6


def test_incremental_feed_does_not_thrash_model_key(tmp_path):
    """End-to-end: a model keyed on the incremental profile is stable across growth."""
    initial = _sales_df(300)
    wh_path = _build_warehouse(tmp_path, initial)
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    profile = profile_dataset(src)
    key_before = fingerprint(src, row_count_aware=False)
    src.close()

    for _ in range(3):
        _append(wh_path, _sales_df(300, seed=1000))
    src = make_warehouse_source("duckdb", wh_path, alias=ALIAS)
    updated = profile_dataset_incremental(
        src, prev=profile, new_rows={f"{ALIAS}.sales": _sales_df(300, seed=1000)}
    )
    key_after = fingerprint(src, row_count_aware=False)
    src.close()

    assert key_after == key_before
    assert updated.columns[0].rows > profile.columns[0].rows
