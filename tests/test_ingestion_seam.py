"""PR-12 — ingestion seam E2E tests.

Covers CsvSource.append_rows / ingest_csv / upsert / row_count and asserts that
the read-only safety posture (safe_sql.py, commit c6cbf83) is preserved: ingest
is an internal write path, run_sql/dsl_query stay read-only, and external file
access is re-locked after a delta load.
"""

from __future__ import annotations

import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.parquet_source import ParquetSource
from demos.analytics.src.analytics.safe_sql import safe_sql_error
from demos.analytics.src.analytics.sql_source import SqlSource


def _write_csv(path, header, rows):
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def _csv_source(tmp_path):
    base = tmp_path / "sales.csv"
    _write_csv(base, "id,region,amount", [(1, "north", 10), (2, "south", 20), (3, "east", 30)])
    return CsvSource(named_csvs={"sales": base})


# --- append_rows -------------------------------------------------------------


def test_append_rows_inserts_and_is_queryable(tmp_path):
    src = _csv_source(tmp_path)
    try:
        assert src.row_count("sales") == 3
        added = src.append_rows(
            "sales",
            [
                {"id": 4, "region": "west", "amount": 40},
                {"id": 5, "region": "north", "amount": 50},
            ],
        )
        assert added == 2
        assert src.row_count("sales") == 5
        rows = {r["id"]: r["amount"] for r in src.native_query("SELECT id, amount FROM sales")}
        assert rows == {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    finally:
        src.close()


def test_append_rows_empty_is_noop(tmp_path):
    src = _csv_source(tmp_path)
    try:
        assert src.append_rows("sales", []) == 0
        assert src.row_count("sales") == 3
    finally:
        src.close()


def test_append_rows_unknown_table_errors(tmp_path):
    src = _csv_source(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.append_rows("ghost", [{"id": 1}])
    finally:
        src.close()


# --- ingest_csv --------------------------------------------------------------


def test_ingest_csv_appends_without_full_reimport(tmp_path):
    src = _csv_source(tmp_path)
    try:
        delta = tmp_path / "sales_delta.csv"
        _write_csv(delta, "id,region,amount", [(4, "west", 40), (5, "north", 50)])
        added = src.ingest_csv("sales", delta, mode="append")
        # Original 3 rows intact (no CREATE OR REPLACE) + 2 new = 5.
        assert added == 2
        assert src.row_count("sales") == 5
        ids = {r["id"] for r in src.native_query("SELECT id FROM sales")}
        assert ids == {1, 2, 3, 4, 5}
    finally:
        src.close()


def test_ingest_csv_creates_table_when_missing(tmp_path):
    src = _csv_source(tmp_path)
    try:
        delta = tmp_path / "newtable.csv"
        _write_csv(delta, "k,v", [(1, "a"), (2, "b")])
        added = src.ingest_csv("newtable", delta, mode="append")
        assert added == 2
        assert src.row_count("newtable") == 2
    finally:
        src.close()


def test_ingest_csv_upsert_mode_not_implemented_in_pr12(tmp_path):
    src = _csv_source(tmp_path)
    try:
        delta = tmp_path / "sales_delta.csv"
        _write_csv(delta, "id,region,amount", [(4, "west", 40)])
        with pytest.raises(NotImplementedError):
            src.ingest_csv("sales", delta, mode="upsert")
    finally:
        src.close()


def test_ingest_relocks_external_access(tmp_path):
    src = _csv_source(tmp_path)
    try:
        delta = tmp_path / "sales_delta.csv"
        _write_csv(delta, "id,region,amount", [(4, "west", 40)])
        src.ingest_csv("sales", delta, mode="append")
        # After the load, external file access must be locked down again: a
        # read_csv against an arbitrary path must be rejected (raises).
        import duckdb

        with pytest.raises(duckdb.Error):
            src.native_query("SELECT * FROM read_csv('/etc/passwd')")
    finally:
        src.close()


# --- upsert (keyed append + dedup) ------------------------------------------


def test_upsert_skips_existing_keys(tmp_path):
    src = _csv_source(tmp_path)
    try:
        # Re-deliver an existing row (id=1) plus a new one (id=4): only the new
        # row should be added.
        added = src.upsert(
            "sales",
            [{"id": 1, "region": "north", "amount": 10}, {"id": 4, "region": "west", "amount": 40}],
            keys=["id"],
        )
        assert added == 1
        assert src.row_count("sales") == 4
        # The duplicate id=1 did not create a second copy.
        dupes = src.native_query("SELECT id FROM sales WHERE id = 1")
        assert len(dupes) == 1
    finally:
        src.close()


def test_upsert_requires_keys(tmp_path):
    src = _csv_source(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.upsert("sales", [{"id": 9, "region": "x", "amount": 1}], keys=[])
    finally:
        src.close()


# --- read-only safety posture preserved -------------------------------------


def test_user_sql_still_cannot_write_or_read_files():
    # The ingest seam is internal; user-facing SQL validation is unchanged.
    assert safe_sql_error("DELETE FROM sales") is not None
    assert safe_sql_error("DROP TABLE sales") is not None
    assert safe_sql_error("SELECT 1; DROP TABLE sales") is not None
    assert safe_sql_error("SELECT * FROM read_csv('/etc/passwd')") is not None
    assert safe_sql_error("SELECT region, COUNT(*) AS n FROM sales GROUP BY region") is None


# --- protocol conformance for read-only backends ----------------------------


def test_readonly_sources_implement_row_count_and_reject_writes(tmp_path):
    # ParquetSource
    pq = tmp_path / "sales.parquet"
    # Build a parquet file via DuckDB so ParquetSource has something to open.
    import duckdb

    duckdb.connect(":memory:").execute(
        f"COPY (SELECT * FROM read_csv_auto('{_make_csv(tmp_path, 'p.csv')}')) "
        f"TO '{pq}' (FORMAT PARQUET)"
    )
    psrc = ParquetSource(named_parquets={"sales": pq})
    try:
        assert psrc.row_count("sales") == 3
        with pytest.raises(NotImplementedError):
            psrc.append_rows("sales", [{"id": 9, "region": "x", "amount": 1}])
    finally:
        psrc.close()

    # SqlSource (in-memory)
    ssrc = SqlSource(db_path=":memory:")
    try:
        ssrc.native_query("CREATE TABLE t (id INTEGER)")
        with pytest.raises(NotImplementedError):
            ssrc.append_rows("t", [{"id": 1}])
    finally:
        ssrc.close()


def _make_csv(tmp_path, name):
    p = tmp_path / name
    _write_csv(p, "id,region,amount", [(1, "north", 10), (2, "south", 20), (3, "east", 30)])
    return p
