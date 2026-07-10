"""CSV data source backed by DuckDB.

Imports one or more CSV files into a DuckDB in-memory (or file-backed) database,
then provides query access through the ``DataSource`` protocol. External file
access is locked down after import so the ``run_sql`` tool can't read arbitrary
files off disk.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import duckdb

from demos.analytics.src.analytics.data_source import (
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
    sql_literal,
    sql_quote,
)

# Production guards: a wall-clock query timeout (seconds) and a hard cap on rows
# returned to the caller. Both are env-overridable.
_QUERY_TIMEOUT_SECONDS = float(os.getenv("ANALYTICS_QUERY_TIMEOUT_SECONDS", "60"))
_MAX_RESULT_ROWS = int(os.getenv("ANALYTICS_MAX_RESULT_ROWS", "500"))


class CsvSource(DataSource):
    """DuckDB-backed CSV data source."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        named_csvs: dict[str, Path] | None = None,
        type_overrides: dict[str, dict[str, str]] | None = None,
        query_timeout_seconds: float | None = None,
        max_result_rows: int | None = None,
    ) -> None:
        self._db_path = str(db_path) if db_path else None
        self._conn = duckdb.connect(self._db_path or ":memory:")
        self._query_timeout = (
            query_timeout_seconds if query_timeout_seconds is not None else _QUERY_TIMEOUT_SECONDS
        )
        self._max_result_rows = max_result_rows if max_result_rows is not None else _MAX_RESULT_ROWS
        self._table_names: list[str] = []
        if named_csvs:
            self._import_csvs(named_csvs, type_overrides or {})

    def _import_csvs(
        self,
        named_csvs: dict[str, Path],
        type_overrides: dict[str, dict[str, str]],
    ) -> None:
        for table_name, csv_path in named_csvs.items():
            name = _sanitize_identifier(table_name)
            overrides = type_overrides.get(name, {})
            read_sql = _read_csv_sql(csv_path, overrides)
            self._conn.execute(
                f"CREATE OR REPLACE TABLE {sql_quote(name)} AS SELECT * FROM {read_sql}"
            )
            self._table_names.append(name)
        # Lock down external access so run_sql can't read arbitrary files
        self._conn.execute("SET enable_external_access = false")

    def _query_connection(self) -> duckdb.DuckDBPyConnection:
        """Connection to run a query on.

        For a file-backed database we open a short-lived *read-only* connection
        per query so it can run on a worker thread (DuckDB connections are not
        safe to share across threads). In-memory databases fall back to the
        shared connection (run synchronously, no timeout).
        """
        if self._db_path:
            con = duckdb.connect(self._db_path, read_only=True)
            try:
                con.execute("SET enable_external_access = false")
            except Exception:
                pass
            return con
        return self._conn

    def _execute(self, sql: str, limit: int | None) -> list[dict[str, Any]]:
        def _run() -> list[dict[str, Any]]:
            con = self._query_connection()
            try:
                cur = con.execute(sql)
                rows = cur.fetchmany(limit) if limit else cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r, strict=False)) for r in rows]
            finally:
                if con is not self._conn:
                    con.close()

        if self._query_timeout and self._query_timeout > 0 and self._db_path:
            result: dict[str, Any] = {}
            error: dict[str, BaseException] = {}

            def _target() -> None:
                try:
                    result["value"] = _run()
                except BaseException as exc:  # noqa: BLE001 - re-raised on main thread
                    error["exc"] = exc

            worker = threading.Thread(target=_target, daemon=True)
            worker.start()
            worker.join(self._query_timeout)
            if worker.is_alive():
                raise TimeoutError(
                    f"query exceeded the {self._query_timeout:g}s limit: {sql[:120]}"
                )
            if "exc" in error:
                raise error["exc"]
            return result["value"]
        return _run()

    def tables(self) -> list[TableSchema]:
        result = []
        for name in self._table_names:
            cols = self._table_columns(name)
            row_count = self._conn.execute(f"SELECT COUNT(*) FROM {sql_quote(name)}").fetchone()[0]
            result.append(TableSchema(name=name, rows=row_count, columns=tuple(cols)))
        return result

    def _table_columns(self, table: str) -> list[ColumnSchema]:
        schema = self._conn.execute(f"DESCRIBE {sql_quote(table)}").fetchall()
        return [ColumnSchema(name=row[0], physical_type=row[1]) for row in schema]

    def relationships(self) -> list[Relationship]:
        return []  # Discovered by RelationshipDiscovery, not the source

    def sample(self, table: str, limit: int) -> list[dict[str, Any]]:
        return self._execute(f"SELECT * FROM {sql_quote(table)}", max(1, limit))

    def native_query(self, sql: str) -> list[dict[str, Any]]:
        return self._execute(sql, None)

    def native_query_with_limit(self, sql: str, max_rows: int) -> list[dict[str, Any]]:
        capped = min(max(1, max_rows), self._max_result_rows)
        return self._execute(sql, capped)

    # --- ingestion seam (PR-12) ---------------------------------------------
    # Internal, server-side writes for continuous / firehose ingestion. Never
    # exposed as a tool. Writes go through the single writer connection
    # (``self._conn``). Delta CSVs are READ IN PYTHON (via pandas/csv) and inserted
    # with parameterized SQL, so DuckDB never needs external file access for
    # ingestion — the ``enable_external_access = false`` lockdown
    # (csv_source.py:68) stays permanently on, and the file-read guard in
    # safe_sql.py (commit c6cbf83) still blocks any user SQL reading files.

    def row_count(self, table: str) -> int:
        name = _sanitize_identifier(table)
        return self._conn.execute(f"SELECT COUNT(*) FROM {sql_quote(name)}").fetchone()[0]

    def _require_table(self, table: str) -> str:
        name = _sanitize_identifier(table)
        if name not in self._table_names:
            raise ValueError(f"unknown table: {table!r} (known: {self._table_names})")
        return name

    def append_rows(self, table: str, rows: list[dict[str, Any]]) -> int:
        """Insert ``rows`` into ``table``. Returns the number of rows inserted."""
        if not rows:
            return 0
        name = self._require_table(table)
        cols = list(rows[0].keys())
        col_sql = ", ".join(sql_quote(c) for c in cols)
        placeholders = ", ".join(["?"] * len(cols))
        params = [tuple(r.get(c) for c in cols) for r in rows]
        # Parameter binding only — never interpolate row values into SQL.
        self._conn.executemany(
            f"INSERT INTO {sql_quote(name)} ({col_sql}) VALUES ({placeholders})", params
        )
        return len(params)

    def ingest_csv(self, table: str, csv_path: Path, *, mode: str = "append") -> int:
        """Load a delta CSV into ``table`` by reading it in Python and inserting.

        The CSV is parsed in the trusted server process (pandas if available,
        else the stdlib csv module) and inserted via parameterized SQL — DuckDB's
        external file access stays disabled. ``mode="append"`` inserts all rows
        (default). ``mode="upsert"`` (keyed merge / watermark / late-arrival) is
        implemented in PR-13 and raises ``NotImplementedError`` here. If
        ``table`` does not yet exist it is created from the delta (so the first
        delta can seed the table).
        """
        if mode == "upsert":
            raise NotImplementedError(
                "upsert mode is implemented in PR-13 (idempotent upsert / dedup / "
                "late-arrival watermark); use mode='append' for PR-12."
            )
        rows = _read_csv_rows(csv_path)
        if not rows:
            return 0
        name = _sanitize_identifier(table)
        if name in self._table_names:
            return self.append_rows(name, rows)
        # Seed a new table from the delta. CREATE TABLE ... AS SELECT infers column
        # types from the (typed) Python values and populates it in one statement.
        cols = list(rows[0].keys())
        col_sql = ", ".join(sql_quote(c) for c in cols)
        row_sql = "(" + ", ".join(["?"] * len(cols)) + ")"
        values_sql = ", ".join([row_sql] * len(rows))
        flat = [v for r in rows for c in cols for v in (r.get(c),)]
        self._conn.execute(
            f"CREATE TABLE {sql_quote(name)} AS "
            f"SELECT * FROM (VALUES {values_sql}) AS _v({col_sql})",
            flat,
        )
        self._table_names.append(name)
        return len(rows)

    def upsert(self, table: str, rows: list[dict[str, Any]], keys: list[str]) -> int:
        """Idempotent keyed insert (append + dedup). Returns rows actually added.

        Rows whose ``keys`` already exist in ``table`` are skipped. Watermark /
        late-arrival handling is added in PR-13; this is the seam-level baseline.
        """
        if not rows:
            return 0
        name = self._require_table(table)
        if not keys:
            raise ValueError("upsert requires at least one key column")
        cols = list(rows[0].keys())
        col_sql = ", ".join(sql_quote(c) for c in cols)
        row_sql = "(" + ", ".join(["?"] * len(cols)) + ")"
        values_sql = ", ".join([row_sql] * len(rows))
        flat = [v for r in rows for c in cols for v in (r.get(c),)]
        key_sql = [sql_quote(k) for k in keys]
        join_on = " AND ".join(f"t.{kq} = _tmp.{kq}" for kq in key_sql)
        tmp = "_ingest_upsert_tmp"
        self._conn.execute(
            f"CREATE TEMP TABLE {tmp} AS "
            f"SELECT * FROM (VALUES {values_sql}) AS _v({col_sql})",
            flat,
        )
        try:
            before = self.row_count(name)
            self._conn.execute(
                f"INSERT INTO {sql_quote(name)} SELECT * FROM {tmp} _tmp "
                f"WHERE NOT EXISTS (SELECT 1 FROM {sql_quote(name)} t WHERE {join_on})"
            )
            return self.row_count(name) - before
        finally:
            self._conn.execute(f"DROP TABLE IF EXISTS {tmp}")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def close(self) -> None:
        self._conn.close()


def _sanitize_identifier(name: str) -> str:
    """Restrict table names to safe identifier characters."""
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    return sanitized  # preserve case


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read a CSV into a list of row dicts, parsed in the trusted server process.

    Uses pandas (type-inferring) when available, otherwise the stdlib csv module.
    Returning typed values lets DuckDB infer proper column types on insert/seed.
    This is how ``ingest_csv`` lands a delta without granting DuckDB external
    file access.
    """
    csv_path = Path(csv_path)
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        return df.where(df.notna(), None).to_dict("records")
    except ImportError:
        import csv

        with open(csv_path, newline="", encoding="utf-8") as fh:
            return [dict(r) for r in csv.DictReader(fh)]


def _read_csv_sql(csv_path: Path, overrides: dict[str, str]) -> str:
    path = sql_literal(str(csv_path.resolve()))
    if not overrides:
        return f"read_csv_auto('{path}', sample_size=-1)"
    types = ", ".join(
        f"'{sql_literal(col)}': '{sql_literal(typ)}'" for col, typ in overrides.items()
    )
    return f"read_csv('{path}', auto_detect=true, sample_size=-1, types={{{types}}})"
