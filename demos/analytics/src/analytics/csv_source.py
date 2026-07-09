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


def _read_csv_sql(csv_path: Path, overrides: dict[str, str]) -> str:
    path = sql_literal(str(csv_path.resolve()))
    if not overrides:
        return f"read_csv_auto('{path}', sample_size=-1)"
    types = ", ".join(
        f"'{sql_literal(col)}': '{sql_literal(typ)}'" for col, typ in overrides.items()
    )
    return f"read_csv('{path}', auto_detect=true, sample_size=-1, types={{{types}}})"
