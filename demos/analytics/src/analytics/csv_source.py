"""CSV data source backed by DuckDB.

Imports one or more CSV files into a DuckDB in-memory (or file-backed) database,
then provides query access through the ``DataSource`` protocol. External file
access is locked down after import so the ``run_sql`` tool can't read arbitrary
files off disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from demos.analytics.src.analytics.data_source import (
    ColumnRole,
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
    sql_literal,
    sql_quote,
)


class CsvSource(DataSource):
    """DuckDB-backed CSV data source."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        named_csvs: dict[str, Path] | None = None,
        type_overrides: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._conn = duckdb.connect(str(db_path) if db_path else ":memory:")
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

    def tables(self) -> list[TableSchema]:
        result = []
        for name in self._table_names:
            cols = self._table_columns(name)
            row_count = self._conn.execute(
                f"SELECT COUNT(*) FROM {sql_quote(name)}"
            ).fetchone()[0]
            result.append(TableSchema(name=name, rows=row_count, columns=tuple(cols)))
        return result

    def _table_columns(self, table: str) -> list[ColumnSchema]:
        schema = self._conn.execute(f"DESCRIBE {sql_quote(table)}").fetchall()
        return [
            ColumnSchema(name=row[0], physical_type=row[1])
            for row in schema
        ]

    def relationships(self) -> list[Relationship]:
        return []  # Discovered by RelationshipDiscovery, not the source

    def sample(self, table: str, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"SELECT * FROM {sql_quote(table)} LIMIT {max(1, limit)}"
        ).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def native_query(self, sql: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def native_query_with_limit(self, sql: str, max_rows: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql).fetchmany(max_rows)
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

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
