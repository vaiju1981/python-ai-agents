"""Parquet data source backed by DuckDB."""

from __future__ import annotations

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


class ParquetSource(DataSource):
    """DuckDB-backed Parquet data source."""

    def __init__(
        self,
        named_parquets: dict[str, Path] | None = None,
        db_path: str = ":memory:",
    ) -> None:
        self._conn = duckdb.connect(db_path)
        self._table_names: list[str] = []
        if named_parquets:
            for table_name, pq_path in named_parquets.items():
                safe_name = _sanitize(table_name)
                path = sql_literal(str(pq_path.resolve()))
                self._conn.execute(
                    f"CREATE OR REPLACE TABLE {sql_quote(safe_name)} "
                    f"AS SELECT * FROM read_parquet('{path}')"
                )
                self._table_names.append(safe_name)
        self._conn.execute("SET enable_external_access = false")

    def tables(self) -> list[TableSchema]:
        result = []
        for name in self._table_names:
            schema = self._conn.execute(f"DESCRIBE {sql_quote(name)}").fetchall()
            cols = [ColumnSchema(name=row[0], physical_type=row[1]) for row in schema]
            row_count = self._conn.execute(
                f"SELECT COUNT(*) FROM {sql_quote(name)}"
            ).fetchone()[0]
            result.append(TableSchema(name=name, rows=row_count, columns=tuple(cols)))
        return result

    def relationships(self) -> list[Relationship]:
        return []

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

    def close(self) -> None:
        self._conn.close()


def _sanitize(name: str) -> str:
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    return sanitized.lower()
