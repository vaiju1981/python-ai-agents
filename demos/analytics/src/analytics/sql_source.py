"""SQL data source: DuckDB files, SQLite, Postgres, or any SQLAlchemy URL.

Tables are read via DuckDB's ``ATTACH`` for supported engines, or directly for
DuckDB native files.
"""

from __future__ import annotations

from typing import Any

import duckdb

from demos.analytics.src.analytics.data_source import (
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
)


class SqlSource(DataSource):
    """DuckDB-backed SQL data source for DuckDB files, SQLite, or attached databases."""

    def __init__(self, db_path: str = ":memory:", attach: dict[str, str] | None = None) -> None:
        """Initialize with a DuckDB file, optionally attaching external databases.

        Args:
            db_path: Path to a DuckDB file (``:memory:`` for ephemeral).
            attach: Optional dict of ``{alias: connection_string}`` to ATTACH
                external databases (e.g. ``{"pg": "postgresql://..."}``).
        """
        self._conn = duckdb.connect(db_path)
        self._table_names: list[str] = []
        self._attached_prefix: dict[str, str] = {}

        if attach:
            for alias, conn_str in attach.items():
                self._conn.execute(f"ATTACH '{conn_str}' AS {alias} (READ_ONLY)")
                self._attached_prefix[alias] = f"{alias}."

        self._discover_tables()

    def _discover_tables(self) -> None:
        rows = self._conn.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
            "ORDER BY table_schema, table_name"
        ).fetchall()
        for schema, name in rows:
            if not name.startswith("duckdb_") and not name.startswith("sqlite_"):
                full = f"{schema}.{name}" if schema != "main" else name
                self._table_names.append(full)

    def tables(self) -> list[TableSchema]:
        result = []
        for name in self._table_names:
            cols = self._table_columns(name)
            row_count = self._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            result.append(TableSchema(name=name, rows=row_count, columns=tuple(cols)))
        return result

    def _table_columns(self, table: str) -> list[ColumnSchema]:
        schema = self._conn.execute(f"DESCRIBE {table}").fetchall()
        return [ColumnSchema(name=row[0], physical_type=row[1]) for row in schema]

    def relationships(self) -> list[Relationship]:
        return []

    def sample(self, table: str, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(f"SELECT * FROM {table} LIMIT {max(1, limit)}").fetchall()
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
