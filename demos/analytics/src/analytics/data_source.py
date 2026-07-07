"""Data source abstraction: protocol, schemas, and types.

A ``DataSource`` is the pluggable seam for any tabular data backend. Implementations
include ``CsvSource`` (DuckDB-backed CSV import), ``SqlSource`` (JDBC databases),
``GraphSource`` (Neo4j graph projection), and ``ParquetSource``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ColumnRole(str, Enum):
    IDENTIFIER = "identifier"
    DIMENSION = "dimension"
    MEASURE_ADDITIVE = "measure_additive"
    MEASURE_RATIO = "measure_ratio"
    TIMESTAMP = "timestamp"
    DATE = "date"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    name: str
    physical_type: str
    role: ColumnRole = ColumnRole.UNKNOWN


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    rows: int
    columns: tuple[ColumnSchema, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Relationship:
    from_table: str
    from_columns: tuple[str, ...]
    to_table: str
    to_columns: tuple[str, ...]
    cardinality: str = "many_to_one"
    coverage: float = 1.0

    @property
    def from_column(self) -> str:
        return self.from_columns[0]

    @property
    def to_column(self) -> str:
        return self.to_columns[0]


class DataSource(Protocol):
    """Pluggable seam for any tabular data backend."""

    def tables(self) -> list[TableSchema]:
        ...

    def relationships(self) -> list[Relationship]:
        ...

    def sample(self, table: str, limit: int) -> list[dict[str, Any]]:
        ...

    def native_query(self, sql: str) -> list[dict[str, Any]]:
        ...

    def native_query_with_limit(self, sql: str, max_rows: int) -> list[dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


def sql_quote(identifier: str) -> str:
    """Quote a SQL identifier (table or column name) for DuckDB."""
    return '"' + identifier.replace('"', '""') + '"'


def sql_qcol(table: str, column: str) -> str:
    """Quote a fully-qualified column reference."""
    return f"{sql_quote(table)}.{sql_quote(column)}"


def sql_literal(value: str) -> str:
    """Escape a value for use inside a single-quoted SQL string literal."""
    return value.replace("'", "''")
