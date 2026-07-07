"""Semantic model: query-oriented model derived from a ``DatasetProfile``.

Measures become metrics (additive -> sum, ratio -> avg); dimensions/booleans
become groupable dimensions; identifiers become entity keys; date/timestamp
columns carry a ``TimeEncoding`` so they can be normalized to a real timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from demos.analytics.src.analytics.data_source import ColumnRole, Relationship
from demos.analytics.src.analytics.profiler import ColumnProfile, DatasetProfile


class TimeEncoding(str, Enum):
    NATIVE = "native"
    EPOCH_SECONDS = "epoch_seconds"
    EPOCH_MILLIS = "epoch_millis"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Metric:
    table: str
    column: str
    aggregation: str

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class Dimension:
    table: str
    column: str

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class TimeColumn:
    table: str
    column: str
    encoding: TimeEncoding

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"

    def to_timestamp_sql(self, column_sql: str) -> str:
        if self.encoding == TimeEncoding.EPOCH_SECONDS:
            return f"to_timestamp({column_sql})"
        if self.encoding == TimeEncoding.EPOCH_MILLIS:
            return f"to_timestamp({column_sql} / 1000)"
        if self.encoding == TimeEncoding.TEXT:
            return f"try_cast({column_sql} AS timestamp)"
        return column_sql


@dataclass(frozen=True, slots=True)
class SemanticModel:
    metrics: tuple[Metric, ...]
    dimensions: tuple[Dimension, ...]
    entity_keys: tuple[str, ...]
    time_columns: tuple[TimeColumn, ...]
    relationships: tuple[Relationship, ...]

    @classmethod
    def from_profile(cls, profile: DatasetProfile) -> SemanticModel:
        stats = {f"{cp.table}.{cp.name}": cp for cp in profile.columns}
        metrics: list[Metric] = []
        dimensions: list[Dimension] = []
        entity_keys: list[str] = []
        time_columns: list[TimeColumn] = []

        for table in profile.tables:
            for col in table.columns:
                if col.role == ColumnRole.MEASURE_ADDITIVE:
                    metrics.append(Metric(table=table.name, column=col.name, aggregation="sum"))
                elif col.role == ColumnRole.MEASURE_RATIO:
                    metrics.append(Metric(table=table.name, column=col.name, aggregation="avg"))
                elif col.role in (ColumnRole.DATE, ColumnRole.TIMESTAMP):
                    dimensions.append(Dimension(table=table.name, column=col.name))
                    encoding = _encoding_of(col, stats.get(f"{table.name}.{col.name}"))
                    time_columns.append(
                        TimeColumn(table=table.name, column=col.name, encoding=encoding)
                    )
                elif col.role in (ColumnRole.DIMENSION, ColumnRole.BOOLEAN):
                    dimensions.append(Dimension(table=table.name, column=col.name))
                elif col.role == ColumnRole.IDENTIFIER:
                    entity_keys.append(f"{table.name}.{col.name}")

        return cls(
            metrics=tuple(metrics),
            dimensions=tuple(dimensions),
            entity_keys=tuple(entity_keys),
            time_columns=tuple(time_columns),
            relationships=profile.relationships,
        )

    def catalog_json(
        self, source_tables: list[Any] | None = None, catalog: Any | None = None
    ) -> str:
        """Compact JSON schema for the agent's system prompt."""
        import json

        return json.dumps(self._catalog_dict(source_tables, catalog), separators=(",", ":"))

    def _catalog_dict(self, source_tables: list[Any] | None, catalog: Any | None) -> dict:
        def desc_ref(ref: str, description: str | None = None) -> str:
            return f"{ref} -- {description}" if description and description.strip() else ref

        tables_out = []
        if source_tables:
            for t in source_tables:
                t_desc = catalog.description_for_table(t.name) if catalog else ""
                cols = [
                    desc_ref(
                        f"{c.name} {c.physical_type}",
                        catalog.description_for_column(t.name, c.name) if catalog else None,
                    )
                    for c in t.columns
                ]
                tables_out.append({"name": t.name, "description": t_desc or "", "columns": cols})

        return {
            "metrics": [
                desc_ref(
                    m.ref, catalog.description_for_column(m.table, m.column) if catalog else None
                )
                for m in self.metrics
            ],
            "dimensions": [
                desc_ref(
                    d.ref, catalog.description_for_column(d.table, d.column) if catalog else None
                )
                for d in self.dimensions
            ],
            "timeColumns": [f"{t.ref} ({t.encoding.value})" for t in self.time_columns],
            "tables": tables_out,
            "relationships": [
                f"{r.from_table}.{','.join(r.from_columns)} ~ "
                f"{r.to_table}.{','.join(r.to_columns)} ({r.cardinality})"
                for r in self.relationships
            ],
        }


def _encoding_of(col: Any, stats: ColumnProfile | None) -> TimeEncoding:
    if col.role == ColumnRole.TIMESTAMP:
        if (
            stats
            and stats.min is not None
            and stats.min >= 1e9
            and stats.max is not None
            and stats.max <= 5e12
        ):
            if stats.max > 1e10:
                return TimeEncoding.EPOCH_MILLIS
            return TimeEncoding.EPOCH_SECONDS
        return TimeEncoding.NATIVE
    if col.role == ColumnRole.DATE:
        if stats and any("date-like" in s for s in [str(stats.signals)]):
            return TimeEncoding.TEXT
        return TimeEncoding.NATIVE
    return TimeEncoding.NATIVE
