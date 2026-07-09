"""Freshness + lineage metadata (P1 defensibility).

Surfaces, for every table, the maximum event date, row count, and how stale the
data is relative to "now" — so an answer can say "is this current?" and "where
did this number come from?". Column-agnostic: it auto-detects each table's date
column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel

from datetime import datetime, timezone


@dataclass
class FreshnessReport:
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    generated_at: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generatedAt": self.generated_at,
            "tables": self.tables,
            "notes": self.notes,
        }


def _date_column(model: SemanticModel, table: str) -> str | None:
    for tc in model.time_columns:
        if tc.table == table:
            return tc.column
    for d in model.dimensions:
        if d.table == table and d.column.lower() in ("day", "date"):
            return d.column
    return None


def freshness(source: DataSource, model: SemanticModel) -> FreshnessReport:
    """Compute max-date / row-count / staleness for every table."""
    notes: list[str] = []
    tables: dict[str, dict[str, Any]] = {}
    now = datetime.now(timezone.utc)
    for t in source.tables():
        dcol = _date_column(model, t.name)
        entry: dict[str, Any] = {"rows": t.rows, "dateColumn": dcol}
        if dcol:
            try:
                ts_expr = None
                for tc in model.time_columns:
                    if tc.table == t.name:
                        ts_expr = tc.to_timestamp_sql(sql_qcol(t.name, tc.column))
                if ts_expr is None:
                    ts_expr = sql_qcol(t.name, dcol)
                row = source.native_query(
                    f"SELECT MAX({ts_expr}) AS max_d, MIN({ts_expr}) AS min_d "
                    f"FROM {sql_quote(t.name)}"
                )
                if row and row[0].get("max_d") is not None:
                    max_d = str(row[0]["max_d"])
                    entry["maxDate"] = max_d
                    try:
                        max_dt = datetime.fromisoformat(max_d.replace("Z", "+00:00"))
                        age = (now - max_dt).total_seconds() / 86400.0
                        entry["staleDays"] = round(age, 2)
                        if age > 1:
                            notes.append(f"{t.name} is {age:.1f}d stale")
                    except ValueError:
                        pass
            except Exception:
                notes.append(f"{t.name}: could not read date column")
        else:
            notes.append(f"{t.name}: no date column detected")
        tables[t.name] = entry
    return FreshnessReport(
        tables=tables, generated_at=now.isoformat(), notes=notes
    )
