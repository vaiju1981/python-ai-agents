"""Freshness + lineage metadata (P1 defensibility).

Surfaces, for every table, the maximum event date, row count, and how stale the
data is relative to "now" — so an answer can say "is this current?" and "where
did this number come from?". Column-agnostic: it auto-detects each table's date
column.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel

from datetime import date, datetime, timezone


# Maximum allowed data staleness (days) before the query path refuses to answer
# silently (PR-16 freshness gate). ``None`` (env ``ANALYTICS_MAX_STALE_DAYS``
# unset) means no gate — the engine answers regardless of age.
_MAX_STALE_ENV = "ANALYTICS_MAX_STALE_DAYS"


def max_stale_days() -> float | None:
    """Configured max-staleness policy in days, or ``None`` if the gate is off."""
    raw = os.getenv(_MAX_STALE_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass
class FreshnessReport:
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    generated_at: str = ""
    notes: list[str] = field(default_factory=list)
    max_stale_days: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generatedAt": self.generated_at,
            "tables": self.tables,
            "notes": self.notes,
            "maxStaleDays": self.max_stale_days,
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
    """Compute max-date / row-count / staleness for every table.

    When the ``ANALYTICS_MAX_STALE_DAYS`` policy is set, each table carries a
    ``stale`` flag (``staleDays`` exceeds the policy) so the query path can refuse
    to serve answers against data older than the policy (PR-16 freshness gate).
    """
    notes: list[str] = []
    tables: dict[str, dict[str, Any]] = {}
    policy = max_stale_days()
    now = datetime.now(timezone.utc)
    for t in source.tables():
        dcol = _date_column(model, t.name)
        entry: dict[str, Any] = {"rows": t.rows, "dateColumn": dcol, "stale": False}
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
                        # DuckDB yields DATE columns as naive ``datetime.date`` and
                        # TIMESTAMP columns as tz-naive/aware ``datetime``; normalize
                        # both to a tz-aware ``datetime`` so the subtraction against
                        # ``now`` (tz-aware) is well-defined.
                        if isinstance(max_dt, date) and not isinstance(max_dt, datetime):
                            max_dt = datetime(max_dt.year, max_dt.month, max_dt.day)
                        if max_dt.tzinfo is None:
                            max_dt = max_dt.replace(tzinfo=timezone.utc)
                        age = (now - max_dt).total_seconds() / 86400.0
                        entry["staleDays"] = round(age, 2)
                        if age > 1:
                            notes.append(f"{t.name} is {age:.1f}d stale")
                        if policy is not None and age > policy:
                            entry["stale"] = True
                    except (ValueError, TypeError):
                        pass
            except Exception:
                notes.append(f"{t.name}: could not read date column")
        else:
            notes.append(f"{t.name}: no date column detected")
        tables[t.name] = entry
    return FreshnessReport(
        tables=tables, generated_at=now.isoformat(), notes=notes, max_stale_days=policy
    )


def stale_tables(source: DataSource, model: SemanticModel, tables: list[str]) -> list[str]:
    """Return the subset of ``tables`` that violate the configured max-stale policy.

    With no policy configured (``ANALYTICS_MAX_STALE_DAYS`` unset) returns ``[]`` —
    the freshness gate is off and every table is considered fresh enough to serve.
    """
    if max_stale_days() is None:
        return []
    rep = freshness(source, model)
    return [t for t in tables if rep.tables.get(t, {}).get("stale")]
