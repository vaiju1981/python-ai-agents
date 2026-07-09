"""Reconciliation tool (P2 defensibility).

Compares a metric computed by the engine against a declared source-of-truth
(e.g. a dashboard number, a control total, or a published report) and reports
the absolute and relative difference plus a pass/fail verdict. This is what lets
an exec/auditor answer "why doesn't this match the dashboard?": the engine shows
the gap instead of silently disagreeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel


@dataclass
class ReconcileResult:
    metric: str
    computed: float
    expected: float
    absolute_diff: float
    relative_diff: float
    tolerance: float
    status: str
    sql: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "computed": round(self.computed, 6),
            "expected": round(self.expected, 6),
            "absoluteDiff": round(self.absolute_diff, 6),
            "relativeDiff": round(self.relative_diff, 6),
            "tolerance": tolerance if (tolerance := self.tolerance) is not None else None,
            "status": self.status,
            "sql": self.sql,
            "notes": self.notes,
        }


def reconcile(
    source: DataSource,
    model: SemanticModel,
    metric: str,
    expected: float,
    tolerance: float = 0.01,
    where: str | None = None,
) -> ReconcileResult:
    """Reconcile a computed aggregate against an expected source-of-truth value."""
    notes: list[str] = []
    table = _table_of(model, metric)
    m = next((x for x in model.metrics if x.ref == metric or x.column == metric), None)
    if m is None:
        return ReconcileResult(
            metric=metric, computed=float("nan"), expected=expected,
            absolute_diff=float("nan"), relative_diff=float("nan"),
            tolerance=tolerance, status="ERROR", notes=[f"metric '{metric}' not found"],
        )
    q = f"{m.aggregation.upper()}({sql_qcol(table, m.column)})"
    sql = f"SELECT {q} AS v FROM {sql_quote(table)}"
    if where:
        sql += f" WHERE {where}"
    rows = source.native_query(sql)
    computed = float(rows[0]["v"]) if rows and rows[0].get("v") is not None else float("nan")
    abs_diff = computed - expected
    rel = abs_diff / expected if expected not in (0.0, float("nan")) else float("nan")
    status = "MATCH" if abs(rel) <= tolerance else "MISMATCH"
    notes.append(f"relative diff {rel:.4%} vs tolerance {tolerance:.2%}")
    return ReconcileResult(
        metric=metric, computed=computed, expected=expected,
        absolute_diff=abs_diff, relative_diff=rel,
        tolerance=tolerance, status=status, sql=sql, notes=notes,
    )


def _table_of(model: SemanticModel, metric: str) -> str:
    for m in model.metrics:
        if m.ref == metric or m.column == metric:
            return m.table
    return metric.split(".", 1)[0] if "." in metric else metric
