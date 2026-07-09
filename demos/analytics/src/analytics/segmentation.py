"""Generic value/intensity segmentation (column-agnostic lift of ATLAS segments).

Groups entities by a measured value and buckets them into value tiers (e.g.
low / mid / high / top), reporting each tier's size, mean value, and share of
total value ("coverage"/intensity). Optionally broken down by a category
dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel


@dataclass
class Segmentation:
    tiers: list[dict[str, Any]]
    coverage: float
    n_entities: int
    by_dimension: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiers": self.tiers,
            "coverage": round(self.coverage, 4),
            "nEntities": self.n_entities,
            "byDimension": self.by_dimension,
            "notes": self.notes,
        }


def segment(
    source: DataSource,
    model: SemanticModel,
    value_col: str,
    entity_col: str,
    dimension_col: str | None = None,
    date_col: str | None = None,
    last_days: int | None = None,
    n_tiers: int = 4,
) -> Segmentation:
    """Segment entities by total value into ``n_tiers`` value tiers."""
    notes: list[str] = []
    vcol = value_col.split(".")[-1]
    ecol = entity_col.split(".")[-1]
    dcol = dimension_col.split(".")[-1] if dimension_col else None
    table = _table_of(model, value_col)
    qv = sql_qcol(table, vcol)
    qe = sql_qcol(table, ecol)
    sel = f"{qe} AS e, SUM({qv}) AS v"
    if dcol:
        sel += f", {sql_qcol(table, dcol)} AS dim"
    where = ""
    if date_col and last_days:
        from demos.analytics.src.analytics.query_planner import _now_expr

        tc = next(
            (t for t in model.time_columns if t.ref == date_col or t.column == date_col),
            None,
        )
        if tc:
            ts_expr = tc.to_timestamp_sql(sql_qcol(table, tc.column))
            where = f" WHERE {ts_expr} >= {_now_expr()} - INTERVAL '{last_days} days'"
    sql = (
        f"SELECT {sel} FROM {sql_quote(table)}{where} GROUP BY e"
        + (f", dim" if dimension_col else "")
    )
    rows = source.native_query(sql)
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return _fail("no data to segment")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["v"])
    if df.empty:
        return _fail("no numeric values to segment")
    if len(df) < n_tiers:
        notes.append("fewer entities than tiers; collapsing")
        n_tiers = max(1, len(df) // 2) or 1

    # Value-based tiers via quantile cut points. qcut assigns labels in
    # ascending value order, so pass labels low→high; we then emit tiers in
    # high→low order so the top (highest-value) tier is first.
    labels_desc = _tier_labels(n_tiers)  # top … low
    labels_asc = list(reversed(labels_desc))  # low … top
    try:
        df["tier"] = pd.qcut(df["v"], n_tiers, labels=labels_asc, duplicates="drop")
    except ValueError:
        df["tier"] = pd.cut(df["v"], n_tiers, labels=labels_asc)
    total_value = float(df["v"].sum())
    present = set(df["tier"].dropna().unique())
    ordered_tiers = [t for t in labels_desc if t in present]
    tiers: list[dict[str, Any]] = []
    for tier in ordered_tiers:
        sub = df[df["tier"] == tier]
        tiers.append(
            {
                "tier": str(tier),
                "entities": int(len(sub)),
                "meanValue": round(float(sub["v"].mean()), 4),
                "totalValue": round(float(sub["v"].sum()), 4),
                "valueShare": round(float(sub["v"].sum()) / total_value, 4) if total_value else 0.0,
            }
        )
    # Coverage = share of value held by the top tier(s) — a concentration measure.
    top_share = float(tiers[0]["valueShare"]) if tiers else 0.0
    by_dim: dict[str, list[dict[str, Any]]] = {}
    if dimension_col and "dim" in df.columns:
        for dim_val, g in df.groupby("dim"):
            dim_total = float(g["v"].sum())
            rows_dim = []
            present_g = set(g["tier"].dropna().unique())
            for tier in [t for t in labels_desc if t in present_g]:
                sg = g[g["tier"] == tier]
                rows_dim.append(
                    {
                        "tier": str(tier),
                        "entities": int(len(sg)),
                        "valueShare": round(float(sg["v"].sum()) / dim_total, 4) if dim_total else 0.0,
                    }
                )
            by_dim[str(dim_val)] = rows_dim

    notes.append(f"split {len(df)} entities into {len(tiers)} value tiers")
    return Segmentation(
        tiers=tiers, coverage=top_share, n_entities=int(len(df)),
        by_dimension=by_dim, notes=notes,
    )


def _tier_labels(n: int) -> list[str]:
    if n <= 1:
        return ["all"]
    if n == 2:
        return ["high", "low"]
    if n == 3:
        return ["high", "mid", "low"]
    if n == 4:
        return ["top", "high", "mid", "low"]
    return [f"t{i+1}" for i in range(n)]


def _table_of(model: SemanticModel, value_col: str) -> str:
    for m in model.metrics:
        if m.ref == value_col or m.column == value_col:
            return m.table
    return value_col.split(".", 1)[0] if "." in value_col else value_col


def _fail(msg: str) -> Segmentation:
    return Segmentation(tiers=[], coverage=0.0, n_entities=0, notes=[msg])
