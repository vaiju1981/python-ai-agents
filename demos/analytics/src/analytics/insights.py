"""Deterministic dataset insights — the "what matters" a customer expects on load.

Every number comes from a DuckDB query over the source (no LLM), so insights are
fast and truthful. Identifiers used in SQL come from the profile/semantic model,
never from user input, so string interpolation here is safe.
"""

from __future__ import annotations

from dataclasses import dataclass

from demos.analytics.src.analytics.charts import ChartSpec, choose_chart
from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.profiler import DatasetProfile
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
    TimeColumn,
)


@dataclass(frozen=True, slots=True)
class Insight:
    title: str
    detail: str
    chart: ChartSpec | None = None
    rows: list[dict] | None = None  # data behind ``chart``, so the UI can render it


def generate_insights(
    source: DataSource,
    model: SemanticModel,
    profile: DatasetProfile,
    *,
    max_metrics: int = 4,
    top_n: int = 6,
) -> list[Insight]:
    """Produce a short, ranked list of deterministic insights about the dataset."""
    out: list[Insight] = [_overview(model, profile)]
    dq = _data_quality(profile)
    if dq is not None:
        out.append(dq)
    for metric in model.metrics[:max_metrics]:
        out.extend(_metric_insights(source, model, profile, metric, top_n))
    return out


def _overview(model: SemanticModel, profile: DatasetProfile) -> Insight:
    n_rows = sum(t.rows for t in profile.tables)
    return Insight(
        title="Dataset overview",
        detail=(
            f"{len(profile.tables)} table(s), {n_rows:,} total rows, "
            f"{len(model.metrics)} metric(s), {len(model.dimensions)} dimension(s), "
            f"{len(model.relationships)} relationship(s)."
        ),
    )


def _data_quality(profile: DatasetProfile) -> Insight | None:
    issues: list[tuple[float, str]] = []
    for c in profile.columns:
        if c.rows and c.nulls:
            pct = c.nulls / c.rows
            if pct >= 0.1:
                issues.append((pct, f"{c.table}.{c.name} ({pct:.0%} null)"))
    if not issues:
        return None
    issues.sort(reverse=True)
    worst = ", ".join(s for _, s in issues[:5])
    return Insight(title="Data quality", detail=f"Columns with notable missing data: {worst}.")


def _metric_insights(
    source: DataSource,
    model: SemanticModel,
    profile: DatasetProfile,
    metric: Metric,
    top_n: int,
) -> list[Insight]:
    out: list[Insight] = []
    tq = sql_quote(metric.table)
    mq = sql_qcol(metric.table, metric.column)
    agg = metric.aggregation.upper()

    # Overall aggregate
    total = _scalar(source, f"SELECT {agg}({mq}) AS v FROM {tq}")
    if total is not None:
        out.append(
            Insight(
                title=f"{metric.column} — overall",
                detail=f"{agg.title()} of {metric.column} across {metric.table}: {_fmt(total)}",
            )
        )

    # Trend over the table's time column, if any
    tc = next((t for t in model.time_columns if t.table == metric.table), None)
    if tc is not None:
        series = _trend_rows(source, tc, metric)
        if len(series) >= 2:
            first_v, last_v = _num(series[0].get("value")), _num(series[-1].get("value"))
            out.append(
                Insight(
                    title=f"{metric.column} — trend",
                    detail=(
                        f"{metric.column} is {_direction(first_v, last_v)} over time "
                        f"({_fmt(first_v)} → {_fmt(last_v)} across {len(series)} periods)."
                    ),
                    chart=choose_chart(series, title=f"{metric.column} over time"),
                    rows=series,
                )
            )

    # Breakdown by a low-cardinality dimension in the same table
    dim = _pick_dimension(model, profile, metric.table)
    if dim is not None:
        rows = _breakdown_rows(source, dim, metric, top_n)
        if rows:
            head = ", ".join(
                f"{r.get(dim.column)} ({_fmt(r.get(metric.column))})" for r in rows[:3]
            )
            out.append(
                Insight(
                    title=f"{metric.column} by {dim.column}",
                    detail=f"Top {dim.column} by {metric.column}: {head}",
                    chart=choose_chart(rows, title=f"{metric.column} by {dim.column}"),
                    rows=rows,
                )
            )
    return out


def _trend_rows(source: DataSource, tc: TimeColumn, metric: Metric) -> list[dict]:
    ts = tc.to_timestamp_sql(sql_qcol(tc.table, tc.column))
    mq = sql_qcol(metric.table, metric.column)
    sql = (
        f"SELECT date_trunc('month', {ts})::date AS period, "
        f"{metric.aggregation.upper()}({mq}) AS value "
        f"FROM {sql_quote(metric.table)} WHERE {ts} IS NOT NULL GROUP BY period ORDER BY period"
    )
    try:
        return source.native_query_with_limit(sql, 60)
    except Exception:
        return []


def _breakdown_rows(source: DataSource, dim: Dimension, metric: Metric, top_n: int) -> list[dict]:
    dq = sql_qcol(dim.table, dim.column)
    mq = sql_qcol(metric.table, metric.column)
    sql = (
        f"SELECT {dq} AS {sql_quote(dim.column)}, "
        f"{metric.aggregation.upper()}({mq}) AS {sql_quote(metric.column)} "
        f"FROM {sql_quote(metric.table)} WHERE {dq} IS NOT NULL "
        f"GROUP BY {dq} ORDER BY 2 DESC LIMIT {max(1, top_n)}"
    )
    try:
        return source.native_query_with_limit(sql, top_n)
    except Exception:
        return []


def _pick_dimension(model: SemanticModel, profile: DatasetProfile, table: str) -> Dimension | None:
    distinct = {(c.table, c.name): c.distinct for c in profile.columns}
    best: Dimension | None = None
    best_card = 1 << 60
    for d in model.dimensions:
        if d.table != table:
            continue
        n = distinct.get((d.table, d.column), 0)
        if 2 <= n <= 50 and n < best_card:
            best, best_card = d, n
    return best


def _scalar(source: DataSource, sql: str) -> float | None:
    try:
        rows = source.native_query(sql)
    except Exception:
        return None
    return _num(rows[0].get("v")) if rows else None


def _fmt(v: object) -> str:
    n = _num(v)
    if n is None:
        return str(v)
    if abs(n) >= 1000:
        return f"{n:,.0f}"
    return f"{n:,.2f}".rstrip("0").rstrip(".")


def _num(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _direction(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "changing"
    if b > a * 1.02:
        return "trending up"
    if b < a * 0.98:
        return "trending down"
    return "roughly flat"
