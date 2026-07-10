"""Column and dataset profiling over any ``DataSource``.

Uses DuckDB SQL (via the source's ``native_query``) for all statistics — fast
and works with any backend that supports standard SQL.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.data_source import (
    ColumnRole,
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
    sql_qtable,
    sql_quote,
)
from demos.analytics.src.analytics.relationships import (
    _DISCOVERY_BUDGET_SECONDS,
)
from demos.analytics.src.analytics.relationships import (
    discover as discover_relationships,
)

# Wall-clock budget for relationship discovery inside an incremental update. We
# reuse the A4 discovery budget so incremental profiling of a growing stream
# stays bounded (PR-11: "tie to the A4 discovery budget").
_INCREMENTAL_DISCOVERY_BUDGET_SECONDS = _DISCOVERY_BUDGET_SECONDS

# When a table is larger than this many rows, profile its statistics on a
# reservoir sample to bound cost. Counts/distinct remain exact. 0 = always full.
_PROFILE_SAMPLE_ROWS = int(os.getenv("ANALYTICS_PROFILE_SAMPLE_ROWS", "0")) or None


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    table: str
    name: str
    physical_type: str
    rows: int
    distinct: int
    nulls: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    stddev: float | None = None
    # Running sufficient statistics (populated for numeric columns) so two
    # profiles can be merged exactly via parallel/Chan variance combination.
    sum: float | None = None
    sumsq: float | None = None
    sample_values: tuple[str, ...] = ()
    signals: frozenset[str] = field(default_factory=frozenset)

    def merge(self, other: ColumnProfile, *, decay: float = 0.0) -> ColumnProfile:
        """Combine this (older) profile with a newer one, returning an updated
        profile that reflects both batches of data.

        ``decay`` (0..1, default 0) discounts the *older* profile before merging
        so drift-sensitive columns can be windowed: ``0.1`` keeps 90% of the old
        weight. With ``decay == 0`` the merge is exact (no windowing).
        """
        return merge_column_profiles(self, other, decay=decay)

    @classmethod
    def from_batch(
        cls,
        table: str,
        col: ColumnSchema,
        values: Iterable[Any],
    ) -> ColumnProfile:
        """Build a profile from an in-memory batch of values (no source query).

        This is the genuinely streaming/incremental path: it costs O(batch) and
        never re-reads the whole table, so it stays within a bounded time budget
        as the stream grows (PR-11).
        """
        vals = list(values)
        rows = len(vals)
        nulls = 0
        numeric = is_numeric(col.physical_type)
        nums: list[float] = []
        distinct_set: set[str] = set()
        samples: list[str] = []
        for v in vals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                nulls += 1
                continue
            s = str(v)
            distinct_set.add(s)
            if len(samples) < 20 and s not in samples:
                samples.append(s)
            fv = _to_float(v) if numeric else None
            if fv is not None:
                nums.append(fv)

        mn = mx = mean = stddev = sum_ = sumsq = None
        if numeric and nums:
            mn = min(nums)
            mx = max(nums)
            sum_ = math.fsum(nums)
            sumsq = math.fsum(x * x for x in nums)
            mean = sum_ / len(nums)
            var = sumsq / len(nums) - mean * mean
            stddev = math.sqrt(var) if var > 0 else 0.0

        signals = _compute_signals(
            tuple(samples), col.name, col.physical_type, rows, len(distinct_set), mn, mx
        )
        return cls(
            table=table,
            name=col.name,
            physical_type=col.physical_type,
            rows=rows,
            distinct=len(distinct_set),
            nulls=nulls,
            min=mn,
            max=mx,
            mean=mean,
            stddev=stddev,
            sum=sum_,
            sumsq=sumsq,
            sample_values=tuple(samples),
            signals=signals,
        )


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    tables: tuple[TableSchema, ...]
    columns: tuple[ColumnProfile, ...]
    relationships: tuple[Relationship, ...]
    import_plan: dict[str, dict[str, str]] = field(default_factory=dict)


_DATE_RE = re.compile(r"\d{4}[/-]\d{2}[/-]\d{2}.*")
_LEADING_ZERO_RE = re.compile(r"0\d+")
_BOOLS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"})


def _compute_signals(
    samples: tuple[str, ...],
    name: str,
    physical_type: str,
    rows: int,
    distinct: int,
    min_val: float | None,
    max_val: float | None,
) -> frozenset[str]:
    """Shared column-signal classifier used by both source profiling and the
    streaming batch profiler so the two paths agree on role hints."""
    signals: set[str] = set()
    if any(_DATE_RE.match(s) for s in samples):
        signals.add("date-like")
    if any(_LEADING_ZERO_RE.match(s) for s in samples):
        signals.add("leading-zeros")
    if distinct == 2 and all(s.lower() in _BOOLS for s in samples):
        signals.add("bool-like")

    name_lower = name.lower()
    numeric = is_numeric(physical_type)
    name_id = bool(re.match(r".*(id|key|code|uuid|guid)$", name_lower))
    high_card = not is_floating_point(physical_type) and rows >= 50 and distinct / rows > 0.95
    if name_id or high_card:
        signals.add("id-like")

    name_time_hint = bool(
        re.search(r"(time|date|epoch|ts|created|updated|timestamp|at)$", name_lower)
    )
    if (
        numeric
        and "id-like" not in signals
        and min_val is not None
        and min_val >= 1e9
        and max_val is not None
        and max_val <= 5e12
        and (distinct > 10 or name_time_hint)
    ):
        signals.add("epoch-like")
    return frozenset(signals)


def profile_column(
    source: DataSource, table: str, col: ColumnSchema, sample_rows: int | None = None
) -> ColumnProfile:
    """Profile a single column using DuckDB SQL."""
    c = sql_quote(col.name)
    t = sql_qtable(table)
    numeric = is_numeric(col.physical_type)

    # Exact row / cardinality / null counts on the full table.
    base = (
        f"SELECT COUNT(*) AS n_rows, COUNT(DISTINCT {c}) AS n_distinct, "
        f"COUNT(*) FILTER (WHERE {c} IS NULL) AS n_nulls"
    )
    exact = source.native_query(f"{base} FROM {t}")[0]
    rows = int(exact.get("n_rows", 0))
    distinct = int(exact.get("n_distinct", 0))
    nulls = int(exact.get("n_nulls", 0))

    # Heavy statistics (min/max/mean/stddev) may run on a reservoir sample of the
    # table to bound cost on very large data; counts above stay exact.
    sample_rows = sample_rows or _PROFILE_SAMPLE_ROWS
    stat_src = (
        f"(SELECT * FROM {t} USING SAMPLE {int(sample_rows)} ROWS)"
        if sample_rows and sample_rows > 0 and rows > sample_rows
        else t
    )
    min_val = max_val = mean = stddev = sum_ = sumsq = None
    if numeric:
        cd = f"CAST({c} AS DOUBLE)"
        stat = _first_ok(
            source,
            [
                f"SELECT MIN({cd}) AS mn, MAX({cd}) AS mx, AVG({cd}) AS av, "
                f"stddev_pop({cd}) AS sd FROM {stat_src}",
                f"SELECT MIN({cd}) AS mn, MAX({cd}) AS mx, AVG({cd}) AS av FROM {stat_src}",
            ],
        )
        min_val = _to_float(stat.get("mn"))
        max_val = _to_float(stat.get("mx"))
        mean = _to_float(stat.get("av"))
        stddev = _to_float(stat.get("sd"))
        if rows > 0 and mean is not None:
            sum_ = mean * rows
            # Recover sum-of-squares from the population variance so merges are
            # exact: var = sumsq/n - mean^2  =>  sumsq = n*(var + mean^2).
            if stddev is not None:
                sumsq = rows * (stddev * stddev + mean * mean)
            else:
                sumsq = rows * mean * mean

    # Sample values
    sample_rows_vals = source.native_query(
        f"SELECT DISTINCT {c} AS v FROM {t} WHERE {c} IS NOT NULL LIMIT 20"
    )
    samples = tuple(str(r["v"]) for r in sample_rows_vals)

    signals = _compute_signals(
        samples, col.name, col.physical_type, rows, distinct, min_val, max_val
    )

    return ColumnProfile(
        table=table,
        name=col.name,
        physical_type=col.physical_type,
        rows=rows,
        distinct=distinct,
        nulls=nulls,
        min=min_val,
        max=max_val,
        mean=mean,
        stddev=stddev,
        sum=sum_,
        sumsq=sumsq,
        sample_values=samples,
        signals=signals,
    )


def profile_dataset(
    source: DataSource,
    catalog: Any | None = None,
    sample_rows: int | None = None,
) -> DatasetProfile:
    """Profile all tables and discover relationships."""
    tables = source.tables()
    columns: list[ColumnProfile] = []
    roles_by_table: dict[str, dict[str, ColumnRole]] = {}
    import_plan: dict[str, dict[str, str]] = {}

    typed_tables: list[TableSchema] = []
    for table in tables:
        typed_cols: list[ColumnSchema] = []
        role_map: dict[str, ColumnRole] = {}
        for col in table.columns:
            cp = profile_column(source, table.name, col, sample_rows)
            columns.append(cp)
            from demos.analytics.src.analytics.semantic_roles import classify_role

            role = classify_role(cp)
            if catalog is not None:
                role = catalog.role_for(table.name, col.name, role)
            typed_cols.append(
                ColumnSchema(name=col.name, physical_type=col.physical_type, role=role)
            )
            role_map[col.name] = role
            if "leading-zeros" in cp.signals:
                import_plan.setdefault(table.name, {})[col.name] = "VARCHAR"
        roles_by_table[table.name] = role_map
        typed_tables.append(
            TableSchema(name=table.name, rows=table.rows, columns=tuple(typed_cols))
        )

    stats_by_ref = {f"{cp.table}.{cp.name}": cp for cp in columns}
    relationships = discover_relationships(source, roles_by_table, stats_by_ref)
    if catalog is not None:
        relationships = catalog.apply(relationships)

    return DatasetProfile(
        tables=tuple(typed_tables),
        columns=tuple(columns),
        relationships=tuple(relationships),
        import_plan=import_plan,
    )


def is_numeric(duck_type: str) -> bool:
    s = duck_type.upper()
    return any(k in s for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC", "HUGEINT"))


def is_floating_point(duck_type: str) -> bool:
    s = duck_type.upper()
    return any(k in s for k in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC"))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_ok(source: DataSource, sqls: list[str]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for sql in sqls:
        try:
            return source.native_query(sql)[0]
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return {}


# -- incremental / streaming profiling (PR-11) --------------------------------


def _apply_decay(cp: ColumnProfile, decay: float) -> ColumnProfile:
    """Discount an *older* profile's weight by ``1 - decay`` for windowing.

    Mean and stddev are scale-invariant, so only the weight-bearing aggregates
    (rows, nulls, sum, sumsq) are discounted; the distribution shape is kept.
    """
    w = 1.0 - decay
    return ColumnProfile(
        table=cp.table,
        name=cp.name,
        physical_type=cp.physical_type,
        rows=int(round(cp.rows * w)),
        distinct=cp.distinct,
        nulls=int(round(cp.nulls * w)),
        min=cp.min,
        max=cp.max,
        mean=cp.mean,
        stddev=cp.stddev,
        sum=(cp.sum * w) if cp.sum is not None else None,
        sumsq=(cp.sumsq * w) if cp.sumsq is not None else None,
        sample_values=cp.sample_values,
        signals=cp.signals,
    )


def merge_column_profiles(
    a: ColumnProfile, b: ColumnProfile, *, decay: float = 0.0
) -> ColumnProfile:
    """Merge two profiles of the *same* column (``a`` older, ``b`` newer).

    Combines exact sufficient statistics via the parallel variance formula, so
    the result equals profiling the union of both batches. ``decay`` windows the
    older profile (see :meth:`ColumnProfile.merge`).
    """
    if a is None:
        return b
    if b is None:
        return a
    if decay and 0.0 < decay < 1.0:
        a = _apply_decay(a, decay)

    rows = a.rows + b.rows
    nulls = a.nulls + b.nulls
    mins = [x for x in (a.min, b.min) if x is not None]
    maxs = [x for x in (a.max, b.max) if x is not None]
    mn = min(mins) if mins else None
    mx = max(maxs) if maxs else None

    if a.sum is not None and b.sum is not None:
        s = a.sum + b.sum
        sq = a.sumsq + b.sumsq
        mean = s / rows if rows else None
        var = (sq / rows - mean * mean) if rows else 0.0
        stddev = math.sqrt(var) if var and var > 0 else 0.0
    else:
        s = sq = mean = stddev = None

    union = set(a.sample_values) | set(b.sample_values)
    distinct = max(a.distinct, b.distinct, len(union))
    sample_values = tuple(dict.fromkeys(list(a.sample_values) + list(b.sample_values)))[:20]
    signals = a.signals | b.signals
    return ColumnProfile(
        table=a.table,
        name=a.name,
        physical_type=a.physical_type,
        rows=rows,
        distinct=distinct,
        nulls=nulls,
        min=mn,
        max=mx,
        mean=mean,
        stddev=stddev,
        sum=s,
        sumsq=sq,
        sample_values=sample_values,
        signals=signals,
    )


def _rows_to_columns(rows: Any) -> dict[str, list[Any]]:
    """Normalize a batch (DataFrame or list[dict]) into per-column value lists."""
    import pandas as pd

    if isinstance(rows, pd.DataFrame):
        out: dict[str, list[Any]] = {}
        for col in rows.columns:
            out[str(col)] = rows[col].tolist()
        return out
    out = {}
    for row in rows:
        for k, v in row.items():
            out.setdefault(k, []).append(v)
    return out


def profile_dataset_incremental(
    source: DataSource,
    *,
    prev: DatasetProfile,
    catalog: Any | None = None,
    new_rows: dict[str, Any] | None = None,
    decay: float = 0.0,
    sample_rows: int | None = None,
) -> DatasetProfile:
    """Update ``prev`` as new data lands, without re-profiling from scratch.

    Two update paths:

    * **Streaming** (``new_rows`` given): each entry is a batch of newly-arrived
      rows for a table (``DataFrame`` or ``list[dict]``). Per-column sufficient
      statistics are updated in O(batch) by merging the batch profile into the
      prior profile (via :func:`merge_column_profiles`). The full table is never
      re-read, so cost stays bounded as the stream grows.
    * **Refresh** (``new_rows`` is ``None``): columns are re-profiled from the
      source and merged into the prior profile (useful when the source changed
      out-of-band but no explicit batch is available).

    Relationship discovery only re-runs when the schema actually changed
    (new/dropped table or column, or a changed type); otherwise the prior
    relationships are reused, which keeps the update within the A4 discovery
    budget. The returned profile carries the same structure, and — because the
    dataset fingerprint is row-count-agnostic (PR-11) — pure row growth does not
    invalidate models keyed on it.
    """
    from demos.analytics.src.analytics.semantic_roles import classify_role

    prev_cols = {(cp.table, cp.name): cp for cp in prev.columns}
    prev_tables = {t.name: t for t in prev.tables}
    prev_schema_key = {(t.name, c.name, c.physical_type) for t in prev.tables for c in t.columns}

    batch_by_table: dict[str, dict[str, list[Any]]] = {}
    if new_rows:
        for table, rows in new_rows.items():
            batch_by_table[table] = _rows_to_columns(rows)

    new_columns: list[ColumnProfile] = []
    new_roles_by_table: dict[str, dict[str, ColumnRole]] = {}
    new_import_plan: dict[str, dict[str, str]] = {}
    new_typed_tables: list[TableSchema] = []
    schema_changed = False

    for table in source.tables():
        typed_cols: list[ColumnSchema] = []
        role_map: dict[str, ColumnRole] = {}
        batch = batch_by_table.get(table.name)
        for col in table.columns:
            prev_cp = prev_cols.get((table.name, col.name))
            if batch is not None and col.name in batch:
                batch_cp = ColumnProfile.from_batch(table.name, col, batch[col.name])
                cp = (
                    batch_cp
                    if prev_cp is None
                    else merge_column_profiles(prev_cp, batch_cp, decay=decay)
                )
            elif new_rows is None:
                fresh = profile_column(source, table.name, col, sample_rows)
                cp = (
                    fresh if prev_cp is None else merge_column_profiles(prev_cp, fresh, decay=decay)
                )
            else:
                # Schema column present in source but not in the provided batch:
                # carry the prior profile forward unchanged.
                cp = (
                    prev_cp
                    if prev_cp is not None
                    else profile_column(source, table.name, col, sample_rows)
                )

            new_columns.append(cp)
            role = classify_role(cp)
            if catalog is not None:
                role = catalog.role_for(table.name, col.name, role)
            typed_cols.append(
                ColumnSchema(name=col.name, physical_type=col.physical_type, role=role)
            )
            role_map[col.name] = role
            if "leading-zeros" in cp.signals:
                new_import_plan.setdefault(table.name, {})[col.name] = "VARCHAR"
        new_roles_by_table[table.name] = role_map
        new_typed_tables.append(
            TableSchema(name=table.name, rows=table.rows, columns=tuple(typed_cols))
        )

        prev_tbl = prev_tables.get(table.name)
        prev_cols_here = set(c.name for c in (prev_tbl.columns if prev_tbl else ()))
        cur_cols_here = set(c.name for c in table.columns)
        if prev_tbl is None or prev_cols_here != cur_cols_here:
            schema_changed = True

    for t in prev_tables:
        if t not in {x.name for x in new_typed_tables}:
            schema_changed = True

    cur_schema_key = {
        (t.name, c.name, c.physical_type) for t in new_typed_tables for c in t.columns
    }
    if cur_schema_key != prev_schema_key:
        schema_changed = True

    if schema_changed:
        stats_by_ref = {f"{cp.table}.{cp.name}": cp for cp in new_columns}
        relationships = discover_relationships(source, new_roles_by_table, stats_by_ref)
        if catalog is not None:
            relationships = catalog.apply(relationships)
    else:
        # Reuse prior relationships unchanged — no extra discovery cost.
        relationships = prev.relationships
        new_import_plan = dict(prev.import_plan)

    return DatasetProfile(
        tables=tuple(new_typed_tables),
        columns=tuple(new_columns),
        relationships=tuple(relationships),
        import_plan=new_import_plan,
    )
