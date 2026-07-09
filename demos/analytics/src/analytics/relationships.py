"""Relationship discovery: finds join paths between tables.

Discovers single-column and composite (multi-column) join keys by testing
overlap between columns across tables. Single-column matches that aren't
confident keys become seeds for composite key search.
"""

from __future__ import annotations

import itertools
import os
import time
from typing import Any

from demos.analytics.src.analytics.data_source import (
    ColumnRole,
    DataSource,
    Relationship,
    sql_qcol,
    sql_quote,
)

MIN_COVERAGE = 0.8
MIN_PARTIAL_COVERAGE = 0.05
# Maximum number of columns in a composite join key. Raised past 2 so tables
# that only join on a (e.g.) (store, product, day) tuple are discovered.
MAX_COMPOSITE_KEY_SIZE = int(os.getenv("ANALYTICS_MAX_COMPOSITE_KEY_SIZE", "3"))

# Global wall-clock budget for relationship discovery. At hundreds of tables the
# pairwise column-overlap probing is O(tables^2 * cols^2); once the budget is
# exceeded we stop and return what we have so profiling stays bounded.
_DISCOVERY_BUDGET_SECONDS = float(os.getenv("ANALYTICS_DISCOVERY_BUDGET_SECONDS", "30"))


def _is_join_candidate(
    role: ColumnRole, distinct: int, rows: int
) -> bool:
    """A column is worth testing as a join key unless it is a measure or a
    high-cardinality free-text column. Identifiers, dimensions, booleans, and
    dates are always candidates (they are the usual join keys); only high-card
    TEXT — which dominates the pairwise cost and almost never joins — is dropped.
    """
    if role in (ColumnRole.MEASURE_ADDITIVE, ColumnRole.MEASURE_RATIO):
        return False
    if role == ColumnRole.TEXT and rows > 0 and (distinct / rows) >= 0.5:
        return False
    return True


def discover(
    source: DataSource,
    roles_by_table: dict[str, dict[str, ColumnRole]],
    stats_by_ref: dict[str, Any],
) -> list[Relationship]:
    """Discover join relationships between tables."""
    confident: list[Relationship] = []
    coarse_by_pair: dict[str, Relationship] = {}
    seeds: dict[str, list[tuple[str, str, str, str]]] = {}

    start = time.monotonic()
    candidates: dict[str, set[str]] = {}
    for t, roles in roles_by_table.items():
        cands: set[str] = set()
        for c, role in roles.items():
            stat = stats_by_ref.get(f"{t}.{c}")
            distinct = int(getattr(stat, "distinct", 0) or 0)
            rows = int(getattr(stat, "rows", 0) or 0)
            if _is_join_candidate(role, distinct, rows):
                cands.add(c)
        candidates[t] = cands

    tables = list(roles_by_table.keys())
    for i, t1 in enumerate(tables):
        if time.monotonic() - start > _DISCOVERY_BUDGET_SECONDS:
            break
        for t2 in tables[i + 1 :]:
            if time.monotonic() - start > _DISCOVERY_BUDGET_SECONDS:
                break
            for c1_name in candidates.get(t1, set()):
                c1_role = roles_by_table[t1][c1_name]
                if c1_role in (ColumnRole.MEASURE_ADDITIVE, ColumnRole.MEASURE_RATIO):
                    continue
                for c2_name in candidates.get(t2, set()):
                    c2_role = roles_by_table[t2][c2_name]
                    if c2_role in (ColumnRole.MEASURE_ADDITIVE, ColumnRole.MEASURE_RATIO):
                        continue
                    rel = _test_pair(source, t1, c1_name, t2, c2_name)
                    if rel is None:
                        continue
                    if rel.coverage >= MIN_COVERAGE:
                        confident.append(rel)
                        # A confident but non-unique (fan-out) single-column
                        # match is also a seed: a composite key built from
                        # several such columns may be selective enough to avoid
                        # the many-to-many fan-out.
                        if rel.cardinality == "many_to_many":
                            seeds.setdefault(f"{t1}\0{t2}", []).append(
                                (t1, c1_name, t2, c2_name)
                            )
                    elif rel.coverage >= MIN_PARTIAL_COVERAGE:
                        pair_key = f"{t1}\0{t2}"
                        coarse_by_pair[pair_key] = rel
                        seeds.setdefault(pair_key, []).append((t1, c1_name, t2, c2_name))

    composites = _discover_composite(source, roles_by_table, seeds, stats_by_ref, start)
    superseded = set(composites.keys())

    result = list(confident)
    for pair_key, rel in coarse_by_pair.items():
        if pair_key not in superseded:
            result.append(rel)
    for composite_rels in composites.values():
        result.extend(composite_rels)
    return result


def _test_pair(
    source: DataSource,
    t1: str,
    c1: str,
    t2: str,
    c2: str,
) -> Relationship | None:
    """Test if columns from two tables can join using a JOIN-based approach."""
    q1 = sql_qcol(t1, c1)
    q2 = sql_qcol(t2, c2)
    try:
        # Stats for t1
        t1_stats = source.native_query(
            f"SELECT COUNT(*) AS total, "
            f"COUNT({q1}) AS non_null, "
            f"COUNT(DISTINCT {q1}) AS distinct_val "
            f"FROM {sql_quote(t1)}"
        )[0]
        # Stats for t2
        t2_stats = source.native_query(
            f"SELECT COUNT(DISTINCT {q2}) AS distinct_val FROM {sql_quote(t2)}"
        )[0]
        # Overlapping distinct values via JOIN
        overlap = source.native_query(
            f"SELECT COUNT(*) AS matched FROM ("
            f"SELECT DISTINCT a.{sql_quote(c1)} AS v "
            f"FROM {sql_quote(t1)} a "
            f"WHERE a.{sql_quote(c1)} IS NOT NULL "
            f"INTERSECT "
            f"SELECT DISTINCT b.{sql_quote(c2)} AS v "
            f"FROM {sql_quote(t2)} b "
            f"WHERE b.{sql_quote(c2)} IS NOT NULL"
            f") AS overlap_result"
        )[0]
    except Exception:
        return None

    t1_non_null = int(t1_stats.get("non_null", 0))
    t1_rows = int(t1_stats.get("total", 0)) or 0
    t1_distinct = int(t1_stats.get("distinct_val", 0))
    t2_distinct = int(t2_stats.get("distinct_val", 0))
    matched = int(overlap.get("matched", 0))

    if t1_non_null == 0 or t1_distinct == 0:
        return None
    coverage = matched / t1_distinct if t1_distinct > 0 else 0
    if coverage < MIN_PARTIAL_COVERAGE:
        return None

    # Cardinality is derived from key uniqueness on each side, not just raw
    # distinct-value counts: a key that is unique on both sides is one-to-one;
    # a key non-unique on the from-side only is many-to-one; non-unique on both
    # is many-to-many (fan-out risk for feature assembly).
    from_unique = t1_distinct >= t1_rows * 0.999 if t1_rows else False
    to_unique = t2_distinct >= t1_rows  # conservative: to-key at least as unique as from
    cardinality = _cardinality(from_unique, to_unique)

    return Relationship(
        from_table=t1,
        from_columns=(c1,),
        to_table=t2,
        to_columns=(c2,),
        cardinality=cardinality,
        coverage=coverage,
    )


def _cardinality(from_unique: bool, to_unique: bool) -> str:
    """Map key-uniqueness on each side to an explicit join cardinality."""
    if from_unique and to_unique:
        return "one_to_one"
    if from_unique and not to_unique:
        return "one_to_one"
    if (not from_unique) and to_unique:
        return "many_to_one"
    return "many_to_many"


def _discover_composite(
    source: DataSource,
    roles_by_table: dict[str, dict[str, ColumnRole]],
    seeds: dict[str, list[tuple[str, str, str, str]]],
    stats_by_ref: dict[str, Any],
    start: float | None = None,
) -> dict[str, list[Relationship]]:
    """Search for composite (multi-column) join keys up to ``MAX_COMPOSITE_KEY_SIZE``.

    Seeds are single-column partial-coverage pairs. Each pair that produced at
    least two seed columns is expanded into combinations of size 2..N (matching
    column counts on both sides) and each candidate composite key is tested. The
    global discovery budget bounds the work.
    """
    result: dict[str, list[Relationship]] = {}
    for pair_key, seed_list in seeds.items():
        if len(seed_list) < 2:
            continue
        t1, c1a, t2, c2a = seed_list[0]
        # Distinct seed columns per side.
        cols1: list[str] = []
        cols2: list[str] = []
        for _, c1, _, c2 in seed_list:
            if c1 not in cols1:
                cols1.append(c1)
            if c2 not in cols2:
                cols2.append(c2)
        if c1a not in cols1:
            cols1.insert(0, c1a)
        if c2a not in cols2:
            cols2.insert(0, c2a)

        max_k = min(MAX_COMPOSITE_KEY_SIZE, len(cols1), len(cols2))
        tested = 0
        # Cap candidate combinations per pair so the search stays bounded even
        # when many columns look join-like (e.g. wide dimension tables).
        PER_PAIR_CAP = int(os.getenv("ANALYTICS_COMPOSITE_PAIR_CAP", "50"))
        for k in range(2, max_k + 1):
            for combo1 in itertools.combinations(cols1, k):
                if tested >= PER_PAIR_CAP:
                    break
                for combo2 in itertools.combinations(cols2, k):
                    if tested >= PER_PAIR_CAP:
                        break
                    # Same-named columns are a valid join across *different*
                    # tables; only skip identical column sets on a self-join.
                    if t1 == t2 and any(a == b for a, b in zip(combo1, combo2)):
                        continue
                    tested += 1
                    rel = _test_composite(source, t1, list(combo1), t2, list(combo2))
                    if rel and rel.coverage >= MIN_COVERAGE:
                        result.setdefault(pair_key, []).append(rel)
            if tested >= PER_PAIR_CAP:
                break
            if start is not None and time.monotonic() - start > _DISCOVERY_BUDGET_SECONDS:
                break
    return result


def _test_composite(
    source: DataSource,
    t1: str,
    cols1: list[str],
    t2: str,
    cols2: list[str],
) -> Relationship | None:
    """Test a composite key join, with explicit cardinality from key uniqueness."""
    q1 = ", ".join(sql_qcol(t1, c) for c in cols1)
    q2 = ", ".join(sql_qcol(t2, c) for c in cols2)
    try:
        result = source.native_query(
            f"SELECT COUNT(*) AS total, "
            f"COUNT(DISTINCT ({q1})) AS from_distinct, "
            f"COUNT(*) FILTER (WHERE ({q1}) IN (SELECT {q2} FROM {sql_quote(t2)})) AS matched "
            f"FROM {sql_quote(t1)}"
        )[0]
        to_distinct = source.native_query(
            f"SELECT COUNT(DISTINCT ({q2})) AS d FROM {sql_quote(t2)}"
        )[0]
    except Exception:
        return None
    total = int(result.get("total", 0))
    matched = int(result.get("matched", 0))
    from_distinct = int(result.get("from_distinct", 0))
    to_distinct_v = int(to_distinct.get("d", 0))
    if total == 0:
        return None
    coverage = matched / total
    if coverage < MIN_PARTIAL_COVERAGE:
        return None
    from_unique = from_distinct >= total * 0.999
    to_unique = to_distinct_v >= total * 0.999
    cardinality = _cardinality(from_unique, to_unique)
    return Relationship(
        from_table=t1,
        from_columns=tuple(cols1),
        to_table=t2,
        to_columns=tuple(cols2),
        cardinality=cardinality,
        coverage=coverage,
    )
