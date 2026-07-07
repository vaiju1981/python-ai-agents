"""Semantic role classification: assigns a ``ColumnRole`` to each column profile."""

from __future__ import annotations

from demos.analytics.src.analytics.data_source import ColumnRole
from demos.analytics.src.analytics.profiler import ColumnProfile


def classify_role(p: ColumnProfile) -> ColumnRole:
    """Classify a column profile into a semantic role."""
    if "bool-like" in p.signals:
        return ColumnRole.BOOLEAN
    if "date-like" in p.signals:
        return ColumnRole.DATE
    if "epoch-like" in p.signals:
        return ColumnRole.TIMESTAMP
    if "id-like" in p.signals or "leading-zeros" in p.signals:
        return ColumnRole.IDENTIFIER
    from demos.analytics.src.analytics.profiler import is_numeric
    if is_numeric(p.physical_type):
        return ColumnRole.MEASURE_RATIO if _is_ratio(p) else ColumnRole.MEASURE_ADDITIVE
    if p.distinct <= 50 or (p.rows > 0 and p.distinct / p.rows < 0.5):
        return ColumnRole.DIMENSION
    return ColumnRole.TEXT


def _is_ratio(p: ColumnProfile) -> bool:
    if p.min is None or p.max is None:
        return False
    bounded_unit = p.min >= 0 and p.max <= 1.0
    name_lower = p.name.lower()
    named_rate = any(k in name_lower for k in ("percent", "rate", "ratio", "pct", "share"))
    bounded_pct = p.min >= 0 and p.max <= 100.0
    return bounded_unit or (named_rate and bounded_pct)
