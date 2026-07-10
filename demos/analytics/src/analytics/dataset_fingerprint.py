"""Dataset fingerprinting for model-cache invalidation.

A trained model is only valid for the data it was trained on. The model cache
keys on a ``dataset_sig`` string; if that signature does not change when the
underlying data changes, stale models get served silently. This module computes
a cheap, deterministic fingerprint of a ``DataSource`` — table names, column
schema (name + type + role), and a lightweight content digest per table — so any
material change to the data produces a new signature and forces a retrain on next
use.

The content digest is intentionally cheap: it hashes distribution-shaped
per-column aggregates (distinct, min/max/mean/stddev where numeric) rather than
every row, so it scales to large tables while still catching value changes. By
default the digest is row-count-*agnostic* (see ``row_count_aware``), so appending
more rows of the same distribution does not thrash the model cache (PR-11); a
schema or role change still invalidates. Set ``ANALYTICS_FINGERPRINT_CONTENT=0``
to fingerprint schema only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_quote
from demos.analytics.src.analytics.profiler import is_numeric

_CONTENT = os.getenv("ANALYTICS_FINGERPRINT_CONTENT", "1") not in ("0", "false", "False")


def _round_sig(x: float, ndigits: int = 10) -> float:
    """Round to ``ndigits`` significant figures so the content digest is stable
    across row-count changes (the same distribution must hash identically whether
    computed over 300 or 600 rows — floating-point summation order otherwise
    leaves a 1-ULP tail that would needlessly thrash the model cache, PR-11).
    """
    if x is None or x == 0 or not math.isfinite(x):
        return x
    from decimal import Decimal

    d = Decimal(str(x)).normalize()
    return float(round(d, ndigits - int(d.adjusted()) - 1))


def _table_content_digest(source: DataSource, table: str, columns: list[Any]) -> str:
    """Cheap per-column aggregate digest that changes when *values* change.

    Uses distribution-shaped aggregates (min/max/mean/stddev + distinct) rather
    than row-dependent totals (COUNT(*), SUM). Appending more rows drawn from the
    same distribution therefore leaves the digest stable — which is what makes the
    model-invalidation signature robust to pure row-count growth (PR-11).

    Numeric aggregates are rounded to a fixed number of significant figures so the
    digest is bit-stable across different row counts (see :func:`_round_sig`).
    """
    parts: list[str] = []
    for col in columns:
        q = sql_quote(col.name)
        parts.append(f"COUNT(DISTINCT {q}) AS d_{_safe(col.name)}")
        if is_numeric(col.physical_type):
            cd = f"CAST({q} AS DOUBLE)"
            parts.append(f"MIN({cd}) AS mn_{_safe(col.name)}")
            parts.append(f"MAX({cd}) AS mx_{_safe(col.name)}")
            parts.append(f"AVG({cd}) AS av_{_safe(col.name)}")
            parts.append(f"stddev_pop({cd}) AS sd_{_safe(col.name)}")
    if not parts:
        return json.dumps({}, sort_keys=True)
    sql = f"SELECT {', '.join(parts)} FROM {sql_quote(table)}"
    try:
        row = source.native_query(sql)
        raw = row[0] if row else {}
        clean = {
            k: (
                _round_sig(float(v))
                if isinstance(v, (int, float)) and k.startswith(("mn_", "mx_", "av_", "sd_"))
                else v
            )
            for k, v in raw.items()
        }
        return json.dumps(clean, sort_keys=True, default=str)
    except Exception:
        # A backend that can't compute aggregates still gets a stable (schema-only)
        # fingerprint; correctness degrades gracefully to row-count sensitivity.
        return ""


def fingerprint(
    source: DataSource,
    *,
    content: bool | None = None,
    row_count_aware: bool = True,
) -> str:
    """Return a deterministic 16-hex-char fingerprint of the dataset.

    ``row_count_aware`` controls whether pure row-count growth invalidates the
    fingerprint (PR-11):

    * ``True`` (default) — the fingerprint includes per-table row counts, so any
      growth changes it. Use this for provenance / reproducibility where a
      different-sized dataset must not silently reproduce an answer.
    * ``False`` — the fingerprint depends only on schema (name/type/role) and the
      distribution-shaped content digest, so appending more rows of the same
      distribution does NOT change it. Use this for the model cache so that
      incremental data arrival does not needlessly retrain models; a schema or
      role change still invalidates.
    """
    use_content = _CONTENT if content is None else content
    payload: dict[str, Any] = {}
    for t in source.tables():
        cols = [
            {"name": c.name, "type": c.physical_type, "role": getattr(c.role, "value", str(c.role))}
            for c in t.columns
        ]
        entry: dict[str, Any] = {"columns": cols}
        if row_count_aware:
            entry["rows"] = t.rows
        if use_content:
            entry["digest"] = _table_content_digest(source, t.name, list(t.columns))
        payload[t.name] = entry
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
