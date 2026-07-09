"""Dataset fingerprinting for model-cache invalidation.

A trained model is only valid for the data it was trained on. The model cache
keys on a ``dataset_sig`` string; if that signature does not change when the
underlying data changes, stale models get served silently. This module computes
a cheap, deterministic fingerprint of a ``DataSource`` — table names, row counts,
column schema (name + type + role), and a lightweight content checksum per table
— so any material change to the data produces a new signature and forces a
retrain on next use.

The content checksum is intentionally cheap: it hashes per-column aggregates
(count, distinct, min/max/sum where numeric) rather than every row, so it scales
to large tables while still catching value changes. Set
``ANALYTICS_FINGERPRINT_CONTENT=0`` to fingerprint schema + row counts only.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_quote
from demos.analytics.src.analytics.profiler import is_numeric

_CONTENT = os.getenv("ANALYTICS_FINGERPRINT_CONTENT", "1") not in ("0", "false", "False")


def _table_content_digest(source: DataSource, table: str, columns: list[Any]) -> str:
    """Cheap per-column aggregate digest that changes when values change."""
    parts: list[str] = ["COUNT(*) AS n"]
    for col in columns:
        q = sql_quote(col.name)
        parts.append(f"COUNT(DISTINCT {q}) AS d_{_safe(col.name)}")
        if is_numeric(col.physical_type):
            cd = f"CAST({q} AS DOUBLE)"
            parts.append(f"MIN({cd}) AS mn_{_safe(col.name)}")
            parts.append(f"MAX({cd}) AS mx_{_safe(col.name)}")
            parts.append(f"SUM({cd}) AS sm_{_safe(col.name)}")
    sql = f"SELECT {', '.join(parts)} FROM {sql_quote(table)}"
    try:
        row = source.native_query(sql)
        return json.dumps(row[0] if row else {}, sort_keys=True, default=str)
    except Exception:
        # A backend that can't compute aggregates still gets a stable (schema-only)
        # fingerprint; correctness degrades gracefully to row-count sensitivity.
        return ""


def fingerprint(source: DataSource, *, content: bool | None = None) -> str:
    """Return a deterministic 16-hex-char fingerprint of the dataset."""
    use_content = _CONTENT if content is None else content
    payload: dict[str, Any] = {}
    for t in source.tables():
        cols = [
            {"name": c.name, "type": c.physical_type, "role": getattr(c.role, "value", str(c.role))}
            for c in t.columns
        ]
        entry: dict[str, Any] = {"rows": t.rows, "columns": cols}
        if use_content:
            entry["digest"] = _table_content_digest(source, t.name, list(t.columns))
        payload[t.name] = entry
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
