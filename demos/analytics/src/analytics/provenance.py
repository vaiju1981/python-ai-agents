"""Answer-provenance envelope + reproducibility (P0 defensibility).

Every governed analytics result is wrapped in a ``ProvenanceEnvelope`` carrying
the SQL that produced it, a deterministic fingerprint of the dataset it ran
against, the row count, a generation timestamp, and the engine version. The
same envelope can deterministically reproduce the answer (re-run from the SQL
+ dataset fingerprint), which is what makes a result audit- and challenge-able.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource
from demos.analytics.src.analytics.dataset_fingerprint import fingerprint

# Bump on any engine change so an envelope's engine_version can flag a result
# produced by an older (possibly buggy) build.
ENGINE_VERSION = "analytics-engine-1.0"


@dataclass
class ProvenanceEnvelope:
    sql: str | None
    dataset_fingerprint: str
    row_count: int | None
    generated_at: str
    engine_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "datasetFingerprint": self.dataset_fingerprint,
            "rowCount": self.row_count,
            "generatedAt": self.generated_at,
            "engineVersion": self.engine_version,
            **self.extra,
        }


def build_envelope(
    source: DataSource,
    sql: str | None = None,
    row_count: int | None = None,
    **extra: Any,
) -> ProvenanceEnvelope:
    """Build a provenance envelope for a result produced against ``source``."""
    try:
        ds_sig = fingerprint(source)
    except Exception:
        ds_sig = "unknown"
    return ProvenanceEnvelope(
        sql=sql,
        dataset_fingerprint=ds_sig,
        row_count=row_count,
        generated_at=datetime.now(timezone.utc).isoformat(),
        engine_version=ENGINE_VERSION,
        extra=extra,
    )


def reproducible(envelope: ProvenanceEnvelope, source: DataSource) -> list[dict[str, Any]]:
    """Re-run a result from its envelope (SQL + dataset fingerprint).

    Recomputes the dataset fingerprint and refuses to reproduce if it no longer
    matches — that is the whole point: a stale dataset cannot silently reproduce
    a previously-valid answer.
    """
    if envelope.sql is None:
        raise ValueError("envelope has no SQL to reproduce")
    current_sig = fingerprint(source)
    if current_sig != envelope.dataset_fingerprint:
        raise ValueError(
            f"dataset fingerprint changed ({envelope.dataset_fingerprint} -> "
            f"{current_sig}); answer is not reproducible on current data"
        )
    return source.native_query(envelope.sql)


def attach(envelope: ProvenanceEnvelope, content: str, data: Any = None,
           error: bool = False) -> Any:
    """Build a ToolResult carrying the provenance envelope.

    Imported lazily to avoid a hard dependency on the core package at module
    import time (the analytics package is also usable standalone).
    """
    from python_ai_agents.core.tool import ToolResult

    return ToolResult(content=content, data=data, error=error,
                      provenance=envelope.to_dict())


def envelope_from_dict(d: dict[str, Any]) -> ProvenanceEnvelope:
    extra = {k: v for k, v in d.items()
             if k not in ("sql", "datasetFingerprint", "rowCount", "generatedAt", "engineVersion")}
    return ProvenanceEnvelope(
        sql=d.get("sql"),
        dataset_fingerprint=d.get("datasetFingerprint", ""),
        row_count=d.get("rowCount"),
        generated_at=d.get("generatedAt", ""),
        engine_version=d.get("engineVersion", ""),
        extra=extra,
    )
