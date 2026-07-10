"""Ingestion orchestration for the analytics engine (PR-13).

Wraps the ``DataSource`` write seam (PR-12) with the production concerns a
firehose needs:

* **Key / time-column resolution** — from an explicit argument, else from the
  ``SemanticModel`` (identifier columns / time columns), else from the source's
  heuristics (``DataSource.primary_keys`` / ``DataSource.time_column``).
* **Idempotent upsert with late-arrival handling** — delegates to
  ``DataSource.upsert``, which tracks a per-table high-water mark and rejects
  rows older than ``watermark - late_window``.
* **Observability** — emits ``analytics.ingest.rows`` / ``late_rejected`` /
  ``errors`` / ``reconcile_diff`` via the ``metrics`` facade (PR-4).
* **Automated reconcile** — optionally compares the ingested batch (or a
  declared source-of-truth metric) against an expected control total and emits a
  reconcile diff + breach counter.

This is internal, server-side code. It is NOT exposed as a user-facing tool;
the read-only posture of ``run_sql`` / ``dsl_query`` is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from demos.analytics.src.analytics.data_source import DataSource, UpsertResult
from demos.analytics.src.analytics.metrics import inc, set_gauge

if TYPE_CHECKING:
    from demos.analytics.src.analytics.semantic_model import SemanticModel


class IngestController:
    """Drive incremental ingestion with keys, watermark, metrics, and reconcile."""

    def __init__(
        self,
        source: DataSource,
        model: SemanticModel | None = None,
        *,
        late_window: float | None = None,
        sink: Any | None = None,
    ) -> None:
        self.source = source
        self.model = model
        self.late_window = late_window
        if sink is not None:
            # Install a controller-scoped metrics sink (e.g. InMemoryMetricsSink
            # in tests, or a Prometheus/OTLP sink in production).
            from demos.analytics.src.analytics.metrics import set_sink

            set_sink(sink)

    # -- resolution ----------------------------------------------------------

    def _resolve_keys(self, table: str, keys: list[str] | None) -> list[str]:
        if keys:
            return list(keys)
        model_keys = self._model_keys(table)
        if model_keys:
            return model_keys
        return self.source.primary_keys(table)

    def _model_keys(self, table: str) -> list[str]:
        if self.model is None:
            return []
        out: list[str] = []
        for d in getattr(self.model, "dimensions", []):
            if getattr(d, "table", None) != table:
                continue
            role = getattr(d, "role", None)
            role_name = role.value if hasattr(role, "value") else str(role)
            if role_name == "identifier":
                out.append(d.column)
        return out

    def _resolve_time_column(self, table: str, time_column: str | None) -> str | None:
        if time_column:
            return time_column
        if self.model is not None:
            for tc in getattr(self.model, "time_columns", []):
                if getattr(tc, "table", None) == table:
                    return tc.column
        return self.source.time_column(table)

    # -- ingest ---------------------------------------------------------------

    def ingest(
        self,
        table: str,
        rows: list[dict[str, Any]],
        *,
        mode: str = "upsert",
        keys: list[str] | None = None,
        time_column: str | None = None,
        expected_rows: int | None = None,
        reconcile: tuple[str, float, float] | None = None,
    ) -> UpsertResult:
        """Ingest ``rows`` into ``table`` and emit ingest metrics.

        Args:
            mode: ``"upsert"`` (keyed merge + watermark, default) or ``"append"``.
            keys: explicit key columns; else resolved from the model or source.
            time_column: explicit event-time column; else resolved.
            expected_rows: optional row-count control total for automated reconcile.
            reconcile: optional ``(metric_ref, expected, tolerance)`` for a
                metric-level control total (requires a model).
        """
        keys = self._resolve_keys(table, keys)
        tcol = self._resolve_time_column(table, time_column)
        tags = {"table": table, "mode": mode}
        try:
            if mode == "append":
                written = self.source.append_rows(table, rows)
                result = UpsertResult(inserted=written, watermark=None)
            elif mode == "upsert":
                if not keys:
                    raise ValueError(
                        "upsert requires key columns (pass keys= or provide a model "
                        "with identifier columns)"
                    )
                result = self.source.upsert(
                    table, rows, keys, time_column=tcol, late_window=self.late_window
                )
            else:
                raise ValueError(f"unknown ingest mode: {mode!r}")
        except Exception:
            inc("analytics.ingest.errors", tags=tags)
            raise

        inc("analytics.ingest.rows", tags=tags, value=result.written)
        if result.late_rejected:
            inc("analytics.ingest.late_rejected", tags=tags, value=result.late_rejected)
        self._reconcile(table, result, expected_rows, reconcile, tags)
        return result

    # -- automated reconcile --------------------------------------------------

    def _reconcile(
        self,
        table: str,
        result: UpsertResult,
        expected_rows: int | None,
        reconcile_arg: tuple[str, float, float] | None,
        tags: dict[str, str],
    ) -> None:
        if expected_rows is not None:
            diff = abs(float(expected_rows) - result.written)
            status = "pass" if diff == 0 else "fail"
            set_gauge(
                "analytics.ingest.reconcile_diff",
                diff,
                tags={**tags, "kind": "row_count", "status": status},
            )
            if status == "fail":
                inc("analytics.ingest.reconcile_breach", tags={**tags, "kind": "row_count"})

        if reconcile_arg is not None:
            if self.model is None:
                return
            metric_ref, expected, tolerance = reconcile_arg
            from demos.analytics.src.analytics.reconcile import reconcile

            res = reconcile(self.source, self.model, metric_ref, float(expected), float(tolerance))
            set_gauge(
                "analytics.ingest.reconcile_diff",
                abs(res.computed - res.expected),
                tags={**tags, "kind": "metric", "metric": metric_ref, "status": res.status},
            )
            if res.status == "fail":
                inc(
                    "analytics.ingest.reconcile_breach",
                    tags={**tags, "kind": "metric", "metric": metric_ref},
                )
