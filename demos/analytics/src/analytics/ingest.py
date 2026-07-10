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
* **Connectors + micro-batch runner** (PR-15) — drain a file/Kafka/CDC feed of
  delta batches into the engine, emitting per-batch metrics and committing
  offsets; replay/backfill is idempotent via the upsert key-dedup (PR-13).

This is internal, server-side code. It is NOT exposed as a user-facing tool;
the read-only posture of ``run_sql`` / ``dsl_query`` is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from demos.analytics.src.analytics.data_source import DataSource, UpsertResult
from demos.analytics.src.analytics.metrics import inc, observe, set_gauge

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


# --- connectors + micro-batch runner (PR-15) -------------------------------


@dataclass
class DeltaBatch:
    """A batch of rows for one table, as polled from a connector."""

    table: str
    rows: list[dict[str, Any]]
    source: str = ""  # provenance (filename, topic+offset, ...)


class Connector(Protocol):
    """A pluggable source of delta batches for the firehose.

    Implementations: ``FileDeltaConnector`` (nightly delta files),
    ``KafkaConnector`` / ``KinesisConnector`` (dependency-gated streaming),
    ``CdcConnector`` (arbitrary change feed). Each ``poll`` returns the batches
    available since the last ``commit``; the runner ingests them and calls
    ``commit`` to advance offsets.
    """

    def poll(self) -> list[DeltaBatch]: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


class FileDeltaConnector:
    """Watch a directory for ``*.delta.csv`` / ``*.delta.parquet`` files.

    The table name is taken from the filename stem before ``.delta`` (e.g.
    ``sales.delta.csv`` -> table ``sales``). ``poll`` returns batches for files
    not yet processed; ``commit`` marks them processed so a later poll (or a
    replay) does not reprocess them. Row-level dedup (PR-13) also makes replay
    idempotent even if a file is re-read.
    """

    def __init__(self, directory: str | Path, *, pattern: str = "*.delta.csv") -> None:
        self._dir = Path(directory)
        self._pattern = pattern
        self._processed: set[str] = set()
        self._pending: set[str] = set()

    def _table_for(self, path: Path) -> str:
        stem = path.stem  # e.g. "sales.delta"
        if stem.endswith(".delta"):
            stem = stem[: -len(".delta")]
        return stem

    def poll(self) -> list[DeltaBatch]:
        batches: list[DeltaBatch] = []
        for path in sorted(self._dir.glob(self._pattern)):
            key = str(path)
            if key in self._processed or key in self._pending:
                continue
            rows = self._read(path)
            self._pending.add(key)
            batches.append(DeltaBatch(table=self._table_for(path), rows=rows, source=key))
        return batches

    def commit(self) -> None:
        self._processed |= self._pending
        self._pending.clear()

    def close(self) -> None:
        pass

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        try:
            import pandas as pd

            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            return df.where(pd.notna(df), None).to_dict("records")  # type: ignore[return-value]
        except ImportError:
            if path.suffix == ".parquet":
                raise RuntimeError(
                    "parquet delta requires pandas/pyarrow; install the analytics-demo deps"
                )
            import csv

            with open(path, newline="", encoding="utf-8") as fh:
                return [dict(r) for r in csv.DictReader(fh)]


class KafkaConnector:
    """Optional Kafka micro-batch connector (dependency-gated).

    Requires the ``kafka`` package (not in the ``analytics-demo`` extra by
    default). Polls a topic and emits one ``DeltaBatch`` per poll; ``commit``
    advances the consumer offset.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str,
        group_id: str = "analytics-ingest",
        *,
        table: str | None = None,
        max_records: int = 1000,
        timeout_ms: int = 1000,
    ) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise RuntimeError(
                "KafkaConnector requires the 'kafka' package; install it (not in the "
                "analytics-demo extra by default)."
            )
        self._topic = topic
        self._table = table or topic
        self._max_records = max_records
        self._timeout_ms = timeout_ms
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )

    def poll(self) -> list[DeltaBatch]:
        msgs = self._consumer.poll(timeout_ms=self._timeout_ms, max_records=self._max_records)
        rows: list[dict[str, Any]] = []
        for records in msgs.values():
            for rec in records:
                value = rec.value
                if isinstance(value, (bytes, bytearray)):
                    value = bytes(value).decode("utf-8")
                rows.append(json.loads(value))
        return [DeltaBatch(table=self._table, rows=rows, source=self._topic)] if rows else []

    def commit(self) -> None:
        self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()


class CdcConnector:
    """Wrap an arbitrary change-feed callable into a ``Connector`` (PR-13 upsert).

    ``feed()`` returns a list of row dicts for ``table`` — handy for a
    JDBC/Debezium-style feed, a generator, or tests. Each poll yields one batch.
    """

    def __init__(self, table: str, feed: Any, *, source: str = "cdc") -> None:
        self._table = table
        self._feed = feed
        self._source = source

    def poll(self) -> list[DeltaBatch]:
        rows = list(self._feed())
        return [DeltaBatch(table=self._table, rows=rows, source=self._source)] if rows else []

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


class MicroBatchRunner:
    """Drain a ``Connector`` into the analytics engine as micro-batches (PR-15).

    Loop: poll -> ingest each batch via ``AnalyticsToolset.ingest`` (PR-13 upsert
    + PR-14 refresh) -> emit per-batch metrics -> run an optional freshness gate
    (PR-16) -> commit offsets. Backfill/replay is the same loop over a historical
    connector; row-level dedup (PR-13) makes replay idempotent.

    Concurrency: the runner owns the single writer (DuckDB is single-writer);
    read queries continue to use the source's per-query read-only connections, so
    ingestion and serving do not contend on the same connection.
    """

    def __init__(self, toolset: Any, *, freshness_check: Any | None = None, tags: dict[str, str] | None = None) -> None:
        self.toolset = toolset
        self.freshness_check = freshness_check
        self.tags = tags or {}

    def process(self, connector: Connector) -> int:
        """Poll one batch cycle, ingest each batch, commit, and return batch count."""
        batches = connector.poll()
        for batch in batches:
            btags = {**self.tags, "table": batch.table}
            try:
                # Coerce raw (often string-typed) delta rows to the live table's
                # schema so temporal/numeric columns compare and insert correctly.
                batch.rows = self._coerce_rows(batch.table, batch.rows)
                result = self.toolset.ingest(batch.table, batch.rows)
                inc("analytics.ingest.batch.rows", tags=btags, value=result.written)
                self._emit_lag(batch, btags)
            except Exception:
                inc("analytics.ingest.batch.errors", tags=btags)
                raise
            if self.freshness_check is not None:
                self.freshness_check()
        # Commit offsets only when we actually drained a batch cycle, so an
        # empty poll does not advance (or redundantly rewrite) the offset marker.
        if batches:
            connector.commit()
        return len(batches)

    def run(self, connector: Connector, *, max_batches: int | None = None) -> int:
        """Run the loop until the connector is drained (or ``max_batches`` reached)."""
        total = 0
        while True:
            n = self.process(connector)
            total += n
            if n == 0 or (max_batches is not None and total >= max_batches):
                break
        return total

    # -- schema coercion (deltas arrive as loosely-typed text) ----------------

    def _table_schema(self, table: str) -> Any | None:
        for t in self.toolset.source.tables():
            if t.name == table:
                return t
        return None

    def _coerce_rows(self, table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Coerce delta rows to the live table's column types before ingest.

        Files/Kafka deliver strings; the live table (seeded by DuckDB's CSV
        auto-detection) usually has typed columns (DATE / TIMESTAMP / numeric).
        Casting here keeps the upsert watermark comparison and the INSERT correct
        without re-reading the file in the database. Columns not yet present in
        the schema (new columns) are left as-is — schema evolution handles those
        (PR-16).
        """
        schema = None
        ts = self._table_schema(table)
        if ts is not None:
            schema = {c.name: c.physical_type for c in ts.columns}
        if not schema:
            return rows
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({k: self._coerce_value(v, schema.get(k, "")) for k, v in r.items()})
        return out

    @staticmethod
    def _coerce_value(value: Any, physical_type: str) -> Any:
        if value is None:
            return None
        pt = physical_type.upper()
        if "DATE" in pt and isinstance(value, str):
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                return value
        if "TIMESTAMP" in pt and isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return value
        if any(k in pt for k in ("BIGINT", "INTEGER", "INT", "SMALLINT", "TINYINT", "UBIGINT", "HUGEINT")):
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return value
        if any(k in pt for k in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")):
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
        if "BOOLEAN" in pt and isinstance(value, str):
            return value.strip().lower() in ("1", "true", "t", "yes", "y")
        return value

    def _emit_lag(self, batch: DeltaBatch, btags: dict[str, str]) -> None:
        tcol = self.toolset.source.time_column(batch.table)
        if not tcol or not batch.rows:
            return
        times = [r.get(tcol) for r in batch.rows if r.get(tcol) is not None]
        if not times:
            return
        now = datetime.now()
        try:
            max_t = max(times)
            if isinstance(max_t, (datetime, date)):
                # ``now`` is a (naive) datetime; a DATE max must be promoted to
                # midnight before subtraction, else ``datetime - date`` raises.
                if isinstance(max_t, date) and not isinstance(max_t, datetime):
                    max_t = datetime.combine(max_t, datetime.min.time())
                lag = (now - max_t).total_seconds()
                observe("analytics.ingest.batch.lag_seconds", value=max(0.0, lag), tags=btags)
        except Exception:
            pass
