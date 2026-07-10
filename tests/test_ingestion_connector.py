"""PR-15 — streaming connector + micro-batch runner + backfill (E2E).

Covers ``FileDeltaConnector`` (nightly delta files), an in-memory
``CdcConnector`` feed, the ``MicroBatchRunner`` drain loop (per-batch metrics,
offset commit, refresh-after-ingest), and idempotent replay/backfill (the
PR-13 upsert key-dedup means re-delivering the same rows adds nothing extra).

The optional Kafka path is gated behind ``RUN_STREAM_TESTS=1`` and is skipped
in CI without the broker / the ``kafka`` package installed, mirroring the
``RUN_WAREHOUSE_TESTS`` pattern (PR-3).
"""

from __future__ import annotations

import datetime as dt

import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.ingest import (
    CdcConnector,
    DeltaBatch,
    FileDeltaConnector,
    MicroBatchRunner,
)
from demos.analytics.src.analytics.metrics import (
    InMemoryMetricsSink,
    LogMetricsSink,
    set_sink,
)
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
)
from demos.analytics.src.analytics.toolset import AnalyticsToolset


@pytest.fixture
def mem_sink():
    """Install an in-memory metrics sink for the test; restore the default after."""
    sink = InMemoryMetricsSink()
    set_sink(sink)
    yield sink
    set_sink(LogMetricsSink())


def _events_source(tmp_path):
    base = tmp_path / "events.csv"
    base.write_text(
        "id,ts,v\n"
        "1,2025-01-01,10\n"
        "2,2025-01-02,20\n"
        "3,2025-01-03,30\n"
    )
    base2 = tmp_path / "events2.csv"
    base2.write_text(
        "id,ts,v\n"
        "1,2025-02-01,100\n"
        "2,2025-02-02,200\n"
    )
    return CsvSource(named_csvs={"events": base, "events2": base2})


def _events_model():
    return SemanticModel(
        metrics=(Metric(table="events", column="v", aggregation="sum"),),
        dimensions=(
            Dimension(table="events", column="id", role="identifier"),
            Dimension(table="events", column="ts"),
        ),
        entity_keys=(),
        time_columns=(Dimension(table="events", column="ts"),),
        relationships=(),
    )


def _events_toolset(tmp_path):
    src = _events_source(tmp_path)
    ts = AnalyticsToolset(src, _events_model(), profile=profile_dataset(src))
    return src, ts


# --- FileDeltaConnector: low-level poll / commit ----------------------------


def test_file_connector_poll_discovers_deltas_and_commit_advances(tmp_path):
    delta_dir = tmp_path / "deltas"
    delta_dir.mkdir()
    (delta_dir / "events.delta.csv").write_text("id,ts,v\n4,2025-01-04,40\n")
    (delta_dir / "events2.delta.csv").write_text("id,ts,v\n5,2025-01-05,50\n")

    conn = FileDeltaConnector(delta_dir)
    try:
        first = conn.poll()
        # Both files surface as their own table (name parsed from the stem).
        assert {b.table for b in first} == {"events", "events2"}
        assert len(first) == 2
        # A second poll within the same cycle returns nothing new (pending held).
        assert conn.poll() == []
        # Commit marks them processed; a later poll leaves them alone.
        conn.commit()
        assert conn.poll() == []
    finally:
        conn.close()


# --- MicroBatchRunner over FileDeltaConnector (dedup + replay) --------------


def test_runner_ingests_two_deltas_deduped_and_commits(tmp_path, mem_sink):
    src, ts = _events_toolset(tmp_path)
    try:
        delta_dir = tmp_path / "deltas"
        delta_dir.mkdir()
        # Two delta files -> two tables. Each delta (re)delivers a key already in
        # the base table to prove PR-13 upsert dedup through the connector path.
        (delta_dir / "events.delta.csv").write_text(
            "id,ts,v\n1,2025-01-01,99\n4,2025-01-04,40\n5,2025-01-05,50\n"
        )
        (delta_dir / "events2.delta.csv").write_text(
            "id,ts,v\n1,2025-02-01,999\n3,2025-02-03,300\n"
        )
        conn = FileDeltaConnector(delta_dir)
        runner = MicroBatchRunner(ts)

        batches = runner.run(conn)
        # One poll cycle yields two files; both ingested.
        assert batches == 2
        # events: base 3 + new(4,5), id=1 updated not duplicated -> 5 rows.
        assert src.row_count("events") == 5
        # events2: base 2 + new(3), id=1 updated -> 3 rows.
        assert src.row_count("events2") == 3
        assert mem_sink.count("analytics.ingest.batch.rows", table="events") == 3
        assert mem_sink.count("analytics.ingest.batch.rows", table="events2") == 2
        # Dedup: the re-delivered id=1 exists exactly once in each table.
        assert len(src.native_query("SELECT id FROM events WHERE id = 1")) == 1
        assert len(src.native_query("SELECT id FROM events2 WHERE id = 1")) == 1
        # Offsets committed -> a replayed run ingests nothing extra.
        assert runner.run(conn) == 0
        assert src.row_count("events") == 5
        assert src.row_count("events2") == 3
        # Each ingest ran refresh_after_ingest (cache invalidated on ingest).
        assert ts._dsl_engine_cache == {}
    finally:
        ts.source.close()


# --- MicroBatchRunner over a fake in-memory connector -----------------------


class _FakeConnector:
    """Drain-once connector: returns its batches until commit, then empty."""

    def __init__(self, batches: list[DeltaBatch]) -> None:
        self._batches = batches
        self.committed = 0

    def poll(self) -> list[DeltaBatch]:
        if self.committed:
            return []
        return list(self._batches)

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        pass


class _RedeliverConnector:
    """Replay connector: always re-delivers the same batches (offsets not moved)."""

    def __init__(self, batches: list[DeltaBatch]) -> None:
        self._batches = batches

    def poll(self) -> list[DeltaBatch]:
        return list(self._batches)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


def _events_rows(start_id: int, count: int, day: str):
    return [
        {"id": start_id + i, "ts": dt.date.fromisoformat(day), "v": 10 * (start_id + i)}
        for i in range(count)
    ]


def test_runner_drains_fake_connector_emits_metrics_and_refreshes(tmp_path, mem_sink):
    src, ts = _events_toolset(tmp_path)
    try:
        batches = [
            DeltaBatch(table="events", rows=_events_rows(10, 2, "2025-01-10")),
            DeltaBatch(table="events", rows=_events_rows(20, 3, "2025-01-11")),
        ]
        conn = _FakeConnector(batches)
        runner = MicroBatchRunner(ts)

        assert runner.run(conn) == 2  # two batch cycles drained, then empty
        assert conn.committed == 1
        # 3 base + (10,11) + (20,21,22) == 8.
        assert src.row_count("events") == 8
        assert mem_sink.count("analytics.ingest.batch.rows", table="events") == 5
        # refresh_after_ingest ran on each ingest (cache cleared).
        assert ts._dsl_engine_cache == {}
        # Replay with the same connector (offsets not yet advanced): dedup means
        # no new rows land even though the batches are re-delivered.
        assert runner.process(conn) == 0
        assert src.row_count("events") == 8
    finally:
        ts.source.close()


def test_runner_idempotent_replay_does_not_double_count(tmp_path, mem_sink):
    src, ts = _events_toolset(tmp_path)
    try:
        batches = [DeltaBatch(table="events", rows=_events_rows(40, 2, "2025-01-20"))]
        conn = _RedeliverConnector(batches)
        runner = MicroBatchRunner(ts)

        n0 = src.row_count("events")
        runner.process(conn)  # first delivery -> 2 rows
        mid = src.row_count("events")
        runner.process(conn)  # replay -> deduped, no new rows
        assert mid == n0 + 2
        assert src.row_count("events") == mid
        # lag metric emitted for the temporal batch.
        assert mem_sink.values("analytics.ingest.batch.lag_seconds", table="events")
    finally:
        ts.source.close()


def test_runner_emits_lag_seconds_for_temporal_batch(tmp_path, mem_sink):
    src, ts = _events_toolset(tmp_path)
    try:
        batches = [DeltaBatch(table="events", rows=_events_rows(50, 1, "2025-02-01"))]
        runner = MicroBatchRunner(ts)
        runner.process(_RedeliverConnector(batches))
        lag = mem_sink.values("analytics.ingest.batch.lag_seconds", table="events")
        assert lag and lag[0] >= 0.0
    finally:
        ts.source.close()


# --- CdcConnector wraps an arbitrary feed -----------------------------------


def test_cdc_connector_feed_lands_via_runner(tmp_path, mem_sink):
    src, ts = _events_toolset(tmp_path)

    def feed():
        return _events_rows(60, 2, "2025-03-01")

    try:
        conn = CdcConnector("events", feed)
        runner = MicroBatchRunner(ts)
        assert runner.process(conn) == 1
        assert src.row_count("events") == 3 + 2
    finally:
        ts.source.close()


# --- Optional gated Kafka path (skipped without broker / package) -----------


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_STREAM_TESTS") != "1",
    reason="set RUN_STREAM_TESTS=1 with a reachable broker to exercise Kafka",
)
def test_kafka_connector_fails_without_broker():
    pytest.importorskip("kafka")
    from demos.analytics.src.analytics.ingest import KafkaConnector
    from kafka.errors import KafkaError

    with pytest.raises(KafkaError):
        # Construction fails fast without a reachable broker.
        KafkaConnector("t", "localhost:9092", group_id="ci")
