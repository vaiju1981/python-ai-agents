"""PR-13 — idempotent upsert / dedup / late-arrival watermark + reconcile E2E.

Covers CsvSource.upsert (true update + insert, watermark, late-window
rejection), key/time-column auto-detection, and IngestController
(metrics + automated reconcile).
"""

from __future__ import annotations

import datetime as dt

import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.ingest import IngestController
from demos.analytics.src.analytics.metrics import InMemoryMetricsSink


def _events_csv(path):
    # ts auto-detected as DATE by DuckDB's read_csv_auto during import.
    path.write_text(
        "id,ts,v\n"
        "1,2025-01-01,10\n"
        "2,2025-01-02,20\n"
        "3,2025-01-03,30\n"
    )


def _events_source(tmp_path):
    base = tmp_path / "events.csv"
    _events_csv(base)
    return CsvSource(named_csvs={"events": base})


# --- true upsert (update existing, insert new) ------------------------------


def test_upsert_updates_existing_and_inserts_new(tmp_path):
    src = _events_source(tmp_path)
    try:
        # Update id=1 (v 10 -> 99) and insert id=4.
        res = src.upsert(
            "events",
            [
                {"id": 1, "ts": dt.date(2025, 1, 1), "v": 99},
                {"id": 4, "ts": dt.date(2025, 1, 4), "v": 40},
            ],
            keys=["id"],
        )
        assert res.updated == 1
        assert res.inserted == 1
        assert src.row_count("events") == 4
        got = {r["id"]: r["v"] for r in src.native_query("SELECT id, v FROM events")}
        assert got[1] == 99  # existing row updated, not duplicated
        assert got[4] == 40
    finally:
        src.close()


# --- late-arrival watermark -------------------------------------------------


def test_upsert_rejects_late_rows_and_advances_watermark(tmp_path):
    src = _events_source(tmp_path)
    try:
        res = src.upsert(
            "events",
            [
                {"id": 4, "ts": dt.date(2025, 1, 2), "v": 40},   # within window -> insert
                {"id": 5, "ts": dt.date(2025, 1, 10), "v": 50},  # newer -> advances wm
                {"id": 6, "ts": dt.date(2024, 12, 30), "v": 60},  # too old -> rejected
            ],
            keys=["id"],
            late_window=2,
        )
        assert res.inserted == 2
        assert res.late_rejected == 1
        assert res.watermark == dt.date(2025, 1, 10)
        assert src.row_count("events") == 5  # 3 original + 2 accepted
        late = src.native_query("SELECT id FROM events WHERE id = 6")
        assert late == []  # rejected row never landed
    finally:
        src.close()


def test_empty_upsert_returns_watermark(tmp_path):
    src = _events_source(tmp_path)
    try:
        res = src.upsert("events", [], keys=["id"])
        assert res.inserted == res.updated == res.late_rejected == 0
        assert res.watermark == dt.date(2025, 1, 3)
    finally:
        src.close()


# --- key / time-column detection --------------------------------------------


def test_primary_keys_auto_detected(tmp_path):
    src = _events_source(tmp_path)
    try:
        assert src.primary_keys("events") == ["id"]
    finally:
        src.close()


def test_time_column_auto_detected(tmp_path):
    src = _events_source(tmp_path)
    try:
        assert src.time_column("events") == "ts"
    finally:
        src.close()


# --- IngestController: resolution, metrics, reconcile -----------------------


def test_controller_resolves_keys_and_time_from_model(tmp_path):
    from types import SimpleNamespace

    fake_model = SimpleNamespace(
        dimensions=[
            SimpleNamespace(table="events", column="id", role=SimpleNamespace(value="identifier"))
        ],
        time_columns=[SimpleNamespace(table="events", column="ts")],
    )
    src = _events_source(tmp_path)
    try:
        ctrl = IngestController(src, model=fake_model)
        assert ctrl._resolve_keys("events", None) == ["id"]
        assert ctrl._resolve_time_column("events", None) == "ts"
        # End-to-end: keys/time resolved from the model, no explicit args.
        res = ctrl.ingest(
            "events",
            [{"id": 4, "ts": dt.date(2025, 1, 4), "v": 40}],
            mode="upsert",
        )
        assert res.inserted == 1
    finally:
        src.close()


def test_controller_emits_ingest_metrics(tmp_path):
    mem = InMemoryMetricsSink()
    src = _events_source(tmp_path)
    try:
        ctrl = IngestController(src, late_window=2, sink=mem)
        res = ctrl.ingest(
            "events",
            [
                {"id": 4, "ts": dt.date(2025, 1, 4), "v": 40},
                {"id": 5, "ts": dt.date(2024, 12, 30), "v": 60},
            ],
            mode="upsert",
        )
        assert mem.count("analytics.ingest.rows", table="events") == res.written
        assert mem.count("analytics.ingest.late_rejected", table="events") == res.late_rejected
    finally:
        src.close()


def test_controller_automated_row_reconcile(tmp_path):
    mem = InMemoryMetricsSink()
    src = _events_source(tmp_path)
    try:
        ctrl = IngestController(src, sink=mem)
        # One row inserted (the other is late-rejected); expected_rows=2 -> fail.
        ctrl.ingest(
            "events",
            [{"id": 4, "ts": dt.date(2025, 1, 4), "v": 40}],
            mode="upsert",
            expected_rows=2,
        )
        diffs = [e for e in mem.events if e["metric"] == "analytics.ingest.reconcile_diff"]
        assert diffs, "reconcile_diff metric not emitted"
        fail = [e for e in diffs if e["tags"]["status"] == "fail"]
        assert fail, "mismatched control total should be a fail"
        assert mem.count("analytics.ingest.reconcile_breach", kind="row_count") >= 1
    finally:
        src.close()


def test_controller_requires_keys_for_upsert(tmp_path, monkeypatch):
    src = _events_source(tmp_path)
    try:
        # Force key resolution to fail (no explicit keys, no model, no detected keys).
        monkeypatch.setattr(src, "primary_keys", lambda table: [])
        ctrl = IngestController(src)
        with pytest.raises(ValueError):
            ctrl.ingest(
                "events",
                [{"id": 9, "ts": dt.date(2025, 1, 9), "v": 1}],
                mode="upsert",
            )
    finally:
        src.close()
