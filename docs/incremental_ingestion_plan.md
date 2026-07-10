# Incremental Data Ingestion & "Firehose" Plan (PR-12 … PR-16)

**Status legend:** ✅ done · 🟡 in-progress · ⬜ planned

**Companion to:** `docs/production_readiness.md` (PR-1 … PR-11). This document
extends that checklist with the work needed to take the analytics engine from
"batch CSV import, re-import the whole file each run" to "continuous,
incremental, idempotent ingestion of a data firehose."

**Scope note (same as the parent doc):** Authentication / multi-tenant isolation
is explicitly out of scope. Ingestion is an **internal, server-side** operation
(trusted caller). It must NEVER be exposed as a user-facing tool — the
`run_sql`/`dsl_query` tools stay strictly read-only (see
`demos/analytics/src/analytics/safe_sql.py` and the hardening in commit
`c6cbf83`).

**Motivating scenario:** Today a deployment imports one or more CSV files once
via `CsvSource._import_csvs` (`csv_source.py:54`), which does a one-shot
`CREATE OR REPLACE TABLE <t> AS SELECT * FROM read_csv(...)` (`csv_source.py:64`).
Every subsequent day, the only way to reflect new data is to re-read the full
file and rebuild the whole table — a full reload, not incremental. A real
firehose (Kafka / Kinesis / CDC / nightly delta files) instead delivers
append-only (and occasionally late/out-of-order, possibly duplicate) records
that must land into the live tables **without** a full re-import, while the
profile, semantic model, DSL grounding, and caches stay consistent.

---

## Current state (what already exists — the bones are good)

These pieces already ship and are the foundation the plan builds on:

- **Incremental profiling** — `profiler.profile_dataset_incremental`
  (`profiler.py`) updates a prior `DatasetProfile` by merging only the arrived
  batch (`ColumnProfile.from_batch` + `merge_column_profiles` via the Chan
  parallel-variance formula), never re-reading the whole table. `ColumnProfile`
  carries running sufficient statistics and a `merge(other, *, decay=0.0)`
  method for drift-windowed updates. Relationship discovery only re-runs on a
  **schema** change. (Implemented as PR-11.)
- **Incremental training** — `train_backend.IncrementalTrainBackend`
  (`train_backend.py`) does partial-fit SGD over the full frame. (PR-10.)
- **Row-count-agnostic model cache key** —
  `dataset_fingerprint.fingerprint(source, row_count_aware=False)` depends only
  on schema + a distribution-shaped digest, so pure row growth does not thrash
  the model cache; `ModelsToolset` already keys its cache on it. (PR-11.)
- **Freshness observability** — `freshness.freshness(source, model)`
  (`freshness.py`) reports per-table max/min date, row count, and staleness in
  days. It reports but does **not** yet gate anything.
- **Control-total reconciliation** — `reconcile.reconcile(...)`
  (`reconcile.py`) compares an engine aggregate against a declared source of
  truth. Currently a manual call, not wired into ingest.

What is **missing** for continuous ingestion:

1. **No write/ingest seam.** `DataSource` (`data_source.py:59`) is read-only —
   `tables`, `sample`, `native_query`, `native_query_with_limit`, `close`. There
   is no `append_rows` / `upsert` / `ingest_csv` / `refresh`. New data physically
   cannot land incrementally.
2. **External-access lockdown blocks delta loads.** `CsvSource._import_csvs`
   calls `SET enable_external_access = false` immediately after the initial
   import (`csv_source.py:68`). A later `INSERT INTO <t> SELECT * FROM
   read_csv(delta.csv)` would therefore be rejected. Ingest needs a controlled,
   server-side re-open of external access for the duration of the load only.
3. **No idempotent upsert / dedup / late-arrival handling.** The firehose
   delivers duplicates and out-of-order rows. There is no primary-key concept
   wired into the source, no watermark on the time column, and no merge/anti-join
   dedup. `CREATE OR REPLACE` only overwrites.
4. **No schema-evolution handling.** A new column arriving in the stream breaks
   `profile_dataset` / `from_profile` / DSL grounding. There is no
   add-column + schema-drift detection path.
5. **Model/semantic refresh is not wired to arrival.** `AnalyticsToolset`
   (`toolset.py`) builds the `SemanticModel` once from a full profile; nothing
   calls `profile_dataset_incremental` or re-grounds the `DslEngine` when data
   lands, so new rows stay invisible to the DSL/NL path until a full rebuild.
6. **No streaming connector.** Only batch CSV / Parquet (`CsvSource`,
   `ParquetSource`) and JDBC (`SqlSource`, `warehouse_sources`) exist. There is
   no micro-batch writer that drains a stream into DuckDB.
7. **Freshness not enforced as a gate.** `freshness.py` computes staleness but
   the query path never blocks serving stale data or auto-selects the correct
   trailing window.

---

## PR-12 — Ingestion seam (the linchpin)  ✅ implemented

**Why:** Without a write path on `DataSource`, every later item is impossible.
This PR adds the minimal, safe, server-side API to land append/upsert batches
and delta CSVs into a live table.

**Status:** ✅ implemented (commit `PR-12`).

**Key design decision (deviation from the original plan):** the plan proposed a
`_allow_external()` context manager that re-enabled DuckDB's
`enable_external_access` for the duration of a delta load. This does **not**
work: in current DuckDB, once `enable_external_access = false` is set it cannot
be re-enabled while the database is running ("Cannot enable external access
while database is running"). Re-enabling it would also be unsafe. Instead,
`ingest_csv` **reads the delta CSV in the trusted Python process** (pandas when
available, else the stdlib `csv` module) and inserts the typed rows via
parameterized SQL. DuckDB never needs external file access for ingestion, so
the `enable_external_access = false` lockdown (`csv_source.py`) stays
**permanently on** — strictly stronger than the plan's toggle-and-relock, and
the `safe_sql.py` file-read guard (commit `c6cbf83`) still blocks any user SQL
reading files.

**What shipped:**

- `DataSource` protocol (`data_source.py`) now declares
  `append_rows` / `ingest_csv` / `upsert` / `row_count` (read-only backends
  raise `NotImplementedError` for the write methods).
- `CsvSource` (`csv_source.py`):
  - `append_rows(table, rows)` — parameterized `executemany` INSERT (no value
    interpolation); returns rows inserted.
  - `ingest_csv(table, csv_path, *, mode="append")` — reads the CSV in Python
    via `_read_csv_rows` and inserts; seeds a new table from the delta when the
    table does not yet exist. `mode="upsert"` raises `NotImplementedError`
    (implemented in PR-13).
  - `upsert(table, rows, keys)` — baseline keyed insert-if-not-exists (append +
    dedup via a temp-table anti-join); watermark / late-arrival handling is
    PR-13.
  - `row_count(table)` — `COUNT(*)` helper.
- `ParquetSource` / `SqlSource` / `GraphSource`: `row_count` implemented (pure
  read); the three write methods raise `NotImplementedError` (documented
  read-only backends).

**Verify (E2E):** `tests/test_ingestion_seam.py` (11 tests, all green)

- `append_rows` grows `row_count` and the rows are queryable.
- `ingest_csv` (append) adds rows **without** a full re-import (original rows
  intact — no `CREATE OR REPLACE`); seeds a new table when missing.
- External-access lockdown holds after ingest: a user `SELECT * FROM
  read_csv('/etc/passwd')` still raises; `safe_sql_error` still blocks writes
  and file reads (read-only posture preserved).
- `upsert` skips existing keys; requires keys.
- Read-only sources implement `row_count` and reject writes.

---

## PR-13 — Idempotent upsert, dedup & late-arrival watermark  ✅ implemented

**Why:** A firehose repeats and reorders. Naive `append_rows` double-counts and
pollutes aggregates. We need a primary key + a watermark so re-delivered or
late events are absorbed correctly, and so control totals stay reconciled.

**Status:** ✅ implemented (commit `PR-13`).

**What shipped:**

- `DataSource` protocol gains `primary_keys(table)` and `time_column(table)`
  (read-only sources return `[]` / `None`); `upsert` now takes `time_column` /
  `late_window` and returns a `UpsertResult` (inserted / updated / late_rejected
  / watermark) — defined in `data_source.py`.
- `CsvSource` (`csv_source.py`): `upsert` performs a true merge — existing keys
  are **updated**, new keys **inserted**; a per-table high-water mark is tracked
  and rows whose event time is older than `watermark - late_window` days are
  rejected as too-late. Late rejection applies only to *new* rows: backfilling /
  updating an existing key is always allowed. `late_window` defaults to
  `ANALYTICS_LATE_WINDOW` (1 day).
- `ingest.py` → `IngestController`: resolves keys / time column from an explicit
  argument, else the `SemanticModel` (identifier dimensions / time columns), else
  the source heuristics; emits `analytics.ingest.rows` / `late_rejected` /
  `errors` / `reconcile_diff` (+ `reconcile_breach`) via the `metrics` facade
  (PR-4); and runs automated reconcile against an `expected_rows` control total
  (or a `(metric_ref, expected, tolerance)` source-of-truth metric).

**Implement (original design as built):**

- **Key detection:** reuse the profiler's role hints — columns classified as
  `IDENTIFIER` (`data_source.ColumnRole.IDENTIFIER`) are candidate keys. Provide
  `DataSource.primary_keys(table) -> list[str]` defaulting to identifier columns
  (fallback: first identifier or explicit config).
- **Watermark:** use the table's detected date/timestamp column
  (`freshness._date_column` logic / `SemanticModel.time_columns`). Track a
  per-table high-water mark (max event time seen) in a small state table
  (`_ingest_watermark`) inside the same DuckDB connection.
- **Upsert semantics:** `upsert(table, rows, keys)` →
  `INSERT INTO <t> SELECT * FROM (VALUES ...) WHERE (keys) NOT IN (SELECT keys
  FROM <t>)` for pure appends, or a merge for updates. For late/out-of-order:
  accept events within a configurable lateness window
  (`ANALYTICS_LATE_WINDOW`) past the watermark; reject/queue anything older as
  "too late" (logged + counted, surfaced via `metrics.py`).
- **Automated reconcile:** after each ingest batch, run `reconcile.reconcile`
  (`reconcile.py`) against a declared control total (row count delta, or a
  source-of-truth sum) and emit `analytics.ingest.reconcile_diff` via
  `metrics.py` (PR-4 sink). A breach beyond tolerance → non-zero exit / alert
  hook (pluggable, off by default).

**Verify (E2E):** `tests/test_ingestion_upsert.py`

- Deliver the same 100 rows twice → final `row_count` == 100 (dedup on key),
  not 200.
- Deliver 100 rows, then 10 late rows within the lateness window → 110; a row
  older than `watermark - late_window` is rejected and counted, not inserted.
- After ingest, `reconcile` against the expected new-row control total shows
  `status == "pass"`; flip the expected total → `status == "fail"` and a metrics
  line is emitted.
- `primary_keys` returns the identifier column(s) from the profile.

---

## PR-14 — Incremental model / semantic refresh wired to ingest  ✅ implemented

**Why:** Even after rows land in DuckDB, the `SemanticModel` and the `DslEngine`
grounding were built once from a full profile and will not reflect the new data
until something refreshes them. This PR makes ingestion trigger an incremental
profile merge + re-ground + cache re-key.

**Status:** ✅ implemented (commit `PR-14`).

**What shipped:**

- `AnalyticsToolset` (`toolset.py`) now holds a live `DatasetProfile`
  (`profile=` arg; built lazily on first refresh when omitted) and a cache of
  `DslEngine` instances.
- `refresh_after_ingest(delta_rows=None, decay=0.0)`: merges the arrived batch
  via `profile_dataset_incremental` (PR-11, O(batch)), rebuilds the
  `SemanticModel` via `SemanticModel.from_profile`, clears the cached DSL
  engines, and returns the `row_count_aware=False` fingerprint (stable under
  pure row growth → model cache not thrashed, PR-11). Relationship discovery
  re-runs only on schema change, inside the A4 budget.
- `AnalyticsToolset.ingest(...)`: one-call convenience that runs
  `IngestController.ingest` (PR-13: idempotent upsert / watermark / metrics /
  reconcile) then `refresh_after_ingest`, so a single ingest leaves the model
  and engine consistent.
- `dsl_query` now caches its engine on `self._dsl_engine_cache` (keyed by
  synonyms) instead of a per-call local dict, so refresh can invalidate it.
  `nl_query` always builds the engine from the current `self.model`, so it picks
  up refreshes automatically.

**Implement (original design as built):**

- `AnalyticsToolset` (`toolset.py`): hold the live `DatasetProfile` and
  `SemanticModel`. Expose `refresh_after_ingest(delta_rows=None, decay=0.0)`
  that:
  1. calls `profiler.profile_dataset_incremental(source, prev=profile,
     new_rows=delta_rows, decay=decay)` (PR-11) — O(batch), no full re-read;
  2. rebuilds `SemanticModel.from_profile(profile)`;
  3. rebuilds the `DslEngine` (`dsl/engine.py`) with the new model + catalog
     (the engine caches by `dataset_sig`; with `row_count_aware=False` the key is
     stable, so the model cache is not thrashed — see PR-11);
  4. re-runs `fingerprint` and, if the schema changed, invalidates downstream
     caches (matches PR-5 schema-contract behavior).
- Relationship discovery re-runs only on schema change inside
  `profile_dataset_incremental`, keeping the update inside the A4 discovery
  budget.
- Wire `CsvSource.ingest_csv`/`append_rows` (PR-12) to optionally call
  `toolset.refresh_after_ingest` (or have the ingestion orchestrator call it) so
  a single ingest call leaves the engine consistent.

**Verify (E2E):** `tests/test_ingestion_refresh.py`

- Ingest rows of the same distribution; assert `SemanticModel` metrics/dimensions
  are unchanged OR advanced correctly, the DSL engine answers reflect the new
  rows (e.g. `SUM(amount)` grows), and `fingerprint(row_count_aware=False)` is
  stable (no needless model retrain).
- Ingest a column addition; assert the new column appears in the model, `dsl_query`
  can reference it, and the fingerprint changes (cache invalidated).
- A growing stream of constant-size batches keeps each refresh O(batch) (assert
  refresh time does not grow with accumulated rows).

---

## PR-15 — Streaming connector / micro-batch writer + backfill  ✅ implemented

**Status:** ✅ implemented (commit `fa515f4`).

**Why:** PR-12/13/14 define *how* a batch lands; this PR defines *where it comes
from* — a connector that drains Kafka/Kinesis/CDC (or a watched delta-directory)
into the ingest seam as micro-batches, plus replay/backfill for gaps.

**What shipped:**

- `Connector` protocol (`ingest.py`) with `poll` / `commit` / `close`; concrete
  adapters `FileDeltaConnector` (watched `*.delta.csv` / `*.delta.parquet`
  directory — the nightly-delta case), `KafkaConnector` (dependency-gated, like
  `ollama` in `pyproject.toml`), and `CdcConnector` (wraps an arbitrary
  change-feed callable into `Connector` for JDBC/Debezium-style feeds).
- `MicroBatchRunner` (`ingest.py`): the drain loop `poll → ingest (PR-13 upsert +
  PR-14 refresh) → emit per-batch metrics → optional freshness gate (PR-16) →
  commit offsets`. It owns the single writer (DuckDB is single-writer); reads use
  the source's per-query read-only connections, so ingest and serve don't contend.
- `_coerce_rows` / `_emit_lag` in the runner: delta rows arrive as loosely-typed
  text, so they are cast to the live table's column types (DATE / TIMESTAMP /
  numeric / boolean) before ingest, and a `analytics.ingest.batch.lag_seconds`
  metric is emitted from the batch's max event time.

**Verify (E2E):** `tests/test_ingestion_connector.py` (6 passed, 1 Kafka test
skipped without a broker / the `kafka` package).

**Implement:**

- A `Connector` protocol (`demos/analytics/src/analytics/ingest.py`, new
  module): `poll() -> list[dict]` / `commit(offsets)`. Concrete adapters:
  - `FileDeltaConnector` — watches a directory for new `*.delta.csv` /
    `*.delta.parquet` files (the nightly-delta case).
  - `KafkaConnector` / `KinesisConnector` — optional, dependency-gated (like
    `ollama` in `pyproject.toml`); micro-batch with max-records / max-lag.
  - `CdcConnector` — wraps a JDBC/Debezium-style change feed into upsert rows
    (delegates to PR-13 `upsert`).
- A `MicroBatchRunner` that loops: poll → `ingest_csv`/`upsert` (PR-12/13) →
  `refresh_after_ingest` (PR-14) → `freshness` check (PR-16) → commit offsets →
  emit `analytics.ingest.batch.{rows,lag_seconds,errors}` via `metrics.py`.
- **Backfill/replay:** given a start offset / date, replay historical deltas
  idempotently (dedup via PR-13 keys), so a gap can be filled without
  double-counting.
- Single-writer coordination: DuckDB is single-writer. The runner owns the
  write connection; read queries use the existing per-query read-only connection
  (`csv_source._query_connection`). Document the concurrency model.

**Verify (E2E):** `tests/test_ingestion_connector.py`

- `FileDeltaConnector` over a temp dir with two dropped delta files → both
  ingested, final row count == union, deduped via PR-13.
- `MicroBatchRunner` with a fake in-memory connector → N micro-batches land,
  `refresh_after_ingest` runs each time, metrics lines emitted, offsets committed
  (replay with same offsets inserts nothing extra).
- Optional gated Kafka/Kinesis test behind `RUN_STREAM_TESTS=1` (skipped in CI
  without a broker), mirroring the `RUN_WAREHOUSE_TESTS` pattern (PR-3).

---

## PR-16 — Schema evolution + freshness gate  ✅ implemented

**Status:** ✅ implemented (commit `82e8a35`).

**Why:** Streams add/retire columns and go stale. We need to evolve the schema
safely and to stop serving answers against data older than a policy.

**What shipped:**

- **Schema evolution** (`csv_source.py`): `append_rows` / `upsert` / `ingest_csv`
  now call `_evolve_schema`, which `ALTER TABLE <t> ADD COLUMN` for any column
  present in the arriving batch but missing from the live table (typed from the
  batch — VARCHAR / BIGINT / DOUBLE / BOOLEAN / DATE / TIMESTAMP). Retired
  columns are left in place (additive, backward-compatible). The batch column
  list is the union across all rows, so a new column arriving in only some rows
  still lands. `upsert`'s insert now uses an explicit column list, so partial
  batches and post-evolution tables with extra columns land correctly.
- **Freshness gate** (`freshness.py` + `toolset.py`): `ANALYTICS_MAX_STALE_DAYS`
  (per-table override available via `freshness()`) drives a per-table `stale`
  flag; the `FreshnessReport` carries `maxStaleDays`. `run_sql` / `dsl_query` /
  `nl_query` plan without executing, extract the referenced tables, and — when
  the policy is set and a table is staler than allowed — return a structured
  `"stale"` result (with the staleness note) instead of a silent number. An
  explicit `allow_stale=True` argument overrides the gate.
- **Auto-window** (`toolset.py` `trend`): a trailing-window query with no
  explicit date filter anchors its lower bound at the table's high-water mark
  (`SELECT MAX(<ts_expr>)`) instead of wall-clock `now`, so in-window late rows
  are included transparently. Works for epoch-encoded time columns.

**Verify (E2E):** `tests/test_ingestion_schema_freshness.py`

- Ingest a batch with a new column → `ALTER` applied, column queryable, model
  updated (no full reload), `dsl_query` can group by it; a partial batch does not
  drop retired columns.
- `ANALYTICS_MAX_STALE_DAYS=0` with no new data → `dsl_query` / `run_sql` return
  a "stale" structured result; with `allow_stale=True` they answer.
- Auto-window: a `trend` query without a date filter aggregates up to the
  watermark (inclusive of in-window late rows), and matches `now`-anchoring for
  fresh data.

**Implement:**

- **Schema evolution:** on ingest, diff the incoming batch's columns vs the live
  table schema; `ALTER TABLE <t> ADD COLUMN` for new columns (typed from the
  profiler); for retired columns, keep the column (backward-compatible) but note
  it in `freshness`/`reconcile` notes. Re-profile incrementally (PR-14).
- **Freshness gate:** extend `freshness.freshness` (`freshness.py`) with a
  policy (`ANALYTICS_MAX_STALE_DAYS` / per-table override). The query path
  (`AnalyticsToolset.dsl_query` / `run_sql`) optionally checks the gate and, when
  violated, returns a structured "stale" result (with the staleness note) instead
  of silently answering — override via an explicit `allow_stale=True` argument.
- **Auto-window:** when a query has no explicit time filter, the planner can
  default the trailing window to "since the watermark minus lateness" so late
  data is naturally included (ties to PR-13 watermark).

**Verify (E2E):** `tests/test_ingestion_schema_freshness.py`

- Ingest a batch with a new column → `ALTER` applied, column queryable, model
  updated, no full reload.
- Set `ANALYTICS_MAX_STALE_DAYS=0` and ingest no new data → a `dsl_query` returns
  a "stale" structured result (not a silent number); with `allow_stale=True` it
  answers.
- Late-window auto-selection: a query without a date filter aggregates up to the
  watermark (inclusive of in-window late rows).

---

## Build order & dependencies

```
PR-12 (ingest seam)  ──►  PR-13 (upsert/dedup/watermark)
       │                        │
       └────────┬───────────────┘
                ▼
         PR-14 (incremental refresh)  ──►  PR-15 (connector/backfill)
                                                 │
                                                 ▼
                                         PR-16 (schema + freshness gate)
```

PR-12 is the critical path and the only hard prerequisite for the rest. Each PR
follows the parent doc's verification cadence: its own E2E test file, `ruff` /
`mypy` clean, and an updated ✅ note here referencing the verifying test path.

## Explicitly OUT OF SCOPE

- **Auth / tenant isolation** — unchanged from parent doc; ingestion is internal.
- **Exactly-once across process crashes mid-batch** — best-effort idempotent
  upsert (PR-13) + offset commit (PR-15) covers the common case; distributed
  transactional exactly-once is a later concern.
- **Streaming SQL (materialized views / continuous aggregates)** — out of scope;
  we refresh on ingest (PR-14), not continuously.
