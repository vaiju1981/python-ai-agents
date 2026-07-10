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

## PR-12 — Ingestion seam (the linchpin)  ⬜ planned

**Why:** Without a write path on `DataSource`, every later item is impossible.
This PR adds the minimal, safe, server-side API to land append/upsert batches
and delta CSVs into a live table, with external access re-opened only for the
load and re-locked immediately after.

**Implement:**

- Extend the `DataSource` protocol (`data_source.py:59`) with:
  - `append_rows(table: str, rows: list[dict[str, Any]]) -> int` — insert N rows
    (best-effort typed). Returns rows inserted.
  - `ingest_csv(table: str, csv_path: Path, *, mode: str = "append") -> int` —
    load a delta CSV into `table`. `mode="append"` → plain insert; `mode="upsert"`
    → merge on detected keys (see PR-13). Returns rows added.
  - `upsert(table: str, rows: list[dict[str, Any]], keys: list[str]) -> int` —
    idempotent merge on `keys` (reserved for PR-13; default impl can raise
    `NotImplementedError` until then, or do an append+dedup).
  - `row_count(table: str) -> int` — cheap `COUNT(*)` helper (profile/freshness
    already do this; promote to the protocol for symmetry).
- `CsvSource` (`csv_source.py`):
  - Add a private context manager `_allow_external()` that does
    `SET enable_external_access = true` (DuckDB's `lock_configuration` defaults
    to false, so this is permitted after the init lockdown), yields, then
    re-runs `SET enable_external_access = false` in a `finally`. If a deployment
    has locked configuration, ingest raises a clear error rather than silently
    no-op'ing.
  - `append_rows`: quote the table (`sql_quote`), build a parameterized
    `INSERT INTO <t> (cols) VALUES (?, ...)` via DuckDB parameter binding (no
    string interpolation of values — reuse `sql_literal` only for identifiers,
    never for data). For large batches prefer `from_df` / `INSERT INTO ... SELECT
    * FROM (VALUES ...)` to avoid per-row round-trips.
  - `ingest_csv`: inside `_allow_external()`, run
    `INSERT INTO <t> SELECT * FROM read_csv_auto('<path>')` (or `read_csv` with
    the table's `type_overrides`). Re-lock on exit.
  - Keep `_import_csvs` (used at construction) unchanged; the new methods operate
    on already-created tables so they compose with the initial load.
- **Safety guardrails (critical):** ingestion is internal only. Do NOT add any
  `Tool` wrapping these methods. Add a module-level note + a unit test asserting
  `run_sql`/`dsl_query` still cannot write and that the file-read guard
  (`safe_sql.py`, commit `c6cbf83`) still blocks `read_csv` in user SQL.
- `ParquetSource` / `SqlSource`: implement `append_rows`/`ingest_csv` analogously
  where the backend supports inserts (Parquet → append to the dataset; SQLSource
  → `INSERT`); raise `NotImplementedError` for read-only warehouses (document
  which sources support writes).

**Verify (E2E):** `tests/test_ingestion_seam.py`

- Build a `CsvSource` from a small CSV; call `append_rows` with a few new dicts;
  assert `row_count` grew by exactly that many and the new values are
  queryable via `native_query`.
- Call `ingest_csv(delta.csv, mode="append")`; assert rows added and the live
  table reflects them (no full re-import — assert the original rows are intact,
  i.e. it did not `CREATE OR REPLACE`).
- Assert external access is **re-locked** after ingest: a follow-up
  `native_query("SELECT * FROM read_csv('/etc/passwd')")` still raises (or
  returns nothing) — the lockdown holds.
- Assert `run_sql`/`dsl_query` remain read-only (re-run `test_safe_sql_*` style
  assertions) and that `safe_sql_error("... read_csv(...)")` still blocks.
- `ParquetSource`/`SqlSource` either ingest correctly or raise
  `NotImplementedError` (documented).

---

## PR-13 — Idempotent upsert, dedup & late-arrival watermark  ⬜ planned

**Why:** A firehose repeats and reorders. Naive `append_rows` double-counts and
pollutes aggregates. We need a primary key + a watermark so re-delivered or
late events are absorbed correctly, and so control totals stay reconciled.

**Implement:**

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

## PR-14 — Incremental model / semantic refresh wired to ingest  ⬜ planned

**Why:** Even after rows land in DuckDB, the `SemanticModel` and the `DslEngine`
grounding were built once from a full profile and will not reflect the new data
until something refreshes them. This PR makes ingestion trigger an incremental
profile merge + re-ground + cache re-key.

**Implement:**

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

## PR-15 — Streaming connector / micro-batch writer + backfill  ⬜ planned

**Why:** PR-12/13/14 define *how* a batch lands; this PR defines *where it comes
from* — a connector that drains Kafka/Kinesis/CDC (or a watched delta-directory)
into the ingest seam as micro-batches, plus replay/backfill for gaps.

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

## PR-16 — Schema evolution + freshness gate  ⬜ planned

**Why:** Streams add/retire columns and go stale. We need to evolve the schema
safely and to stop serving answers against data older than a policy.

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
