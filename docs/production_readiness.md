# Analytics Engine — Production Readiness

**Purpose:** A one-by-one implementation checklist for taking the generic
analytics engine (`demos/analytics/`) from "demo + test-covered" to "safe to
run in production." Each item is self-contained, and every item has an explicit
**end-to-end verification** gate that must pass before moving to the next.

**Scope note:** Authentication and multi-tenant isolation are explicitly
**out of scope** here — a separate system in front of this engine owns auth.
Items below assume a trusted, single-tenant (or already-auth'd) caller.

**How to use this doc:** Implement one item, then run its verification gate
(`pytest` scenario, manual check, or both). Only mark it ✅ after the gate is
green. Keep the numbered list stable so PRs can reference items (e.g. "closes
PR-1").

**Legend:** ✅ done · 🟡 in-progress · ⬜ planned

---

## Priority order

1. File-store concurrency + durability (PR-1)
2. Remove pickle / harden model cache (PR-2)
3. Warehouse integration tests + creds handling (PR-3)
4. Observability / metrics export (PR-4)
5. Partial-failure robustness for fan-out joins (PR-5)
6. LLM trust-grade coupling test (PR-6)
7. Cross-answer lineage graph (PR-7)  *(was §G G4)*
8. Warehouse-side model scoring (PR-8)  *(was §G G1)*
9. Auto-apply feedback-loop threshold tuning (PR-9)  *(was §G G2)*
10. Out-of-core / distributed row-level ML (PR-10)  *(was §G G3)*
11. Streaming / incremental profiling (PR-11)  *(was §G G5)*

---

## PR-1 — File-store concurrency + durability

**Why:** `model_store.py` writes pickled model files with no lock; concurrent
workers (multi-process server, serverless, multiple replicas) can corrupt or
clobber each other's writes. `decision_store.py` has only an in-process
`threading.Lock` — safe within one process, unsafe across processes.

**Implement:**
- Add cross-process file locking (`fcntl.flock` on POSIX) around every
  read-modify-write of `model_store` and `decision_store` (and `audit_store`,
  `catalog` overrides if persisted).
- Prefer atomic writes: write to `<name>.tmp` then `os.replace()` onto the
  final path (rename is atomic on POSIX/NTFS).
- Add a small stress test helper that spawns N threads/processes doing
  concurrent save/load and asserts no corruption / no lost writes.

**Verify (E2E):**
- New test `tests/test_production_concurrency.py`: launch ≥8 concurrent writers
  against `ModelStore.save`/`DecisionStore.record_*`; after join, every written
  record is readable and byte-identical to what was saved (no truncation, no
  cross-talk). Run under `pytest-xdist` (`-n 4`) so it also exercises multiple
  processes.
- Manual: run the analytics app behind a 4-worker gunicorn/uvicorn and fire a
  burst of `predict` + `reconcile` calls; confirm model cache dir has no `.tmp`
  leftovers and no `pickle` `EOFError` in logs.

---

## PR-2 — Remove pickle / harden model cache

**Why:** `model_store.py:93` does `pickle.load` from a disk path. If the cache
directory is shared or writable by another principal, deserializing attacker-
controlled bytes is remote-code-execution. Even without that, pickle is brittle
across library versions (breaks reproducibility).

**Status:** ✅ implemented (commit `158cf35` + PR-2 commit).

**Chosen implementation:** the *signing* branch. Cached models are written as a
versioned JSON envelope (`<key>.json`) containing `engine_version`,
`lib_versions`, the pickled model bytes (base64), and an optional HMAC signature.
`pickle` is invoked **only after** the envelope passes its checks:
- `engine_version` / `lib_versions` mismatch → silent cache miss (retrain, no
  stale-format load);
- when `ANALYTICS_MODEL_CACHE_KEY` is set, a missing/invalid signature raises
  `ModelCacheIntegrityError` and the bytes never reach `pickle.loads` (kills the
  RCE vector for a shared/writable cache dir);
- with no key (local/dev), unsigned envelopes still load, but a warning notes
  integrity was not verified.

The JSON-weights branch was considered but rejected: the only cached models are
sklearn RandomForest estimators, whose tree state isn't cleanly JSON-serializable
without private APIs; signing gives the same security property with zero
cross-version fragility.

**Verify (E2E):** (covered in `tests/test_production_concurrency.py`)
- Test: save with `ANALYTICS_MODEL_CACHE_KEY` set, flip a byte in the on-disk
  `payload`, reload → raises `ModelCacheIntegrityError` (no `pickle.loads`).
- Test: unsigned file + key configured → raises `ModelCacheIntegrityError`.
- Test: bump `ENGINE_VERSION` → `get` returns `None` (miss, forces retrain).
- Test: no key → unsigned envelope round-trips (dev mode).

---

## PR-3 — Warehouse integration tests + creds handling

**Status:** ✅ implemented.

**Why:** `warehouse_sources.py` (P2 pushdown) had no integration test; all 32
existing tests run on synthetic DuckDB. The Snowflake/BigQuery/Postgres paths
were unverified end-to-end, and there was no secret-manager integration for
credentials.

**What shipped:**
- `secrets.py`: a `SecretProvider` protocol + `EnvSecretProvider` (pluggable for
  a vault later) and `redact_secrets()` which scrubs `user:password@` from any
  string (logs, error messages, provenance).
- `warehouse_sources.make_warehouse_source` now resolves the URI from a named
  secret (`secret_name` + provider) instead of a literal; validates the attach
  `alias` is a safe SQL identifier; adds `duckdb`/`duckdb_file` kinds.
- SQL identifier quoting fixed for catalog-qualified table names
  (`sql_qtable` / `sql_quote` now split on `.` and quote per part), and
  `SqlSource` discovers attached tables via `duckdb_tables()` so they are
  referenced as `catalog.schema.table` (previously attached tables surfaced
  under the host `main` schema and were unreferenceable).
- `ModelsToolset._resolve` now splits refs on the *last* dot, so a metric like
  `wh.sales.amount` resolves to table `wh.sales` (not `wh`).

**Verify (E2E):** `tests/test_warehouse_integration.py`
- Local DuckDB-file "warehouse" stand-in exercises the real ATTACH pushdown
  path; `summarize`, grouped `run_query`, SQL-native `ab_test` (Welch's t), and
  a fan-out join all match an in-memory baseline on the same data.
- `SecretProvider` resolution + `redact_secrets` unit tests; bad-alias rejected.
- Real Snowflake/BigQuery/Postgres paths are the same suite, gated behind
  `RUN_WAREHOUSE_TESTS=1` + the matching `WAREHOUSE_*_URI` secret (skipped in CI
  without creds). With creds, run `RUN_WAREHOUSE_TESTS=1 pytest -k warehouse`.

**Implement:**
- Add a creds abstraction: read warehouse secrets from env vars / a pluggable
  `SecretProvider` (env by default, vault/secret-manager later) — never log
  them, redact in provenance.
- Add an integration test suite gated behind an env flag
  (`RUN_WAREHOUSE_TESTS=1`) + available creds, so CI stays green without secrets
  but can be run in a secured pipeline.
- Cover: connect, profile a sample, run `ab_test` (SQL-native Welch's t),
  `summarize`, and one fan-out join; assert results match the DuckDB baseline on
  the same data.

**Verify (E2E):**
- `RUN_WAREHOUSE_TESTS=0 pytest` → warehouse tests skipped, suite still green.
- With creds present: `RUN_WAREHOUSE_TESTS=1 pytest -k warehouse` → all pass and
  numeric results match the DuckDB reference within tolerance.
- Manual: confirm no secret string appears in `--log-level=DEBUG` output or in
  any `provenance` envelope.

---

## PR-4 — Observability / metrics export

**Status:** ✅ implemented.

**Why:** Trust grades, abstentions, drift signals, latency, and error rates
currently live only inside answer payloads. In production you cannot monitor or
alert on them.

**What shipped:**
- `metrics.py`: a tiny metrics facade with a pluggable `MetricsSink` protocol:
  - `LogMetricsSink` (default): one JSON line per metric via `logging`.
  - `InMemoryMetricsSink`: for tests / in-process rollups.
  - A real Prometheus / OTLP sink can be dropped in later behind the same
    protocol with **no change to call sites** and **no new hard dependency**.
  - Helpers `inc` / `set_gauge` / `observe` plus a `ContextVar` so the answer
    path tags metrics with the current tool name without threading it through
    every `_ok` call site.
- Emitted from the answer path (`toolset._ok` / `_make_tool`):
  - `analytics.answer.by_trust_tier{tier}` — answers by trust tier
  - `analytics.answer.abstained{tool}` — abstention rate per causal tool
  - `analytics.tool.calls{tool}` / `analytics.tool.latency_seconds{tool}`
    (histogram) / `analytics.tool.errors{tool}` — per-tool latency + errors
  - `analytics.drift.breaches{feature,tool}` — PSI/KS drift breaches
  - `analytics.model_cache.{hits,misses,writes}` — `model_store` cache stats
- `readiness(directories=None)` probe that reports the active sink **and** store
  connectivity (existence / readability / writability of the configured store
  roots, via `ANALYTICS_MODEL_CACHE_DIR` when no dirs are passed).

**Verify (E2E):** `tests/test_metrics.py`
- `summarize` on a large, fully-covered sample → `by_trust_tier{tier=DIRECTIONAL}`
  + `tool.calls` + `tool.latency_seconds` recorded via the in-memory sink.
- Thin-evidence `matched_impact` → `abstained{tool=matched_impact}` bumped.
- `model_store` get/put loop → `model_cache.{hits,misses,writes}` counters.
- `readiness()` returns `ok` with per-directory connectivity when pointed at a
  writable store root.
- Manual: run the app, exercise each tool once, and confirm a JSON metrics line
  per call with the right tier label (`pytest -s` + `logging` at INFO, or swap
  in a Prometheus/OTLP sink via `metrics.set_sink(...)`).

---

## PR-5 — Partial-failure robustness for fan-out joins

**Status:** ✅ implemented.

**Why:** Multi-fact / fan-out queries assemble several CTEs and joins. If one
upstream table is missing, schema-drifts, or times out mid-query, behavior is
undefined today.

**What shipped:**
- `query_planner.py`:
  - `QueryPlanError` (+ `MissingTableError` / `SchemaContractError`) — a scoped
    error naming the offending `table` and the `reason`, instead of a generic SQL
    failure at execution time.
  - `validate_plan(model, spec, source)` — fails fast against the live source
    schema: every referenced table exists, and every referenced column
    (metric / dimension / filter / join key) is present.
  - `plan_query(model, spec, source=None, best_effort=False)` — the
    partial-failure wrapper. With no `source` it is identical to `plan`; with a
    `source` it validates first; with `best_effort=True` a failing table is
    dropped from the spec and the partial query is returned as a `PlanResult`
    whose `warnings` / `dropped_tables` name what was excluded.
- `toolset.run_query` / `compare` now call `plan_query(self.model, spec,
  self.source, best_effort=...)` and surface any `warnings` in the provenance
  envelope. Best-effort is opt-in via `ANALYTICS_QUERY_BEST_EFFORT=1`.
- Schema-contract breaches tie into PR-2: dropping a column changes
  `dataset_sig` (it fingerprints column schema), so cached models keyed on the
  old signature become a miss.

**Verify (E2E):** `tests/test_query_planner_partial_failure.py`
- Model referencing a non-existent `ghost` table → `MissingTableError` with
  `table == "ghost"` (scoped, not a generic SQL error).
- Same model + `best_effort=True` → `PlanResult` with `dropped_tables == ["ghost"]`,
  non-empty `warnings`, and SQL that actually executes against the source.
- Model referencing an existing table with a missing column → `SchemaContractError`
  naming that table.
- Drop a column post-profiling → `dataset_sig` changes → a PR-2 cached model
  under the old signature is a miss (new signature → `get` returns `None`).

---

## PR-6 — LLM trust-grade coupling test

**Status:** ✅ implemented.

**Why:** The tracker claims "the graded tier is surfaced to the model," but
nothing tests that the agent loop actually injects the trust grade into the
prompt and that the model respects abstentions. This is the linchpin of the
"defensible answers" story.

**What shipped:**
- `python_ai_agents.core.tool.ToolResult` gained a structured `trust` field
  (tier/confidence/abstain) carried on every result, not just as prose in the
  answer body or the provenance envelope.
- `default_agent._tool_result_for_model` (the tool-result formatter) now renders
  a machine-checkable `[TRUST:TIER]` token from `result.trust` — and for
  `INSUFFICIENT` it appends an explicit non-assertion directive. `_invoke_tool`
  preserves `trust` (and `provenance`) when capping the result for the model.
- `AnalyticsToolset._ok` / `ModelsToolset._ok` set `trust=` on the `ToolResult`.
- `agent.create_agent` system prompt instructs the model to honor
  `[TRUST:...]`: abstain from causal/confident claims on `[TRUST:INSUFFICIENT]`,
  flag `[TRUST:DIRECTIONAL]` as directional not definitive.

**Verify (E2E):** `tests/test_trust_grade_coupling.py`
- `ModelsToolset._ok` at each tier → rendered message contains `[TRUST:{TIER}]`;
  `INSUFFICIENT` additionally contains the "do not assert causal" directive.
- An `INSUFFICIENT` causal-style result rendered through the agent formatter
  carries `[TRUST:INSUFFICIENT]` + non-assertion instruction, and
  `create_agent(...).system_prompt` instructs the model to honor it.
- Optional live check (flagged, `PAA_RUN_OLLAMA_TESTS=1`): the agent on
  thin-evidence data answers with an abstention rather than a confident causal
  claim.

---

## PR-7 — Cross-answer lineage graph  *(tracker §G G4)*

**Status:** ✅ implemented.

**Why:** Provenance envelopes are per-answer. A derived answer (forecast built
on a reconciled metric) cannot currently be traced back through every upstream
SQL + fingerprint.

**What shipped:**
- `lineage.py`: a file-backed `LineageGraph` (persisted as `lineage.json` so it
  lives alongside `audit_store`) linking `dataset_sig` → SQL → answer id →
  downstream answer id. `record(answer_id, dataset_sig, sql, parents=...)` adds a
  node; `trace_lineage(answer_id)` walks parents upstream to raw sources
  (dependency-ordered); `upstream_dataset_sigs` collects the distinct fingerprints
  in a chain.
- Both `AnalyticsToolset` and `ModelsToolset` take an optional shared
  `lineage` graph. Every answer allocates an `answerId`, records a lineage node
  (with its `dataset_sig` + SQL), and links to the answers produced earlier in
  the same conversation via the graph's shared `scope`. The `answerId` is also
  written into the provenance envelope.
- `ReconcileResult` now carries the `sql` it ran, so the `reconcile` tool can
  record it; the `forecast` tool records its aggregation SQL too.
- `create_agent` accepts a shared `lineage` graph passed to both toolsets.

**Verify (E2E):** `tests/test_lineage.py`
- `reconcile` (answer A) then `forecast` (answer B) on a shared graph: B's node
  lists A as a parent; `trace_lineage(B)` returns A with its original
  `dataset_sig` + SQL. A reloaded `LineageGraph` from disk still resolves the
  chain (persistence).
- A missing answer id yields an empty trace (no crash).

---

## PR-8 — Warehouse-side model scoring  *(tracker §G G1)*

**Status:** ✅ implemented.

**Why:** `predict`/`forecast` serving still materializes a bounded sample into
pandas for tree/linear models. G1 removes that last row-level pull by pushing
scoring to the warehouse (SQL/UDF scoring for linear/tree; vendor ML optional).

**What shipped:**
- `warehouse_sources.score_warehouse(source, frame_sql, model, feature_cols,
  *, task)` emits a single in-warehouse `SELECT ... AS prediction FROM
  (<frame>) AS _f`. Linear models are expressed as exact arithmetic SQL
  (`intercept + Σ coef_i * col_i`), which matches local pandas scoring to
  float tolerance. The function **only builds a string** — it never pulls rows,
  so no frame is materialized into the Python process.
- `ModelsToolset.predict` now tries the warehouse path first via
  `_predict_in_warehouse`: if the source is a real warehouse (`make_warehouse_source`
  marks it with `_is_warehouse`), the model is a linear regressor, and
  `score_warehouse` can express it, scoring + drift run entirely as aggregate
  SQL and the result carries `scored_in_warehouse: True`. Otherwise it falls
  back to the existing local bounded-sample path (unchanged behavior for
  non-warehouse sources and tree/RF/classification models).
- Documented limit: tree / random-forest / k-means / isolation-forest (and
  classification linear models) return `None` from `score_warehouse`, so they
  fall back to local scoring — clustering/scoring needs a different strategy
  (PR-10).
- `forecast` already operates on SQL-native monthly aggregates (not the full
  frame), so it was left as-is; it pulls no row-level data either.

**Verify (E2E):** `tests/test_warehouse_scoring.py`
- `(DuckDB-backed "warehouse")` train on a 400-row sample, score the full 3000
  row frame in-warehouse → predictions match local pandas `LinearRegression`
  scoring within `rtol/atol=1e-6`, and `score_warehouse` makes **zero**
  `native_query` calls (no frame materialization).
- `predict` on a warehouse linear model issues aggregate SQL only and returns
  `scored_in_warehouse: True` with `n_scored` equal to the full table size.
- Tree model → `score_warehouse` returns `None` (falls back to local).
- Non-warehouse (CSV) source with a linear model → `_predict_in_warehouse`
  returns `None`, preserving local predict behavior (regression guard).

**Implement:**
- Extend `warehouse_sources.py` with a `score(frame, model)` that emits SQL
  (linear coefficients as arithmetic; tree as CASE/WHEN or a warehouse UDF) so
  inference runs in-warehouse.
- Fall back to local bounded sample only when the warehouse can't express it.
- Cover k-means/isolation-forest separately (clustering/scoring needs a
  different strategy — document the limit).

**Verify (E2E):**
- Test (DuckDB-backed "warehouse"): train on sample, score a large frame
  in-warehouse, assert results match the local pandas scoring within tolerance
  AND that zero rows were pulled into the Python process (assert no frame
  materialization call).
- Manual: with `RUN_WAREHOUSE_TESTS=1`, confirm `predict` issues exactly one
  SELECT and returns without loading the scored table locally.

---

## PR-9 — Auto-apply feedback-loop threshold tuning  *(tracker §G G2)*

**Status:** ✅ implemented.

**Why:** Today `tune_trust_thresholds()` only *recommends* raise/lower/hold.
Operators want opt-in self-calibration from labeled outcomes, with guardrails
and an audit entry.

**What shipped:**
- `trust.py`:
  - `TRUST_BOUNDS` — per-threshold safe-clamp ranges so auto-tuning can never
    move a bar into unsafe territory.
  - `current_thresholds()` — live snapshot of the (runtime-mutable) thresholds.
  - `apply_trust_tuning(action, *, step)` — `raise` makes the bar *stricter*
    (thresholds grow), `lower` makes it *looser* (thresholds shrink); every
    value is clamped to `TRUST_BOUNDS`. It mutates the module globals that
    `grade()` reads at call time, so grading changes live with no caller change.
  - `auto_tune_trust_thresholds(suggestion, *, min_samples, enabled, audit)` —
    the gate. Off by default (`ANALYTICS_TRUST_AUTO_TUNE` unset) → recommend-only
    path. When enabled and the suggestion is `raise`/`lower` with `>= min_samples`
    labeled outcomes, it applies the change and writes an audit entry
    (old→new + evidence) via `audit.record_trust_tuning` if the sink supports it.
- `audit_store.SqliteAuditStore.record_trust_tuning(old, new, evidence)` — durable
  `trust_tuning` event (survives restarts), satisfying the audit requirement.

**Verify (E2E):** `tests/test_trust_auto_tune.py`
- Feed labeled TRUSTED outcomes that are wrong → `DecisionStore.tune_trust_thresholds`
  suggests `raise`. With auto-tune **on**, the live `ANALYTICS_TRUST_*` thresholds
  shift (stricter) and a `trust_tuning` audit entry (old→new + evidence) is written.
- With auto-tune **off**, feeding the same suggestion leaves thresholds untouched
  (recommend-only default preserved).
- Below `min_samples` → not applied.
- Pinning thresholds at their upper clamp and applying `raise` → bounded (no net
  change, no value exceeds the clamp); repeated raises converge at the clamp.

**Implement:**
- Add an opt-in policy (env `ANALYTICS_TRUST_AUTO_TUNE=1`) that applies the
  suggestion to `ANALYTICS_TRUST_*` with:
  - bounds/clamps so thresholds can't move past safe limits,
  - a minimum sample size before applying,
  - an audit entry in `decision_store`/`audit_store` recording old→new value
    and the evidence.
- Keep the recommend-only path as default.

**Verify (E2E):**
- Test: feed labeled outcomes that clearly warrant a threshold change; with
  auto-tune on, assert the env-derived thresholds shift and an audit entry is
  written; with auto-tune off, assert they do NOT change.
- Test: attempt to push a threshold past its clamp → assert it is bounded.

---

## PR-10 — Out-of-core / distributed row-level ML  *(tracker §G G3)*

**Status:** ✅ implemented.

**Why:** Row-level models (random forest / k-means / isolation forest) train on
a `ANALYTICS_MAX_TRAIN_ROWS` reservoir sample. For populations beyond that,
quality degrades.

**What shipped:**
- `train_backend.py` — a pluggable `TrainBackend` (Protocol) with four concrete
  implementations:
  - `BoundedTrainBackend` (default) — trains a RandomForest on the reservoir
    sample (the existing behavior).
  - `FullPopulationBackend` — trains a RandomForest on the **entire** population
    in memory (no cap) when it fits.
  - `IncrementalTrainBackend` — trains with partial-fit `SGD` estimators in
    batches over the full frame (the incremental/partial-fit strategy PR-10
    calls for); features and target are scaled so online SGD stays stable.
  - `DaskBackend` (optional) — out-of-core training via `dask-ml` `Incremental`;
    raises a clear `RuntimeError` if `dask`/`dask-ml` are not installed.
- `ModelsToolset(..., train_backend=...)` + `get_train_backend(name)` (env
  `ANALYTICS_TRAIN_BACKEND`, default `bounded`). A backend with
  `uses_full_population = True` also tells `_assemble_features` to skip the row
  cap when assembling the **training** frame (serving/predict still honors the
  cap for speed). `_fit` delegates estimator training to the backend and prefers
  a backend-provided CV score.
- Result: with no backend (or `bounded`), behavior is unchanged — models still
  train on the bounded reservoir sample.

**Verify (E2E):** `tests/test_train_backend.py`
- Synthetic population (6000 rows) well above the reservoir cap (400): the
  `full` backend trains on all 6000 rows and its `cv_r2` **improves** vs the
  `bounded` backend's capped-sample fit; `n_rows` reflects the full population.
- Default backend (no `train_backend` passed) → `bounded`, `n_rows` capped, no
  "full population" training.
- `IncrementalTrainBackend` trains on the full frame (n_rows == full population)
  and reports a sane CV.
- `get_train_backend` resolves bounded/full/incremental/dask and rejects unknown
  names; `DaskBackend` raises `RuntimeError` when the optional deps are absent.

**Implement:**
- Add incremental/partial-fit estimators (where the algorithm supports it) or a
  pluggable Dask/Spark backend selected by config, so tree/cluster models can
  train on the full population when a backend is configured.
- Keep the bounded-sample default when no backend is set (no behavior change for
  existing users).

**Verify (E2E):**
- Test (local, synthetic > `MAX_TRAIN_ROWS`): with a Dask/local-backend config,
  assert the model trains on the full frame (not just the reservoir) and
  metrics improve vs the sampled baseline.
- Test: with no backend, assert behavior is unchanged (still bounded sample).

---

## PR-11 — Streaming / incremental profiling  *(tracker §G G5)*

**Why:** `profiler.py` re-profiles from scratch on each `profile_dataset`. As
new data lands, this is wasteful and can thrash `dataset_sig`/caches.

**Implement:**
- Add incremental update of column stats + relationship discovery as new rows
  arrive (reservoir-counted stats; decay or windowing for drift-sensitive
  columns).
- Reconcile `dataset_sig` semantics so incremental updates don't needlessly
  invalidate models (only invalidate on schema/role change, not on row growth).

**Verify (E2E):**
- Test: profile a dataset, append rows, run incremental update; assert stats
  update and that a pure row-count growth does NOT change the model-invalidation
  signature (while a schema change still does).
- Test: incremental profiling of a growing stream stays within a bounded time
  budget (tie to the A4 discovery budget).

---

## Explicitly OUT OF SCOPE (handled elsewhere)

- **Authentication / multi-tenant isolation** — a separate system in front of
  this engine owns auth and tenant scoping. No work item here.

## Verification cadence (global)

After **every** PR:
1. `pytest` green (including the new E2E gate for that PR).
2. `ruff check` + `ruff format --check` + `mypy` clean (CI gates).
3. Update this doc: move the item ✅ and note the verifying test path.
4. PR title references the item (e.g. `PR-1: ...`).
