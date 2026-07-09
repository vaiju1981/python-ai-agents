# Analytics Engine — New Code Tracker

**Goal:** Harden the generic, domain-agnostic analytics engine (`demos/analytics/`)
and lift reusable, column-agnostic capabilities from the gaming `ATLAS` engine
(`/Users/vaijanath.rao/GA/ml-docs/game-recs/engine`) into generic tools, so the
same engine can serve casino, ecommerce, health, education, and real-estate
workloads with zero domain knowledge.

**Status legend:** ✅ done · 🟡 in-progress · ⬜ planned

---

## A. Production hardening (from engine review)

| # | Item | File(s) | Status | Notes |
|---|------|---------|--------|-------|
| A1 | Dataset-fingerprint model invalidation | `model_store.py`, `models_tools.py`, `data_source.py` | ✅ | Hash table names + row counts + column schema + cheap content checksum; bake into `dataset_sig` so a changed dataset invalidates cached models instead of silently serving stale ones. |
| A2 | PSI / KS drift (upgrade mean-shift heuristic) | `models_tools.py` | ✅ | Store per-feature training quantiles; compute PSI at serving time against scored-frame quantiles; report KS too. |
| A3 | Safe `run_sql` via `sqlglot` parser | `analytics/safe_sql.py`, `toolset.py` | ✅ | Parse AST; allow only SELECT/WITH read-only; reject writes/DDL/multi-statement. Falls back to the keyword scanner if `sqlglot` is absent. |
| A4 | Relationship-discovery scaling | `relationships.py` | ✅ | Candidate-key pre-filter (drop measures + high-cardinality free-text) + global time budget (`ANALYTICS_DISCOVERY_BUDGET_SECONDS`, default 30s) so discovery stays bounded at hundreds of tables. |
| A5 | Typed composite keys + many-to-many handling | `relationships.py`, `query_planner.py` | ✅ | Composite search up to `ANALYTICS_MAX_COMPOSITE_KEY_SIZE` (default 3), seeded from partial *and* confident-but-non-unique single-column matches; explicit cardinality (incl. `many_to_many`) so `feature_frame_sql` avoids fan-out. Tested in `test_analytics_tracker.py`. |
| A6 | Catalog role override (measure/key) | `catalog.py`, `profiler.py`, `semantic_model.py` | ✅ | `Catalog.role_overrides` forces a column's role (e.g. `sales.price` → measure); applied in `profile_dataset` and surfaced on `SemanticModel.columns`. Tested. |

## B. Generic tools inspired by the ATLAS engine

Reusable, column-agnostic algorithms identified in `atlas/` (see §C). Each becomes
a governed `READ_ONLY` tool parameterized by `valueCol` / `entityCol` / `dateCol`
instead of hardcoded gaming vocabulary.

| # | Tool | Source pattern (atlas) | Status | Notes |
|---|------|------------------------|--------|-------|
| B1 | `change_point` — regime/break detection | `changepoint._binseg` (binary segmentation) | ✅ | Generic 1-D daily-series break detection + Cohen-d effect on the driver metric. Trust-graded. |
| B2 | `matched_impact` — matched-control DiD | `backtest/harness.py` | ✅ | `backtest.py`: caliper-matched never-treated controls + A/A synthetic-null + parallel-trends gate; trust-graded with abstention. Tested. |
| B3 | Walk-forward + conformal intervals | `estimators/_core.py` (CQR) | ✅ | `conformal.py`: honest low/high bands for forecasts, leakage-guarded; trust-graded on calibration coverage. Tested. |
| B4 | `segment` — value/intensity segmentation | `segments.build`, `coverage` | ✅ | `segmentation.py`: generic cohort segmentation on any category + numeric value column (top→low tiers). Tested. |
| B5 | Backtest harness | `backtest/harness.run` | ✅ | `backtest.py` drives the pre/post vs matched-control validation behind B2. Tested. |
| B6 | Decision/approval governance store | `decisions.py` | ✅ | `decision_store.py`: JSON-backed approval lifecycle + host-notify + persistence. Tested. |
| B7 | Portfolio / budget optimizer | `optimizer/portfolio.py` | ✅ | `portfolio.py`: greedy ROI frontier + Pareto (value/risk) selection over scored items. Tested. |

## C. ATLAS engine summary (source of inspiration)

Standalone, deterministic **slot-floor optimization / recommendation engine**
(gaming/casino). Reads floor meters, runs matched-control causal backtests, trains
quantile uplift models, serves recommendations + what-ifs + a portfolio optimizer
via FastAPI. ~50 modules under `atlas/`.

**Generic (liftable) capabilities:**
- Matched-control difference-in-differences backtest (`backtest/harness.py`).
- Walk-forward day-grouped OOF + split-conformal intervals (`estimators/_core.py`).
- Binary-segmentation change-point detection (`changepoint._binseg`).
- Causal borrowing + trust tiering (`causal/estimator.py`, `causal/neighborhood.py`).
- OOS validation / "does a lever move it" gates (`coverage`, `catalog`).
- Multi-objective portfolio + Monte-Carlo plan risk (`optimizer/portfolio.py`).
- Decision/approval governance (`decisions.py`).

**Domain-hardcoded (leave behind / wrap behind column-mapping config):**
`denom`, `jackpot`, `coin-in`, `hold/payback`, `theoWin`, `handlePulls`,
machine `model`/`cabinetType`, VIP/club tiers, and the gaming term dictionaries
in `metrics.py` / `metric_registry.py` / `metric_lexicon.py`.

## D. Implementation inventory

**First pass (A1–A4, B1):**
- `dataset_fingerprint.py` — table checksum. `safe_sql.py` — sqlglot guard.
- `model_store.py` / `models_tools.py` — A1 fingerprint, A2 PSI/KS drift.
- `relationships.py` — A4 budget + candidate filter.
- `toolset.py` — A3 safe `run_sql`, B1 `change_point`.
- `tests/test_analytics_production.py` — covers A1–A4, B1.

**Second pass (A5–A6, B2–B7, P0–P3, cross-cut):**
- `relationships.py` — A5 composite keys (seeded from confident non-unique
  matches; fixed `COUNT(DISTINCT (...))` and cross-table same-name joins).
- `catalog.py` / `profiler.py` / `semantic_model.py` — A6 role overrides +
  `SemanticModel.columns`.
- `backtest.py` (B2/B5), `conformal.py` (B3), `segmentation.py` (B4),
  `decision_store.py` (B6, + feedback loop), `portfolio.py` (B7).
- `provenance.py` (P0 envelope + reproducibility), `trust.py` (P0 grade +
  abstention), `freshness.py` (P1), `warehouse_sources.py` (P2), `reconcile.py`
  (P2), `verify.py` (cross-cut).
- `toolset.py` — `_ok` now attaches a trust grade to **every** answer-producing
  tool (n / coverage / gates), abstains on thin evidence for causal tools, and
  surfaces the tier in answer content.
- `models_tools.py` — `ModelsToolset._ok` gives predictive/causal answers the
  same provenance + trust envelope; `ab_test` is SQL-native (Welch's t from
  group aggregates); row-based ML bounded by `ANALYTICS_MAX_TRAIN_ROWS`.
- `tests/test_analytics_tracker.py` / `tests/test_models_tools.py` — cover
  A5–A6, B2–B7, P0–P3, cross-cut, trust-on-every-answer, abstention, and the
  SQL-native `ab_test` matching a direct scipy t-test.

## E. Open risks / follow-ups

- `sqlglot` added to `requirements-analytics-demo.txt` (optional import; graceful fallback).
- Row-level ML (random forest / k-means / isolation forest) materializes a
  bounded reservoir sample (`ANALYTICS_MAX_TRAIN_ROWS`, default 200k); this is
  inherent to those algorithms. Aggregate-only tools (`ab_test`, `summarize`,
  `correlate`, `forecast`) are already SQL-native.
- Trust thresholds (`ANALYTICS_TRUST_*`) are env-tunable; the P3 feedback loop
  (`tune_trust_thresholds`) suggests adjustments but does not auto-apply them.
- Discovery scaling (A4) changes *which* relationships are found only when the
  budget is exceeded; default budget is generous so existing tests are unaffected.

---

## F. Gap analysis & production roadmap

**Target (decided):** all personas (analyst, operator, engineer, exec), all
scales, with the **highest correctness bar: answers must be audit/defensible**
(lineage, reconciliation, reproducibility). Defensibility is the through-line
that satisfies every persona at once.

### Where it stands
A strong **analyst copilot / self-serve BI accelerator** with real governance and
honesty primitives, now extended into a **defensible decision system**: every
answer carries a provenance envelope + trust grade, causal claims are validated
with matched controls, freshness/lineage/reconciliation are surfaced, and tools
abstain when evidence is thin. The remaining ceiling is pure scale (SQL-native
feature assembly).

### What's good (keep / lean into)
- Zero-config generalization (auto profile → relationship discovery → cross-table
  joins) — the moat vs. dbt/LookML/Cube.
- Governance is first-class: READ_ONLY tools, sqlglot SQL firewall, audited calls.
- Honesty primitives: effect sizes + p-values, causation caveats, PSI/KS drift,
  train-once/serve-many with fingerprint invalidation.
- Deterministic + test-covered → analytical correctness is regression-testable.

### The defensibility gaps (status against the chosen bar)
- ✅ **Provenance on answers** — every result carries SQL + dataset fingerprint +
  row count + timestamp + engine version (`provenance.py`) and is reproducible.
- ✅ **Trust grade** — every answer graded TRUSTED/DIRECTIONAL/INSUFFICIENT from
  coverage + n + gates (`trust.py`, `_ok`); tier surfaced in the answer content.
- ✅ **Causal claims validated** — matched controls + A/A null + parallel-trends
  gate (`backtest.py`).
- ✅ **Freshness / lineage / reconciliation** — `freshness.py`, `reconcile.py`.
- ✅ **Abstention** — inferential tools return `[ABSTAIN]` below evidence thresholds.
- 🟡 **Scale ceiling** — warehouse pushdown added (`warehouse_sources.py`),
  profiling is sampled, `ab_test` is SQL-native, and row-based ML is memory-
  bounded by default (`ANALYTICS_MAX_TRAIN_ROWS`). True row-level models
  (random forest / k-means / isolation forest) still materialize a *bounded*
  sample — inherent to the algorithm, not a whole-table load.

### Roadmap (phased; defensibility first)

| Phase | Item | Serves | Tracker link | Status |
|-------|------|--------|--------------|--------|
| P0 — Defensible answers | Answer-provenance envelope: every tool result carries `{sql, dataset_fingerprint, row_count, generated_at, engine_version}` | all | `provenance.py` | ✅ |
| P0 | Trust-grading on every answer (TRUSTED/DIRECTIONAL/INSUFFICIENT) from coverage, n, and validation gates | all | `trust.py`, `toolset.py` `_ok` | ✅ | 
| P0 | Abstention path: tools return INSUFFICIENT (not a guess) below evidence thresholds | analyst/exec | `toolset.py`, `models_tools.py` | ✅ | 
| P0 | Reproducibility: deterministic re-run of any answer from its provenance envelope | exec/audit | `provenance.py` (`reproducible`) + test | ✅ |
| P1 — Validated causal | Matched-control DiD `matched_impact` + A/A synthetic-null + parallel-trends gates | operator | `backtest.py` (B2/B5) | ✅ |
| P1 | Conformal prediction intervals for forecast/predict (leakage-guarded) | operator/analyst | `conformal.py` (B3) | ✅ |
| P1 | Freshness + lineage metadata surfaced on answers (max event date, source, staleness) | all | `freshness.py` | ✅ |
| P2 — Scale | Warehouse pushdown (Snowflake/BigQuery/Postgres) behind the `DataSource` port | engineer | `warehouse_sources.py` | ✅ |
| P2 | Incremental / sampled profiling + SQL-native feature assembly (drop whole-table pandas) | engineer | `profiler.py`, `models_tools.py` | ✅ | Sampled profiling via `ANALYTICS_PROFILE_SAMPLE_ROWS` (counts stay exact); `ab_test` now computes Welch's t from SQL group aggregates (no row materialization); row-based ML (RF/k-means/IsolationForest) is memory-bounded by default via `ANALYTICS_MAX_TRAIN_ROWS` (reservoir sample). |
| P2 | Reconciliation tool: compare a computed metric to a declared source-of-truth | exec/audit | `reconcile.py` | ✅ |
| P3 — Decision system | Approval/governance store for high-impact recommendations | operator | `decision_store.py` (B6) | ✅ |
| P3 | Portfolio/budget optimizer over scored actions (Pareto/greedy ROI) | operator | `portfolio.py` (B7) | ✅ |
| P3 | Feedback loop: capture answer/recommendation outcomes to tune trust thresholds | all | `decision_store.py` (`record_outcome`, `tune_trust_thresholds`) | ✅ |
| Cross-cut | Semantic verification: check the LLM chose answerable, correct metrics/dims | all | `verify.py` | ✅ |
| Cross-cut | Typed composite / many-to-many keys (fan-out-safe features) | engineer | A5 | ✅ |
| Cross-cut | Catalog role overrides (force measure vs. identifier) | analyst/eng | A6 | ✅ |

**Status note (this pass):** all P0/P1/P2/P3 and cross-cut items are implemented
and test-covered (`tests/test_analytics_tracker.py`, `tests/test_models_tools.py`).
Trust-grading + abstention apply to **every** answer-producing tool — both the
descriptive/statistical tools (`AnalyticsToolset._ok`) and the predictive/causal
tools (`ModelsToolset._ok`) — and the graded tier is surfaced to the model. For
P2, aggregate tools (`ab_test`) are now SQL-native and row-based ML is memory-
bounded by default; genuinely row-level models still materialize a *bounded*
sample (tree/k-means cannot be expressed as pure SQL aggregates).

**Everything on the tracker is now implemented and test-covered** (A1–A6,
B1–B7, P0–P3, cross-cut). The only inherent limit is that row-level ML models
(random forest / k-means / isolation forest) materialize a *bounded* reservoir
sample rather than pure SQL aggregates — a property of the algorithms, not a
gap. Extensions beyond this tracker are captured in §G.

---

## G. Future / backlog (post-tracker; not yet scheduled)

Items intentionally deferred — the tracker above is complete without them. Pick
up when the need arises; each should ship with its own test before merge.

| # | Item | Serves | Likely file(s) | Notes |
|---|------|--------|----------------|-------|
| G1 | Warehouse-side model scoring | engineer/scale | `models_tools.py`, `warehouse_sources.py` | Push `predict` scoring down to the warehouse (e.g. UDF / SQL-scoring for linear/tree models, or vendor ML) so serving never pulls rows into pandas. Removes the last row-level materialization for inference. |
| G2 | Auto-apply feedback-loop threshold tuning | all | `decision_store.py`, `trust.py` | Today `tune_trust_thresholds()` only *recommends* raise/lower/hold. Add an opt-in policy that applies the suggestion to `ANALYTICS_TRUST_*` (with an audit entry + guardrails) so trust bars self-calibrate from labeled outcomes. |
| G3 | Distributed / out-of-core training for row-level ML | engineer/scale | `models_tools.py` | For datasets beyond the `ANALYTICS_MAX_TRAIN_ROWS` sample, add incremental/partial-fit estimators or a Dask/Spark backend so tree/cluster models can train on the full population instead of a reservoir sample. |
| G4 | Lineage graph across derived answers | exec/audit | new `lineage.py` | Chain provenance envelopes so a derived answer (e.g. a forecast built on a reconciled metric) links back through every upstream SQL + fingerprint, giving full end-to-end lineage. |
| G5 | Streaming / incremental profiling | engineer | `profiler.py` | Update column stats and relationship discovery incrementally as new data lands, instead of re-profiling from scratch. |

**When picking one up:** move it into a numbered phase above, add its status
(⬜→🟡→✅), and mirror the pattern already in the codebase (governed READ_ONLY
tool, provenance + trust envelope, dedicated test).

