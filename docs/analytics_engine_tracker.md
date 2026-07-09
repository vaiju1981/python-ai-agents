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
| A5 | Typed composite keys + many-to-many handling | `relationships.py`, `query_planner.py` | ⬜ | Extend composite search past size 2; explicit cardinality so `feature_frame_sql` avoids fan-out on many-to-many. |
| A6 | Catalog role override (measure/key) | `catalog.py`, `profiler.py` | ⬜ | Let a user/config force a column's role (e.g. `price` is a measure, not an identifier) so profiling never mis-inferres domain meaning. |

## B. Generic tools inspired by the ATLAS engine

Reusable, column-agnostic algorithms identified in `atlas/` (see §C). Each becomes
a governed `READ_ONLY` tool parameterized by `valueCol` / `entityCol` / `dateCol`
instead of hardcoded gaming vocabulary.

| # | Tool | Source pattern (atlas) | Status | Notes |
|---|------|------------------------|--------|-------|
| B1 | `change_point` — regime/break detection | `changepoint._binseg` (binary segmentation) | ✅ | Generic 1-D daily-series break detection + Cohen-d effect on the driver metric. |
| B2 | `matched_impact` — matched-control DiD | `backtest/harness.py` | ⬜ | Generalize `event_impact` with caliper-matched never-treated controls + A/A synthetic-null + parallel-trends gate. |
| B3 | Walk-forward + conformal intervals | `estimators/_core.py` (CQR) | ⬜ | Honest low/high bands for `forecast`/`build_model` predictions, leakage-guarded. |
| B4 | `segment` — value/intensity segmentation | `segments.build`, `coverage` | ⬜ | Generic cohort segmentation on any category + numeric value column. |
| B5 | Backtest harness | `backtest/harness.run` | ⬜ | Reusable pre/post vs matched-control validation driver behind B2. |
| B6 | Decision/approval governance store | `decisions.py` | ⬜ | Generic approval workflow for high-impact recommendations. |
| B7 | Portfolio / budget optimizer | `optimizer/portfolio.py` | ⬜ | Generic NSGA-II Pareto / greedy ROI frontier over scored items (e.g. which models/levers to act on). |

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

## D. Implementation inventory (done this pass)

- `demos/analytics/src/analytics/dataset_fingerprint.py` — new; table checksum.
- `demos/analytics/src/analytics/safe_sql.py` — new; sqlglot-based guard.
- `demos/analytics/src/analytics/model_store.py` — A1: fingerprint in `dataset_sig`.
- `demos/analytics/src/analytics/models_tools.py` — A1 wiring, A2 PSI/KS drift.
- `demos/analytics/src/analytics/relationships.py` — A4 budget + candidate filter.
- `demos/analytics/src/analytics/toolset.py` — A3 safe `run_sql`, B1 `change_point`.
- `tests/test_analytics_production.py` — new; covers A1–A4, B1.

## E. Open risks / follow-ups

- `sqlglot` added to `requirements-analytics-demo.txt` (optional import; graceful fallback).
- A5/A6/B2–B7 are scoped but not yet implemented (see status above); each gets its
  own test before merge.
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
honesty primitives. It reliably answers "what happened and what's related?" but is
**not yet a decision system**: it cannot yet grade its own confidence, validate a
causal claim, prove data freshness, or refuse when unsure — the things that make
an answer defensible in a finance/regulatory review.

### What's good (keep / lean into)
- Zero-config generalization (auto profile → relationship discovery → cross-table
  joins) — the moat vs. dbt/LookML/Cube.
- Governance is first-class: READ_ONLY tools, sqlglot SQL firewall, audited calls.
- Honesty primitives: effect sizes + p-values, causation caveats, PSI/KS drift,
  train-once/serve-many with fingerprint invalidation.
- Deterministic + test-covered → analytical correctness is regression-testable.

### The defensibility gaps (blockers for the chosen bar)
- **No provenance on answers** — no answer-level record of SQL + dataset fingerprint
  + row counts + timestamp, so results aren't reproducible or challengeable.
- **No trust grade** — a 24%-coverage join reads as authoritative as a 100% one.
- **Causal claims unvalidated** — OLS with a caveat, but no matched controls / A-A
  null / parallel-trends gate.
- **No freshness / lineage / reconciliation** — can't answer "is this current?",
  "where did this number come from?", "why doesn't it match the dashboard?".
- **No abstention** — no "insufficient evidence, I won't answer" path.
- **Scale ceiling** — single-node DuckDB + whole-table pandas; no warehouse pushdown.

### Roadmap (phased; defensibility first)

| Phase | Item | Serves | Tracker link | Status |
|-------|------|--------|--------------|--------|
| P0 — Defensible answers | Answer-provenance envelope: every tool result carries `{sql, dataset_fingerprint, row_count, generated_at, engine_version}` | all | new `provenance.py` | ⬜ |
| P0 | Trust-grading on every answer (TRUSTED/DIRECTIONAL/INSUFFICIENT) from coverage, n, and validation gates | all | ATLAS `causal/estimator.py` | ⬜ |
| P0 | Abstention path: tools return INSUFFICIENT (not a guess) below evidence thresholds | analyst/exec | `toolset.py`, `models_tools.py` | ⬜ |
| P0 | Reproducibility: deterministic re-run of any answer from its provenance envelope | exec/audit | `provenance.py` + test | ⬜ |
| P1 — Validated causal | Matched-control DiD `matched_impact` + A/A synthetic-null + parallel-trends gates | operator | B2/B5 | ⬜ |
| P1 | Conformal prediction intervals for forecast/predict (leakage-guarded) | operator/analyst | B3 | ⬜ |
| P1 | Freshness + lineage metadata surfaced on answers (max event date, source, staleness) | all | new `freshness.py` | ⬜ |
| P2 — Scale | Warehouse pushdown (Snowflake/BigQuery/Postgres) behind the `DataSource` port | engineer | `data_source.py` adapters | ⬜ |
| P2 | Incremental / sampled profiling + SQL-native feature assembly (drop whole-table pandas) | engineer | `profiler.py`, `models_tools.py` | ⬜ |
| P2 | Reconciliation tool: compare a computed metric to a declared source-of-truth | exec/audit | new `reconcile.py` | ⬜ |
| P3 — Decision system | Approval/governance store for high-impact recommendations | operator | B6 (`decisions.py`) | ⬜ |
| P3 | Portfolio/budget optimizer over scored actions (Pareto/greedy ROI) | operator | B7 (`optimizer`) | ⬜ |
| P3 | Feedback loop: capture answer/recommendation outcomes to tune trust thresholds | all | `audit_store.py` + new | ⬜ |
| Cross-cut | Semantic verification: check the LLM chose answerable, correct metrics/dims | all | new `verify.py` | ⬜ |
| Cross-cut | Typed composite / many-to-many keys (fan-out-safe features) | engineer | A5 | ⬜ |
| Cross-cut | Catalog role overrides (force measure vs. identifier) | analyst/eng | A6 | ⬜ |

**Recommended first slice (highest credibility per unit effort):** P0
provenance envelope + trust-grading + abstention. It reuses numbers the engine
already computes (coverage, n, cv scores, drift), touches every answer, and is
the minimum bar for "audit/defensible". Ship P1 matched-control impact next to
make the causal story defensible.

