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
