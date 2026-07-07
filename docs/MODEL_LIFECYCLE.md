# Model Lifecycle

How the analytics demo persists, reuses, and re-trains models. Applies to the
predictive tools (`build_model`, and — as a next step — `forecast`).

## Problem

Tools that train ad-hoc on every call retrain identical queries, aren't
reproducible, and never separate *training* from *serving*. Fine at demo scale;
wrong for production.

## Design

**ModelStore seam** (`demos/analytics/src/analytics/model_store.py`) — ML stays
out of the thin core. Backends: `InMemoryModelStore` (dev), `FileModelStore`
(local, pickle to a directory), and — for production — an **MLflow registry
adapter** (compose, don't compete). *Next step.*

**Model key** — a deterministic hash of `(dataset signature, task, target, sorted
predictors, algorithm, params)`. Same inputs → same key → cache hit. The dataset
signature is the one already computed for the DuckDB import, so **a data change
changes the key**.

## When we train — on demand, cached

A `build_model` call computes the key and asks the store for a fresh model:
- **hit** → return the stored metrics (`cached: true`, `trained_at`);
- **miss** → train, persist the model + metrics, return (`cached: false`).

## When we re-train — invalidation triggers

| Trigger | Mechanism | Status |
|---|---|---|
| **Data changed** | dataset signature is part of the key → new data ⇒ new key ⇒ retrain | ✅ implemented |
| **Staleness (TTL)** | `model_ttl`; a cached model older than the TTL is ignored | ✅ implemented |
| **Explicit** | `retrain: true` argument forces a fresh fit | ✅ implemented |
| **Drift** | per-feature training stats travel with the model; `predict` compares the scored rows against them (standardized mean shift, threshold 0.5) and recommends `build_model(retrain=true)` when it flags. Retraining stays an explicit call — no silent refits. | ✅ implemented |
| **Scheduled** | by composition: any scheduler (cron, CI, Airflow) calls `build_model(retrain=true)` on its cadence. No in-process scheduler to babysit. | ✅ mechanism documented |

## Versioning & serving

- Each record stores `{key, trained_at, metadata (metrics + train stats), model}`.
  One record per key; last-N history/rollback is deferred until something needs it.
- **Serving — implemented:** the `predict` tool loads the stored model for a spec
  (training once if absent) and scores rows — optionally filtered
  (`filters: [{column, op, value}]`) — without retraining. It reports the
  prediction summary, `model_cached`/`trained_at`, and the drift check.

## Implemented / deferred

Implemented: `ModelStore` protocol + `InMemory`/`File` backends; key-based
train-once caching with TTL + explicit `retrain`; `predict` serving with drift
detection. The demo wires a `FileModelStore` under the dataset's working
directory.

Deferred until a real consumer exists: MLflow registry adapter (when deploying
alongside MLflow, use it directly), last-N version rollback, and `forecast`
caching (the Holt-Winters fit is sub-second on monthly aggregates — a cache
would only add staleness).
