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
| **Drift** | watch live prediction error / input-distribution shift (via observer + eval hooks) and retrain past a threshold | 📋 planned |
| **Scheduled** | cron / `CronCreate` retrains served models on new data on a cadence | 📋 planned |

## Versioning & serving

- Each record stores `{key, trained_at, metadata (metrics), model}`. Keep last N
  per key; allow rollback/compare. *(first cut keeps one per key)*
- **Serving** splits train from inference: a future `predict(model_key, rows)`
  loads a persisted model and scores without retraining.

## First cut (implemented)

`ModelStore` protocol + `InMemoryModelStore`/`FileModelStore`; `build_model` does
key-based train-once caching with TTL and an explicit `retrain` flag, and reports
`cached` + `trained_at`. The demo wires a `FileModelStore` under the dataset's
working directory. Drift, scheduled retrain, `forecast` caching, `predict`
serving, and the MLflow adapter are the documented next steps.
