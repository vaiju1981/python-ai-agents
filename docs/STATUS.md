# Status & Production Readiness

_Last updated: 2026-07-06._

## Summary

`python_ai_agents` core is a thin, composing trust/runtime layer. The analytics
demo is a truthful, governed NL-analytics app with descriptive **and** predictive
tools. The **core is production-hardened** (P0–P3 closed, `mypy --strict` clean,
CI gate) and the **demo is hardened** (structured charts, resource cleanup, NL
eval). Only auth / multi-user hosting is deferred by request.

Local gate (conda `mlv2Py3`, Python 3.10): `ruff check` + `ruff format --check` +
`mypy` (0 issues) + `pytest` (**188 passed, 22 skipped**), all green. CI runs the
same on every push/PR.

## Core — DONE

- Thin, composing surface (109 symbols): reinvented RAG / tokenizer / DAG runner /
  prompt templating / memory summarizers removed in favor of libraries.
- Ordered trust seam (unavailable → validate → approve → timed-invoke → cap),
  guardrails, audit, checkpoints, observers, eval, token budget, SQLite persistence.
- **P0** tool timeout now bounds blocking/CPU tools (worker-thread + abandon-on-timeout);
  regression-tested. **P0** broken a2a adapter removed.
- **P1** within-step deadline checks; streamed tool calls assembled by id; OTel span
  per-async-task (ContextVar); eval scorer offloaded off the loop.
- **P2** conversation-store race fixed (built in `__post_init__`); `RedactingObserver`
  rebuilds by keyword; **`py.typed` + `mypy --strict` clean**; CI added.
- **P3** `.gitignore` added; `ruff format` adopted repo-wide; lint clean.
- Proven adapters: Ollama, LangGraph.

## Demo (analytics) — DONE

- CSV → DuckDB import, deterministic profiling + semantic model + relationship
  discovery; built once per session (no per-message rebuild, no connection leak).
- Descriptive tools (9) + **predictive/causal tools (7)**: `build_model`, `forecast`,
  `ab_test`, `causal_effect`, `uplift`, `cluster`, `anomaly_detection` — correct
  implementations, honest caveats, behavioral tests.
- Real Plotly charts (Insights, Chat, SQL); deterministic on-load insights.
- Truthful copy and dependencies.

## Demo hardening — DONE

- **Structured chart channel:** `ToolResult` carries an optional `data` payload
  (never sent to the model); row-returning tools attach their rows and the app
  charts them exactly — no more best-effort text parsing.
- **Resource cleanup:** re-import (and `mkdtemp`) only when the upload changes;
  the prior dataset's DuckDB connection + tempdir are reclaimed on load (fixes a
  per-rerun tempdir leak).
- **NL→tool eval:** gated (`PAA_RUN_OLLAMA_TESTS`) test wiring the core
  `EvalRunner` over the analytics agent, as a regression guard on answer quality.
- **Season-aware forecast:** Holt-Winters with an additive seasonal component
  (auto-detected period), falling back to trend-only / linear.
- **Model lifecycle (first cut):** `ModelStore` (InMemory/File) with train-once
  caching keyed by dataset signature + task + target + predictors + algorithm;
  retrain on data change, TTL, or explicit request. Full plan, incl. drift /
  scheduled retrain / `predict` serving / MLflow adapter, in
  [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md).

## Deferred (by request)

- **Auth / multi-user isolation and durable per-user persistence.** The demo runs
  single-user; uploaded data lives for the session. Add authentication and a
  per-user store when hosting it as a shared service.

## Run / verify (env: conda `mlv2Py3`, Python 3.10)

```bash
MLV=/opt/homebrew/Caskroom/miniconda/base/envs/mlv2Py3/bin/python
"$MLV" -m pip install -e '.[dev,analytics-demo]'
"$MLV" -m ruff check src tests demos/analytics/src && "$MLV" -m mypy && "$MLV" -m pytest -q
ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \
    streamlit run demos/analytics/src/analytics/app.py
```
