# Status & Production Readiness

_Last updated: 2026-07-06._

## Summary

`python_ai_agents` core is a thin, composing trust/runtime layer. The analytics
demo is a truthful, governed NL-analytics app with descriptive **and** predictive
tools. The **core is now production-hardened** (P0–P3 closed, `mypy --strict`
clean, CI gate). The remaining work is the **demo's multi-user hardening (P1)**.

Local gate (conda `mlv2Py3`, Python 3.10): `ruff check` + `ruff format --check` +
`mypy` (0 issues) + `pytest` (**187 passed, 21 skipped**), all green. CI runs the
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

## Remaining — Demo multi-user hardening (P1, in progress)

| Pri | Gap | Plan |
|-----|-----|------|
| P1 | No auth / multi-user isolation (single-process Streamlit) | Add login + per-user session scoping. |
| P1 | No durable per-user persistence (data + catalog live in a tempdir, lost on restart) | Per-user persistent DuckDB file + saved catalog; cleanup on session end. |
| P2 | Chat charts are best-effort (regex-extract JSON from tool text) | Structured tool-result channel so chart data is exact. |
| P2 | No NL→tool accuracy eval for the demo | Wire the core eval harness over a fixed question set. |

## Run / verify (env: conda `mlv2Py3`, Python 3.10)

```bash
MLV=/opt/homebrew/Caskroom/miniconda/base/envs/mlv2Py3/bin/python
"$MLV" -m pip install -e '.[dev,analytics-demo]'
"$MLV" -m ruff check src tests demos/analytics/src && "$MLV" -m mypy && "$MLV" -m pytest -q
ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \
    streamlit run demos/analytics/src/analytics/app.py
```
