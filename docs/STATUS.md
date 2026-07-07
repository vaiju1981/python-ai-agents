# Status & Production Readiness

_Last updated: 2026-07-06._

## Summary

`python_ai_agents` core is a thin trust/runtime layer that now **composes with the
Python ecosystem instead of reinventing it**. The analytics demo is a **truthful,
governed NL-analytics app**. Both have green tests (179 passed, 22 skipped under
`mlv2Py3`), but neither is production-hardened yet — the gaps below are the work
remaining. Priorities: **P0** = blocks production / correctness, **P1** = needed for
a real deployment, **P2** = quality/robustness, **P3** = hygiene.

## Core — status

**Done**
- Thin, composing surface: 109 public symbols. Reinvented RAG / tokenizer / DAG
  runner / prompt templating / memory summarizers removed in favor of libraries
  (LlamaIndex, tiktoken, LangGraph, stdlib).
- Agent loop with an ordered trust seam (unavailable → validate → approve → timed
  invoke → cap), guardrails, audit, checkpoints, observers, eval, token budget,
  SQLite conversation persistence.
- Proven adapters: **Ollama**, **LangGraph** (real tests against the libraries).

**Gaps to production** (from the earlier deep review; not fixed this pass)

| Pri | Gap | Where | Impact / fix |
|-----|-----|-------|--------------|
| P0 | Per-tool timeout can't cancel sync/blocking tools (`anyio.move_on_after` only cancels at an `await`) | `core/default_agent.py` `_invoke_with_timeout`, `core/resilient.py` | A blocking tool (sync HTTP/DB/`time.sleep`) ignores `tool_timeout_seconds` and hangs the turn. Run sync tools via `anyio.to_thread.run_sync` under `fail_after`, or require truly-async tools. |
| P0 | a2a adapter is dead even when installed — `except ImportError: pass` hides a real API mismatch | `adapters/a2a.py` | Always returns "A2A SDK not installed". Rewrite against the real `a2a-sdk` API, or remove the adapter. |
| P1 | Deadline enforced only at the loop top, not within a step | `core/default_agent.py` | One step firing several tool calls can run far past the deadline. |
| P1 | OTel observer shares one `_current_span` across concurrent turns | `adapters/otel.py` | Wrong-trace attributes + leaked (never-ended) spans under concurrency. Use OTel context, not instance state. |
| P1 | Streamed tool calls aren't assembled by id | `core/streaming.py` | Real delta-streaming providers yield fragmented/duplicated tool calls. |
| P1 | mcp / otel / guardrails_ai / deepeval adapters tested only against fakes; `deepeval.score()` runs blocking `measure()` on the loop | `adapters/*`, `tests/*` | Real integration paths unproven; offload deepeval with `to_thread`. |
| P2 | Data race on lazy `conversation_store` init on a shared agent | `core/default_agent.py` | Concurrent runs can drop history. Build in `__post_init__`. |
| P2 | `RedactingObserver` rebuilds `AgentResponse`/`ToolResult` positionally, dropping fields | `core/observe.py` | Downstream sees wrong `retryable`. Rebuild by keyword. |
| P2 | No `py.typed` marker → strict mypy can't type-check the installed package | packaging | Type safety configured but not enforced. Add `py.typed` and ship it in the wheel. |
| P2 | No CI; ~90 pre-existing repo-wide ruff findings | tooling | Add a CI gate (pytest + ruff + mypy on `mlv2Py3`/3.10). |
| P3 | `.gitignore` misses `__pycache__/` and `uv.lock` | repo | Build artifacts show as untracked. |

## Demo (analytics) — status

**Done**
- CSV → DuckDB import (external file access locked down), deterministic profiling +
  semantic model + cross-file relationship discovery.
- 9 governed read-only tools; `run_sql` guarded (statement allowlist + literal/
  comment stripping). Structured-filter injection removed with the deleted tools.
- Real Plotly charts (Insights, Chat, SQL); deterministic on-load insights
  (totals, trends, top breakdowns, data quality) — every number from a query.
- Built **once per session** (no per-message rebuild, no DuckDB connection leak);
  copy and dependencies match what actually ships.

**Gaps to production**

| Pri | Gap | Impact / fix |
|-----|-----|--------------|
| P1 | Single-process Streamlit; no auth, no multi-user/tenant isolation | Not deployable as a shared service as-is. |
| P1 | No persistence — uploaded data + catalog live in a `mkdtemp` tempdir, lost on restart; tempdirs/DuckDB connections are only reclaimed on a new upload, not on session end | Data loss on restart; resource growth over long uptime. Add a cleanup hook / persistent store. |
| P2 | Chat charts are best-effort (regex-extract the JSON array from framed tool text) | Fragile. Add a structured tool-result channel so the chart data is exact. |
| P2 | No NL→tool accuracy eval for the demo | Answer quality rides on the model with no regression guard. Wire the core eval harness over a fixed question set. |
| P2 | `SqlSource` / `GraphSource` exist but aren't wired into the UI or tested | CSV is the only exercised source path. |
| P3 | No cost/rate controls on model calls; no programmatic API (FastAPI was removed as unused) | Add if at-scale or headless use is needed. |

## Run / verify (env: conda `mlv2Py3`, Python 3.10)

```bash
MLV=/opt/homebrew/Caskroom/miniconda/base/envs/mlv2Py3/bin/python
"$MLV" -m pip install -e '.[dev,analytics-demo]'
"$MLV" -m pytest -q
ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \
    streamlit run demos/analytics/src/analytics/app.py
```
