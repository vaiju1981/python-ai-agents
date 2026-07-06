# python-ai-agents Design

## North Star

Build a thin, vendor-neutral orchestration and trust layer for Python agents. The
project should compose with the existing Python ecosystem rather than compete with it.

The first proof is an end-to-end analytics app over user-provided CSV files. Future
sources can include SQLite, Postgres, DuckDB files, and Neo4j/graph data.

## Principles

1. Use the ecosystem. OpenAI Agents SDK, Pydantic AI, LangGraph, LlamaIndex,
   AutoGen, DuckDB, Polars, pandas, scikit-learn, statsmodels, and Plotly are
   substrates or capability libraries.
2. Keep the core thin. Core owns seams, policy, audit, checkpoints, and product
   runtime primitives.
3. Trust is the default. Tool effects, approval policy, audit, and request context
   are first-class.
4. Long tasks are normal. Workspaces, checkpoints, resumability, and artifacts are
   part of the expected product shape.
5. Demos must be real. The analytics demo should be deterministic where cheap,
   visibly artifact-producing, and useful on messy user datasets.

## Layers

| Layer | Owned By | Examples |
| --- | --- | --- |
| L0 Substrate | Dependencies | OpenAI Agents SDK, Pydantic AI, LangGraph, LlamaIndex, AutoGen |
| L1 Core Runtime | `python-ai-agents` | `Agent`, `RequestContext`, workspaces, checkpoints, adapters |
| L2 Trust & Ops | `python-ai-agents` | tool effects, approval, guardrails, audit, observers, replay, evals |
| L3 Capability Packs | Optional packages/demos | analytics/data science, graph analytics, document workflows |

## Core Dependency Boundary

Core may depend on small runtime libraries such as Pydantic and AnyIO. It should not
depend on DuckDB, pandas, Polars, Plotly, Streamlit, FastAPI, scikit-learn, or
statsmodels.

SQLite can be used by core via the Python standard library for local coordination,
checkpointing, and audit storage.

## Initial Milestones

1. Core protocols and dataclasses: `Agent`, `AgentRequest`, `AgentResponse`,
   `RequestContext`, `ModelPort`, `Message`, `ToolCall`, `Tool`, `ToolEffect`,
   `ToolApprover`, `AuditSink`.
2. Trust wrapper: input/output guardrails, deadlines, tool policy hooks, audit.
   (Initial agent and tool wrappers exist.)
3. Tool hardening: selected-tools enforcement, argument validation seams, per-tool
   timeout, and framed/capped tool results.
4. Long-agent memory: tenant/session-scoped conversation memory, history browsing,
   windowed short-term memory, and SQLite conversation persistence for local rollout.
5. Observability: observer hooks for turn/model/tool/error/usage events, with
   recording, redaction, and token accounting kept separate from audit. **Done.**
6. Durable local stores: SQLite checkpoint store and audit store. **Done.**
7. Substrate adapters: start with one real adapter, then add more. (Ollama now
   supports both the `Agent` and `ModelPort` seams.) **Done.**
8. Recoverable workflow: thin LangGraph adapter that bridges our `CheckpointStore`
   to LangGraph's `BaseCheckpointSaver`, wraps a compiled graph as our `Agent`
   protocol with `RequestContext`/audit, and supports interrupt/resume.
   (`langgraph` optional extra.) **Done.**
9. Smaller core improvements: built-in guardrails (keyword blocklist, PII scrub,
   injection heuristic), token-budget enforcement over `Usage`, Pydantic
   structured-output helper, streaming `ModelPort` seam, and eval harness.
   Ecosystem adapters for Presidio (PII), Guardrails AI (injection), and
   DeepEval (eval scoring) are optional extras; core built-ins are
   zero-dependency fallbacks. **Done.**
10. Analytics demo: multi-CSV import, profile, semantic model, governed tools,
    artifact workspace, and chat/API.

The Python port tracks `java-ai-agent` deliberately; see `docs/JAVA_PARITY.md` for the
conversion map and sequencing.

## Test Model Substrates

Ollama is the first lightweight local/cloud model test substrate. The adapter lives
outside core in `python_ai_agents.adapters` and talks to Ollama's HTTP API without
adding a client dependency.

The opt-in live smoke-test matrix is:

- `gemma4:31b-cloud`
- `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0`
- `ornith:latest`
