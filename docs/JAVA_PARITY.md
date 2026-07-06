# Java Parity Notes

`java-ai-agent` is the reference implementation for this Python project. The Python port should
keep the same philosophy, while staying Pythonic and thin over ecosystem substrates.

## Port First

- Universal seams: `Agent`, `AgentRequest`, `AgentResponse`, `RequestContext`.
- Trust at the seam: `Trust.govern`, guardrails, deadlines, tool approval, audit.
- Replay safety: typed stop reasons, retryability, idempotency keys, deterministic stores.
- Local runtime stores: in-memory and SQLite audit/checkpoint/idempotency stores.
- Ollama-first live smoke tests for local/cloud model substrates.
- Analytics demo: CSV/DuckDB catalog, safe SQL tools, relationship discovery, query planning.

## Port Later

- ModelPort abstractions, streaming, token budgets, and structured output helpers.
- Conversation and episodic memory with tenant boundaries.
- Observers, redaction, recording, and replay.
- RAG, skills, reflection, supervision, and deep-agent planning.
- MCP, OpenTelemetry, durable SQL stores, and production service packaging.

## Keep Out Of Core

- DuckDB, pandas, Polars, Plotly, Streamlit/FastAPI, ML/statistics libraries.
- Vendor SDKs and heavyweight substrate packages.
- Analytics-specific schema inference and query-planning dependencies.
