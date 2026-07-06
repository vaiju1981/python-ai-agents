# Java Parity Notes

`java-ai-agent` is the reference implementation for this Python project. The Python port should
keep the same philosophy, while staying Pythonic and thin over ecosystem substrates.

## Port First

- Universal seams: `Agent`, `AgentRequest`, `AgentResponse`, `RequestContext`.
- Model seams: `ModelPort`, `ModelRequest`, `ModelResponse`, `Message`, `ToolCall`.
- Trust at the seam: `Trust.govern`, guardrails, deadlines, tool approval, audit.
- Default runtime loop: `DefaultAgent` over model calls and governed tool calls.
- Tool hardening: selector-enforced tools, argument validation, timeouts, framed/capped results.
- Replay safety: typed stop reasons, retryability, idempotency keys, deterministic stores.
- Long-agent memory: conversation stores, browsable history, and windowed memory.
- Local runtime stores: in-memory and SQLite audit/checkpoint/idempotency stores.
- Ollama-first live smoke tests for local/cloud model substrates.
- Analytics demo: CSV/DuckDB catalog, safe SQL tools, relationship discovery, query planning.

## Port Later

- Streaming, token budgets, and structured output helpers.
- Episodic memory with tenant boundaries.
- Observers, redaction, recording, and replay.
- RAG, skills, reflection, supervision, and deep-agent planning.
- MCP, OpenTelemetry, durable SQL stores, and production service packaging.

## Keep Out Of Core

- DuckDB, pandas, Polars, Plotly, Streamlit/FastAPI, ML/statistics libraries.
- Vendor SDKs and heavyweight substrate packages.
- Analytics-specific schema inference and query-planning dependencies.
