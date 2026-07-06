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
- Observability: observer hooks, recording, redaction, and token accounting.
- Local runtime stores: in-memory and SQLite audit/checkpoint/idempotency stores.
- Ollama-first live smoke tests for local/cloud model substrates.
- Recoverable workflow: thin LangGraph adapter bridging our checkpoint store to
  LangGraph's `BaseCheckpointSaver`, with `RequestContext`/audit and interrupt/resume.
- Built-in guardrails (zero-dep fallbacks): keyword blocklist, PII scrub, injection heuristic.
- Ecosystem guardrail adapters: Presidio (PII), Guardrails AI (injection/content validation).
- Token-budget enforcement over `Usage` events.
- Pydantic structured-output helper with retry-on-validation-failure.
- Streaming `ModelPort` seam (`StreamingModelPort` + `StreamingModelAdapter`).
- Eval harness: `EvalCase`, `EvalRunner`, pluggable scorers.
- DeepEval eval-scoring adapter for LLM-as-judge metrics.
- Analytics demo: CSV/DuckDB catalog, safe SQL tools, relationship discovery, query planning.

## Port Later

- Episodic memory with tenant boundaries.
- Replay adapters that consume recorded model/tool events.
- RAG, skills, reflection, supervision, and deep-agent planning.
- MCP, OpenTelemetry, durable SQL stores, and production service packaging.

## Keep Out Of Core

- DuckDB, pandas, Polars, Plotly, Streamlit/FastAPI, ML/statistics libraries.
- Vendor SDKs and heavyweight substrate packages.
- Analytics-specific schema inference and query-planning dependencies.
