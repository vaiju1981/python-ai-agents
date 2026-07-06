# python-ai-agents

A thin orchestration, trust, and product-runtime layer for building Python AI agents.

You bring the substrate: OpenAI Agents SDK, Pydantic AI, LangGraph, LlamaIndex,
AutoGen, local models, cloud models, and Python data tools.

`python-ai-agents` gives you the product layer above them: request context,
tool authorization, guardrails, audit, durable workspaces, checkpoints, replay,
eval discipline, and adapters that let agents compose behind one seam.

This project is not a replacement for LangChain, LangGraph, Pydantic AI, OpenAI
Agents SDK, or AutoGen. It uses them as dependencies and adds a small,
opinionated layer for building long-running, governed, observable products.

## Dependency Shape

The core stays intentionally small.

- `python_ai_agents.core`: no analytics, UI, dataframe, plotting, or ML dependencies.
- `python_ai_agents.adapters`: optional substrate adapters.
- `demos/analytics`: owns DuckDB, Polars/Pandas, Plotly, Streamlit/FastAPI, and ML dependencies.

SQLite is acceptable in core because it is part of Python's standard library and is useful for
coordination, recovery, checkpoints, and audit.

## Core Runtime Shape

The first core runtime slice includes:

- a universal `Agent` seam;
- a small `ModelPort` seam for chat-model adapters;
- a `DefaultAgent` loop that can call selected tools through validation, approval,
  timeout, and result-capping hooks;
- tenant/session-scoped conversation memory for long-running agents;
- in-memory and SQLite audit/checkpoint stores;
- idempotency and retryability primitives for replay-safe products.

## Flagship Demo: Analytics Agent

The analytics demo is a complete application, not a notebook:

- upload one or more CSVs;
- import them into DuckDB;
- profile columns and infer metrics, dimensions, time columns, keys, and relationships;
- ask natural-language questions through governed read-only tools;
- generate SQL, tables, charts, exports, and workspace artifacts;
- refine the semantic catalog with LLM assistance.

It exists to prove the thin core can support real products.

## Local Environment

The initial development environment is the existing Conda env:

```bash
conda activate mlv2Py3
python --version
```

The current target is Python 3.10+.

### Install Core

From the repo root:

```bash
python -m pip install -r requirements-core.txt
```

For core development:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Or create a Conda environment:

```bash
conda env create -f environment.yml
conda activate python-ai-agents
```

### Install Analytics Demo

From the repo root:

```bash
python -m pip install -r requirements-analytics-demo.txt
```

Or create a demo-focused Conda environment:

```bash
conda env create -f environment-analytics-demo.yml
conda activate python-ai-agents-analytics
```

The analytics dependencies are intentionally demo-scoped. They should not be imported by
`python_ai_agents.core`.

### Ollama Model Tests

The Ollama adapter uses Ollama's local HTTP API with the Python standard library, so it
does not add a dependency to core.

The live model smoke tests are opt-in:

```bash
ollama list
PAA_RUN_OLLAMA_TESTS=1 python -m pytest tests/test_ollama_adapter.py
```

The default live test matrix is:

- `gemma4:31b-cloud`
- `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0`
- `ornith:latest`

Normal test runs use a fake Ollama transport and do not require Ollama to be running.

### Optional Tool Validation

Core includes a pluggable `ToolArgumentValidator` seam and lightweight defaults. For
full JSON Schema validation, install the optional extra:

```bash
python -m pip install -e .[tools-jsonschema]
```

Then use `python_ai_agents.adapters.JsonSchemaToolArgumentValidator`.
