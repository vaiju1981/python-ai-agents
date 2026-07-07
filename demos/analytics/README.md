# Analytics Demo

A truthful, end-to-end analytics agent built on the thin core: point it at your
CSVs and it understands them, answers questions in plain English with **real
numbers**, and shows charts and proactive insights. Numbers are never invented —
every chart and insight is derived from a governed read-only query.

## Run it

```bash
python -m pip install -r requirements-analytics-demo.txt
ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \
    streamlit run demos/analytics/src/analytics/app.py
```

## What it does

- **Upload CSVs** — imported into DuckDB with external file access locked down.
- **Understands the data, fast** — deterministic profiling and semantic-model
  inference (metrics, dimensions, time columns, keys) plus cross-file
  relationship discovery. No LLM on the load path, so it's quick.
- **Insights tab** — on load, deterministic insights per metric: overall total,
  trend direction over time, top breakdowns, and data-quality flags.
- **Chat** — ask in plain English; the agent answers with governed read-only
  tools (`run_query`, `trend`, `compare`, `summarize`, `correlate`, `outliers`,
  `regression`, `run_sql`) and renders a chart when the result shape fits.
- **SQL tab** — a read-only DuckDB escape hatch, with an automatic chart.
- **Optional** (off by default) — an LLM pass to relabel column roles.

The source, profile, semantic model, and agent are built **once per dataset**
and cached in the session, so questions don't re-import or reconnect.

## Dependencies

Demo-only — never imported by `python_ai_agents.core`:

- **DuckDB** — analytical SQL and CSV ingestion
- **Pandas** — dataframe glue for rendering
- **Plotly** — charts
- **Streamlit** — UI
- **scikit-learn** — the linear-regression tool
