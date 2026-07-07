# Analytics Demo

A truthful, end-to-end analytics agent built on the thin core: point it at your
CSVs and it understands them, answers questions in plain English with **real
numbers**, and shows charts and proactive insights. Numbers are never invented —
every chart and insight is derived from a governed read-only query.

## Run it

```bash
python -m pip install -r requirements-analytics-demo.txt
ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \
    streamlit run demos/analytics/src/analytics/app.py --server.maxUploadSize=4096
```

## What it does

- **Two ways to load data** — *Upload CSV* (small files, through the browser), or
  *Local file path* for **large/production files**: DuckDB reads them **in place**
  (no upload) into a **file-backed database**, so a multi-GB CSV stays out-of-core
  (on disk, not RAM). Import locks external file access down afterward.
- **Understands the data, fast** — deterministic profiling and semantic-model
  inference (metrics, dimensions, time columns, keys) plus cross-file
  relationship discovery. No LLM on the load path, so it's quick.
- **Insights tab** — on load, deterministic insights per metric: overall total,
  trend direction over time, top breakdowns, and data-quality flags.
- **Chat** — ask in plain English; the agent answers with governed read-only
  tools and renders a chart when the shape fits.
  - *Descriptive:* `run_query`, `trend`, `compare`, `summarize`, `correlate`,
    `outliers`, `regression`, `run_sql`.
  - *Predictive & causal:* `build_model`, `forecast`, `ab_test`, `causal_effect`,
    `uplift`, `cluster`, `anomaly_detection` — each reports its method and the
    number of rows it used; causal/uplift carry an explicit "not proof of
    causation" caveat.
- **Scale** — descriptive answers/insights run as SQL over the *full* table
  (exact). ML tools also use the full table **by default**; on very large data you
  can cap the training rows via the sidebar ("Max ML training rows", 0 = all) to
  trade accuracy for speed. The chosen count is always reported.
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
- **scikit-learn / scipy / statsmodels** — modeling, statistical tests, forecasting
