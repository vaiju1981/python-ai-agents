# Analytics Demo

The analytics demo is the flagship end-to-end product built on the thin core.

It should support:

- one or more CSV uploads;
- DuckDB-backed ingestion and safe SQL execution;
- deterministic profiling and semantic model inference;
- relationship discovery across files;
- governed read-only tools for analysis;
- generated artifacts: SQL, tables, charts, exports, model summaries;
- LLM-assisted catalog refinement.

Demo-only dependencies belong here, not in `python_ai_agents.core`.

Planned demo libraries:

- DuckDB for analytical SQL;
- Polars/Pandas for data profiling and transformations;
- Plotly for charts;
- Streamlit or FastAPI for the UI/API;
- scikit-learn/statsmodels for ML/statistical tools.

