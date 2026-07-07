"""Streamlit app: point at your CSVs → understand them → ask → answers, charts, insights.

Run with::

    ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \\
        streamlit run demos/analytics/src/analytics/app.py

The data source, profile, semantic model, and agent are built once per dataset
and cached in the session, so questions are answered without re-importing or
reconnecting. Every number comes from a governed read-only query; charts and
insights are derived from those results, never invented.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import anyio
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, "src")
sys.path.insert(0, "demos/analytics/src")

from demos.analytics.src.analytics.agent import create_agent  # noqa: E402
from demos.analytics.src.analytics.charts import ChartSpec, choose_chart  # noqa: E402
from demos.analytics.src.analytics.csv_source import CsvSource  # noqa: E402
from demos.analytics.src.analytics.insights import generate_insights  # noqa: E402
from demos.analytics.src.analytics.models import from_env as model_from_env  # noqa: E402
from demos.analytics.src.analytics.profiler import profile_dataset  # noqa: E402
from demos.analytics.src.analytics.schema_builder import refine_profile_with_llm  # noqa: E402
from demos.analytics.src.analytics.semantic_model import SemanticModel  # noqa: E402
from python_ai_agents import AgentRequest, NoopAgentObserver  # noqa: E402
from python_ai_agents.core.tool import ToolResult  # noqa: E402


def _render_chart(spec: ChartSpec | None, rows: list[dict] | None) -> None:
    """Render a ChartSpec + its rows as a Plotly figure, if renderable."""
    if spec is None or not rows:
        return
    df = pd.DataFrame(rows)
    if spec.x not in df.columns or (spec.y is not None and spec.y not in df.columns):
        return
    try:
        if spec.kind == "line":
            fig = px.line(df, x=spec.x, y=spec.y, title=spec.title, markers=True)
        elif spec.kind == "bar":
            fig = px.bar(df, x=spec.x, y=spec.y, title=spec.title)
        elif spec.kind == "histogram":
            fig = px.histogram(df, x=spec.x, title=spec.title)
        else:
            return
    except Exception:
        return
    st.plotly_chart(fig, use_container_width=True)


class _RowCapture(NoopAgentObserver):
    """Captures rows from the last successful tool call so the chat can chart them."""

    def __init__(self) -> None:
        self.rows: list[dict] | None = None

    async def on_tool_result(self, tool_name: str, result: ToolResult, latency: object) -> None:
        if result.error:
            return
        rows = _extract_rows(result.content)
        if rows:
            self.rows = rows


def _extract_rows(text: object) -> list[dict] | None:
    """Best-effort: pull the JSON array of row objects out of a framed tool result."""
    import json

    if not isinstance(text, str):
        return None
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    return None


def _load_dataset(named_csvs: dict[str, Path], *, refine: bool) -> None:
    """Import CSVs, profile, model, and cache everything once per dataset."""
    sig = tuple(sorted((n, p.stat().st_size) for n, p in named_csvs.items()))
    if st.session_state.get("data_sig") == sig and "source" in st.session_state:
        return

    old = st.session_state.get("source")
    if old is not None:
        try:
            old.close()
        except Exception:
            pass

    source = CsvSource(named_csvs=named_csvs)
    profile = profile_dataset(source)
    if refine:
        try:
            with st.spinner("Refining semantic schema with the model..."):
                profile = anyio.run(refine_profile_with_llm, profile, model_from_env())
        except Exception as exc:
            st.warning(f"LLM schema refinement skipped: {exc}")
    semantic = SemanticModel.from_profile(profile)

    st.session_state.update(
        data_sig=sig,
        source=source,
        profile=profile,
        semantic=semantic,
        insights=generate_insights(source, semantic, profile),
        messages=[],
    )
    st.session_state.pop("agent", None)  # force agent rebuild against new data


def _ensure_agent(provider: str, model_name: str) -> None:
    """Build the agent once per (dataset, model); reuse it across questions.

    A persistent ``_RowCapture`` observer rides along so the chat can chart the
    rows behind each answer; it is reset before every question.
    """
    model_sig = (st.session_state.get("data_sig"), provider, model_name)
    if st.session_state.get("model_sig") == model_sig and "agent" in st.session_state:
        return
    capture = _RowCapture()
    st.session_state.capture = capture
    st.session_state.agent = create_agent(
        st.session_state.source, model_from_env(), st.session_state.semantic, observers=[capture]
    )
    st.session_state.model_sig = model_sig


def main() -> None:
    st.set_page_config(page_title="Analytics Agent", page_icon="📊", layout="wide")
    st.title("📊 Analytics Agent")
    st.caption("Upload CSVs, ask in plain English, get governed answers with charts.")

    with st.sidebar:
        st.header("Model")
        provider = st.text_input("Provider", value="ollama")
        model_name = st.text_input("Model", value="ornith:latest")
        import os

        os.environ["ANALYTICS_MODEL_PROVIDER"] = provider
        os.environ["ANALYTICS_MODEL"] = model_name
        refine = st.checkbox(
            "LLM schema refinement (slower)",
            value=False,
            help="Deterministic profiling is used by default; enable for an LLM relabel pass.",
        )

        st.divider()
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
        if uploaded:
            named_csvs: dict[str, Path] = {}
            tmp_dir = Path(tempfile.mkdtemp())
            for f in uploaded:
                path = tmp_dir / f.name
                path.write_bytes(f.getbuffer())
                named_csvs[Path(f.name).stem] = path
            with st.spinner("Importing into DuckDB and profiling..."):
                _load_dataset(named_csvs, refine=refine)
            sm = st.session_state.semantic
            st.success(
                f"Loaded {len(named_csvs)} table(s) · {len(sm.metrics)} metric(s) · "
                f"{len(sm.relationships)} relationship(s)."
            )

    if "source" not in st.session_state:
        st.info("Upload one or more CSV files to get started.")
        return

    _ensure_agent(provider, model_name)
    profile = st.session_state.profile
    semantic = st.session_state.semantic
    source = st.session_state.source

    tab_insights, tab_chat, tab_profile, tab_sql = st.tabs(["Insights", "Chat", "Profile", "SQL"])

    # --- Insights: proactive, deterministic ---
    with tab_insights:
        st.subheader("What stands out")
        for ins in st.session_state.insights:
            st.markdown(f"**{ins.title}** — {ins.detail}")
            _render_chart(ins.chart, ins.rows)

    # --- Chat: ask in plain English ---
    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart") is not None:
                    _render_chart(msg["chart"], msg.get("rows"))

        if prompt := st.chat_input("e.g. Which category has the highest total?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"), st.spinner("Analyzing..."):
                capture = st.session_state.capture
                capture.rows = None
                try:
                    response = anyio.run(st.session_state.agent.run, AgentRequest.ephemeral(prompt))
                    answer = response.output
                except Exception as exc:
                    answer = f"Error: {exc}"
                st.markdown(answer)
                spec = choose_chart(capture.rows) if capture.rows else None
                _render_chart(spec, capture.rows)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "chart": spec, "rows": capture.rows}
                )

    # --- Profile: tables, columns, relationships ---
    with tab_profile:
        for t in profile.tables:
            with st.expander(f"{t.name} ({t.rows:,} rows)"):
                col_data = [
                    {
                        "Column": cp.name,
                        "Type": cp.physical_type,
                        "Role": next(
                            (c.role.value for c in t.columns if c.name == cp.name), "unknown"
                        ),
                        "Distinct": cp.distinct,
                        "Nulls": cp.nulls,
                        "Mean": round(cp.mean, 2) if cp.mean is not None else None,
                    }
                    for cp in profile.columns
                    if cp.table == t.name
                ]
                st.dataframe(col_data, use_container_width=True)
        if semantic.relationships:
            st.subheader("Relationships")
            for r in semantic.relationships:
                st.write(
                    f"{r.from_table}.{','.join(r.from_columns)} → "
                    f"{r.to_table}.{','.join(r.to_columns)} ({r.cardinality}, {r.coverage:.0%})"
                )

    # --- SQL: read-only escape hatch, with an auto chart ---
    with tab_sql:
        sql = st.text_area(
            "Read-only DuckDB SQL",
            value=f"SELECT * FROM {profile.tables[0].name} LIMIT 100",
            height=100,
        )
        if st.button("Run"):
            try:
                rows = source.native_query_with_limit(sql, 500)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    _render_chart(choose_chart(rows), rows)
                else:
                    st.info("(no rows returned)")
            except Exception as exc:
                st.error(f"SQL error: {exc}")


if __name__ == "__main__":
    main()
