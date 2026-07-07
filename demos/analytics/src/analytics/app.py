"""Streamlit app: CSV upload → profile → chat → charts.

Run with::

    ANALYTICS_MODEL_PROVIDER=ollama ANALYTICS_MODEL=ornith:latest \\
        streamlit run demos/analytics/src/analytics/app.py

Supports multiple CSV uploads, automatic profiling, semantic model inference,
relationship discovery, and a chat interface where the LLM agent answers
analytics questions using governed read-only tools.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import anyio
import streamlit as st

sys.path.insert(0, "src")
sys.path.insert(0, "demos/analytics/src")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.catalog import Catalog
from demos.analytics.src.analytics.agent import create_agent, create_supervisor
from demos.analytics.src.analytics.models import from_env as model_from_env
from demos.analytics.src.analytics.schema_builder import refine_profile_with_llm


def main() -> None:
    st.set_page_config(page_title="Analytics Agent", page_icon="📊", layout="wide")
    st.title("📊 Analytics Agent")
    st.caption("Upload CSV files, ask questions in plain English, get governed analytics answers.")

    # --- Sidebar: data source configuration ---
    with st.sidebar:
        st.header("Model")
        os.environ.setdefault("ANALYTICS_MODEL_PROVIDER", "ollama")
        os.environ.setdefault("ANALYTICS_MODEL", "ornith:latest")
        model_provider = st.text_input("Provider", value=os.environ.get("ANALYTICS_MODEL_PROVIDER", "ollama"))
        model_name = st.text_input("Model", value=os.environ.get("ANALYTICS_MODEL", "ornith:latest"))
        os.environ["ANALYTICS_MODEL_PROVIDER"] = model_provider
        os.environ["ANALYTICS_MODEL"] = model_name

        use_supervisor = st.checkbox("Use supervisor (multi-agent)", value=False)
        refine_csv_schema = st.checkbox("Use LLM schema refinement for CSV", value=True)

        st.divider()
        st.header("Data Source")
        source_type = st.radio("Source type", ["CSV files", "Neo4j graph", "SQL database"], disabled=False)

        source = None
        profile = None
        semantic = None

        if source_type == "CSV files":
            uploaded = st.file_uploader(
                "Upload CSV files",
                type="csv",
                accept_multiple_files=True,
            )
            if uploaded:
                named_csvs = {}
                tmp_dir = Path(tempfile.mkdtemp())
                for f in uploaded:
                    path = tmp_dir / f.name
                    path.write_bytes(f.getbuffer())
                    table_name = Path(f.name).stem
                    named_csvs[table_name] = path

                with st.spinner("Importing CSVs into DuckDB and profiling..."):
                    source = CsvSource(named_csvs=named_csvs)
                    catalog_file = tmp_dir / "catalog.json"
                    catalog = Catalog.load(catalog_file)
                    profile = profile_dataset(source, catalog)
                    if refine_csv_schema:
                        try:
                            with st.spinner("Refining CSV semantic schema with the model..."):
                                profile = anyio.run(
                                    refine_profile_with_llm,
                                    profile,
                                    model_from_env(),
                                )
                        except Exception as exc:
                            st.warning(f"LLM schema refinement skipped: {exc}")
                    semantic = SemanticModel.from_profile(profile)
                st.success(f"Imported {len(named_csvs)} table(s), discovered {len(semantic.relationships)} relationship(s).")

        elif source_type == "Neo4j graph":
            neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
            neo4j_user = st.text_input("Neo4j user", value="neo4j")
            neo4j_pass = st.text_input("Neo4j password", type="password", value="neo4j")
            if st.button("Connect & Profile"):
                with st.spinner("Connecting to Neo4j and projecting graph..."):
                    from demos.analytics.src.analytics.graph_source import GraphSource
                    source = GraphSource(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
                    profile = profile_dataset(source)
                    semantic = SemanticModel.from_profile(profile)
                st.success(f"Projected {len(source.tables())} table(s).")

        elif source_type == "SQL database":
            db_path = st.text_input("DuckDB file path", value=":memory:")
            if st.button("Connect & Profile"):
                with st.spinner("Connecting and profiling..."):
                    from demos.analytics.src.analytics.sql_source import SqlSource
                    source = SqlSource(db_path=db_path)
                    profile = profile_dataset(source)
                    semantic = SemanticModel.from_profile(profile)
                st.success(f"Found {len(source.tables())} table(s).")

    # --- Main area: tabs ---
    if source is None or semantic is None:
        st.info("Upload data or connect a source to get started.")
        return

    tab_profile, tab_chat, tab_sql = st.tabs(["Profile", "Chat", "SQL"])

    # --- Profile tab ---
    with tab_profile:
        st.subheader("Tables & Schema")
        for t in profile.tables:
            with st.expander(f"{t.name} ({t.rows:,} rows)", expanded=False):
                col_data = []
                for cp in profile.columns:
                    if cp.table == t.name:
                        col_data.append({
                            "Column": cp.name,
                            "Type": cp.physical_type,
                            "Role": next((c.role.value for c in t.columns if c.name == cp.name), "unknown"),
                            "Distinct": cp.distinct,
                            "Nulls": cp.nulls,
                            "Min": cp.min,
                            "Max": cp.max,
                            "Mean": round(cp.mean, 2) if cp.mean is not None else None,
                            "Signals": ", ".join(sorted(cp.signals)) if cp.signals else "",
                        })
                st.dataframe(col_data, use_container_width=True)

        if semantic.relationships:
            st.subheader("Discovered Relationships")
            for r in semantic.relationships:
                st.write(
                    f"{r.from_table}.{','.join(r.from_columns)} → "
                    f"{r.to_table}.{','.join(r.to_columns)} ({r.cardinality}, {r.coverage:.0%})"
                )

        if semantic.metrics:
            st.subheader("Metrics (measures)")
            st.write(", ".join(f"{m.ref} ({m.aggregation})" for m in semantic.metrics))

        if semantic.time_columns:
            st.subheader("Time columns")
            st.write(", ".join(f"{t.ref} ({t.encoding.value})" for t in semantic.time_columns))

    # --- Chat tab ---
    with tab_chat:
        st.subheader("Ask a question")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("e.g. What is the top metric by dimension? Show the latest trend."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        model = model_from_env()
                        if use_supervisor:
                            agent = create_supervisor(source, model, semantic)
                        else:
                            agent = create_agent(source, model, semantic)

                        import anyio
                        from python_ai_agents import AgentRequest

                        response = anyio.run(
                            agent.run,
                            AgentRequest.ephemeral(prompt),
                        )
                        answer = response.output
                    except Exception as exc:
                        answer = f"Error: {exc}"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- SQL tab ---
    with tab_sql:
        st.subheader("Run read-only SQL")
        sql = st.text_area(
            "DuckDB SQL",
            value=f"SELECT * FROM {profile.tables[0].name} LIMIT 10",
            height=100,
        )
        if st.button("Run"):
            try:
                rows = source.native_query_with_limit(sql, 500)
                if rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("(no rows returned)")
            except Exception as exc:
                st.error(f"SQL error: {exc}")


if __name__ == "__main__":
    main()
