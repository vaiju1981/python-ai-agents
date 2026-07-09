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

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, "src")
sys.path.insert(0, "demos/analytics/src")

from demos.analytics.src.analytics.agent import create_agent  # noqa: E402
from demos.analytics.src.analytics.audit_store import SqliteAuditStore  # noqa: E402
from demos.analytics.src.analytics.charts import ChartSpec, choose_chart  # noqa: E402
from demos.analytics.src.analytics.csv_source import CsvSource  # noqa: E402
from demos.analytics.src.analytics.insights import generate_insights  # noqa: E402
from demos.analytics.src.analytics.model_store import FileModelStore  # noqa: E402
from demos.analytics.src.analytics.models import from_env as model_from_env  # noqa: E402
from demos.analytics.src.analytics.profiler import profile_dataset  # noqa: E402
from demos.analytics.src.analytics.schema_builder import refine_profile_with_llm  # noqa: E402
from demos.analytics.src.analytics.semantic_model import SemanticModel  # noqa: E402
from python_ai_agents import AgentRequest, NoopAgentObserver, RequestContext  # noqa: E402
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
        # Prefer the tool's structured payload (exact); fall back to parsing framed text.
        rows = _extract_rows(result.data if result.data is not None else result.content)
        if rows:
            self.rows = rows


def _extract_rows(payload: object) -> list[dict] | None:
    """Best-effort: find a chartable list of row dicts in a tool result.

    Accepts either a list payload directly, a dict payload that embeds a
    list-of-dicts field, or a framed text string containing a JSON array.
    """
    import json

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return None
    if not isinstance(payload, str):
        return None
    match = re.search(r"\[.*\]", payload, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    return None


def _load_dataset(named_csvs: dict[str, Path], sig: tuple, tmp_dir: Path, *, refine: bool) -> None:
    """Import CSVs, profile, model, and cache once per dataset; reclaim the prior dataset."""
    old = st.session_state.get("source")
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
    old_dir = st.session_state.get("tmp_dir")
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)

    # File-backed DuckDB so large tables spill to disk (out-of-core) instead of RAM.
    source = CsvSource(db_path=str(tmp_dir / "data.duckdb"), named_csvs=named_csvs)
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
        tmp_dir=tmp_dir,
        session_id=str(uuid4()),
        source=source,
        profile=profile,
        semantic=semantic,
        insights=generate_insights(source, semantic, profile),
        messages=[],
    )
    st.session_state.pop("agent", None)  # force agent rebuild against new data


def _ensure_agent(provider: str, model_name: str, max_train_rows: int | None) -> None:
    """Build the agent once per (dataset, model); reuse it across questions.

    A persistent ``_RowCapture`` observer rides along so the chat can chart the
    rows behind each answer; it is reset before every question.
    """
    model_sig = (st.session_state.get("data_sig"), provider, model_name, max_train_rows)
    if st.session_state.get("model_sig") == model_sig and "agent" in st.session_state:
        return
    capture = _RowCapture()
    st.session_state.capture = capture
    # Durable, SQLite-backed audit store (survives restarts, scalable, queryable).
    audit = SqliteAuditStore(Path(st.session_state.tmp_dir) / "audit.db")
    st.session_state.audit = audit
    store = FileModelStore(Path(st.session_state.tmp_dir) / "models")
    st.session_state.agent = create_agent(
        st.session_state.source,
        model_from_env(),
        st.session_state.semantic,
        observers=[capture, audit],
        model_store=store,
        dataset_sig=str(st.session_state.data_sig),
        max_train_rows=max_train_rows,
        audit_sink=audit,
    )
    st.session_state.model_sig = model_sig


def _run_tool_audited(
    tool_name: str, arguments: dict[str, Any], *, input_hint: str = ""
) -> ToolResult:
    """Drive a specific tool through the agent's governed + audited pipeline.

    Used by the SQL/guided/compare panels so every execution — not just chat —
    flows through the same validation, approval, read-only guard, timeout, result
    framing, and audit/observer notification as an in-turn tool call.
    """
    audit = st.session_state.get("audit")
    if audit is not None:
        audit.session_id = st.session_state.session_id
    request = AgentRequest(
        input=input_hint,
        context=RequestContext.session(st.session_state.session_id),
    )

    async def _run() -> ToolResult:
        return await st.session_state.agent.run_tool(tool_name, arguments, request)

    return anyio.run(_run)


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
        max_train_rows = st.number_input(
            "Max ML training rows (0 = all)",
            min_value=0,
            value=0,
            step=10_000,
            help="Rows ML tools train on; 0 = the full table (slower on big data).",
        )

        st.divider()
        st.header("Data")
        mode = st.radio("Source", ["Upload CSV", "Local file path"], horizontal=True)

        if mode == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
            max_files = int(os.getenv("ANALYTICS_MAX_UPLOAD_FILES", "20"))
            max_bytes = int(os.getenv("ANALYTICS_MAX_UPLOAD_BYTES", str(4 * 1024**3)))
            if uploaded and len(uploaded) > max_files:
                st.error(f"Too many files (max {max_files}).")
                uploaded = []
            elif uploaded:
                oversized = [f.name for f in uploaded if f.size > max_bytes]
                if oversized:
                    st.error(f"File(s) exceed the size limit: {', '.join(oversized)}")
                    uploaded = []
            if uploaded:
                # name+size signature: only re-import when the upload actually changes.
                sig = tuple(sorted((f.name, f.size) for f in uploaded))
                if st.session_state.get("data_sig") != sig:
                    tmp_dir = Path(tempfile.mkdtemp())
                    named_csvs: dict[str, Path] = {}
                    for f in uploaded:
                        path = tmp_dir / f.name
                        path.write_bytes(f.getbuffer())
                        named_csvs[Path(f.name).stem] = path
                    with st.spinner("Importing into DuckDB and profiling..."):
                        _load_dataset(named_csvs, sig, tmp_dir, refine=refine)
        else:
            st.caption("Read CSVs on disk (no upload) — read in place; best for large files.")
            paths_text = st.text_area(
                "CSV path(s), one per line", height=100, placeholder="/data/sales.csv"
            )
            if st.button("Load", type="primary") and paths_text.strip():
                paths = [Path(p.strip()).expanduser() for p in paths_text.splitlines() if p.strip()]
                missing = [str(p) for p in paths if not p.is_file()]
                if missing:
                    st.error("Not found: " + ", ".join(missing))
                else:
                    sig = tuple(sorted((str(p), p.stat().st_size) for p in paths))
                    if st.session_state.get("data_sig") != sig:
                        tmp_dir = Path(tempfile.mkdtemp())
                        named = {p.stem: p for p in paths}
                        with st.spinner("Reading in place and profiling..."):
                            _load_dataset(named, sig, tmp_dir, refine=refine)

        if "semantic" in st.session_state:
            sm = st.session_state.semantic
            st.success(
                f"{len(st.session_state.profile.tables)} table(s) · {len(sm.metrics)} metric(s) · "
                f"{len(sm.relationships)} relationship(s)."
            )

    if "source" not in st.session_state:
        st.info("Upload one or more CSV files to get started.")
        return

    _ensure_agent(provider, model_name, int(max_train_rows) or None)
    profile = st.session_state.profile
    semantic = st.session_state.semantic

    tab_insights, tab_chat, tab_profile, tab_sql, tab_audit = st.tabs(
        ["Insights", "Chat", "Profile", "SQL", "Audit"]
    )

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
                audit = st.session_state.audit
                audit.session_id = st.session_state.session_id
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
                result = _run_tool_audited("run_sql", {"sql": sql}, input_hint="SQL tab")
                if result.error:
                    st.error(result.content)
                elif result.data:
                    rows = result.data
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    _render_chart(choose_chart(rows), rows)
                else:
                    st.info("(no rows returned)")
            except Exception as exc:
                st.error(f"SQL error: {exc}")

        with st.expander("Guided query (run_query + derived metrics)"):
            g_metrics = st.text_input("Metrics (comma-separated refs)", value="sales.amount")
            g_dims = st.text_input("Dimensions (optional)", value="")
            g_derived = st.text_area(
                "Derived metrics (optional JSON, e.g. "
                '[{"name":"avg_price","expression":"sales.amount/sales.quantity"}])',
                height=70,
            )
            if st.button("Run guided query"):
                try:
                    derived = json.loads(g_derived) if g_derived.strip() else []
                    args = {
                        "metrics": [m.strip() for m in g_metrics.split(",") if m.strip()],
                        "dimensions": [d.strip() for d in g_dims.split(",") if d.strip()],
                        "derivedMetrics": derived,
                    }
                    result = _run_tool_audited("run_query", args, input_hint="guided query")
                    if result.error:
                        st.error(result.content)
                    elif result.data:
                        st.dataframe(pd.DataFrame(result.data), use_container_width=True)
                        _render_chart(choose_chart(result.data), result.data)
                    else:
                        st.info(result.content)
                except Exception as exc:
                    st.error(f"Query error: {exc}")

        with st.expander("Period comparison (compare)"):
            c_metrics = st.text_input(
                "Metrics (comma-separated refs)", value="sales.amount", key="cmp_metrics"
            )
            default_tc = semantic.time_columns[0].ref if semantic.time_columns else ""
            c_time = st.text_input("Time column", value=default_tc, key="cmp_time")
            c_days = st.number_input("Last N days", min_value=1, value=30, key="cmp_days")
            c_dims = st.text_input("Dimensions (optional)", value="", key="cmp_dims")
            if st.button("Run compare"):
                try:
                    args = {
                        "metrics": [m.strip() for m in c_metrics.split(",") if m.strip()],
                        "timeColumn": c_time.strip(),
                        "lastDays": int(c_days),
                        "dimensions": [d.strip() for d in c_dims.split(",") if d.strip()],
                    }
                    result = _run_tool_audited("compare", args, input_hint="compare")
                    if result.error:
                        st.error(result.content)
                    else:
                        m = re.search(r"\{.*\}", result.content, re.DOTALL)
                        if m:
                            st.json(json.loads(m.group(0)))
                        if result.data:
                            st.dataframe(pd.DataFrame(result.data), use_container_width=True)
                            _render_chart(choose_chart(result.data), result.data)
                except Exception as exc:
                    st.error(f"Compare error: {exc}")

    # --- Audit: durable, SQLite-backed tool-call observability for this session ---
    with tab_audit:
        store = st.session_state.get("audit")
        records = (
            store.records(session_id=st.session_state.get("session_id"))
            if store is not None
            else []
        )
        if not records:
            st.info("No tool calls yet. Ask a question or run a query to populate the audit log.")
        else:
            total = len(records)
            errors = sum(1 for r in records if not r["ok"])
            st.caption(f"{total} tool call(s) · {errors} error(s) · persisted to SQLite")
            st.dataframe(pd.DataFrame(records), use_container_width=True)


if __name__ == "__main__":
    main()
