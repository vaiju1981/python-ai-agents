"""Analytics agent and supervisor: wires the model + tools + system prompt.

The agent is provider-agnostic — the ``ModelPort`` is injected. The system
prompt embeds the generated schema so the model can write queries without
calling ``describe_dataset`` first.
"""

from __future__ import annotations

from typing import Any

from demos.analytics.src.analytics.data_source import DataSource
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents import Agent, AgentObserver, DefaultAgent, ModelPort
from python_ai_agents.core.supervise import Router, SupervisorAgent


def create_agent(
    source: DataSource,
    model: ModelPort,
    semantic: SemanticModel | None = None,
    catalog: Any = None,
    observers: list[AgentObserver] | None = None,
    model_store: Any = None,
    dataset_sig: str = "",
    max_train_rows: int | None = None,
) -> Agent:
    """Create a schema-driven analytics agent with governed read-only tools."""
    if semantic is None:
        profile = profile_dataset(source, catalog)
        semantic = SemanticModel.from_profile(profile)

    tools = AnalyticsToolset(source, semantic, catalog)
    models = ModelsToolset(
        source, semantic, store=model_store, dataset_sig=dataset_sig, max_train_rows=max_train_rows
    )
    system_prompt = (
        "You are a careful data analyst over this dataset. Use these EXACT refs.\n"
        f"SCHEMA: {tools.catalog_json()}\n"
        "Answer each question by calling exactly ONE tool, then STOP and reply in plain text "
        "with the numbers — do not make more tool calls after the first result.\n"
        "Descriptive tools:\n"
        "- run_query: metrics grouped by dimensions; 'last N days' -> lastDays + timeColumn; "
        "'top N' -> limit; simple filters {column,op,value}.\n"
        "- compare: period-over-period comparison; requires metrics, timeColumn, lastDays.\n"
        "- trend: single-table trend by day/week/month.\n"
        "- summarize(metric): distribution/percentiles of one measure.\n"
        "- correlate(target): 'what correlates with' a metric.\n"
        "- outliers(metric): unusual values by z-score.\n"
        "- regression(target): which measures linearly predict a target.\n"
        "- run_sql: read-only DuckDB SQL for custom filters, time buckets, window functions, and "
        "any derived or ratio metric (per-unit, rate, share) — run_query only aggregates a metric "
        "by dimensions, it cannot compute ratios.\n"
        "Predictive & causal tools (use when the question needs modeling or an experiment):\n"
        "- build_model(target, predictors?): train a model, rank feature importance.\n"
        "- predict(target, filters?): score rows with the stored trained model (no retrain) — "
        "use for 'predict/estimate X for <subset>'; reports drift vs training data.\n"
        "- forecast(metric, timeColumn, horizon?): project a metric forward.\n"
        "- ab_test(metric, groupColumn, groupA, groupB): compare two groups (t-test).\n"
        "- causal_effect(target, treatment, controls?): effect adjusting for confounders.\n"
        "- uplift(target, treatment, predictors?): who benefits most from a treatment.\n"
        "- cluster(columns, k?): segment rows. anomaly_detection(columns): flag outliers.\n"
        "After the tool returns, STOP calling tools and answer with the numbers. Never present "
        "unfiltered numbers as if filtered; do not claim causation from correlation — cite the "
        "tool's caveat when reporting causal_effect or uplift."
    )

    return DefaultAgent(
        model=model,
        tools=tools.all_tools() + models.all_tools(),
        system_prompt=system_prompt,
        max_steps=8,
        tool_timeout_seconds=120.0,
        max_tool_result_chars=16_000,
        observers=observers or [],
    )


def create_supervisor(
    source: DataSource,
    model: ModelPort,
    semantic: SemanticModel | None = None,
    catalog: Any = None,
    router: Router | None = None,
) -> Agent:
    """Create a multi-agent that routes between an analyst and a schema guide."""
    if semantic is None:
        profile = profile_dataset(source, catalog)
        semantic = SemanticModel.from_profile(profile)

    tools = AnalyticsToolset(source, semantic, catalog)
    analyst = create_agent(source, model, semantic, catalog)

    schema_guide = DefaultAgent(
        model=model,
        tools=[tools.describe_dataset()],
        system_prompt=(
            "You explain the dataset's structure — its tables, columns, types, and "
            "relationships — in plain English. Call describe_dataset once, then summarize what "
            "data is available. Do not compute metrics."
        ),
        max_steps=4,
        tool_timeout_seconds=120.0,
        max_tool_result_chars=16_000,
    )

    from python_ai_agents.core.supervise import KeywordRouter

    return (
        SupervisorAgent.builder()
        .specialist(
            "analyst",
            "Answers questions that need numbers: metrics, aggregations, comparisons, rankings.",
            analyst,
        )
        .specialist(
            "schema_guide",
            "Explains what data exists: tables, columns, types, and relationships.",
            schema_guide,
        )
        .router(router or KeywordRouter())
        .fallback("analyst")
        .build()
    )
