"""Analytics agent and supervisor: wires the model + tools + system prompt.

The agent is provider-agnostic — the ``ModelPort`` is injected. The system
prompt embeds the generated schema so the model can write queries without
calling ``describe_dataset`` first.
"""

from __future__ import annotations

from typing import Any

from python_ai_agents import Agent, DefaultAgent, ModelPort, AgentObserver
from python_ai_agents.core.supervise import SupervisorAgent, Router

from demos.analytics.src.analytics.data_source import DataSource
from demos.analytics.src.analytics.advanced_tools import AdvancedToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from demos.analytics.src.analytics.ml_tools import MLToolset


def create_agent(
    source: DataSource,
    model: ModelPort,
    semantic: SemanticModel | None = None,
    catalog: Any = None,
    observers: list[AgentObserver] | None = None,
) -> Agent:
    """Create a schema-driven analytics agent with governed read-only tools."""
    if semantic is None:
        profile = profile_dataset(source, catalog)
        semantic = SemanticModel.from_profile(profile)

    tools = AnalyticsToolset(source, semantic, catalog)
    advanced_tools = AdvancedToolset(source, semantic)
    ml_tools = MLToolset(source, semantic)
    system_prompt = (
        "You are a careful data analyst over this dataset. Use these EXACT refs.\n"
        f"SCHEMA: {tools.catalog_json()}\n"
        "Answer each question by calling ONE tool, then replying in plain text with the numbers:\n"
        "- run_query: metrics grouped by dimensions; 'last N days' -> lastDays + timeColumn; "
        "'top N' -> limit; simple filters {column,op,value}. Cross-table joins are fan-out-safe.\n"
        "- compare: period-over-period comparison; requires metrics, timeColumn, lastDays.\n"
        "- trend: single-table trend by day/week/month.\n"
        "- summarize(metric): distribution/percentiles of one measure.\n"
        "- correlate(target): 'what correlates with / drives' a metric.\n"
        "- outliers(metric): unusual values by z-score.\n"
        "- regression(target): which measures linearly predict a target.\n"
        "- run_sql: read-only DuckDB SQL for custom filters, time buckets, window functions.\n"
        "\n"
        "Advanced analytics tools:\n"
        "- cohort_analysis(entityColumn, timeColumn, cohortGrain?): retention by first activity cohort.\n"
        "- funnel_analysis(entityColumn, steps): conversion through filtered steps.\n"
        "- rfm_segmentation(entityColumn, timeColumn, monetaryColumn): recency/frequency/monetary segments.\n"
        "- time_series_decomposition(metric, timeColumn, period?): trend, seasonality, residuals.\n"
        "- correlation_matrix(table?, columns?): full numeric correlation matrix.\n"
        "- data_quality(table): nulls, distinctness, type quality report.\n"
        "- percentile_ranking(entityColumn, metric): percentile distribution by entity.\n"
        "- benchmark_comparison(entityColumn, metric, targetEntity): compare one entity to peers.\n"
        "- pca_analysis(table?, columns?, nComponents?): principal component analysis.\n"
        "- survival_analysis(entityColumn, timeColumn): time-to-event estimate.\n"
        "- granger_causality(cause, effect, timeColumn, maxLag?): predictive causality check.\n"
        "\n"
        "ML & statistical tools (for deeper analysis):\n"
        "- ml_regression(target, predictors?, method?): train linear/ridge/lasso/random forest/gradient boosting.\n"
        "- classification(target, predictors?, method?): train logistic/rf/gbrt classifier with metrics.\n"
        "- clustering(columns, method?, nClusters?): k-means or DBSCAN with silhouette score.\n"
        "- forecast(metric, timeColumn, horizon, method?): ARIMA/exponential smoothing/linear trend.\n"
        "- causal_analysis(target, treatment, controls?, method?): estimate treatment effects (ATE with CI).\n"
        "- uplift_modeling(target, treatment, predictors?, method?): identify who benefits most from treatment.\n"
        "- feature_importance(target, method?): rank features by permutation or tree importance.\n"
        "- statistical_test(test, column, groupColumn): t-test, chi-square, Mann-Whitney, ANOVA.\n"
        "- anomaly_detection(columns, method?, contamination?): isolation forest or LOF anomaly detection.\n"
        "- cross_validate(target, method?, cvFolds?, task?): k-fold cross-validation for model evaluation.\n"
        "After the tool returns, STOP calling tools and answer. "
        "Never present unfiltered numbers as if filtered; do not claim causation from correlation."
    )

    return DefaultAgent(
        model=model,
        tools=_unique_tools(tools.all_tools() + advanced_tools.all_tools() + ml_tools.all_tools()),
        system_prompt=system_prompt,
        max_steps=12,
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
    """Create a multi-agent: routes to analyst or schema_guide."""
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

    return SupervisorAgent.builder() \
        .specialist("analyst", "Answers questions that need numbers: metrics, aggregations, comparisons, rankings.", analyst) \
        .specialist("schema_guide", "Explains what data exists: tables, columns, types, and relationships.", schema_guide) \
        .router(router or KeywordRouter()) \
        .fallback("analyst") \
        .build()


def _unique_tools(tools: list[Any]) -> list[Any]:
    """Fail fast if two toolsets expose the same model-facing name."""
    seen: set[str] = set()
    result = []
    for tool in tools:
        name = tool.spec.name
        if name in seen:
            raise ValueError(f"duplicate analytics tool name: {name}")
        seen.add(name)
        result.append(tool)
    return result
