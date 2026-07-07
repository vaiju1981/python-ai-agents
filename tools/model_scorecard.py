"""Live Ollama scorecard using the production analytics demo stack.

Run from the repository root:

    PYTHONPATH=src:. python tools/model_scorecard.py

The harness compares candidate Ollama models by running the actual analytics
demo path: CSV import, deterministic profiling, semantic model construction,
the demo agent prompt, AnalyticsToolset, and ModelsToolset. It writes both JSON
and Markdown scorecards.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio

from demos.analytics.src.analytics.agent import create_agent
from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from python_ai_agents import AgentRequest, RecordingObserver, RequestContext
from python_ai_agents.adapters import OllamaModelPort

DEFAULT_MODELS = ("gemma4:31b-cloud", "ornith:latest")
DEFAULT_OUTPUT_DIR = Path("model_scorecards")
PASS_THRESHOLD = 80.0
MAX_OLLAMA_CONTEXT = 65_536

RUBRIC_WEIGHTS = {
    "answer": 35.0,
    "completion": 10.0,
    "tool_selection": 20.0,
    "tool_arguments": 20.0,
    "tool_efficiency": 5.0,
    "output_hygiene": 10.0,
}


@dataclass(frozen=True, slots=True)
class ScorecardCase:
    name: str
    category: str
    prompt: str
    expected_terms: tuple[str, ...]
    required_tools: tuple[str, ...]
    argument_check: Callable[[tuple[dict[str, Any], ...]], tuple[bool, str]] | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseResult:
    name: str
    category: str
    score: float
    max_score: float
    passed: bool
    output: str
    tool_calls: tuple[dict[str, Any], ...]
    duration_seconds: float
    stop_reason: str
    detail: str
    rubric: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelResult:
    model: str
    available: bool
    score: float
    max_score: float
    duration_seconds: float
    cases: tuple[CaseResult, ...] = ()
    error: str = ""

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.passed) / len(self.cases)

    @property
    def percent(self) -> float:
        return self.score / self.max_score if self.max_score else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Ollama models through the analytics demo stack."
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--num-ctx", type=int, default=MAX_OLLAMA_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    num_ctx = min(args.num_ctx, MAX_OLLAMA_CONTEXT)

    results = anyio.run(
        run_scorecard,
        tuple(args.models),
        args.base_url,
        args.timeout,
        args.max_steps,
        num_ctx,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"ollama_model_scorecard_{timestamp}.json"
    md_path = args.output_dir / f"ollama_model_scorecard_{timestamp}.md"

    payload = {
        "generated_at": timestamp,
        "harness": "demos.analytics",
        "ollama_options": {"temperature": 0, "top_p": 0.1, "num_ctx": num_ctx},
        "models": [model_result_to_dict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")

    print(render_markdown(results))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


async def run_scorecard(
    models: tuple[str, ...],
    base_url: str,
    timeout: float,
    max_steps: int,
    num_ctx: int,
) -> tuple[ModelResult, ...]:
    cases = build_cases()
    results: list[ModelResult] = []
    for model_name in models:
        print(f"[scorecard] model={model_name} starting", flush=True)
        model_start = perf_counter()
        model = OllamaModelPort(
            model_name,
            base_url=base_url,
            options={"temperature": 0, "top_p": 0.1, "num_ctx": num_ctx},
            timeout=timeout,
        )
        try:
            if not await model.has_model():
                print(f"[scorecard] model={model_name} unavailable", flush=True)
                results.append(
                    ModelResult(
                        model=model_name,
                        available=False,
                        score=0,
                        max_score=0,
                        duration_seconds=perf_counter() - model_start,
                        error="model is not available in Ollama",
                    )
                )
                continue
            case_results = []
            for case in cases:
                print(f"[scorecard] model={model_name} case={case.name} starting", flush=True)
                case_result = await run_case(model_name, model, case, max_steps)
                case_results.append(case_result)
                mark = "PASS" if case_result.passed else "FAIL"
                print(
                    f"[scorecard] model={model_name} case={case.name} "
                    f"{mark} score={case_result.score:.1f}/{case_result.max_score:.1f} "
                    f"duration={case_result.duration_seconds:.1f}s",
                    flush=True,
                )
            results.append(
                ModelResult(
                    model=model_name,
                    available=True,
                    score=sum(result.score for result in case_results),
                    max_score=sum(result.max_score for result in case_results),
                    duration_seconds=perf_counter() - model_start,
                    cases=tuple(case_results),
                )
            )
            print(f"[scorecard] model={model_name} finished", flush=True)
        except Exception as exc:
            print(f"[scorecard] model={model_name} error={exc.__class__.__name__}: {exc}", flush=True)
            results.append(
                ModelResult(
                    model=model_name,
                    available=True,
                    score=0,
                    max_score=sum(case_max_score(case) for case in cases),
                    duration_seconds=perf_counter() - model_start,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            )
    return tuple(results)


async def run_case(
    model_name: str,
    model: OllamaModelPort,
    case: ScorecardCase,
    max_steps: int,
) -> CaseResult:
    observer = RecordingObserver()
    source: CsvSource | None = None
    started = perf_counter()
    output = ""
    stop_reason = ""
    try:
        source, semantic = build_demo_dataset()
        agent = create_agent(source, model, semantic, observers=[observer])
        agent.max_steps = max_steps
        context = RequestContext(session_id=f"analytics-scorecard-{model_name}-{case.name}")
        response = await agent.run(AgentRequest(case.prompt, context))
        output = response.output
        stop_reason = response.stop_reason
    except Exception as exc:
        output = f"{exc.__class__.__name__}: {exc}"
        stop_reason = "error"
    finally:
        if source is not None:
            source.close()

    tool_calls = tuple(
        {"name": call.name, "arguments": call.arguments} for call in observer.tool_calls
    )
    score, max_score, passed, detail, rubric, warnings = score_case(
        case,
        output,
        tool_calls,
        stop_reason,
    )
    return CaseResult(
        name=case.name,
        category=case.category,
        score=score,
        max_score=max_score,
        passed=passed,
        output=output,
        tool_calls=tool_calls,
        duration_seconds=perf_counter() - started,
        stop_reason=stop_reason,
        detail=detail,
        rubric=rubric,
        warnings=warnings,
    )


def build_demo_dataset() -> tuple[CsvSource, SemanticModel]:
    tmp_dir = tempfile.TemporaryDirectory()
    csv_path = Path(tmp_dir.name) / "scorecard_sales.csv"
    csv_path.write_text(_scorecard_csv(), encoding="utf-8")
    source = CsvSource(named_csvs={"scorecard_sales": csv_path})
    source._scorecard_tmp_dir = tmp_dir  # type: ignore[attr-defined]
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    return source, semantic


def _scorecard_csv() -> str:
    rows = ["day,region,product,campaign,treatment,revenue,cost,profit,units,visits"]
    months = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
    for idx, month in enumerate(months, start=1):
        rows.extend(
            [
                f"{month}-05,North,Widget,control,0,{100 + idx * 10},{60 + idx * 4},{40 + idx * 6},{20 + idx},{80 + idx * 3}",
                f"{month}-10,South,Widget,test,1,{110 + idx * 12},{65 + idx * 4},{45 + idx * 8},{22 + idx},{88 + idx * 4}",
                f"{month}-15,East,Gadget,test,1,{140 + idx * 18},{70 + idx * 5},{70 + idx * 13},{26 + idx},{95 + idx * 5}",
                f"{month}-20,West,Gadget,control,0,{120 + idx * 8},{75 + idx * 4},{45 + idx * 4},{18 + idx},{82 + idx * 3}",
            ]
        )
    return "\n".join(rows) + "\n"


def build_cases() -> tuple[ScorecardCase, ...]:
    table = "scorecard_sales"
    revenue = f"{table}.revenue"
    profit = f"{table}.profit"
    units = f"{table}.units"
    visits = f"{table}.visits"
    day = f"{table}.day"
    region = f"{table}.region"
    product = f"{table}.product"
    campaign = f"{table}.campaign"
    treatment = f"{table}.treatment"
    return (
        ScorecardCase(
            name="schema_discovery",
            category="analytics-schema",
            prompt=(
                "Use describe_dataset once. What table is available and what are two useful "
                "metrics for analysis?"
            ),
            expected_terms=("scorecard_sales", "revenue", "profit"),
            required_tools=("describe_dataset",),
            notes="Checks the real demo schema-discovery tool.",
        ),
        ScorecardCase(
            name="revenue_by_region",
            category="analytics-query",
            prompt=(
                f"Use run_query with metric {revenue} grouped by dimension {region}. "
                "Which region has the highest total revenue?"
            ),
            expected_terms=("east", "1218"),
            required_tools=("run_query",),
            argument_check=_requires_refs(
                ("run_query", "metrics", revenue),
                ("run_query", "dimensions", region),
            ),
            notes="Metric/dimension planning through AnalyticsToolset.run_query.",
        ),
        ScorecardCase(
            name="profit_by_product_sql",
            category="analytics-sql",
            prompt=(
                "Use run_sql to compute total profit by product from scorecard_sales. "
                "Which product has the higher total profit and what is the total?"
            ),
            expected_terms=("gadget", "1047"),
            required_tools=("run_sql",),
            argument_check=lambda calls: _has_sql_terms(calls, ("scorecard_sales", "product", "profit")),
            notes="Read-only SQL fallback through the real analytics SQL guard.",
        ),
        ScorecardCase(
            name="revenue_trend",
            category="analytics-time",
            prompt=(
                f"Use trend for metric {revenue} with timeColumn {day}, grain month, "
                "and lastDays 220. Summarize whether revenue is rising."
            ),
            expected_terms=("rising", "revenue"),
            required_tools=("trend",),
            argument_check=_requires_refs(
                ("trend", "metrics", revenue),
                ("trend", "timeColumn", day),
            ),
            notes="Time-series tool selection and date-column argument handling.",
        ),
        ScorecardCase(
            name="summarize_profit",
            category="analytics-stats",
            prompt=f"Use summarize for metric {profit}. Report the mean and range.",
            expected_terms=("mean", "profit"),
            required_tools=("summarize",),
            argument_check=_requires_refs(("summarize", "metric", profit)),
            notes="Descriptive statistics via the real summarize tool.",
        ),
        ScorecardCase(
            name="build_predictive_model",
            category="analytics-modeling",
            prompt=(
                f"Use build_model to predict {profit} from predictors {revenue}, {cost_ref()}, "
                f"{units}, and {visits}. Which feature is most important?"
            ),
            expected_terms=("feature", "importance"),
            required_tools=("build_model",),
            argument_check=_requires_refs(("build_model", "target", profit)),
            notes="Predictive ModelsToolset path with train-once-ready API.",
        ),
        ScorecardCase(
            name="forecast_revenue",
            category="analytics-forecast",
            prompt=(
                f"Use forecast for metric {revenue}, timeColumn {day}, horizon 2. "
                "Report the forecast method and first forecast value."
            ),
            expected_terms=("forecast", "method"),
            required_tools=("forecast",),
            argument_check=_requires_refs(
                ("forecast", "metric", revenue),
                ("forecast", "timeColumn", day),
            ),
            notes="Forecasting through the demo predictive toolset.",
        ),
        ScorecardCase(
            name="campaign_ab_test",
            category="analytics-experiment",
            prompt=(
                f"Use ab_test for metric {revenue}, groupColumn {campaign}, groupA test, "
                "and groupB control. Report the difference and p-value."
            ),
            expected_terms=("difference", "p"),
            required_tools=("ab_test",),
            argument_check=_requires_refs(
                ("ab_test", "metric", revenue),
                ("ab_test", "groupColumn", campaign),
                ("ab_test", "groupA", "test"),
                ("ab_test", "groupB", "control"),
            ),
            notes="Experiment-style analytics via ModelsToolset.ab_test.",
        ),
        ScorecardCase(
            name="causal_effect",
            category="analytics-causal",
            prompt=(
                f"Use causal_effect with target {profit}, treatment {treatment}, "
                f"and controls {units}, {visits}. Report the caveat."
            ),
            expected_terms=("causation", "confound"),
            required_tools=("causal_effect",),
            argument_check=_requires_refs(
                ("causal_effect", "target", profit),
                ("causal_effect", "treatment", treatment),
            ),
            notes="Causal caveat and treatment-effect tool behavior.",
        ),
        ScorecardCase(
            name="cluster_segments",
            category="analytics-segmentation",
            prompt=f"Use cluster with columns {revenue}, {profit}, and {visits}, k 3.",
            expected_terms=("cluster", "silhouette"),
            required_tools=("cluster",),
            argument_check=_requires_refs(("cluster", "columns", revenue), ("cluster", "columns", profit)),
            notes="Segmentation tool routing and argument construction.",
        ),
    )


def cost_ref() -> str:
    return "scorecard_sales.cost"


def score_case(
    case: ScorecardCase,
    output: str,
    tool_calls: tuple[dict[str, Any], ...],
    stop_reason: str,
) -> tuple[float, float, bool, str, dict[str, float], tuple[str, ...]]:
    max_score = case_max_score(case)
    details: list[str] = []
    warnings: list[str] = []
    rubric = {name: 0.0 for name in RUBRIC_WEIGHTS}

    normalized_output = _normalize_answer_text(output)
    missing_terms = [term for term in case.expected_terms if term.lower() not in normalized_output]
    if not missing_terms:
        rubric["answer"] = RUBRIC_WEIGHTS["answer"]
        details.append("answer terms ok")
    else:
        details.append(f"missing answer terms: {', '.join(missing_terms)}")

    if stop_reason == "completed":
        rubric["completion"] = RUBRIC_WEIGHTS["completion"]
        details.append("completed")
    else:
        details.append(f"stop_reason={stop_reason}")

    missing_tools = _missing_required_tools(case.required_tools, tool_calls)
    if not missing_tools:
        rubric["tool_selection"] = RUBRIC_WEIGHTS["tool_selection"]
        details.append("required tools ok")
    else:
        details.append(f"missing tools: {', '.join(missing_tools)}")

    if case.argument_check is None:
        rubric["tool_arguments"] = RUBRIC_WEIGHTS["tool_arguments"]
    else:
        ok, message = case.argument_check(tool_calls)
        if ok:
            rubric["tool_arguments"] = RUBRIC_WEIGHTS["tool_arguments"]
        details.append(f"arguments: {message}")

    efficient, efficiency_detail = _tool_efficiency(case.required_tools, tool_calls)
    if efficient:
        rubric["tool_efficiency"] = RUBRIC_WEIGHTS["tool_efficiency"]
    details.append(f"efficiency: {efficiency_detail}")

    hygienic, hygiene_detail = _output_hygiene(output)
    if hygienic:
        rubric["output_hygiene"] = RUBRIC_WEIGHTS["output_hygiene"]
    else:
        warnings.append(hygiene_detail)
    details.append(f"hygiene: {hygiene_detail}")

    score = sum(rubric.values())
    passed = score >= PASS_THRESHOLD
    return score, max_score, passed, "; ".join(details), rubric, tuple(warnings)


def case_max_score(case: ScorecardCase) -> float:
    return sum(RUBRIC_WEIGHTS.values())


def _missing_required_tools(
    required_tools: tuple[str, ...],
    tool_calls: tuple[dict[str, Any], ...],
) -> list[str]:
    actual = Counter(call["name"] for call in tool_calls)
    required = Counter(required_tools)
    missing: list[str] = []
    for name, count in required.items():
        if actual[name] < count:
            missing.extend([name] * (count - actual[name]))
    return missing


def _tool_efficiency(
    required_tools: tuple[str, ...],
    tool_calls: tuple[dict[str, Any], ...],
) -> tuple[bool, str]:
    required_count = len(required_tools)
    actual_count = len(tool_calls)
    if actual_count == 0:
        return False, "no tool calls"
    allowed_count = required_count + 1
    repeated = [
        name
        for name, count in Counter(call["name"] for call in tool_calls).items()
        if count > Counter(required_tools)[name] + 1
    ]
    if actual_count > allowed_count:
        return False, f"too many calls ({actual_count} > {allowed_count})"
    if repeated:
        return False, f"repeated calls: {', '.join(repeated)}"
    return True, "tool calls efficient"


def _output_hygiene(output: str) -> tuple[bool, str]:
    stripped = output.strip()
    lower = stripped.lower()
    if not stripped:
        return False, "empty final answer"
    if "<think>" not in lower and "</think>" not in lower:
        return True, "clean final answer"
    after_think = _text_after_last_think_block(stripped).strip()
    if not after_think:
        return False, "thinking trace leaked without final answer"
    return False, "thinking trace leaked before final answer"


def _text_after_last_think_block(text: str) -> str:
    lower = text.lower()
    marker = "</think>"
    index = lower.rfind(marker)
    if index < 0:
        return ""
    return text[index + len(marker) :]


def _normalize_answer_text(text: str) -> str:
    return text.lower().replace(",", "")


def _requires_refs(
    *checks: tuple[str, str, str],
) -> Callable[[tuple[dict[str, Any], ...]], tuple[bool, str]]:
    def check(calls: tuple[dict[str, Any], ...]) -> tuple[bool, str]:
        missing = [
            f"{tool_name}.{key} contains {value}"
            for tool_name, key, value in checks
            if not _argument_contains(calls, tool_name, key, value)
        ]
        if missing:
            return False, f"missing refs: {', '.join(missing)}"
        return True, "argument refs ok"

    return check


def _argument_contains(
    calls: tuple[dict[str, Any], ...],
    tool_name: str,
    key: str,
    value: str,
) -> bool:
    expected = value.lower()
    for call in calls:
        if call["name"] != tool_name:
            continue
        actual = call["arguments"].get(key)
        if _value_contains(actual, expected):
            return True
    return False


def _value_contains(actual: Any, expected: str) -> bool:
    if isinstance(actual, str):
        return actual.lower() == expected or expected in actual.lower()
    if isinstance(actual, (list, tuple)):
        return any(_value_contains(item, expected) for item in actual)
    return str(actual).lower() == expected


def _has_sql_terms(
    calls: tuple[dict[str, Any], ...],
    terms: tuple[str, ...],
) -> tuple[bool, str]:
    sql_values = [
        str(call["arguments"].get("sql", "")).lower()
        for call in calls
        if call["name"] == "run_sql"
    ]
    if not sql_values:
        return False, "no SQL argument"
    sql = sql_values[-1]
    missing = [term for term in terms if term not in sql]
    return not missing, "sql terms ok" if not missing else f"missing SQL terms: {missing}"


def model_result_to_dict(result: ModelResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "available": result.available,
        "score": result.score,
        "max_score": result.max_score,
        "percent": round(result.percent * 100, 1),
        "pass_rate": round(result.pass_rate * 100, 1),
        "duration_seconds": round(result.duration_seconds, 2),
        "error": result.error,
        "cases": [
            {
                "name": case.name,
                "category": case.category,
                "score": case.score,
                "max_score": case.max_score,
                "passed": case.passed,
                "duration_seconds": round(case.duration_seconds, 2),
                "stop_reason": case.stop_reason,
                "detail": case.detail,
                "rubric": case.rubric,
                "warnings": list(case.warnings),
                "tool_calls": list(case.tool_calls),
                "output": case.output,
            }
            for case in result.cases
        ],
    }


def render_markdown(results: tuple[ModelResult, ...]) -> str:
    fastest = _fastest_available_duration(results)
    lines = [
        "# Ollama Analytics Demo Model Scorecard",
        "",
        "Harness: `demos.analytics` CSV source, profiler, semantic model, demo agent, "
        "`AnalyticsToolset`, and `ModelsToolset`.",
        "",
        "| Model | Score | Pass Rate | Duration | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if not result.available:
            lines.append(
                f"| `{result.model}` | n/a | n/a | {result.duration_seconds:.1f}s | unavailable |"
            )
            continue
        status = result.error or "ok"
        lines.append(
            f"| `{result.model}` | {result.score:.1f}/{result.max_score:.1f} "
            f"({result.percent * 100:.1f}%) | {result.pass_rate * 100:.1f}% | "
            f"{result.duration_seconds:.1f}s | {status} |"
        )

    lines.extend(["", "## Speed Comparison", ""])
    lines.extend(["| Model | Duration | Relative To Fastest |", "| --- | ---: | ---: |"])
    for result in results:
        if not result.available or fastest is None:
            continue
        ratio = result.duration_seconds / fastest if fastest else 0.0
        lines.append(f"| `{result.model}` | {result.duration_seconds:.1f}s | {ratio:.2f}x |")

    lines.extend(["", "## Rubric", ""])
    lines.extend(["| Component | Points | Meaning |", "| --- | ---: | --- |"])
    lines.append("| Answer correctness | 35 | Expected facts/terms appear in the final answer. |")
    lines.append("| Completion | 10 | Agent turn completed without hitting an error/step limit. |")
    lines.append("| Tool selection | 20 | Required analytics demo tools were called. |")
    lines.append("| Tool arguments | 20 | Tool arguments used discovered table/column refs. |")
    lines.append("| Tool efficiency | 5 | No excessive repeated tool calls. |")
    lines.append(
        "| Output hygiene | 10 | Final answer is clean: no visible `<think>` trace and not empty. |"
    )

    lines.extend(["", "## Case Details", ""])
    for result in results:
        lines.append(f"### {result.model}")
        if result.error:
            lines.extend(["", f"- Error: {result.error}", ""])
            continue
        lines.extend(
            [
                "",
                "| Case | Category | Score | Tools | Warnings | Detail |",
                "| --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for case in result.cases:
            tools = ", ".join(call["name"] for call in case.tool_calls) or "-"
            mark = "PASS" if case.passed else "FAIL"
            warnings = ", ".join(case.warnings) or "-"
            lines.append(
                f"| {case.name} | {case.category} | {case.score:.1f}/{case.max_score:.1f} "
                f"{mark} | {tools} | {warnings} | {case.detail} |"
            )
        lines.append("")
    lines.extend(["## Selection Heuristic", ""])
    lines.append(
        "- Prefer the highest score for the production analytics demo, but inspect "
        "modeling, forecast, SQL, and causal rows first."
    )
    lines.append("- Use the fastest passing model when analytics-tool accuracy is tied.")
    lines.append(
        "- Treat failures to call required demo tools as blockers, even if the final "
        "answer sounds plausible."
    )
    return "\n".join(lines)


def _fastest_available_duration(results: tuple[ModelResult, ...]) -> float | None:
    durations = [
        result.duration_seconds for result in results if result.available and not result.error
    ]
    return min(durations) if durations else None


if __name__ == "__main__":
    main()
