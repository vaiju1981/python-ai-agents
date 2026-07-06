"""Live Ollama model scorecard for production-demo model selection.

Run from the repository root:

    PYTHONPATH=src python tools/model_scorecard.py

The harness compares candidate Ollama models on simple prompting, complex
reasoning, multi-turn memory, single-tool use, multi-tool use, and
analytics-style tool calls. It writes both JSON and Markdown scorecards.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio

from python_ai_agents import (
    AgentRequest,
    DefaultAgent,
    InMemoryConversationStore,
    RecordingObserver,
    RequestContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from python_ai_agents.adapters import DEFAULT_OLLAMA_TEST_MODELS, OllamaModelPort
from python_ai_agents.core.tool import Tool


DEFAULT_MODELS = DEFAULT_OLLAMA_TEST_MODELS
DEFAULT_OUTPUT_DIR = Path("model_scorecards")


@dataclass(frozen=True, slots=True)
class ScorecardCase:
    name: str
    category: str
    prompt: str
    expected_terms: tuple[str, ...]
    tools: tuple[Tool, ...] = ()
    turns: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
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


@dataclass(slots=True)
class FunctionTool:
    _spec: ToolSpec
    _fn: Callable[[dict[str, Any]], str]

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, arguments: dict[str, Any], context: RequestContext) -> ToolResult:
        try:
            return ToolResult.ok(self._fn(arguments))
        except Exception as exc:
            return ToolResult.failed(f"{self.spec.name} failed: {exc}")


SALES_ROWS = (
    {"region": "North", "product": "Widget", "revenue": 120, "cost": 80, "units": 6},
    {"region": "South", "product": "Widget", "revenue": 90, "cost": 60, "units": 5},
    {"region": "East", "product": "Gadget", "revenue": 150, "cost": 90, "units": 10},
    {"region": "West", "product": "Gadget", "revenue": 130, "cost": 95, "units": 8},
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Ollama models for the production demo.")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    results = anyio.run(
        run_scorecard,
        tuple(args.models),
        args.base_url,
        args.timeout,
        args.max_steps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"ollama_model_scorecard_{timestamp}.json"
    md_path = args.output_dir / f"ollama_model_scorecard_{timestamp}.md"

    payload = {
        "generated_at": timestamp,
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
) -> tuple[ModelResult, ...]:
    cases = build_cases()
    results: list[ModelResult] = []
    for model_name in models:
        model_start = perf_counter()
        model = OllamaModelPort(
            model_name,
            base_url=base_url,
            options={"temperature": 0, "top_p": 0.1},
            timeout=timeout,
        )
        try:
            if not await model.has_model():
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
            case_results = [
                await run_case(model_name, model, case, max_steps)
                for case in cases
            ]
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
        except Exception as exc:
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
    agent = DefaultAgent(
        model=model,
        tools=list(case.tools),
        system_prompt=system_prompt_for(case),
        max_steps=max_steps,
        observers=[observer],
        conversation_store=InMemoryConversationStore(),
        tool_timeout_seconds=30.0,
    )
    context = RequestContext(session_id=f"scorecard-{model_name}-{case.name}", tenant="scorecard")
    started = perf_counter()
    output = ""
    stop_reason = ""
    try:
        if case.turns:
            for turn in case.turns:
                response = await agent.run(AgentRequest(turn, context))
                output = response.output
                stop_reason = response.stop_reason
        else:
            response = await agent.run(AgentRequest(case.prompt, context))
            output = response.output
            stop_reason = response.stop_reason
    except Exception as exc:
        output = f"{exc.__class__.__name__}: {exc}"
        stop_reason = "error"

    tool_calls = tuple(
        {"name": call.name, "arguments": call.arguments}
        for call in observer.tool_calls
    )
    score, max_score, passed, detail = score_case(case, output, tool_calls, stop_reason)
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
    )


def build_cases() -> tuple[ScorecardCase, ...]:
    return (
        ScorecardCase(
            name="simple_exact",
            category="simple",
            prompt="Reply with exactly: PASS",
            expected_terms=("pass",),
            notes="Basic instruction following.",
        ),
        ScorecardCase(
            name="complex_reasoning",
            category="complex",
            prompt=(
                "Facts: East revenue is 150 and South revenue is 90. "
                "Which region leads and by how much? Answer as 'East leads by 60'."
            ),
            expected_terms=("east", "60"),
            notes="Small arithmetic with constrained wording.",
        ),
        ScorecardCase(
            name="multi_turn_memory",
            category="multi-turn",
            prompt="",
            turns=(
                "Remember for this session: North revenue is 120 and South revenue is 90. Reply OK.",
                "Using only what I told you earlier, which region has higher revenue and by how much?",
            ),
            expected_terms=("north", "30"),
            notes="Checks conversation store plus model recall.",
        ),
        ScorecardCase(
            name="single_tool_addition",
            category="single-tool",
            prompt="Use the add_numbers tool to add 17 and 25. Then answer with the sum.",
            expected_terms=("42",),
            tools=(add_numbers_tool(),),
            required_tools=("add_numbers",),
            argument_check=lambda calls: _has_numbers(calls, "add_numbers", (17, 25)),
            notes="Canonical function-call reliability.",
        ),
        ScorecardCase(
            name="multi_tool_metric_delta",
            category="multi-tool",
            prompt=(
                "Use lookup_metric for North revenue and South revenue, then use subtract_numbers "
                "to compute North minus South. Final answer should say North leads by the difference."
            ),
            expected_terms=("north", "30"),
            tools=(lookup_metric_tool(), subtract_numbers_tool()),
            required_tools=("lookup_metric", "lookup_metric", "subtract_numbers"),
            argument_check=_checks_metric_delta,
            notes="Requires sequencing two lookup calls plus math.",
        ),
        ScorecardCase(
            name="analytics_groupby",
            category="analytics",
            prompt=(
                "Use run_metric_query to calculate total revenue by region. "
                "Which region has the highest total revenue and what is the amount?"
            ),
            expected_terms=("east", "150"),
            tools=(run_metric_query_tool(),),
            required_tools=("run_metric_query",),
            argument_check=_requires_argument_pairs(
                ("run_metric_query", "metric", "revenue"),
                ("run_metric_query", "dimension", "region"),
            ),
            notes="Analytics agent: choose metric + dimension.",
        ),
        ScorecardCase(
            name="analytics_margin_rank",
            category="analytics-complex",
            prompt=(
                "Use rank_metric with metric margin_rate grouped by product. "
                "Which product has the higher margin rate?"
            ),
            expected_terms=("gadget",),
            tools=(rank_metric_tool(),),
            required_tools=("rank_metric",),
            argument_check=_requires_argument_pairs(
                ("rank_metric", "metric", "margin_rate"),
                ("rank_metric", "dimension", "product"),
            ),
            notes="Derived metric over grouped records.",
        ),
        ScorecardCase(
            name="complex_sql_tool",
            category="complex-tool",
            prompt=(
                "Use read_only_sql to calculate gross margin by product from sales. "
                "Then tell me which product has the highest gross margin."
            ),
            expected_terms=("gadget", "95"),
            tools=(read_only_sql_tool(),),
            required_tools=("read_only_sql",),
            argument_check=lambda calls: _has_sql_terms(calls, ("product", "revenue", "cost")),
            notes="SQL-shaped complex tool use for analytics fallback.",
        ),
    )


def system_prompt_for(case: ScorecardCase) -> str:
    if not case.tools:
        return (
            "You are being evaluated. Follow the user instructions exactly. "
            "Be terse and do not add caveats."
        )
    return (
        "You are being evaluated for production agent tool use. "
        "When the user asks you to use a tool, call the appropriate tool with valid arguments. "
        "After tool results are available, stop calling tools and answer tersely with the requested value."
    )


def add_numbers_tool() -> Tool:
    return FunctionTool(
        ToolSpec(
            name="add_numbers",
            description="Add two numbers.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        lambda args: str(float(args["a"]) + float(args["b"])).rstrip("0").rstrip("."),
    )


def subtract_numbers_tool() -> Tool:
    return FunctionTool(
        ToolSpec(
            name="subtract_numbers",
            description="Subtract b from a.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        lambda args: str(float(args["a"]) - float(args["b"])).rstrip("0").rstrip("."),
    )


def lookup_metric_tool() -> Tool:
    return FunctionTool(
        ToolSpec(
            name="lookup_metric",
            description="Look up one metric value for one region.",
            input_schema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": ["North", "South", "East", "West"]},
                    "metric": {"type": "string", "enum": ["revenue", "cost", "units"]},
                },
                "required": ["region", "metric"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        lambda args: str(_sum(args["metric"], region=args["region"])),
    )


def run_metric_query_tool() -> Tool:
    return FunctionTool(
        ToolSpec(
            name="run_metric_query",
            description="Group an analytics metric by one dimension.",
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["revenue", "cost", "units", "gross_margin", "margin_rate"],
                    },
                    "dimension": {"type": "string", "enum": ["region", "product"]},
                },
                "required": ["metric", "dimension"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        lambda args: json.dumps(_group_metric(args["metric"], args["dimension"]), sort_keys=True),
    )


def rank_metric_tool() -> Tool:
    return FunctionTool(
        ToolSpec(
            name="rank_metric",
            description="Rank dimension values by metric descending.",
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["revenue", "cost", "units", "gross_margin", "margin_rate"],
                    },
                    "dimension": {"type": "string", "enum": ["region", "product"]},
                    "top_n": {"type": "integer", "default": 3},
                },
                "required": ["metric", "dimension"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        lambda args: json.dumps(
            _group_metric(args["metric"], args["dimension"])[: int(args.get("top_n", 3))],
            sort_keys=True,
        ),
    )


def read_only_sql_tool() -> Tool:
    return FunctionTool(
        ToolSpec(
            name="read_only_sql",
            description=(
                "Run a read-only SQL query over table sales(region, product, revenue, cost, units)."
            ),
            input_schema={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        _run_synthetic_sql,
    )


def _run_synthetic_sql(args: dict[str, Any]) -> str:
    sql = str(args.get("sql", "")).lower()
    forbidden = ("insert", "update", "delete", "drop", "alter", "create")
    if any(word in sql for word in forbidden):
        raise ValueError("only SELECT queries are allowed")
    if "product" in sql and "revenue" in sql and "cost" in sql:
        rows = _group_metric("gross_margin", "product")
        return json.dumps(rows, sort_keys=True)
    return json.dumps(SALES_ROWS, sort_keys=True)


def _sum(metric: str, *, region: str | None = None, product: str | None = None) -> float:
    rows = SALES_ROWS
    if region is not None:
        rows = tuple(row for row in rows if row["region"].lower() == region.lower())
    if product is not None:
        rows = tuple(row for row in rows if row["product"].lower() == product.lower())
    if metric == "gross_margin":
        return sum(row["revenue"] - row["cost"] for row in rows)
    return sum(row[metric] for row in rows)


def _group_metric(metric: str, dimension: str) -> list[dict[str, Any]]:
    groups = sorted({str(row[dimension]) for row in SALES_ROWS})
    result = []
    for group in groups:
        rows = tuple(row for row in SALES_ROWS if row[dimension] == group)
        revenue = sum(row["revenue"] for row in rows)
        cost = sum(row["cost"] for row in rows)
        if metric == "gross_margin":
            value = revenue - cost
        elif metric == "margin_rate":
            value = round((revenue - cost) / revenue, 4)
        else:
            value = sum(row[metric] for row in rows)
        result.append({dimension: group, metric: value})
    result.sort(key=lambda item: float(item[metric]), reverse=True)
    return result


def score_case(
    case: ScorecardCase,
    output: str,
    tool_calls: tuple[dict[str, Any], ...],
    stop_reason: str,
) -> tuple[float, float, bool, str]:
    max_score = case_max_score(case)
    details: list[str] = []
    score = 0.0

    normalized_output = output.lower()
    missing_terms = [term for term in case.expected_terms if term.lower() not in normalized_output]
    if not missing_terms:
        score += 6.0
        details.append("answer terms ok")
    else:
        details.append(f"missing answer terms: {', '.join(missing_terms)}")

    if stop_reason == "completed":
        score += 1.0
        details.append("completed")
    else:
        details.append(f"stop_reason={stop_reason}")

    if case.required_tools:
        missing_tools = _missing_required_tools(case.required_tools, tool_calls)
        if not missing_tools:
            score += 2.0
            details.append("required tools ok")
        else:
            details.append(f"missing tools: {', '.join(missing_tools)}")

        if case.argument_check is None:
            score += 1.0
        else:
            ok, message = case.argument_check(tool_calls)
            if ok:
                score += 1.0
            details.append(f"arguments: {message}")
    else:
        score += 3.0
        details.append("no tool required")

    passed = score >= max_score * 0.8
    return score, max_score, passed, "; ".join(details)


def case_max_score(case: ScorecardCase) -> float:
    return 10.0


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


def _has_numbers(
    calls: tuple[dict[str, Any], ...],
    tool_name: str,
    expected: tuple[float, float],
) -> tuple[bool, str]:
    for call in calls:
        if call["name"] != tool_name:
            continue
        args = call["arguments"]
        values = sorted(float(value) for value in args.values() if _is_number(value))
        if values == sorted(expected):
            return True, "numeric arguments ok"
    return False, f"expected numeric arguments {expected}"


def _checks_metric_delta(calls: tuple[dict[str, Any], ...]) -> tuple[bool, str]:
    lookup_args = [call["arguments"] for call in calls if call["name"] == "lookup_metric"]
    has_north = any(str(args.get("region", "")).lower() == "north" for args in lookup_args)
    has_south = any(str(args.get("region", "")).lower() == "south" for args in lookup_args)
    subtract_ok, _message = _has_numbers(calls, "subtract_numbers", (120, 90))
    ok = has_north and has_south and subtract_ok
    return ok, f"north_lookup={has_north} south_lookup={has_south} subtract_120_90={subtract_ok}"


def _has_argument_pair(
    calls: tuple[dict[str, Any], ...],
    tool_name: str,
    key: str,
    value: str,
) -> bool:
    for call in calls:
        if call["name"] != tool_name:
            continue
        actual = call["arguments"].get(key)
        if isinstance(actual, str) and actual.lower() == value.lower():
            return True
    return False


def _requires_argument_pairs(
    *pairs: tuple[str, str, str],
) -> Callable[[tuple[dict[str, Any], ...]], tuple[bool, str]]:
    def check(calls: tuple[dict[str, Any], ...]) -> tuple[bool, str]:
        missing = [
            f"{tool_name}.{key}={value}"
            for tool_name, key, value in pairs
            if not _has_argument_pair(calls, tool_name, key, value)
        ]
        if missing:
            return False, f"missing argument pairs: {', '.join(missing)}"
        return True, "argument pairs ok"

    return check


def _has_sql_terms(
    calls: tuple[dict[str, Any], ...],
    terms: tuple[str, ...],
) -> tuple[bool, str]:
    sql_values = [
        str(call["arguments"].get("sql", "")).lower()
        for call in calls
        if call["name"] == "read_only_sql"
    ]
    if not sql_values:
        return False, "no SQL argument"
    sql = sql_values[-1]
    missing = [term for term in terms if term not in sql]
    return not missing, "sql terms ok" if not missing else f"missing SQL terms: {missing}"


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


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
                "tool_calls": list(case.tool_calls),
                "output": case.output,
            }
            for case in result.cases
        ],
    }


def render_markdown(results: tuple[ModelResult, ...]) -> str:
    lines = [
        "# Ollama Model Scorecard",
        "",
        "| Model | Score | Pass Rate | Duration | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if not result.available:
            lines.append(f"| `{result.model}` | n/a | n/a | {result.duration_seconds:.1f}s | unavailable |")
            continue
        status = result.error or "ok"
        lines.append(
            f"| `{result.model}` | {result.score:.1f}/{result.max_score:.1f} "
            f"({result.percent * 100:.1f}%) | {result.pass_rate * 100:.1f}% | "
            f"{result.duration_seconds:.1f}s | {status} |"
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
                "| Case | Category | Score | Tools | Detail |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for case in result.cases:
            tools = ", ".join(call["name"] for call in case.tool_calls) or "-"
            mark = "PASS" if case.passed else "FAIL"
            lines.append(
                f"| {case.name} | {case.category} | {case.score:.1f}/{case.max_score:.1f} "
                f"{mark} | {tools} | {case.detail} |"
            )
        lines.append("")
    lines.extend(["## Selection Heuristic", ""])
    lines.append(
        "- Prefer the highest score for the production analytics demo, but inspect analytics, "
        "multi-tool, and complex-tool rows first; those are more predictive than simple echo tests."
    )
    lines.append(
        "- Use the fastest passing model for local demos when analytics-tool accuracy is tied."
    )
    lines.append(
        "- Treat failures to call required tools as blockers for the analytics demo, even if the "
        "final answer looks plausible."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
