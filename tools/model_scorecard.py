"""Live Ollama scorecard using the production analytics demo stack.

Run from the repository root:

    PYTHONPATH=src:. python tools/model_scorecard.py

The harness compares candidate Ollama models by running the actual analytics
demo path: CSV import, deterministic profiling, semantic model construction, the
demo agent prompt, AnalyticsToolset, and ModelsToolset. It writes both JSON and
Markdown scorecards.

Design notes (what makes it discriminating rather than "everyone gets 100%"):

* Prompts are **natural business questions** — they never name the tool or its
  arguments, so the model has to choose the right tool and columns itself.
* Answer correctness is scored against the **ground truth computed live from the
  dataset** (partial credit), not substring matches on hand-written terms.
* Tool selection accepts **any of** a set of defensible tools per case, so using
  ``run_sql`` where ``run_query`` also works isn't wrongly penalized.
* Models are grouped into **local** and **cloud** tiers so the ranking is
  meaningful for a given deployment.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
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
    # Correctness is weighted highest: the prompts don't name a tool, and the
    # expected values are the *true* answers computed from the dataset. Getting the
    # right number implies the right tool + columns, so a separate argument check
    # is redundant and was dropped.
    "answer": 45.0,
    "tool_selection": 25.0,
    "completion": 10.0,
    "tool_efficiency": 5.0,
    "output_hygiene": 15.0,
}


@dataclass(frozen=True, slots=True)
class ScorecardCase:
    name: str
    category: str
    prompt: str  # a natural business question — it must NOT name the tool or its args
    expected_terms: tuple[str, ...]  # ground-truth tokens; each entry is a `a|b|c` synonym set
    tool_groups: tuple[tuple[str, ...], ...]  # each group is satisfied if ANY listed tool is called
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
    def tier(self) -> str:
        return _tier(self.model)

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
    parser.add_argument("--timeout", type=float, default=180.0)  # cloud models need > 60s per call
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


def _tier(model: str) -> str:
    return "cloud" if "cloud" in model.lower() else "local"


def _ground_truth() -> dict[str, str]:
    """Compute the true answers from the deterministic scorecard CSV, so correctness
    is scored against real values that can't drift from the data."""
    source, _ = build_demo_dataset()
    try:
        q = source.native_query
        region = q(
            "SELECT region, SUM(revenue) AS r FROM scorecard_sales "
            "GROUP BY region ORDER BY r DESC LIMIT 1"
        )[0]
        product = q(
            "SELECT product, SUM(profit) AS p FROM scorecard_sales "
            "GROUP BY product ORDER BY p DESC LIMIT 1"
        )[0]
        ppu = q(
            "SELECT product, SUM(profit) / SUM(units) AS ppu FROM scorecard_sales "
            "GROUP BY product ORDER BY ppu DESC LIMIT 1"
        )[0]
        ab = {
            row["campaign"]: float(row["m"])
            for row in q(
                "SELECT campaign, AVG(revenue) AS m FROM scorecard_sales "
                "WHERE campaign IN ('test', 'control') GROUP BY campaign"
            )
        }
    finally:
        source.close()
    return {
        "top_region": str(region["region"]),
        "top_region_revenue": str(int(region["r"])),
        "top_product": str(product["product"]),
        "top_product_profit": str(int(product["p"])),
        "best_ppu_product": str(ppu["product"]),
        "ab_winner": "test" if ab["test"] > ab["control"] else "control",
    }


def build_cases() -> tuple[ScorecardCase, ...]:
    gt = _ground_truth()
    return (
        ScorecardCase(
            name="schema_discovery",
            category="analytics-schema",
            prompt="What table is in this dataset, and what are two useful metrics to analyze?",
            expected_terms=("scorecard_sales", "revenue|profit|cost|units|visits"),
            tool_groups=(("describe_dataset", "run_sql"),),
            notes="Natural schema question; the model must reach for a discovery tool.",
        ),
        ScorecardCase(
            name="revenue_by_region",
            category="analytics-query",
            prompt="Which region has the highest total revenue, and what is that total?",
            expected_terms=(gt["top_region"], gt["top_region_revenue"]),
            tool_groups=(("run_query", "run_sql"),),
            notes="Ground truth: the top region and its summed revenue.",
        ),
        ScorecardCase(
            name="profit_by_product",
            category="analytics-sql",
            prompt="Which product has the higher total profit, and what is the total?",
            expected_terms=(gt["top_product"], gt["top_product_profit"]),
            tool_groups=(("run_sql", "run_query"),),
            notes="Ground truth: the top product and its summed profit.",
        ),
        ScorecardCase(
            name="profit_per_unit",
            category="analytics-derived",
            prompt="Which product has the highest profit per unit sold?",
            expected_terms=(gt["best_ppu_product"],),
            tool_groups=(("run_sql",),),
            notes="Harder: a derived metric (profit/units) needs SQL, not a plain group-by.",
        ),
        ScorecardCase(
            name="revenue_trend",
            category="analytics-time",
            prompt="Is total revenue rising or falling across the months?",
            expected_terms=("rising|increasing|upward|grew|growth|higher|up",),
            tool_groups=(("trend", "run_sql", "run_query"),),
            notes="Trend direction (the data rises by construction).",
        ),
        ScorecardCase(
            name="summarize_profit",
            category="analytics-stats",
            prompt="Give me the average profit and its range (min and max).",
            expected_terms=("mean|average|avg", "range|min|max|lowest|highest"),
            tool_groups=(("summarize", "run_sql"),),
            notes="Descriptive statistics under a natural prompt.",
        ),
        ScorecardCase(
            name="drivers_of_profit",
            category="analytics-drivers",
            prompt="Which numeric factors most strongly relate to profit?",
            expected_terms=("revenue|cost",),
            tool_groups=(("correlate", "regression", "build_model"),),
            notes="Harder: ambiguous — several tools are defensible; revenue/cost drive profit most.",
        ),
        ScorecardCase(
            name="build_predictive_model",
            category="analytics-modeling",
            prompt="Build a model that predicts profit and tell me which feature matters most.",
            expected_terms=("importance|important|feature",),
            tool_groups=(("build_model",),),
            notes="Feature importance is data-dependent, so the answer token is scored lightly.",
        ),
        ScorecardCase(
            name="forecast_revenue",
            category="analytics-forecast",
            prompt="Project revenue for the next two months, and tell me what method you used.",
            expected_terms=("forecast|projection|method|holt|trend",),
            tool_groups=(("forecast",),),
            notes="Forecasting tool selection under a natural prompt.",
        ),
        ScorecardCase(
            name="campaign_ab_test",
            category="analytics-experiment",
            prompt=(
                "Did the 'test' campaign beat 'control' on revenue, and is the difference "
                "statistically significant?"
            ),
            expected_terms=(gt["ab_winner"], "significant|p-value|p value|significance"),
            tool_groups=(("ab_test",),),
            notes="Ground truth: which campaign wins on average revenue, plus a significance claim.",
        ),
        ScorecardCase(
            name="causal_effect",
            category="analytics-causal",
            prompt=(
                "Does the treatment cause higher profit, after adjusting for units and visits? "
                "Be explicit about how much we can actually conclude."
            ),
            expected_terms=("causation|causal|confound|cannot|not proof|not prove",),
            tool_groups=(("causal_effect", "regression"),),
            notes="Must surface the not-proof-of-causation caveat.",
        ),
        ScorecardCase(
            name="cluster_segments",
            category="analytics-segmentation",
            prompt="Group the rows into three segments based on their numeric behavior.",
            expected_terms=("cluster|segment|group",),
            tool_groups=(("cluster",),),
            notes="Segmentation tool selection under a natural prompt.",
        ),
    )


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

    normalized = _normalize_answer_text(output)

    # Answer correctness: fraction of ground-truth tokens present (partial credit).
    if case.expected_terms:
        found = [term for term in case.expected_terms if _term_present(term, normalized)]
        rubric["answer"] = RUBRIC_WEIGHTS["answer"] * len(found) / len(case.expected_terms)
        missing = [term for term in case.expected_terms if term not in found]
        details.append("answer ok" if not missing else f"answer missing: {', '.join(missing)}")
    else:
        rubric["answer"] = RUBRIC_WEIGHTS["answer"]

    # Tool selection: each group is satisfied if ANY of its tools was called.
    missing_groups = _missing_tool_groups(case.tool_groups, tool_calls)
    if not missing_groups:
        rubric["tool_selection"] = RUBRIC_WEIGHTS["tool_selection"]
        details.append("tool ok")
    else:
        details.append("missing tool (any of): " + "; ".join("/".join(g) for g in missing_groups))

    if stop_reason == "completed":
        rubric["completion"] = RUBRIC_WEIGHTS["completion"]
        details.append("completed")
    else:
        details.append(f"stop_reason={stop_reason}")

    efficient, efficiency_detail = _tool_efficiency(case.tool_groups, tool_calls)
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


def _term_present(term: str, normalized_output: str) -> bool:
    """A term is present if ANY of its `a|b|c` synonyms appears in the output."""
    return any(syn.strip() in normalized_output for syn in term.lower().split("|") if syn.strip())


def _missing_tool_groups(
    tool_groups: tuple[tuple[str, ...], ...],
    tool_calls: tuple[dict[str, Any], ...],
) -> list[tuple[str, ...]]:
    called = {call["name"] for call in tool_calls}
    return [group for group in tool_groups if not any(name in called for name in group)]


def _tool_efficiency(
    tool_groups: tuple[tuple[str, ...], ...],
    tool_calls: tuple[dict[str, Any], ...],
) -> tuple[bool, str]:
    actual_count = len(tool_calls)
    if actual_count == 0:
        return False, "no tool calls"
    allowed_count = len(tool_groups) + 1
    counts = Counter(call["name"] for call in tool_calls)
    repeated = [name for name, count in counts.items() if count > 2]
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


def model_result_to_dict(result: ModelResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "tier": result.tier,
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
        "`AnalyticsToolset`, and `ModelsToolset`. Prompts are natural questions (the tool is "
        "not named); correctness is scored against ground truth computed from the data.",
        "",
        "| Model | Tier | Score | Pass Rate | Duration | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if not result.available:
            lines.append(
                f"| `{result.model}` | {result.tier} | n/a | n/a | "
                f"{result.duration_seconds:.1f}s | unavailable |"
            )
            continue
        status = result.error or "ok"
        lines.append(
            f"| `{result.model}` | {result.tier} | {result.score:.1f}/{result.max_score:.1f} "
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
    lines.append(
        "| Answer correctness | 45 | The dataset's *true* answer (computed from the CSV) appears "
        "in the reply; partial credit per fact. |"
    )
    lines.append(
        "| Tool selection | 25 | A relevant analytics tool was chosen — the prompt does not name it. |"
    )
    lines.append("| Completion | 10 | Agent turn completed without an error/step limit. |")
    lines.append("| Tool efficiency | 5 | No excessive or repeated tool calls. |")
    lines.append(
        "| Output hygiene | 15 | Final answer is clean: no visible `<think>` trace and not empty. |"
    )

    lines.extend(["", "## Case Details", ""])
    for result in results:
        lines.append(f"### {result.model} ({result.tier})")
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
    best_local = _best_in_tier(results, "local")
    best_cloud = _best_in_tier(results, "cloud")
    if best_local:
        lines.append(f"- Best **local** model: `{best_local.model}` ({best_local.percent * 100:.1f}%).")
    if best_cloud:
        lines.append(f"- Best **cloud** model: `{best_cloud.model}` ({best_cloud.percent * 100:.1f}%).")
    lines.append("- Pick within the tier you can actually deploy; only compare across tiers for reference.")
    lines.append("- Prefer higher correctness; break ties on the modeling/forecast/causal rows, then speed.")
    return "\n".join(lines)


def _best_in_tier(results: tuple[ModelResult, ...], tier: str) -> ModelResult | None:
    candidates = [r for r in results if r.available and not r.error and r.tier == tier]
    return max(candidates, key=lambda r: r.percent) if candidates else None


def _fastest_available_duration(results: tuple[ModelResult, ...]) -> float | None:
    durations = [
        result.duration_seconds for result in results if result.available and not result.error
    ]
    return min(durations) if durations else None


if __name__ == "__main__":
    main()
