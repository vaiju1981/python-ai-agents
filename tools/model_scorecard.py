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
from python_ai_agents.adapters import RECOMMENDED_SAMPLING, OllamaModelPort

DEFAULT_MODELS = ("gemma4:31b-cloud", "ornith:latest")
DEFAULT_OUTPUT_DIR = Path("model_scorecards")
PASS_THRESHOLD = 80.0
# Both candidate models support 262K ctx. This benchmark only uses ~3K tokens of prompt,
# so this cap is headroom for large production schemas, not a limit the suite hits.
MAX_OLLAMA_CONTEXT = 131_072

# Real-data suite: three related casino tables (11.5M rows, ~170 columns total) with trap
# columns (coinIn vs coinInCarded, netWin vs grossWin vs theoWin) AND cross-table joins
# (sessions bridges players<->machines) that punish schema-navigation mistakes. This is
# what actually separates a 9B from a 31B. Opt in with `--dataset redrock`.
REDROCK_DEFAULT_DIR = "/Users/vaijanath.rao/ga_cache/training_data"
REDROCK_FILES = {
    "asset_daily": "STATION_assetDaily_REDROCK.csv",  # machine-day performance (assetId)
    "player_visits": "STATION_playerVisits_REDROCK.csv",  # player demographics (playerId)
    "sessions": "STATION_sessions_REDROCK.csv",  # bridge: has BOTH playerId and assetId
}
REDROCK_TABLE = "asset_daily"  # primary machine table for the single-table cases
REDROCK_MAX_TRAIN_ROWS = 50_000

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
    # Multi-turn cases: a sequence of follow-up prompts run on ONE session so history
    # carries. The LAST turn is what's scored, and its answer is designed to depend on
    # resolving "that / those / it" against earlier turns (context-free answer differs).
    turns: tuple[str, ...] = ()


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
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "redrock"),
        default="synthetic",
        help="synthetic = built-in deterministic CSV; redrock = real 1.3M-row casino data",
    )
    parser.add_argument("--redrock-dir", default=REDROCK_DEFAULT_DIR)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=REDROCK_MAX_TRAIN_ROWS,
        help="row cap for ML tools on large tables (0 = no cap)",
    )
    # Defaults are the vendor-recommended sampling (temp 0 made ornith loop tool calls
    # into the step budget). Pass --temperature 0 --top-p 0.1 --top-k 0 for a
    # deterministic reproducibility run.
    parser.add_argument(
        "--temperature", type=float, default=float(RECOMMENDED_SAMPLING["temperature"])
    )
    parser.add_argument("--top-p", type=float, default=float(RECOMMENDED_SAMPLING["top_p"]))
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(RECOMMENDED_SAMPLING["top_k"]),
        help="0 = omit (Ollama default)",
    )
    args = parser.parse_args()
    num_ctx = min(args.num_ctx, MAX_OLLAMA_CONTEXT)
    max_train_rows = args.max_train_rows or None

    # One options dict used for BOTH the model calls and the recorded payload, so the
    # scorecard always reflects the exact sampling settings that produced it.
    options: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": num_ctx,
    }
    if args.top_k > 0:
        options["top_k"] = args.top_k

    results = anyio.run(
        run_scorecard,
        tuple(args.models),
        args.base_url,
        args.timeout,
        args.max_steps,
        options,
        args.dataset,
        args.redrock_dir,
        max_train_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"ollama_model_scorecard_{args.dataset}_{timestamp}.json"
    md_path = args.output_dir / f"ollama_model_scorecard_{args.dataset}_{timestamp}.md"

    payload = {
        "generated_at": timestamp,
        "harness": "demos.analytics",
        "dataset": args.dataset,
        "ollama_options": options,
        "models": [model_result_to_dict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results, args.dataset), encoding="utf-8")

    print(render_markdown(results, args.dataset))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


async def run_scorecard(
    models: tuple[str, ...],
    base_url: str,
    timeout: float,
    max_steps: int,
    options: dict[str, Any],
    dataset: str = "synthetic",
    redrock_dir: str = REDROCK_DEFAULT_DIR,
    max_train_rows: int | None = None,
) -> tuple[ModelResult, ...]:
    # Build the dataset ONCE and reuse it read-only across every case and model. For the
    # 4.7GB real data this is mandatory: re-importing per case would dominate the run.
    print(f"[scorecard] dataset={dataset} building", flush=True)
    source, semantic = build_dataset(dataset, redrock_dir)
    try:
        cases = build_cases(dataset, source)
        print(f"[scorecard] dataset={dataset} ready: {len(cases)} cases", flush=True)
        results: list[ModelResult] = []
        for model_name in models:
            print(f"[scorecard] model={model_name} starting", flush=True)
            model_start = perf_counter()
            model = OllamaModelPort(
                model_name,
                base_url=base_url,
                options=dict(options),
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
                    case_result = await run_case(
                        model_name, model, source, semantic, case, max_steps, max_train_rows
                    )
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
                print(
                    f"[scorecard] model={model_name} error={exc.__class__.__name__}: {exc}",
                    flush=True,
                )
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
    finally:
        source.close()
    return tuple(results)


async def run_case(
    model_name: str,
    model: OllamaModelPort,
    source: CsvSource,
    semantic: SemanticModel,
    case: ScorecardCase,
    max_steps: int,
    max_train_rows: int | None = None,
) -> CaseResult:
    observer = RecordingObserver()
    started = perf_counter()
    output = ""
    stop_reason = ""
    try:
        agent = create_agent(
            source, model, semantic, observers=[observer], max_train_rows=max_train_rows
        )
        agent.max_steps = max_steps
        # One agent + one session_id across all turns, so the framework's conversation store
        # carries history. For single-turn cases this is just one prompt. The LAST turn's
        # answer is what gets scored; the observer accumulates tool calls across turns.
        context = RequestContext(session_id=f"analytics-scorecard-{model_name}-{case.name}")
        for turn_prompt in case.turns or (case.prompt,):
            response = await agent.run(AgentRequest(turn_prompt, context))
            output = response.output
            stop_reason = response.stop_reason
            if stop_reason != "completed":
                break  # a broken turn breaks the chain — stop and score what we have
    except Exception as exc:
        output = f"{exc.__class__.__name__}: {exc}"
        stop_reason = "error"

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


def build_dataset(dataset: str, redrock_dir: str) -> tuple[CsvSource, SemanticModel]:
    if dataset == "redrock":
        return build_redrock_dataset(redrock_dir)
    return build_demo_dataset()


def build_demo_dataset() -> tuple[CsvSource, SemanticModel]:
    tmp_dir = tempfile.TemporaryDirectory()
    csv_path = Path(tmp_dir.name) / "scorecard_sales.csv"
    csv_path.write_text(_scorecard_csv(), encoding="utf-8")
    source = CsvSource(named_csvs={"scorecard_sales": csv_path})
    source._scorecard_tmp_dir = tmp_dir  # type: ignore[attr-defined]
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    return source, semantic


def build_redrock_dataset(redrock_dir: str) -> tuple[CsvSource, SemanticModel]:
    """Import all three real casino tables into a file-backed DuckDB (out-of-core) once.

    ~4.7GB / 11.5M rows across asset_daily, player_visits, and sessions — imports in ~1min,
    then every case queries it read-only. Loading all three is what makes cross-table
    questions ("what kind of players play the top machines") answerable: `sessions` bridges
    `player_visits` (playerId -> demographics) and machines (assetId/gameTitle). The wide,
    3-table, trap-heavy schema is the point — it is what actually separates a 9B from a 31B.
    """
    base = Path(redrock_dir)
    named = {table: base / filename for table, filename in REDROCK_FILES.items()}
    missing = [str(p) for p in named.values() if not p.exists()]
    if missing:
        raise SystemExit("REDROCK data not found:\n  " + "\n  ".join(missing))
    tmp_dir = tempfile.TemporaryDirectory()
    db_path = Path(tmp_dir.name) / "redrock.duckdb"
    source = CsvSource(db_path=str(db_path), named_csvs=named)
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


def _ground_truth_synthetic(source: CsvSource) -> dict[str, str]:
    """Compute the true answers from the deterministic scorecard CSV, so correctness
    is scored against real values that can't drift from the data."""
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
    margin = float(q("SELECT 100.0 * SUM(profit) / SUM(revenue) AS m FROM scorecard_sales")[0]["m"])
    test_rev = int(q("SELECT SUM(revenue) AS r FROM scorecard_sales WHERE campaign = 'test'")[0]["r"])
    best_margin = q(
        "SELECT product FROM scorecard_sales GROUP BY product "
        "ORDER BY SUM(profit) / SUM(revenue) DESC LIMIT 1"
    )[0]["product"]
    months = q(
        "SELECT date_trunc('month', day) AS mo, SUM(revenue) AS r "
        "FROM scorecard_sales GROUP BY mo ORDER BY mo"
    )
    first_r, last_r = float(months[0]["r"]), float(months[-1]["r"])
    mom_pct = 100.0 * (last_r - first_r) / first_r
    top2 = q("SELECT region FROM scorecard_sales GROUP BY region ORDER BY SUM(revenue) DESC LIMIT 2")
    return {
        "top_region": str(region["region"]),
        "top_region_revenue": str(int(region["r"])),
        "top_product": str(product["product"]),
        "top_product_profit": str(int(product["p"])),
        "best_ppu_product": str(ppu["product"]),
        "ab_winner": "test" if ab["test"] > ab["control"] else "control",
        "overall_margin_pct": str(int(margin)),
        "test_revenue": str(test_rev),
        "best_margin_product": str(best_margin),
        "mom_growth_pct": str(int(mom_pct)),
        "top_region_2": str(top2[1]["region"]),
    }


def build_cases(dataset: str, source: CsvSource) -> tuple[ScorecardCase, ...]:
    if dataset == "redrock":
        return build_redrock_cases(source)
    return build_synthetic_cases(source)


def build_synthetic_cases(source: CsvSource) -> tuple[ScorecardCase, ...]:
    gt = _ground_truth_synthetic(source)
    return (
        ScorecardCase(
            name="schema_discovery",
            category="analytics-schema",
            prompt="What table is in this dataset, and what are two useful metrics to analyze?",
            expected_terms=("scorecard_sales", "revenue|profit|cost|units|visits"),
            tool_groups=(),  # the schema is already in the prompt — answering directly is correct
            notes="Scored on the answer only; no tool is required when the schema is in context.",
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
        # --- Harder, capability-discriminating cases (fair, exact ground truth) ---
        ScorecardCase(
            name="profit_margin",
            category="analytics-ratio",
            prompt="What is the overall profit margin (profit divided by revenue) as a percentage?",
            expected_terms=(gt["overall_margin_pct"],),
            tool_groups=(("run_sql",),),
            notes="Hard: a ratio over all rows — run_query can't divide two sums.",
        ),
        ScorecardCase(
            name="filtered_test_revenue",
            category="analytics-filter",
            prompt="What is the total revenue for the 'test' campaign only?",
            expected_terms=(gt["test_revenue"],),
            tool_groups=(("run_query", "run_sql"),),
            notes="Hard: must apply the campaign='test' filter, not total everything.",
        ),
        ScorecardCase(
            name="best_margin_product",
            category="analytics-ratio",
            prompt="Which product has the better profit margin (profit divided by revenue)?",
            expected_terms=(gt["best_margin_product"],),
            tool_groups=(("run_sql",),),
            notes="Hard: ratio + argmax across products.",
        ),
        ScorecardCase(
            name="revenue_mom_growth",
            category="analytics-temporal",
            prompt="By what percent did total revenue change from the first month to the last month?",
            expected_terms=(gt["mom_growth_pct"],),
            tool_groups=(("run_sql", "trend"),),
            notes="Hard: multi-step — first vs last month totals, then percent change.",
        ),
        ScorecardCase(
            name="top_two_regions",
            category="analytics-query",
            prompt="List the top two regions by total revenue.",
            expected_terms=(gt["top_region"], gt["top_region_2"]),
            tool_groups=(("run_query", "run_sql"),),
            notes="Hard: multi-row ordered extraction (both region names must appear).",
        ),
        ScorecardCase(
            name="causation_honesty",
            category="analytics-honesty",
            prompt="Prove that the treatment causes higher profit.",
            expected_terms=(
                "cannot|can't|does not prove|not prove|no proof|observational|confound|"
                "randomized|correlation",
            ),
            tool_groups=(),
            notes="Honesty: a good model refuses to claim proof of causation from this data.",
        ),
    )


def _ground_truth_redrock(source: CsvSource) -> dict[str, str]:
    """True answers computed live from the real casino asset table.

    Answers are names / small ints / precise percents so exact-match scoring is fair on
    huge production numbers (no giant sums to recite). The discrimination lives in picking
    the RIGHT column among near-duplicates (coinIn vs *Carded; netWin vs grossWin vs
    theoWin) and the right aggregation — not in reciting a 10-digit total.
    """
    q = source.native_query
    t = REDROCK_TABLE
    top2_mfr = q(
        f"SELECT manufacturer m FROM {t} GROUP BY 1 ORDER BY SUM(coinIn) DESC LIMIT 2"
    )
    n_titles = int(q(f"SELECT COUNT(DISTINCT gameTitle) n FROM {t}")[0]["n"])
    konami_pb = float(
        q(f"SELECT AVG(paybackPercent) p FROM {t} WHERE manufacturer = 'KONAMI'")[0]["p"]
    )
    hold = float(q(f"SELECT 100.0 * SUM(netWin) / SUM(coinIn) h FROM {t}")[0]["h"])
    top_zone = q(
        f"SELECT zone z FROM {t} GROUP BY 1 ORDER BY COUNT(DISTINCT assetId) DESC LIMIT 1"
    )[0]["z"]
    lease = {
        bool(r["k"]): float(r["m"])
        for r in q(f"SELECT isLeased k, AVG(netWin) m FROM {t} GROUP BY 1")
    }
    top_game = q(
        f"SELECT gameTitle g FROM {t} GROUP BY 1 ORDER BY SUM(theoWin) DESC LIMIT 1"
    )[0]["g"]

    # --- cross-table ground truth (needs all three tables joined) ---
    chairman = q(
        "SELECT clubLevel c FROM player_visits GROUP BY 1 ORDER BY SUM(coinIn) DESC LIMIT 1"
    )[0]["c"]
    top_state = q(
        "SELECT state s FROM player_visits GROUP BY 1 "
        "ORDER BY COUNT(DISTINCT playerId) DESC LIMIT 1"
    )[0]["s"]
    top_game_ci = q(
        "SELECT gameTitle g FROM sessions GROUP BY 1 ORDER BY SUM(coinIn) DESC LIMIT 1"
    )[0]["g"]
    game_lit = str(top_game_ci).replace("'", "''")
    # ageGroup bracket numbers drift per player, so dedupe to one label per player, then
    # match on the stable generation name (Gen X / Millennials / ...), not the bracket.
    top_gen_label = q(
        "WITH pdim AS (SELECT playerId, MAX(ageGroup) ag FROM player_visits GROUP BY 1) "
        "SELECT p.ag ag FROM sessions s JOIN pdim p ON s.playerId = p.playerId "
        f"WHERE s.gameTitle = '{game_lit}' GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1"
    )[0]["ag"]
    gen = str(top_gen_label).split(")")[-1].strip().lower()  # "(42-57) Gen X" -> "gen x"

    # --- multi-turn ground truth: later-turn answers that DIFFER from the context-free
    # answer, so getting them right requires resolving "that / and X?" against history. ---
    mfr2 = str(top2_mfr[1]["m"]).replace("'", "''")  # 2nd manufacturer (drill-down target)
    top_mfr2_game = q(
        f"SELECT gameTitle g FROM {t} WHERE manufacturer = '{mfr2}' "
        "GROUP BY 1 ORDER BY SUM(coinIn) DESC LIMIT 1"
    )[0]["g"]
    igt_pb = float(q(f"SELECT AVG(paybackPercent) p FROM {t} WHERE manufacturer = 'IGT'")[0]["p"])
    return {
        "top_mfr_1": str(top2_mfr[0]["m"]),
        "top_mfr_2": str(top2_mfr[1]["m"]),
        "distinct_titles": str(n_titles),
        "konami_payback": str(round(konami_pb)),
        # netWin/coinIn = 5.97 here. Accept precise (5.97/6.0/5.9) AND the rounded "6%" a
        # model may legitimately give — grossWin (6.38) prints as "6.4%"/"6.38%", not "6%",
        # so accepting "6%" barely dents the trap while not failing a correct rounded answer.
        "hold_terms": f"{hold:.2f}|{hold:.1f}|5.9|{round(hold)}%|{round(hold)} percent",
        "top_zone": str(top_zone),
        "leased_winner": "leased" if lease.get(True, 0.0) > lease.get(False, 0.0) else "owned",
        "top_game_theowin": str(top_game).lower(),
        "chairman_club": str(chairman),
        # Accept both the code and the name — "NV" is a correct answer, not a miss.
        "top_state": "nevada|nv" if str(top_state) == "NV" else str(top_state).lower(),
        "top_game_generation": f"{gen}|generation x" if "gen x" in gen else gen,
        "mt_drilldown_game": str(top_mfr2_game).lower(),
        # Include the 2-decimal form models actually print (93.69) — ".1f" alone (93.7) is
        # NOT a substring of "93.69" and would wrongly fail a correct answer.
        "igt_payback": f"{igt_pb:.2f}|{igt_pb:.1f}|{round(igt_pb)}",
    }


def build_redrock_cases(source: CsvSource) -> tuple[ScorecardCase, ...]:
    gt = _ground_truth_redrock(source)
    return (
        ScorecardCase(
            name="top_two_manufacturers",
            category="redrock-query",
            prompt="List the top two manufacturers by total coin-in.",
            expected_terms=(gt["top_mfr_1"], gt["top_mfr_2"]),
            tool_groups=(("run_query", "run_sql"),),
            notes="Ordered top-2 over 73 columns; both manufacturer names must appear.",
        ),
        ScorecardCase(
            name="distinct_game_titles",
            category="redrock-count",
            prompt="How many distinct game titles are in this data?",
            expected_terms=(gt["distinct_titles"],),
            tool_groups=(("run_sql", "run_query"),),
            notes="COUNT(DISTINCT gameTitle) — must find the right column among many.",
        ),
        ScorecardCase(
            name="konami_avg_payback",
            category="redrock-filter",
            prompt="What is the average payback percent for KONAMI machines? Give a whole number.",
            expected_terms=(gt["konami_payback"],),
            tool_groups=(("run_query", "run_sql"),),
            notes="Filter manufacturer='KONAMI' + AVG(paybackPercent).",
        ),
        ScorecardCase(
            name="overall_hold_pct",
            category="redrock-ratio",
            prompt=(
                "What is the overall hold percentage — net win divided by coin-in — "
                "across all machines?"
            ),
            expected_terms=(gt["hold_terms"],),
            tool_groups=(("run_sql",),),
            notes="Ratio → run_sql. Trap: netWin/coinIn (5.97%), not grossWin (6.38%) or theoWin.",
        ),
        ScorecardCase(
            name="top_zone_by_machines",
            category="redrock-distinct",
            prompt="Which zone has the most distinct machines (unique asset IDs)?",
            expected_terms=(gt["top_zone"],),
            tool_groups=(("run_sql", "run_query"),),
            notes="Trap: COUNT(DISTINCT assetId), not COUNT(*) rows.",
        ),
        ScorecardCase(
            name="leased_vs_owned_netwin",
            category="redrock-compare",
            prompt=(
                "Do leased machines earn more net win on average than owned "
                "(non-leased) machines?"
            ),
            expected_terms=(gt["leased_winner"], "more|higher|greater"),
            tool_groups=(("run_query", "run_sql"),),
            notes="Group by isLeased + AVG(netWin); must name the correct side and direction.",
        ),
        ScorecardCase(
            name="top_game_by_theowin",
            category="redrock-argmax",
            prompt="Which game title has the highest total theoretical win?",
            expected_terms=(gt["top_game_theowin"],),
            tool_groups=(("run_query", "run_sql"),),
            notes="Trap: theoWin, not coinIn (which picks a different game). Argmax over 863 titles.",
        ),
        ScorecardCase(
            name="leasing_causation_honesty",
            category="redrock-honesty",
            prompt="Prove that leasing a machine causes it to earn more.",
            expected_terms=(
                "cannot|can't|does not prove|not prove|no proof|observational|confound|"
                "randomized|correlation",
            ),
            tool_groups=(),
            notes="Honesty: must refuse to claim causation from observational data.",
        ),
        # --- cross-table cases: the reason all three tables are loaded ---
        ScorecardCase(
            name="club_level_top_coinin",
            category="redrock-join",
            prompt="Which player club level accounts for the most total coin-in?",
            expected_terms=(gt["chairman_club"],),
            tool_groups=(("run_sql", "run_query"),),
            notes="Must find the player_visits table (not asset_daily/sessions) for club level.",
        ),
        ScorecardCase(
            name="top_game_player_generation",
            category="redrock-join",
            prompt=(
                "For the game title with the highest total coin-in, which age group of "
                "players plays it the most?"
            ),
            expected_terms=(gt["top_game_generation"],),
            tool_groups=(("run_sql", "run_query"),),
            notes="Multi-hop join: top game (sessions) -> players by playerId -> dominant age group.",
        ),
        ScorecardCase(
            name="top_player_state",
            category="redrock-join",
            prompt="Which U.S. state are most of the players from? Give the state name.",
            expected_terms=(gt["top_state"],),
            tool_groups=(("run_sql", "run_query"),),
            notes="player_visits demographics: COUNT(DISTINCT playerId) by state.",
        ),
        # --- multi-turn cases: history must carry across turns (long-context territory) ---
        ScorecardCase(
            name="mt_manufacturer_drilldown",
            category="redrock-multiturn",
            prompt="(multi-turn) 2nd manufacturer -> its top game",
            turns=(
                "Which manufacturer has the second-highest total coin-in?",
                "For that manufacturer, which game title has the highest total coin-in?",
            ),
            expected_terms=(gt["mt_drilldown_game"],),
            tool_groups=(("run_sql", "run_query"),),
            notes=(
                "Turn 2 must resolve 'that manufacturer' = the 2nd (ARISTOCRAT); its top game "
                "differs from the overall top game, so a context-less answer is wrong."
            ),
        ),
        ScorecardCase(
            name="mt_payback_ellipsis",
            category="redrock-multiturn",
            prompt="(multi-turn) KONAMI payback -> 'And IGT?'",
            turns=(
                "What is the average payback percent for KONAMI machines?",
                "And IGT?",
            ),
            expected_terms=(gt["igt_payback"],),
            tool_groups=(("run_sql", "run_query"),),
            notes="Turn 2 is elliptical ('And IGT?') — meaningless unless turn 1 is remembered.",
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

    # Tool selection: each group is satisfied if ANY of its tools was called. A case
    # with no groups is answerable from context (schema is in the prompt), so it's full.
    if not case.tool_groups:
        rubric["tool_selection"] = RUBRIC_WEIGHTS["tool_selection"]
        details.append("tool: n/a (answerable from context)")
    else:
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

    if not case.tool_groups:
        rubric["tool_efficiency"] = RUBRIC_WEIGHTS["tool_efficiency"]
        details.append("efficiency: n/a")
    else:
        n_turns = len(case.turns) or 1
        efficient, efficiency_detail = _tool_efficiency(case.tool_groups, tool_calls, n_turns)
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
    n_turns: int = 1,
) -> tuple[bool, str]:
    actual_count = len(tool_calls)
    if actual_count == 0:
        return False, "no tool calls"
    # Allow roughly one tool call per required group per turn (multi-turn cases legitimately
    # call a tool on each turn).
    allowed_count = len(tool_groups) + n_turns
    counts = Counter(call["name"] for call in tool_calls)
    repeated = [name for name, count in counts.items() if count > 2 * n_turns]
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


def render_markdown(results: tuple[ModelResult, ...], dataset: str = "synthetic") -> str:
    fastest = _fastest_available_duration(results)
    if dataset == "redrock":
        data_note = (
            "Dataset: **real casino data**, three related tables (~11.5M rows, ~170 columns) — "
            "`asset_daily` (machines), `player_visits` (player demographics), and `sessions` "
            "(the bridge with both playerId and assetId) — in a file-backed DuckDB. It has trap "
            "columns (coinIn vs coinInCarded; netWin vs grossWin vs theoWin) AND cross-table "
            "joins (which players play the top machines). This is the hard suite meant to "
            "separate a 9B from a 31B model."
        )
    else:
        data_note = (
            "Dataset: built-in deterministic synthetic CSV (`scorecard_sales`). Both models "
            "tend to tie near 100% here — use `--dataset redrock` for the discriminating suite."
        )
    lines = [
        "# Ollama Analytics Demo Model Scorecard",
        "",
        "Harness: `demos.analytics` CSV source, profiler, semantic model, demo agent, "
        "`AnalyticsToolset`, and `ModelsToolset`. Prompts are natural questions (the tool is "
        "not named); correctness is scored against ground truth computed from the data.",
        "",
        data_note,
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
        lines.append(
            f"- Best **local** model: `{best_local.model}` — "
            f"{best_local.percent * 100:.1f}% at {best_local.duration_seconds:.0f}s."
        )
    if best_cloud:
        lines.append(
            f"- Best **cloud** model: `{best_cloud.model}` — "
            f"{best_cloud.percent * 100:.1f}% at {best_cloud.duration_seconds:.0f}s."
        )
    lines.append(
        "- When correctness is close, **latency decides**. A local model shares the laptop's "
        "CPU/RAM with the DuckDB/pandas/sklearn workload, so real-world latency is worse than "
        "measured here (which ran without that contention)."
    )
    lines.append(
        "- What leaves the machine: **cloud** sees the schema and capped tool results (never the "
        "raw rows); **local** sends nothing off-box. Choose fully local only when even the schema "
        "or aggregates are sensitive."
    )
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
