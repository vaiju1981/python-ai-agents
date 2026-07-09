"""Matched-control difference-in-differences backtest (generic, column-agnostic).

Lifts the ATLAS ``backtest/harness.py`` capability into a generic tool
parameterized by ``valueCol`` / ``entityCol`` / ``dateCol`` instead of gaming
vocabulary. Given a "treatment" table of (entity, date) events, it estimates the
causal impact on a measured value using:

* caliper-matched **never-treated** controls,
* an **A/A synthetic null** to size the noise floor (no-effect bias),
* a **parallel-trends gate** that detrends systematic pre-existing drift.

Everything is read-only against the ``DataSource``; the heavy math runs in
pandas on a fetched frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel

# --- Tunables (mirror ATLAS config.py defaults; override via env) ---
K_MATCH = int(os.getenv("ANALYTICS_DID_KMATCH", "40"))
MIN_CONTROLS = int(os.getenv("ANALYTICS_DID_MIN_CONTROLS", "8"))
CALIPER_FRAC = float(os.getenv("ANALYTICS_DID_CALIPER_FRAC", "1.0"))
CALIPER_ABS = float(os.getenv("ANALYTICS_DID_CALIPER_ABS", "75"))
MIN_WINDOW_DAYS = int(os.getenv("ANALYTICS_DID_MIN_WINDOW", "3"))
PRETREND_TOL = float(os.getenv("ANALYTICS_DID_PRETREND_TOL", "10"))
PRE_WINDOW_DAYS = int(os.getenv("ANALYTICS_DID_PRE_WINDOW", "14"))
N_SYNTH_NULL = int(os.getenv("ANALYTICS_DID_NULL_N", "300"))
BACKTEST_MAX_ROWS = int(os.getenv("ANALYTICS_BACKTEST_MAX_ROWS", "0")) or None


@dataclass
class BacktestResult:
    did_effect: float
    did_ci: tuple[float, float]
    p_value: float
    n_treated: int
    n_controls: int
    treated_pre: float
    treated_post: float
    control_pre: float
    control_post: float
    null_median: float
    noise_floor: float
    parallel_trends_bias: float
    detrended: bool
    verdict: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "didEffect": round(self.did_effect, 4),
            "ci": [round(x, 4) for x in self.did_ci],
            "pValue": round(self.p_value, 4),
            "nTreated": self.n_treated,
            "nControls": self.n_controls,
            "treatedPre": round(self.treated_pre, 4),
            "treatedPost": round(self.treated_post, 4),
            "controlPre": round(self.control_pre, 4),
            "controlPost": round(self.control_post, 4),
            "nullMedian": round(self.null_median, 4),
            "noiseFloor": round(self.noise_floor, 4),
            "parallelTrendsBias": round(self.parallel_trends_bias, 4),
            "detrended": self.detrended,
            "verdict": self.verdict,
            "notes": self.notes,
        }


def _fetch_frame(
    source: DataSource,
    table: str,
    entity_col: str,
    date_col: str,
    value_col: str,
    exposure_col: str | None,
) -> "np.ndarray | Any":
    cols = [entity_col, date_col, value_col]
    if exposure_col:
        cols.append(exposure_col)
    select = ", ".join(sql_qcol(table, c) for c in cols)
    sql = f"SELECT {select} FROM {sql_quote(table)}"
    if BACKTEST_MAX_ROWS:
        sql += f" USING SAMPLE {int(BACKTEST_MAX_ROWS)} ROWS"
    rows = source.native_query(sql)
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if exposure_col:
        df[exposure_col] = pd.to_numeric(df[exposure_col], errors="coerce").fillna(0)
    df = df.dropna(subset=[entity_col, date_col])
    return df


def _rate(df: "Any", entity_col: str, value_col: str, exposure_col: str | None,
          lo: "Any", hi: "Any") -> dict[Any, float]:
    """Per-entity value-per-exposure rate within [lo, hi] (closed-open)."""
    sub = df[(df["_d"] >= lo) & (df["_d"] < hi)]
    if sub.empty:
        return {}
    if exposure_col:
        g = sub.groupby(entity_col).apply(
            lambda x: x[value_col].sum() / max(x[exposure_col].sum(), 1e-9)
        )
    else:
        g = sub.groupby(entity_col)[value_col].mean()
    return g.to_dict()


def _prepost(df: "Any", entity_col: str, value_col: str, exposure_col: str | None,
             c: "Any", pre_days: int, post_days: int) -> dict[str, dict[Any, float]]:
    lo_pre = c - np.timedelta64(pre_days, "D")
    hi_pre = c
    lo_post = c
    hi_post = c + np.timedelta64(post_days, "D")
    pre = _rate(df, entity_col, value_col, exposure_col, lo_pre, hi_pre)
    post = _rate(df, entity_col, value_col, exposure_col, lo_post, hi_post)
    return {"pre": pre, "post": post}


def _matched_controls(
    pool_rates: dict[Any, float], treated_pre: float
) -> tuple[list[Any], float, int]:
    """Caliper-match never-treated entities to the treated pre-rate."""
    caliper = max(CALIPER_ABS, CALIPER_FRAC * abs(treated_pre))
    dists = sorted(
        ((abs(r - treated_pre), e) for e, r in pool_rates.items()), key=lambda x: x[0]
    )
    # Drop any outside the caliper, then keep the K_MATCH nearest.
    within = [e for d, e in dists if d <= caliper]
    matched = within[:K_MATCH]
    if len(matched) < MIN_CONTROLS:
        return [], float("nan"), 0
    median_pre = float(np.median([pool_rates[e] for e in matched]))
    return matched, median_pre, len(matched)


def _did(pre: dict[Any, float], post: dict[Any, float]) -> float:
    """Difference-in-differences of per-entity pre/post changes (mean of diffs)."""
    entities = set(pre) & set(post)
    if not entities:
        return float("nan")
    treated_diff = np.array([post[e] - pre[e] for e in entities])
    return float(np.mean(treated_diff))


def _did_with_se(
    treated_pre: dict[Any, float], treated_post: dict[Any, float],
    control_pre: dict[Any, float], control_post: dict[Any, float],
):
    """DiD with a t-test based CI/p-value across per-entity changes."""
    t_entities = set(treated_pre) & set(treated_post)
    c_entities = set(control_pre) & set(control_post)
    if not t_entities or not c_entities:
        return float("nan"), (float("nan"), float("nan")), 1.0
    t_diff = np.array([treated_post[e] - treated_pre[e] for e in t_entities])
    c_diff = np.array([control_post[e] - control_pre[e] for e in c_entities])
    did = float(t_diff.mean() - c_diff.mean())
    # Welch two-sample statistic on the per-entity diffs.
    se = float(np.sqrt(t_diff.var(ddof=1) / len(t_diff) + c_diff.var(ddof=1) / len(c_diff)))
    if se <= 0:
        return did, (did, did), 0.0
    from math import erf, sqrt

    def _norm_cdf(z: float) -> float:
        return 0.5 * (1 + erf(z / sqrt(2)))

    z = did / se
    p = 2 * (1 - _norm_cdf(abs(z)))
    return did, (did - 1.96 * se, did + 1.96 * se), float(p)


def matched_impact(
    source: DataSource,
    model: SemanticModel,
    value_col: str,
    entity_col: str,
    date_col: str,
    treatment_table: str,
    treatment_key: str,
    treatment_date_col: str,
    treatment_filter: str | None = None,
    exposure_col: str | None = None,
    pre_days: int = PRE_WINDOW_DAYS,
    post_days: int = 14,
) -> BacktestResult:
    """Estimate causal impact of treatments on ``value_col`` via matched-control DiD."""
    notes: list[str] = []
    vcol = value_col.split(".")[-1]
    ecol = entity_col.split(".")[-1]
    dcol = date_col.split(".")[-1]
    xcol = exposure_col.split(".")[-1] if exposure_col else None
    df = _fetch_frame(source, _table_of(model, value_col), ecol, dcol, vcol, xcol)
    if df is None or getattr(df, "empty", True):
        return _fail("no data available for the value table")
    df = df.rename(columns={date_col: "_d"})

    # Treated units + their (first) treatment date.
    ev_sql = (
        f"SELECT {sql_qcol(treatment_table, treatment_key)} AS k, "
        f"{sql_qcol(treatment_table, treatment_date_col)} AS d FROM {sql_quote(treatment_table)}"
    )
    if treatment_filter:
        ev_sql += f" WHERE {treatment_filter}"
    ev_rows = source.native_query(ev_sql)
    import pandas as pd

    ev = pd.DataFrame(ev_rows)
    if ev.empty:
        return _fail("no treatment events found")
    ev["d"] = pd.to_datetime(ev["d"], errors="coerce")
    ev = ev.dropna(subset=["k", "d"])
    treated: dict[Any, Any] = dict(zip(ev["k"], ev["d"]))

    all_entities = set(df[entity_col].unique())
    control_pool = all_entities - set(treated)

    # Pre/post rates for treated units.
    t_pre: dict[Any, float] = {}
    t_post: dict[Any, float] = {}
    for e, c in treated.items():
        pp = _prepost(df, ecol, vcol, xcol, c, pre_days, post_days)
        if pp["pre"] and pp["post"]:
            t_pre[e] = np.mean(list(pp["pre"].values()))
            t_post[e] = np.mean(list(pp["post"].values()))

    if not t_pre:
        return _fail("no treated unit has data in both pre and post windows")

    treated_pre_rate = float(np.mean(list(t_pre.values())))

    # Build the never-treated pool's pre rates (relative to each treated date,
    # using a representative window before "now"). Use the latest treated date
    # as the reference so pool rates are comparable.
    ref_c = max(treated.values())
    pool_pp = _prepost(df, ecol, vcol, xcol, ref_c, pre_days, post_days)
    pool_pre = {e: r for e, r in pool_pp["pre"].items() if e in control_pool}
    if not pool_pre:
        return _fail("no never-treated control units available")

    matched, _median_pool_pre, n_controls = _matched_controls(pool_pre, treated_pre_rate)
    if n_controls == 0:
        return _fail("too few caliper-matched controls (increase CALIPER or pool size)")
    notes.append(f"caliper-matched {n_controls} never-treated controls")

    # Control pre/post: reuse matched units' pre (from pool) and their post at ref_c.
    c_pre = {e: pool_pre[e] for e in matched}
    c_post = {e: pool_pp["post"].get(e, pool_pre[e]) for e in matched}

    did, ci, p = _did_with_se(t_pre, t_post, c_pre, c_post)

    # Parallel-trends gate: shift the treatment into its own pre-period.
    pt_bias = _parallel_trends_bias(df, ecol, vcol, xcol, treated, pre_days, post_days)
    detrended = abs(pt_bias) > PRETREND_TOL
    if detrended:
        did = did - pt_bias
        notes.append(f"detrended by parallel-trends bias {pt_bias:.3f}")

    # A/A synthetic null.
    null_median, noise_floor = _synthetic_null(df, ecol, vcol, xcol,
                                               pool_pre, pre_days, post_days)

    verdict = _verdict(n_controls, noise_floor, null_median, detrended, p)
    notes.append(f"verdict={verdict}")

    return BacktestResult(
        did_effect=did, did_ci=ci, p_value=p,
        n_treated=len(t_pre), n_controls=n_controls,
        treated_pre=treated_pre_rate, treated_post=float(np.mean(list(t_post.values()))),
        control_pre=float(np.mean(list(c_pre.values()))),
        control_post=float(np.mean(list(c_post.values()))),
        null_median=null_median, noise_floor=noise_floor,
        parallel_trends_bias=pt_bias, detrended=detrended, verdict=verdict, notes=notes,
    )


def _parallel_trends_bias(df, entity_col, value_col, exposure_col, treated,
                          pre_days, post_days) -> float:
    """DiD computed at c - pre_days (nothing should happen) → pre-existing drift."""
    shifted: dict[Any, Any] = {e: (c - np.timedelta64(pre_days, "D")) for e, c in treated.items()}
    shifted = {e: c for e, c in shifted.items()
               if c >= df["_d"].min() + np.timedelta64(pre_days, "D")}
    if not shifted:
        return 0.0
    pre: dict[Any, float] = {}
    post: dict[Any, float] = {}
    for e, c in shifted.items():
        pp = _prepost(df, entity_col, value_col, exposure_col, c, pre_days, post_days)
        if pp["pre"] and pp["post"]:
            pre[e] = np.mean(list(pp["pre"].values()))
            post[e] = np.mean(list(pp["post"].values()))
    if not pre:
        return 0.0
    pool_pp = _prepost(df, entity_col, value_col, exposure_col,
                       max(shifted.values()), pre_days, post_days)
    ref_pre = float(np.mean(list(pre.values())))
    caliper = max(CALIPER_ABS, CALIPER_FRAC * abs(ref_pre))
    pool_pre = {e: r for e, r in pool_pp["pre"].items()
                if e not in shifted and abs(r - ref_pre) <= caliper}
    if not pool_pre:
        return 0.0
    matched = list(pool_pre)[:K_MATCH]
    c_pre = {e: pool_pre[e] for e in matched}
    c_post = {e: pool_pp["post"].get(e, pool_pre[e]) for e in matched}
    return _did(pre, post) - _did(c_pre, c_post)


def _synthetic_null(df, entity_col, value_col, exposure_col, pool_pre,
                    pre_days, post_days):
    """A/A: permute random (entity, date) pairs through the same DiD."""
    rng = np.random.default_rng(42)
    entities = list(pool_pre.keys())
    dates = df["_d"].dropna().sort_values()
    if len(entities) < 2 or dates.empty:
        return 0.0, 0.0
    lifts = []
    lo = dates.min() + np.timedelta64(pre_days, "D")
    hi = dates.max() - np.timedelta64(post_days, "D")
    if lo >= hi:
        return 0.0, 0.0
    span = (hi - lo).days
    for _ in range(N_SYNTH_NULL):
        e = entities[rng.integers(0, len(entities))]
        c = lo + np.timedelta64(int(rng.integers(0, max(1, span))), "D")
        pp = _prepost(df, entity_col, value_col, exposure_col, c, pre_days, post_days)
        if not (pp["pre"] and pp["post"]):
            continue
        pre_rate = np.mean(list(pp["pre"].values()))
        caliper = max(CALIPER_ABS, CALIPER_FRAC * abs(pre_rate))
        pool = {x: r for x, r in pool_pre.items()
                if x != e and abs(r - pre_rate) <= caliper}
        if len(pool) < MIN_CONTROLS:
            continue
        matched = list(pool)[:K_MATCH]
        c_pre = {x: pool[x] for x in matched}
        c_post = {x: pp["post"].get(x, pool[x]) for x in matched}
        lifts.append(_did(pp["pre"], pp["post"]) - _did(c_pre, c_post))
    if not lifts:
        return 0.0, 0.0
    lifts = np.array(lifts)
    return float(np.median(lifts)), float(np.subtract(*np.percentile(lifts, [75, 25])) / 2)


def _verdict(n_controls: int, noise_floor: float, null_median: float,
             detrended: bool, p: float) -> str:
    if n_controls < MIN_CONTROLS:
        return "INSUFFICIENT"
    if noise_floor <= 0:
        return "INSUFFICIENT"
    if abs(null_median) > PRETREND_TOL:
        return "NOISY"
    if detrended and p < 0.05:
        return "TRUSTED"
    if p < 0.05:
        return "DIRECTIONAL"
    return "INSUFFICIENT"


def _table_of(model: SemanticModel, value_col: str) -> str:
    for m in model.metrics:
        if m.ref == value_col or m.column == value_col:
            return m.table
    return value_col.split(".", 1)[0] if "." in value_col else value_col


def _fail(msg: str) -> BacktestResult:
    return BacktestResult(
        did_effect=float("nan"), did_ci=(float("nan"), float("nan")), p_value=float("nan"),
        n_treated=0, n_controls=0, treated_pre=float("nan"), treated_post=float("nan"),
        control_pre=float("nan"), control_post=float("nan"), null_median=float("nan"),
        noise_floor=float("nan"), parallel_trends_bias=float("nan"), detrended=False,
        verdict="INSUFFICIENT", notes=[msg],
    )
