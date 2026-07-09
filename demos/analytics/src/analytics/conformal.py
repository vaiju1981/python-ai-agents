"""Walk-forward forecasting with split-conformal (CQR) prediction intervals.

Generic, column-agnostic lift of ATLAS ``estimators/_core.py``. Trains quantile
regressors in day-grouped, leakage-safe walk-forward folds, then builds honest
low/high bands via split-conformal quantile regression (CQR). Bands are never
narrower than the measured per-series noise floor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel

INTERVAL_ALPHA = float(os.getenv("ANALYTICS_CQR_ALPHA", "0.20"))
N_BLOCKS = int(os.getenv("ANALYTICS_CQR_BLOCKS", "6"))
MIN_N = int(os.getenv("ANALYTICS_CQR_MIN_N", "14"))
SEED = 42


@dataclass
class ConformalForecast:
    mean: list[float]
    lo: list[float]
    hi: list[float]
    dates: list[str]
    interval_coverage_cal: float
    noise_floor: float
    n_train: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": [round(x, 4) for x in self.mean],
            "lo": [round(x, 4) for x in self.lo],
            "hi": [round(x, 4) for x in self.hi],
            "dates": self.dates,
            "intervalCoverageCal": round(self.interval_coverage_cal, 4),
            "noiseFloor": round(self.noise_floor, 4),
            "nTrain": self.n_train,
            "notes": self.notes,
        }


def _day_grouped_folds(dates: "np.ndarray", n_blocks: int = N_BLOCKS) -> list[tuple["np.ndarray", "np.ndarray"]]:
    """Leakage-safe folds: each test block trains only on earlier days."""
    uniq = np.unique(dates)
    if len(uniq) < n_blocks:
        n_blocks = max(2, len(uniq))
    edges = np.array_split(np.arange(len(uniq)), n_blocks)
    rowblock = np.empty(len(dates), dtype=int)
    for bi, idx in enumerate(edges):
        for j in idx:
            rowblock[dates == uniq[j]] = bi
    folds = []
    for k in range(1, n_blocks):
        tr = np.where(rowblock < k)[0]
        te = np.where(rowblock == k)[0]
        if len(tr) and len(te):
            folds.append((tr, te))
    return folds


def _build_features(series: "np.ndarray", lags: int = 7) -> "np.ndarray":
    """Lag + calendar-free features for quantile regression."""
    n = len(series)
    feat = np.zeros((n, lags))
    for l in range(1, lags + 1):
        feat[:, l - 1] = np.concatenate([[np.nan] * l, series[: n - l]])
    return feat


def conformal_forecast(
    source: DataSource,
    model: SemanticModel,
    value_col: str,
    date_col: str,
    entity_col: str | None = None,
    feature_cols: list[str] | None = None,
    horizon: int = 14,
    lags: int = 7,
) -> ConformalForecast:
    """Walk-forward conformal forecast of a daily value series."""
    notes: list[str] = []
    table = _table_of(model, value_col)
    vcol = value_col.split(".")[-1]
    dcol = date_col.split(".")[-1]
    ecol = entity_col.split(".")[-1] if entity_col else None
    qv = sql_qcol(table, vcol)
    qd = sql_qcol(table, dcol)
    agg = f"AVG({qv})" if not ecol else f"SUM({qv})"
    select_cols = [f"{qd}::date AS d", f"{agg} AS v"]
    if ecol:
        select_cols.append(f"{sql_qcol(table, ecol)} AS e")
    inner = f'SELECT {", ".join(select_cols)} FROM {sql_quote(table)}'
    inner += " GROUP BY "
    inner += "d, e" if ecol else "d"
    sql = (
        f"WITH s AS ({inner}) "
        f"SELECT d, AVG(v) AS v FROM s GROUP BY d ORDER BY d"
    )
    rows = source.native_query(sql)
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return _fail("no data for forecast")
    df["d"] = pd.to_datetime(df["d"])
    df = df.dropna().sort_values("d")
    y = df["v"].to_numpy(dtype=float)
    dates = df["d"].to_numpy()
    feat = _build_features(y, lags)
    valid = ~np.any(np.isnan(feat), axis=1)
    if valid.sum() < MIN_N:
        return _fail(f"need >= {MIN_N} daily points after lag window; got {int(valid.sum())}")
    if np.any(np.isnan(y[:lags])):
        notes.append("series begins with NaNs (dropped for training)")
    X = feat[valid]
    Y = y[valid]
    D = dates[valid]

    from sklearn.ensemble import GradientBoostingRegressor

    folds = _day_grouped_folds(D, N_BLOCKS)
    if not folds:
        return _fail("not enough time blocks for walk-forward")
    oof_q10 = np.full(len(Y), np.nan)
    oof_q50 = np.full(len(Y), np.nan)
    oof_q90 = np.full(len(Y), np.nan)
    for tr, te in folds:
        if len(tr) < lags or len(te) == 0:
            continue
        for alpha, target in ((0.1, oof_q10), (0.5, oof_q50), (0.9, oof_q90)):
            mdl = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, random_state=SEED, n_estimators=100, max_depth=3
            )
            mdl.fit(X[tr], Y[tr])
            target[te] = mdl.predict(X[te])

    scored = ~np.isnan(oof_q10)
    if scored.sum() < MIN_N:
        return _fail("too few out-of-fold scored rows for conformal bands")
    E = np.maximum(oof_q10[scored] - Y[scored], Y[scored] - oof_q90[scored])
    n = scored.sum()
    k = min(1.0, np.ceil((n + 1) * (1 - INTERVAL_ALPHA)) / n)
    conformal_q = float(np.quantile(E, k))

    # Noise floor: residual spread of the median model.
    noise_floor = float(np.std(Y[scored] - oof_q50[scored]) or 0.0)
    coverage = float(
        np.mean(
            (Y[scored] >= oof_q10[scored] - conformal_q)
            & (Y[scored] <= oof_q90[scored] + conformal_q)
        )
    )

    # Fit final model on all data and forecast the horizon forward.
    final = {}
    for alpha in (0.1, 0.5, 0.9):
        mdl = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, random_state=SEED, n_estimators=100, max_depth=3
        )
        mdl.fit(X, Y)
        final[alpha] = mdl
    fut_dates: list[str] = []
    fut_mean: list[float] = []
    fut_lo: list[float] = []
    fut_hi: list[float] = []
    buf = list(y)
    for h in range(1, horizon + 1):
        fx = np.array(buf[-lags:][::-1]).reshape(1, -1)
        p10 = float(final[0.1].predict(fx)[0])
        p50 = float(final[0.5].predict(fx)[0])
        p90 = float(final[0.9].predict(fx)[0])
        lo = p50 - max(p50 - p10 + conformal_q, noise_floor)
        hi = p50 + max(p90 - p50 + conformal_q, noise_floor)
        fut_mean.append(p50)
        fut_lo.append(lo)
        fut_hi.append(hi)
        buf.append(p50)
        fut_dates.append(str((df["d"].max() + np.timedelta64(h, "D")).date()))

    notes.append(f"{len(folds)} walk-forward folds; conformal q={conformal_q:.3f}")
    return ConformalForecast(
        mean=fut_mean, lo=fut_lo, hi=fut_hi, dates=fut_dates,
        interval_coverage_cal=coverage, noise_floor=noise_floor,
        n_train=int(n), notes=notes,
    )


def _table_of(model: SemanticModel, value_col: str) -> str:
    for m in model.metrics:
        if m.ref == value_col or m.column == value_col:
            return m.table
    return value_col.split(".", 1)[0] if "." in value_col else value_col


def _fail(msg: str) -> ConformalForecast:
    return ConformalForecast(
        mean=[], lo=[], hi=[], dates=[], interval_coverage_cal=float("nan"),
        noise_floor=float("nan"), n_train=0, notes=[msg],
    )
