"""Predictive & causal analytics tools: modeling, forecasting, A/B tests, causal
effect, uplift, clustering, anomalies.

These are governed read-only tools (they read data and compute; they never write).
Correctness and honesty are the point: every result reports its method and sample
size, statistical tests report effect size + p-value, and causal/uplift tools
carry an explicit "this is not proof of causation" caveat. Heavy libraries
(scikit-learn, scipy, statsmodels) are imported lazily so the module stays cheap
to import.

SQL identifiers come from the semantic model / table schema, never from free-form
user text, so the interpolation here is safe.
"""

from __future__ import annotations

import json
import time
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_literal, sql_qcol, sql_quote
from demos.analytics.src.analytics.model_store import ModelRecord, ModelStore, model_key
from demos.analytics.src.analytics.query_planner import OPERATORS
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import (
    MAX_RESULT_CHARS,
    _filter_schema,
    _frame,
    _make_tool,
    _object_schema,
    _string_array,
)
from python_ai_agents.core.tool import Tool, ToolResult

_MIN_ROWS = 20
# Drift = standardized mean shift of a feature vs its training distribution.
# ponytail: mean-shift heuristic; upgrade to PSI/KS tests if this proves too coarse.
_DRIFT_THRESHOLD = 0.5


class ModelsToolset:
    """Predictive/causal tools over a ``DataSource`` + ``SemanticModel``."""

    def __init__(
        self,
        source: DataSource,
        model: SemanticModel,
        *,
        store: ModelStore | None = None,
        dataset_sig: str = "",
        model_ttl: float | None = None,
        max_train_rows: int | None = None,
    ) -> None:
        self.source = source
        self.model = model
        self.store = store
        self.dataset_sig = dataset_sig
        self.model_ttl = model_ttl
        # None = train on the full table (the default). A positive value caps rows
        # (reservoir sample) so the user can trade accuracy for speed on big data.
        self.max_train_rows = max_train_rows

    # -- tool registry --------------------------------------------------------

    def all_tools(self) -> list[Tool]:
        return [
            self.build_model(),
            self.predict(),
            self.forecast(),
            self.ab_test(),
            self.causal_effect(),
            self.uplift(),
            self.cluster(),
            self.anomaly_detection(),
        ]

    # -- build_model / predict (train once, serve many) ------------------------

    def _prepare_model_spec(
        self, args: dict[str, Any]
    ) -> tuple[str, str, list[str], bool, str, Any]:
        """Resolve target/predictors, load training data, and compute the model key."""
        import pandas as pd

        t_table, t_col = self._resolve(args["target"])
        predictors = args.get("predictors") or self._numeric_columns(t_table, exclude=t_col)
        pcols = [c for p in predictors for (pt, c) in [self._resolve(p)] if pt == t_table]
        if not pcols:
            raise ValueError("no predictors in the target's table")

        df = self._frame(t_table, [t_col, *dict.fromkeys(pcols)])
        x = df[pcols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        data = pd.concat([df[t_col], x], axis=1).dropna()
        if len(data) < _MIN_ROWS:
            raise ValueError(f"need {_MIN_ROWS}+ rows to model (got {len(data)})")

        task = args.get("task", "auto")
        is_clf = task == "classification" or (
            task == "auto" and self._looks_categorical(data[t_col])
        )
        feature_cols = list(x.columns)
        algo = "random_forest_classifier" if is_clf else "random_forest_regressor"
        key = model_key(
            dataset_sig=self.dataset_sig,
            task="classification" if is_clf else "regression",
            target=t_col,
            predictors=feature_cols,
            algorithm=algo,
        )
        return t_table, t_col, feature_cols, is_clf, key, data

    def _train_or_load(
        self, args: dict[str, Any]
    ) -> tuple[str, str, list[str], Any, dict[str, Any], bool, float]:
        """Train-once cache: reuse a fresh stored model unless the caller forces a retrain."""
        t_table, t_col, feature_cols, is_clf, key, data = self._prepare_model_spec(args)
        if self.store is not None and not args.get("retrain", False):
            cached = self.store.get(key, max_age=self.model_ttl)
            if cached is not None:
                return (
                    t_table,
                    t_col,
                    feature_cols,
                    cached.model,
                    dict(cached.metadata),
                    True,
                    cached.trained_at,
                )
        result, fitted = self._fit(data[t_col], data[feature_cols], feature_cols, is_clf)
        trained_at = time.time()
        if self.store is not None:
            self.store.put(
                ModelRecord(key=key, model=fitted, metadata=result, trained_at=trained_at)
            )
        return t_table, t_col, feature_cols, fitted, result, False, trained_at

    def build_model(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            _, _, _, _, meta, cached, trained_at = self._train_or_load(args)
            return _ok("build_model", {**meta, "cached": cached, "trained_at": trained_at})

        return _make_tool(
            "build_model",
            "Train a model to predict a target and rank feature importance. "
            "Auto-detects classification vs regression. Args: target, predictors?, task?.",
            _guarded("build_model", impl),
            _object_schema(
                {
                    "target": {"type": "string", "description": "Column to predict."},
                    "predictors": _string_array("Numeric predictors (default: measures)."),
                    "task": {
                        "type": "string",
                        "enum": ["auto", "classification", "regression"],
                        "default": "auto",
                    },
                    "retrain": {"type": "boolean", "default": False},
                },
                required=("target",),
            ),
        )

    def predict(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import pandas as pd

            # Serving: load the stored model for this spec (training once if absent),
            # then score rows WITHOUT retraining — the train/serve split.
            t_table, t_col, feature_cols, fitted, meta, cached, trained_at = self._train_or_load(
                args
            )
            filters = list(args.get("filters") or [])
            score_df = self._frame(t_table, feature_cols, filters=filters)
            score_df = score_df.apply(pd.to_numeric, errors="coerce").dropna()
            if score_df.empty:
                return ToolResult.failed("no rows to score after filters")

            preds = fitted.predict(score_df[feature_cols].astype(float).values)
            if meta.get("task") == "classification":
                classes = list(meta.get("classes") or [])
                labels = [classes[int(p)] if 0 <= int(p) < len(classes) else str(p) for p in preds]
                counts: dict[str, int] = {}
                for label in labels:
                    counts[label] = counts.get(label, 0) + 1
                summary: dict[str, Any] = {"class_distribution": counts}
            else:
                summary = {
                    "mean": round(float(preds.mean()), 4),
                    "min": round(float(preds.min()), 4),
                    "max": round(float(preds.max()), 4),
                }

            return _ok(
                "predict",
                {
                    "target": t_col,
                    "task": meta.get("task"),
                    "n_scored": int(len(score_df)),
                    "prediction": summary,
                    "model_cached": cached,
                    "trained_at": trained_at,
                    "drift": _drift_check(meta.get("train_stats") or {}, score_df, feature_cols),
                    "method": "serves the stored model; build_model(retrain=true) refreshes it",
                },
            )

        return _make_tool(
            "predict",
            "Score rows with the stored trained model (trains once if missing) — no retraining "
            "per call. Reports predictions and drift vs the training data. "
            "Args: target, predictors?, task?, filters?.",
            _guarded("predict", impl),
            _object_schema(
                {
                    "target": {"type": "string", "description": "Column the model predicts."},
                    "predictors": _string_array("Numeric predictors (default: measures)."),
                    "task": {
                        "type": "string",
                        "enum": ["auto", "classification", "regression"],
                        "default": "auto",
                    },
                    "filters": _filter_schema(),
                },
                required=("target",),
            ),
        )

    def _fit(self, y: Any, x: Any, pcols: list[str], is_clf: bool) -> tuple[dict[str, Any], Any]:
        xv = x.astype(float).values
        # Per-feature training stats travel with the model so `predict` can flag drift.
        train_stats = {
            c: {"mean": round(float(x[c].mean()), 4), "std": round(float(x[c].std(ddof=0)), 4)}
            for c in pcols
        }
        if is_clf:
            from sklearn.ensemble import RandomForestClassifier

            codes, uniques = _encode(y)
            n_classes = len(uniques)
            model = RandomForestClassifier(n_estimators=200, random_state=0)
            model.fit(xv, codes)
            score = _safe_cv(model, xv, codes, "accuracy", is_clf=True)
            return {
                "task": "classification",
                "n_rows": int(len(y)),
                "n_classes": int(n_classes),
                "classes": [str(u) for u in uniques],
                "cv_accuracy": score,
                "feature_importance": _importances(pcols, model.feature_importances_),
                "train_stats": train_stats,
                "method": "RandomForestClassifier, 5-fold CV accuracy on historical data",
            }, model
        from sklearn.ensemble import RandomForestRegressor

        yv = y.astype(float).values
        model = RandomForestRegressor(n_estimators=200, random_state=0)
        model.fit(xv, yv)
        score = _safe_cv(model, xv, yv, "r2", is_clf=False)
        return {
            "task": "regression",
            "n_rows": int(len(y)),
            "cv_r2": score,
            "feature_importance": _importances(pcols, model.feature_importances_),
            "train_stats": train_stats,
            "method": "RandomForestRegressor, 5-fold CV R^2 on historical data",
        }, model

    # -- forecast -------------------------------------------------------------

    def forecast(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import numpy as np

            metric = args["metric"]
            horizon = max(1, int(args.get("horizon", 6)))
            tc = self._time_column(args["timeColumn"])
            if tc is None:
                return ToolResult.failed(f"time column '{args['timeColumn']}' not found")
            m_table, m_col = self._resolve(metric)
            ts = tc.to_timestamp_sql(sql_qcol(tc.table, tc.column))
            sql = (
                f"SELECT date_trunc('month', {ts})::date AS period, "
                f"SUM({sql_qcol(m_table, m_col)}) AS value "
                f"FROM {sql_quote(m_table)} WHERE {ts} IS NOT NULL GROUP BY period ORDER BY period"
            )
            rows = self.source.native_query(sql)
            values = [float(r["value"]) for r in rows if r.get("value") is not None]
            if len(values) < 4:
                return ToolResult.failed(f"need >= 4 periods to forecast (got {len(values)})")

            fc, method = _forecast_values(np.array(values), horizon)
            resid_std = float(np.std(np.diff(values))) if len(values) > 1 else 0.0
            band = 1.96 * resid_std
            forecast = [
                {
                    "step": i + 1,
                    "value": round(float(v), 2),
                    "low": round(float(v - band), 2),
                    "high": round(float(v + band), 2),
                }
                for i, v in enumerate(fc)
            ]
            return _ok(
                "forecast",
                {
                    "metric": m_col,
                    "history_periods": len(values),
                    "horizon": horizon,
                    "method": method,
                    "forecast": forecast,
                    "note": "Monthly aggregation; interval ≈ ±1.96×std of month-over-month change.",
                },
                data=forecast,
            )

        return _make_tool(
            "forecast",
            "Forecast a metric forward over time. Args: metric, timeColumn, horizon?.",
            _guarded("forecast", impl),
            _object_schema(
                {
                    "metric": {"type": "string", "description": "Measure to forecast."},
                    "timeColumn": {"type": "string", "description": "Time column ref."},
                    "horizon": {"type": "integer", "minimum": 1, "default": 6},
                },
                required=("metric", "timeColumn"),
            ),
        )

    # -- ab_test --------------------------------------------------------------

    def ab_test(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import numpy as np
            from scipy import stats

            metric = args["metric"]
            m_table, m_col = self._resolve(metric)
            g_table, g_col = self._resolve(args["groupColumn"])
            if g_table != m_table:
                return ToolResult.failed("metric and groupColumn must be in the same table")
            a, b = str(args["groupA"]), str(args["groupB"])
            df = self._frame(m_table, [m_col, g_col])
            va = df[df[g_col].astype(str) == a][m_col].apply(_to_float).dropna().values
            vb = df[df[g_col].astype(str) == b][m_col].apply(_to_float).dropna().values
            if len(va) < 2 or len(vb) < 2:
                return ToolResult.failed(f"need >= 2 rows per group (A={len(va)}, B={len(vb)})")

            t, p = stats.ttest_ind(va, vb, equal_var=False)  # Welch's
            ma, mb = float(va.mean()), float(vb.mean())
            pooled = float(np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)) or 1.0
            return _ok(
                "ab_test",
                {
                    "metric": m_col,
                    "groupA": a,
                    "groupB": b,
                    "nA": int(len(va)),
                    "nB": int(len(vb)),
                    "meanA": round(ma, 4),
                    "meanB": round(mb, 4),
                    "difference": round(ma - mb, 4),
                    "welch_t": round(float(t), 3),
                    "p_value": round(float(p), 4),
                    "cohens_d": round((ma - mb) / pooled, 3),
                    "verdict": "significant at p<0.05" if p < 0.05 else "not significant (p>=0.05)",
                },
            )

        return _make_tool(
            "ab_test",
            "Compare a metric between two groups with a Welch's t-test (effect size + p-value). "
            "Args: metric, groupColumn, groupA, groupB.",
            _guarded("ab_test", impl),
            _object_schema(
                {
                    "metric": {"type": "string"},
                    "groupColumn": {"type": "string"},
                    "groupA": {"type": "string"},
                    "groupB": {"type": "string"},
                },
                required=("metric", "groupColumn", "groupA", "groupB"),
            ),
        )

    # -- causal_effect --------------------------------------------------------

    def causal_effect(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import pandas as pd
            import statsmodels.api as sm

            t_table, t_col = self._resolve(args["target"])
            tr_table, tr_col = self._resolve(args["treatment"])
            if tr_table != t_table:
                return ToolResult.failed("target and treatment must be in the same table")
            controls = [
                c
                for x in (args.get("controls") or [])
                for (ct, c) in [self._resolve(x)]
                if ct == t_table
            ]
            cols = list(dict.fromkeys([t_col, tr_col, *controls]))
            df = self._frame(t_table, cols).apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) < _MIN_ROWS:
                return ToolResult.failed(f"need {_MIN_ROWS}+ numeric rows (got {len(df)})")

            xd = sm.add_constant(df[[tr_col, *controls]])
            fit = sm.OLS(df[t_col], xd).fit()
            ci = fit.conf_int().loc[tr_col]
            return _ok(
                "causal_effect",
                {
                    "target": t_col,
                    "treatment": tr_col,
                    "controls": controls,
                    "n": int(len(df)),
                    "estimated_effect": round(float(fit.params[tr_col]), 4),
                    "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
                    "p_value": round(float(fit.pvalues[tr_col]), 4),
                    "r_squared": round(float(fit.rsquared), 3),
                    "caveat": (
                        "Observational OLS adjusting only for the named controls. "
                        "This is NOT proof of "
                        "causation — unmeasured confounders can bias it. For a causal claim, run a "
                        "randomized experiment (ab_test)."
                    ),
                },
            )

        return _make_tool(
            "causal_effect",
            "Estimate a treatment's effect on a target, adjusting for confounders (OLS). "
            "Args: target, treatment, controls?. Reports effect + CI with a causation caveat.",
            _guarded("causal_effect", impl),
            _object_schema(
                {
                    "target": {"type": "string"},
                    "treatment": {"type": "string"},
                    "controls": _string_array("Confounder columns to adjust for."),
                },
                required=("target", "treatment"),
            ),
        )

    # -- uplift ---------------------------------------------------------------

    def uplift(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor

            t_table, t_col = self._resolve(args["target"])
            tr_table, tr_col = self._resolve(args["treatment"])
            if tr_table != t_table:
                return ToolResult.failed("target and treatment must be in the same table")
            preds = args.get("predictors") or self._numeric_columns(t_table, exclude=t_col)
            pcols = [
                c for p in preds for (pt, c) in [self._resolve(p)] if pt == t_table and c != tr_col
            ]
            if not pcols:
                return ToolResult.failed("no predictor columns for uplift")

            df = self._frame(t_table, list(dict.fromkeys([t_col, tr_col, *pcols])))
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            treated = df[df[tr_col] > df[tr_col].median()]
            control = df[df[tr_col] <= df[tr_col].median()]
            if len(treated) < 10 or len(control) < 10:
                return ToolResult.failed("need >= 10 rows in each of treated/control")

            def _rf(g):
                rf = RandomForestRegressor(n_estimators=200, random_state=0)
                return rf.fit(g[pcols], g[t_col])

            mt, mc = _rf(treated), _rf(control)
            up = mt.predict(df[pcols]) - mc.predict(df[pcols])
            order = np.argsort(up)[::-1]
            top_decile = up[order[: max(1, len(up) // 10)]]
            corr = np.corrcoef(np.c_[df[pcols].values, up].T)[-1, :-1]
            drivers = _importances(pcols, np.abs(corr))
            return _ok(
                "uplift",
                {
                    "target": t_col,
                    "treatment": tr_col,
                    "n": int(len(df)),
                    "mean_uplift": round(float(up.mean()), 4),
                    "top_decile_mean_uplift": round(float(top_decile.mean()), 4),
                    "drivers_of_uplift": drivers,
                    "caveat": (
                        "T-learner on observational data; directional, not proof of causation."
                    ),
                },
            )

        return _make_tool(
            "uplift",
            "Estimate who benefits most from a treatment. Args: target, treatment, predictors?.",
            _guarded("uplift", impl),
            _object_schema(
                {
                    "target": {"type": "string"},
                    "treatment": {"type": "string"},
                    "predictors": _string_array("Predictor columns (default: measures)."),
                },
                required=("target", "treatment"),
            ),
        )

    # -- cluster --------------------------------------------------------------

    def cluster(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import pandas as pd
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler

            table = None
            cols: list[str] = []
            for ref in args["columns"]:
                t, c = self._resolve(ref)
                table = table or t
                if t == table:
                    cols.append(c)
            if not cols:
                return ToolResult.failed("no valid columns")
            df = self._frame(table, cols).apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) < _MIN_ROWS:
                return ToolResult.failed(f"need >= {_MIN_ROWS} rows (got {len(df)})")
            k = max(2, min(int(args.get("k", 3)), len(df) - 1))
            xs = StandardScaler().fit_transform(df[cols].values)
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(xs)
            sizes = {int(c): int(n) for c, n in zip(*_unique_counts(labels), strict=False)}
            sil = float(silhouette_score(xs, labels)) if k < len(df) else 0.0
            return _ok(
                "cluster",
                {
                    "columns": cols,
                    "k": k,
                    "n": int(len(df)),
                    "silhouette": round(sil, 3),
                    "cluster_sizes": sizes,
                    "method": "k-means on standardized features",
                },
            )

        return _make_tool(
            "cluster",
            "Segment rows into k clusters over numeric columns (k-means). Args: columns, k?.",
            _guarded("cluster", impl),
            _object_schema(
                {
                    "columns": _string_array("Numeric columns to cluster on."),
                    "k": {"type": "integer", "minimum": 2, "default": 3},
                },
                required=("columns",),
            ),
        )

    # -- anomaly_detection ----------------------------------------------------

    def anomaly_detection(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            import pandas as pd
            from sklearn.ensemble import IsolationForest

            table = None
            cols: list[str] = []
            for ref in args["columns"]:
                t, c = self._resolve(ref)
                table = table or t
                if t == table:
                    cols.append(c)
            if not cols:
                return ToolResult.failed("no valid columns")
            contamination = float(args.get("contamination", 0.05))
            df = self._frame(table, cols).apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) < _MIN_ROWS:
                return ToolResult.failed(f"need >= {_MIN_ROWS} rows (got {len(df)})")
            iso = IsolationForest(contamination=contamination, random_state=0)
            flags = iso.fit_predict(df[cols].values)
            anomalies = df[flags == -1]
            sample = anomalies.head(10).to_dict("records")
            return _ok(
                "anomaly_detection",
                {
                    "columns": cols,
                    "n": int(len(df)),
                    "n_anomalies": int((flags == -1).sum()),
                    "contamination": contamination,
                    "examples": sample,
                    "method": "IsolationForest",
                },
            )

        return _make_tool(
            "anomaly_detection",
            "Flag anomalous rows via IsolationForest. Args: columns, contamination?.",
            _guarded("anomaly_detection", impl),
            _object_schema(
                {
                    "columns": _string_array("Numeric columns."),
                    "contamination": {"type": "number", "default": 0.05},
                },
                required=("columns",),
            ),
        )

    # -- shared helpers -------------------------------------------------------

    def _resolve(self, ref: str) -> tuple[str, str]:
        if "." in ref:
            table, col = ref.split(".", 1)
            return table, col
        for t in self.source.tables():
            for c in t.columns:
                if c.name == ref:
                    return t.name, ref
        raise ValueError(f"unknown column: {ref}")

    def _numeric_columns(self, table: str, *, exclude: str = "") -> list[str]:
        return [m.ref for m in self.model.metrics if m.table == table and m.column != exclude]

    def _time_column(self, ref: str) -> Any:
        for tc in self.model.time_columns:
            if tc.ref == ref or tc.column == ref:
                return tc
        return None

    def _frame(
        self,
        table: str,
        columns: list[str],
        max_rows: int | None = None,
        filters: list[dict[str, Any]] | None = None,
    ) -> Any:
        import pandas as pd

        cap = max_rows if max_rows is not None else self.max_train_rows
        select = ", ".join(sql_qcol(table, c) for c in columns)
        parts = [f"{sql_qcol(table, c)} IS NOT NULL" for c in columns]
        for f in filters or []:
            f_table, f_col = self._resolve(str(f["column"]))
            if f_table != table:
                raise ValueError(f"filter column '{f['column']}' is not in table '{table}'")
            op = str(f["op"])
            if op not in OPERATORS:
                raise ValueError(f"unsupported filter op: {op}")
            parts.append(f"{sql_qcol(table, f_col)} {op} '{sql_literal(str(f['value']))}'")
        sql = f"SELECT {select} FROM {sql_quote(table)} WHERE {' AND '.join(parts)}"
        if cap:
            # User opted to cap training rows; reservoir-sample so it fits in memory.
            sql = f"SELECT * FROM ({sql}) USING SAMPLE {int(cap)} ROWS"
        return pd.DataFrame(self.source.native_query(sql))

    @staticmethod
    def _looks_categorical(series: Any) -> bool:
        if series.dtype == object or str(series.dtype) == "category":
            return True
        return series.nunique() <= 10


def _guarded(name: str, fn: Any) -> Any:
    async def invoke(arguments: dict[str, Any], context: Any) -> ToolResult:
        try:
            return await fn(arguments, context)
        except Exception as exc:  # noqa: BLE001 — tools must fail as a result, never crash the agent
            return ToolResult.failed(f"{name} failed: {exc}")

    return invoke


def _ok(name: str, obj: dict[str, Any], data: Any = None) -> ToolResult:
    return ToolResult.ok(
        _frame(name, json.dumps(obj, default=str)[:MAX_RESULT_CHARS]), data=data
    )


def _drift_check(train_stats: dict[str, Any], df: Any, feature_cols: list[str]) -> dict[str, Any]:
    """Standardized mean shift of each scoring feature vs its training distribution."""
    shifts: dict[str, float] = {}
    for col in feature_cols:
        stats = train_stats.get(col)
        if not stats:
            continue
        std = float(stats.get("std") or 0.0) or 1.0
        shifts[col] = round(abs(float(df[col].mean()) - float(stats["mean"])) / std, 3)
    if not shifts:
        return {"checked": False, "note": "no training statistics stored with this model"}
    worst = max(shifts, key=lambda c: shifts[c])
    detected = shifts[worst] > _DRIFT_THRESHOLD
    result: dict[str, Any] = {
        "checked": True,
        "detected": detected,
        "score": shifts[worst],
        "threshold": _DRIFT_THRESHOLD,
        "worst_feature": worst,
        "feature_shift": shifts,
    }
    if detected:
        result["recommendation"] = (
            "scored rows' distribution shifted vs training data; "
            "retrain via build_model(retrain=true)"
        )
    return result


def _importances(names: list[str], values: Any) -> list[dict[str, Any]]:
    pairs = sorted(zip(names, (float(v) for v in values), strict=False), key=lambda x: -abs(x[1]))
    return [{"feature": n, "importance": round(v, 3)} for n, v in pairs]


def _encode(y: Any) -> tuple[Any, list[Any]]:
    cat = y.astype("category")
    return cat.cat.codes.values, list(cat.cat.categories)


def _safe_cv(model: Any, x: Any, y: Any, scoring: str, *, is_clf: bool) -> float | str:
    from sklearn.model_selection import cross_val_score

    try:
        folds = 5
        if is_clf:
            _, counts = _unique_counts(y)
            folds = max(2, min(5, int(min(counts))))
        return round(float(cross_val_score(model, x, y, cv=folds, scoring=scoring).mean()), 3)
    except Exception:
        return "n/a (too few samples for cross-validation)"


def _unique_counts(arr: Any) -> tuple[Any, Any]:
    import numpy as np

    return np.unique(arr, return_counts=True)


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _forecast_values(series: Any, horizon: int) -> tuple[list[float], str]:
    import numpy as np

    n = len(series)
    # Season-aware: additive trend + seasonality when at least two full cycles are
    # present (period 12 = yearly on monthly buckets, 4 = quarterly). Falls back to
    # trend-only Holt-Winters, then a linear trend, if a fit fails or data is short.
    for period in (12, 4):
        if n >= 2 * period:
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                fit = ExponentialSmoothing(
                    series, trend="add", seasonal="add", seasonal_periods=period
                ).fit()
                return (
                    list(fit.forecast(horizon)),
                    f"Holt-Winters (additive trend + seasonality, period={period})",
                )
            except Exception:
                pass
    if n >= 4:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            fit = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
            return list(fit.forecast(horizon)), "Holt-Winters (additive trend, no seasonality)"
        except Exception:
            pass
    # Fallback: linear trend
    idx = np.arange(n)
    slope, intercept = np.polyfit(idx, series, 1)
    future = np.arange(n, n + horizon)
    return list(slope * future + intercept), "linear trend (ordinary least squares)"
