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
import os
import time
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_literal, sql_qcol, sql_quote
from demos.analytics.src.analytics.metrics import inc
from demos.analytics.src.analytics.model_store import ModelRecord, ModelStore, model_key
from demos.analytics.src.analytics.query_planner import OPERATORS, feature_frame_sql
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import (
    MAX_RESULT_CHARS,
    _filter_schema,
    _frame,
    _make_tool,
    _object_schema,
    _string_array,
)
from demos.analytics.src.analytics.train_backend import TrainBackend, get_train_backend
from demos.analytics.src.analytics.warehouse_sources import score_warehouse
from python_ai_agents.core.tool import Tool, ToolResult

_MIN_ROWS = 20
# Row-based ML (random forests, k-means, isolation forest) must materialize rows
# into memory. To keep that bounded on large tables we default to a generous
# reservoir-sample cap; callers can override per-instance via ``max_train_rows``
# or globally via ``ANALYTICS_MAX_TRAIN_ROWS`` (0 = no cap / full table).
_DEFAULT_MAX_TRAIN_ROWS = int(os.getenv("ANALYTICS_MAX_TRAIN_ROWS", "200000")) or None
# Drift detection: PSI (population stability index) is the primary signal,
# with a two-sample KS test and a standardized mean-shift fallback. PSI
# convention: <0.1 stable, 0.1-0.25 moderate, >0.25 significant drift.
_PSI_THRESHOLD = 0.25
_MEAN_SHIFT_TO_PSI = 1.0  # keeps the fallback mean-shift score on a comparable scale
_MEAN_SHIFT_DRIFT = 2.0  # standardized mean-shift magnitude treated as drift


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
        lineage: Any = None,
        train_backend: Any = None,
    ) -> None:
        self.source = source
        self.model = model
        self.store = store
        # Default the dataset signature to a content fingerprint so a changed
        # dataset invalidates cached models instead of serving stale fits.
        if not dataset_sig:
            from demos.analytics.src.analytics.dataset_fingerprint import fingerprint

            # Row-count-agnostic signature: pure data growth (more rows of the
            # same distribution) must not needlessly invalidate trained models,
            # only schema/role changes do (PR-11).
            dataset_sig = fingerprint(source, row_count_aware=False)
        self.dataset_sig = dataset_sig
        self.model_ttl = model_ttl
        # None = train on the full table (the default). A positive value caps rows
        # (reservoir sample) so the user can trade accuracy for speed on big data.
        self.max_train_rows = (
            max_train_rows if max_train_rows is not None else _DEFAULT_MAX_TRAIN_ROWS
        )
        # Pluggable training backend (PR-10). Controls whether row-level models
        # train on a reservoir sample (default) or the full population / out-of-core.
        self.train_backend: TrainBackend = (
            train_backend if train_backend is not None else get_train_backend()
        )
        # Optional cross-answer lineage graph (PR-7); shared with the descriptive
        # toolset so a forecast can link back to the reconcile it was built from.
        self.lineage = lineage

    # -- tool registry --------------------------------------------------------

    def _ok(
        self,
        name: str,
        obj: dict[str, Any],
        data: Any = None,
        *,
        n: int | None = None,
        coverage: float | None = None,
        gates: dict[str, bool] | None = None,
        trust: dict[str, Any] | None = None,
        sql: str | None = None,
    ) -> ToolResult:
        """Wrap a predictive/causal result with a provenance envelope + trust grade.

        Mirrors ``AnalyticsToolset._ok`` so every answer (descriptive *or*
        predictive) is defensible: it carries where/how it was produced and how
        much to trust it. ``n`` is inferred from common sample-size keys when not
        given explicitly.
        """
        from demos.analytics.src.analytics.provenance import build_envelope

        if n is None:
            for k in ("n", "n_rows", "n_scored", "history_periods"):
                v = obj.get(k)
                if isinstance(v, (int, float)):
                    n = int(v)
                    break
        if trust is None and (n is not None or coverage is not None or gates):
            from demos.analytics.src.analytics.trust import grade

            trust = grade(coverage=coverage, n=n, gates=gates).to_dict()

        extra: dict[str, Any] = {}
        if trust is not None:
            # Embed trust in the JSON body (machine-readable) and the provenance
            # envelope, rather than appending prose that would break parsers.
            obj = {**obj, "trust": trust}
            extra["trust"] = trust
        from demos.analytics.src.analytics.lineage import LineageGraph

        answer_id = LineageGraph.new_id()
        extra["answerId"] = answer_id
        content = _frame(name, json.dumps(obj, default=str)[:MAX_RESULT_CHARS])
        env = build_envelope(self.source, **extra)
        if self.lineage is not None:
            self.lineage.record(
                answer_id, env.dataset_fingerprint, sql or "", parents=self.lineage.scope
            )
            self.lineage.scope.append(answer_id)
        return ToolResult.ok(
            content, data=data, provenance=env.to_dict(), trust=trust if trust is not None else None
        )

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
    ) -> tuple[str, str, list[tuple[str, str]], list[str], bool, str, Any]:
        """Resolve target/predictors, load training data, and compute the model key.

        Predictors may live in *related* tables; they are assembled into the
        target's row grain via discovered relationships (multi-table features),
        so a model can use attributes from anywhere in the dataset.
        """
        import pandas as pd

        t_table, t_col = self._resolve(args["target"])
        pred_refs = args.get("predictors") or [
            m.ref for m in self.model.metrics if m.table == t_table and m.column != t_col
        ]
        pred_cols = [self._resolve(p) for p in pred_refs]  # list[(table, col)]
        if not pred_cols:
            raise ValueError("no predictors available for the target")

        df = self._assemble_features(t_table, t_col, pred_cols, for_training=True)
        x = (
            df[[c for (_t, c) in pred_cols]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(axis=1, how="all")
        )
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
        return t_table, t_col, pred_cols, feature_cols, is_clf, key, data

    def _train_or_load(
        self, args: dict[str, Any]
    ) -> tuple[str, str, list[tuple[str, str]], Any, dict[str, Any], bool, float]:
        """Train-once cache: reuse a fresh stored model unless the caller forces a retrain."""
        t_table, t_col, pred_cols, feature_cols, is_clf, key, data = self._prepare_model_spec(args)
        if self.store is not None and not args.get("retrain", False):
            cached = self.store.get(key, max_age=self.model_ttl)
            if cached is not None:
                return (
                    t_table,
                    t_col,
                    pred_cols,
                    cached.model,
                    dict(cached.metadata),
                    True,
                    cached.trained_at,
                )
        result, fitted = self._fit(data[t_col], data[feature_cols], feature_cols, is_clf)
        result["feature_refs"] = [f"{t}.{c}" for (t, c) in pred_cols]
        trained_at = time.time()
        if self.store is not None:
            self.store.put(
                ModelRecord(key=key, model=fitted, metadata=result, trained_at=trained_at)
            )
        return t_table, t_col, pred_cols, fitted, result, False, trained_at

    def build_model(self) -> Tool:
        async def impl(args: dict[str, Any], ctx: Any) -> ToolResult:
            _, _, _, _, meta, cached, trained_at = self._train_or_load(args)
            return self._ok("build_model", {**meta, "cached": cached, "trained_at": trained_at})

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
            t_table, t_col, pred_cols, fitted, meta, cached, trained_at = self._train_or_load(args)
            feature_cols = [c for (_t, c) in pred_cols]
            filters = list(args.get("filters") or [])

            # Warehouse path: push scoring to the warehouse when the model is
            # expressible (linear) so we never materialize the scored frame.
            wh = self._predict_in_warehouse(t_table, t_col, pred_cols, fitted, meta, filters)
            if wh is not None:
                return wh

            # Local fallback (tree/RF, classification, or a non-warehouse source).
            score_df = self._assemble_features(t_table, t_col, pred_cols, filters=filters)
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

            return self._ok(
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

    def _predict_in_warehouse(
        self,
        t_table: str,
        t_col: str,
        pred_cols: list[tuple[str, str]],
        fitted: Any,
        meta: dict[str, Any],
        filters: list[dict[str, Any]],
    ) -> ToolResult | None:
        """Score in-warehouse when the model is expressible (linear regression).

        Returns a ToolResult if the warehouse could express the model, else
        ``None`` so the caller falls back to the local bounded-sample path. The
        frame is never materialized into the Python process — predictions and
        drift are computed with aggregate SQL on the warehouse.

        Only runs for an actual warehouse source (``make_warehouse_source``);
        local/CSV sources keep their existing local predict behavior.
        """
        if not getattr(self.source, "_is_warehouse", False):
            return None
        if meta.get("task") != "regression":
            return None
        feature_cols = [c for (_t, c) in pred_cols]
        frame_sql = self._feature_frame_sql(t_table, t_col, pred_cols, filters=filters)
        score_sql = score_warehouse(self.source, frame_sql, fitted, feature_cols, task="regression")
        if score_sql is None:
            return None
        agg = self.source.native_query(
            f"SELECT COUNT(*) AS n, AVG(prediction) AS mean, "
            f"MIN(prediction) AS min, MAX(prediction) AS max "
            f"FROM ({score_sql}) AS _s"
        )
        if not agg or int(agg[0]["n"]) == 0:
            return ToolResult.failed("no rows to score after filters")
        row = agg[0]
        summary = {
            "mean": round(float(row["mean"]), 4),
            "min": round(float(row["min"]), 4),
            "max": round(float(row["max"]), 4),
        }
        payload = {
            "target": t_col,
            "task": meta.get("task"),
            "n_scored": int(row["n"]),
            "prediction": summary,
            "scored_in_warehouse": True,
            "drift": self._warehouse_drift(frame_sql, feature_cols, meta),
            "method": "in-warehouse SQL scoring (linear); build_model(retrain=true) refreshes it",
        }
        return self._ok("predict", payload, data=payload)

    def _warehouse_drift(
        self, frame_sql: str, feature_cols: list[str], meta: dict[str, Any]
    ) -> Any:
        """Per-feature mean-shift drift computed with aggregate SQL (no materialization)."""
        sel = ", ".join(f"AVG({sql_qcol('_f', c)}) AS {sql_quote(c)}" for c in feature_cols)
        sql = f"SELECT {sel} FROM ({frame_sql}) AS _f"
        row = self.source.native_query(sql)
        if not row:
            return {"checked": False, "note": "no rows"}
        vals = {c: float(row[0].get(c) or 0.0) for c in feature_cols}
        import pandas as pd

        df = pd.DataFrame({c: [vals[c]] for c in feature_cols})
        train_stats = meta.get("train_stats") or {}
        reduced = {
            c: {
                "mean": (train_stats.get(c) or {}).get("mean", 0.0),
                "std": (train_stats.get(c) or {}).get("std", 0.0),
            }
            for c in feature_cols
        }
        return _drift_check(reduced, df, feature_cols)

    def _fit(self, y: Any, x: Any, pcols: list[str], is_clf: bool) -> tuple[dict[str, Any], Any]:
        # Per-feature training stats travel with the model so `predict` can flag
        # drift. We keep mean/std (fast mean-shift) plus decile edges + a value
        # sample so serving can compute PSI and a KS test against training.
        train_stats = {
            c: {
                "mean": round(float(x[c].mean()), 4),
                "std": round(float(x[c].std(ddof=0)), 4),
                "quantiles": _deciles(x[c]),
                "sample": _value_sample(x[c]),
            }
            for c in pcols
        }
        algo = "random_forest_classifier" if is_clf else "random_forest_regressor"
        res = self.train_backend.fit(y=y, x=x, feature_cols=pcols, is_clf=is_clf, algo=algo)
        model = res.model
        xv = x.astype(float).values
        if is_clf:
            codes, uniques = _encode(y)
            n_classes = len(uniques)
            # Backends that train out-of-core may return their own CV; otherwise
            # compute it here on the (in-memory) training frame.
            score = (
                res.cv
                if res.cv is not None
                else _safe_cv(model, xv, codes, "accuracy", is_clf=True)
            )
            return {
                "task": "classification",
                "n_rows": int(len(y)),
                "n_classes": int(n_classes),
                "classes": [str(u) for u in uniques],
                "cv_accuracy": score,
                "feature_importance": res.feature_importance or [],
                "train_stats": train_stats,
                "method": res.method or "RandomForestClassifier, 5-fold CV accuracy",
            }, model
        yv = y.astype(float).values
        score = res.cv if res.cv is not None else _safe_cv(model, xv, yv, "r2", is_clf=False)
        return {
            "task": "regression",
            "n_rows": int(len(y)),
            "cv_r2": score,
            "feature_importance": res.feature_importance or [],
            "train_stats": train_stats,
            "method": res.method or "RandomForestRegressor, 5-fold CV R^2",
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
            return self._ok(
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
                sql=sql,
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

            # SQL-native: compute per-group n / mean / sample-variance in the
            # engine instead of pulling every row into pandas. Welch's t-test and
            # Cohen's d are then derived from these aggregates.
            qv = sql_qcol(m_table, m_col)
            qg = sql_qcol(m_table, g_col)
            sql = (
                f"SELECT CAST({qg} AS VARCHAR) AS grp, COUNT({qv}) AS n, "
                f"AVG(CAST({qv} AS DOUBLE)) AS mean, "
                f"var_samp(CAST({qv} AS DOUBLE)) AS var "
                f"FROM {sql_quote(m_table)} "
                f"WHERE {qv} IS NOT NULL AND CAST({qg} AS VARCHAR) IN "
                f"('{sql_literal(a)}', '{sql_literal(b)}') GROUP BY grp"
            )
            by = {str(r["grp"]): r for r in self.source.native_query(sql)}
            ra, rb = by.get(a), by.get(b)
            nA = int(ra["n"]) if ra else 0
            nB = int(rb["n"]) if rb else 0
            if nA < 2 or nB < 2:
                return ToolResult.failed(f"need >= 2 rows per group (A={nA}, B={nB})")

            ma, mb = float(ra["mean"]), float(rb["mean"])
            vA = float(ra["var"] or 0.0)
            vB = float(rb["var"] or 0.0)
            se = np.sqrt(vA / nA + vB / nB)
            t = (ma - mb) / se if se > 0 else 0.0
            # Welch–Satterthwaite degrees of freedom.
            denom = (vA / nA) ** 2 / max(nA - 1, 1) + (vB / nB) ** 2 / max(nB - 1, 1)
            dof = ((vA / nA + vB / nB) ** 2 / denom) if denom > 0 else float(nA + nB - 2)
            p = float(2 * stats.t.sf(abs(t), dof)) if se > 0 else 1.0
            pooled = float(np.sqrt((vA + vB) / 2)) or 1.0
            return self._ok(
                "ab_test",
                {
                    "metric": m_col,
                    "groupA": a,
                    "groupB": b,
                    "nA": nA,
                    "nB": nB,
                    "meanA": round(ma, 4),
                    "meanB": round(mb, 4),
                    "difference": round(ma - mb, 4),
                    "welch_t": round(float(t), 3),
                    "p_value": round(p, 4),
                    "cohens_d": round((ma - mb) / pooled, 3),
                    "verdict": "significant at p<0.05" if p < 0.05 else "not significant (p>=0.05)",
                    "method": "Welch's t-test from SQL group aggregates (no row materialization)",
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
            controls = [self._resolve(x) for x in (args.get("controls") or [])]
            control_cols = [c for (_t, c) in controls]
            df = (
                self._assemble_features(t_table, t_col, [(tr_table, tr_col), *controls])
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            if len(df) < _MIN_ROWS:
                return ToolResult.failed(f"need {_MIN_ROWS}+ numeric rows (got {len(df)})")

            xd = sm.add_constant(df[[tr_col, *control_cols]])
            fit = sm.OLS(df[t_col], xd).fit()
            ci = fit.conf_int().loc[tr_col]
            return self._ok(
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
            preds = args.get("predictors") or [
                m.ref for m in self.model.metrics if m.table == t_table and m.column != t_col
            ]
            pred_pairs = [self._resolve(p) for p in preds]
            pcols = [c for (_t, c) in pred_pairs]
            df = (
                self._assemble_features(t_table, t_col, [(tr_table, tr_col), *pred_pairs])
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
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
            return self._ok(
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
            return self._ok(
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
            return self._ok(
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
            table, col = ref.rsplit(".", 1)
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

    def _feature_frame_sql(
        self,
        target_table: str,
        target_col: str,
        pred_cols: list[tuple[str, str]],
        filters: list[dict[str, Any]] | None = None,
    ) -> str:
        """SQL for the row-grain feature frame (target + predictors), no sampling.

        Used by both the local path (which materializes a bounded sample) and the
        warehouse path (which scores the full frame in-warehouse).
        """
        cols = [(target_table, target_col), *pred_cols]
        sql = feature_frame_sql(self.model, target_table, cols)
        if filters:
            clauses: list[str] = []
            for f in filters:
                f_table, f_col = self._resolve(str(f["column"]))
                if f_table != target_table:
                    raise ValueError(
                        f"filter column '{f['column']}' is not in target table '{target_table}'"
                    )
                op = str(f["op"])
                if op not in OPERATORS:
                    raise ValueError(f"unsupported filter op: {op}")
                lit = sql_literal(str(f["value"]))
                clauses.append(f"{sql_qcol(target_table, f_col)} {op} '{lit}'")
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
        return sql

    def _assemble_features(
        self,
        target_table: str,
        target_col: str,
        pred_cols: list[tuple[str, str]],
        filters: list[dict[str, Any]] | None = None,
        *,
        for_training: bool = False,
    ) -> Any:
        """Row-grain feature frame for modeling: join ``target_table`` to related
        tables (via discovered relationships) and select the target + predictors.

        Unlike ``_frame`` (single table), this assembles predictors that live in
        *other* tables, so a model can use features from across the dataset.

        When ``for_training`` and the configured ``train_backend`` trains on the
        full population (PR-10), the reservoir row cap is skipped so the model is
        fit on every row. Serving/predict keeps the cap for speed.
        """
        import pandas as pd

        sql = self._feature_frame_sql(target_table, target_col, pred_cols, filters=filters)
        # Honor the user's row cap (reservoir sample) so big data can trade
        # accuracy for speed; matches the sampling already done in ``_frame``.
        cap = self.max_train_rows
        if for_training and getattr(self.train_backend, "uses_full_population", False):
            cap = None
        if cap:
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
    return ToolResult.ok(_frame(name, json.dumps(obj, default=str)[:MAX_RESULT_CHARS]), data=data)


def _drift_check(train_stats: dict[str, Any], df: Any, feature_cols: list[str]) -> dict[str, Any]:
    """Per-feature distribution drift of scored rows vs the training data.

    Uses PSI (population stability index) over the training deciles as the
    primary signal, a two-sample KS test where a training value sample is
    available, and a standardized mean-shift as a fallback. PSI convention:
    <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
    """
    features: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for col in feature_cols:
        stats = train_stats.get(col)
        if not stats:
            continue
        col_vals = df[col].dropna()
        if col_vals.empty:
            continue
        std = float(stats.get("std") or 0.0) or 1.0
        mean_shift = round(abs(float(col_vals.mean()) - float(stats["mean"])) / std, 3)
        entry: dict[str, Any] = {"mean_shift": mean_shift}
        score = mean_shift / _MEAN_SHIFT_TO_PSI  # comparable scale when PSI absent

        quantiles = stats.get("quantiles")
        if quantiles:
            psi = _psi(quantiles, col_vals)
            entry["psi"] = round(psi, 3)
            score = psi
        sample = stats.get("sample")
        if sample:
            ks = _ks(sample, col_vals)
            if ks is not None:
                entry["ks"] = round(ks, 3)
        features[col] = entry
        scores[col] = score

    if not scores:
        return {"checked": False, "note": "no training statistics stored with this model"}
    worst = max(scores, key=lambda c: scores[c])
    worst_psi = features[worst].get("psi")
    detected = (worst_psi is not None and worst_psi > _PSI_THRESHOLD) or (
        worst_psi is None and scores[worst] > _MEAN_SHIFT_DRIFT
    )
    result: dict[str, Any] = {
        "checked": True,
        "detected": detected,
        "method": "PSI over training deciles (+ KS, mean-shift)",
        "score": round(scores[worst], 3),
        "threshold": _PSI_THRESHOLD,
        "worst_feature": worst,
        "features": features,
    }
    if detected:
        result["recommendation"] = (
            "scored rows' distribution shifted vs training data; "
            "retrain via build_model(retrain=true)"
        )
        inc(
            "analytics.drift.breaches",
            tags={"feature": worst, "tool": "predict"},
        )
    return result


def _deciles(series: Any) -> list[float]:
    """Training decile edges (10 bins → 11 edges) used as PSI/histogram bins."""
    import numpy as np

    vals = series.dropna().astype(float).values
    if len(vals) == 0:
        return []
    edges = np.quantile(vals, [i / 10 for i in range(11)])
    # De-duplicate collapsed edges (near-constant features) while staying sorted.
    out: list[float] = []
    for e in edges:
        e = float(e)
        if not out or e > out[-1]:
            out.append(e)
    return out


def _value_sample(series: Any, n: int = 500) -> list[float]:
    import numpy as np

    vals = series.dropna().astype(float).values
    if len(vals) <= n:
        return [float(v) for v in vals]
    rng = np.random.default_rng(0)
    return [float(v) for v in rng.choice(vals, size=n, replace=False)]


def _as_array(series: Any) -> Any:
    """Coerce a pandas Series or numpy array to a 1-D float array (NaNs dropped)."""
    import numpy as np

    if hasattr(series, "dropna"):
        return series.dropna().astype(float).values
    arr = np.asarray(series, dtype=float)
    return arr[~np.isnan(arr)]


def _psi(edges: list[float], series: Any) -> float:
    """Population stability index of ``series`` against training bin ``edges``."""
    import numpy as np

    if len(edges) < 2:
        return 0.0
    vals = _as_array(series)
    if len(vals) == 0:
        return 0.0
    bins = np.array(edges, dtype=float)
    n_bins = len(bins) - 1
    # Expected mass is uniform across deciles by construction (1/n_bins each).
    expected = np.full(n_bins, 1.0 / n_bins)
    counts, _ = np.histogram(vals, bins=bins)
    # Include out-of-range values in the nearest edge bin.
    below = int((vals < bins[0]).sum())
    above = int((vals > bins[-1]).sum())
    counts = counts.astype(float)
    counts[0] += below
    counts[-1] += above
    actual = counts / max(1, counts.sum())
    eps = 1e-6
    actual = np.clip(actual, eps, None)
    expected = np.clip(expected, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _ks(train_sample: list[float], series: Any) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic (0..1); None if scipy is absent."""
    try:
        from scipy import stats as _stats
    except Exception:
        return None
    vals = _as_array(series)
    if len(vals) == 0 or not train_sample:
        return None
    try:
        return float(_stats.ks_2samp(train_sample, vals).statistic)
    except Exception:
        return None


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
