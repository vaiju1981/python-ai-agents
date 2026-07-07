"""Advanced ML and statistical analysis tools for the analytics agent.

Goes beyond simple SQL analytics to support model building, causal inference,
uplift modeling, forecasting, clustering, and statistical testing — all as
governed read-only tools that pull data from the ``DataSource`` and return
structured results.

Libraries used: scikit-learn, statsmodels, numpy.
"""

from __future__ import annotations

import json
from typing import Any

from demos.analytics.src.analytics.data_source import DataSource, sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import SemanticModel

from python_ai_agents.core.tool import Tool, ToolEffect, ToolResult, ToolSpec

MAX_RESULT_CHARS = 16_000

__all__ = ["MLToolset"]


class MLToolset:
    """Builds governed ML and statistical analysis tools."""

    def __init__(self, source: DataSource, model: SemanticModel) -> None:
        self.source = source
        self.model = model

    def regression(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                target = args.get("target", "")
                predictors = args.get("predictors", [])
                method = args.get("method", "linear")  # linear, ridge, lasso, rf, gbrt
                filters = args.get("filters", [])

                t = _resolve_metric(self.model, target)
                if t is None:
                    return ToolResult.failed(f"target metric '{target}' not found")
                if not predictors:
                    predictors = [m.column for m in self.model.metrics
                                  if m.table == t.table and m.ref != t.ref]
                if not predictors:
                    return ToolResult.failed("no predictor metrics found in same table")

                df = _fetch_dataframe(self.source, t.table, [t.column] + predictors, filters)
                if len(df) < 10:
                    return ToolResult.failed(f"not enough data: {len(df)} rows (need 10+)")

                import numpy as np
                y = df[t.column].values.astype(float)
                X = df[predictors].values.astype(float)
                mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
                X, y = X[mask], y[mask]

                regressor = _make_regressor(method)
                regressor.fit(X, y)
                score = regressor.score(X, y)
                y_pred = regressor.predict(X)
                rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

                result = {
                    "method": method,
                    "target": t.column,
                    "predictors": [],
                    "r_squared": round(score, 4),
                    "rmse": round(rmse, 4),
                    "n_rows": len(y),
                }
                if hasattr(regressor, "coef_"):
                    result["predictors"] = [
                        {"name": p, "coefficient": round(float(c), 4)}
                        for p, c in zip(predictors, regressor.coef_)
                    ]
                    if hasattr(regressor, "intercept_"):
                        result["intercept"] = round(float(regressor.intercept_), 4)
                elif hasattr(regressor, "feature_importances_"):
                    result["predictors"] = [
                        {"name": p, "importance": round(float(i), 4)}
                        for p, i in sorted(
                            zip(predictors, regressor.feature_importances_),
                            key=lambda x: x[1], reverse=True
                        )
                    ]

                return ToolResult.ok(_frame("regression", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"regression failed: {exc}")

        return _tool("ml_regression", "Train a regression model. Args: target, predictors (optional), method (linear/ridge/lasso/rf/gbrt).", invoke)

    def classification(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                target = args.get("target", "")
                predictors = args.get("predictors", [])
                method = args.get("method", "logistic")  # logistic, rf, gbrt
                filters = args.get("filters", [])

                # Target can be a dimension (categorical) or metric
                resolved = _resolve_column(self.model, target)
                if resolved is None:
                    return ToolResult.failed(f"target '{target}' not found in schema")
                table, target_col = resolved

                if not predictors:
                    predictors = [m.column for m in self.model.metrics if m.table == table]
                    predictors += [d.column for d in self.model.dimensions
                                   if d.table == table and d.column != target_col]
                if not predictors:
                    return ToolResult.failed("no predictor columns found")

                df = _fetch_dataframe(self.source, table, [target_col] + predictors, filters)
                if len(df) < 10:
                    return ToolResult.failed(f"not enough data: {len(df)} rows (need 10+)")

                from sklearn.preprocessing import LabelEncoder
                import numpy as np

                le = LabelEncoder()
                y = le.fit_transform(df[target_col].astype(str))
                X = df[predictors].values
                # Encode non-numeric predictors
                for i, p in enumerate(predictors):
                    if X[:, i].dtype == object:
                        X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))
                X = X.astype(float)
                mask = ~np.any(np.isnan(X), axis=1)
                X, y = X[mask], y[mask]

                if len(set(y)) < 2:
                    return ToolResult.failed(f"target has only one class: {set(y)}")

                clf = _make_classifier(method)
                clf.fit(X, y)
                y_pred = clf.predict(X)
                from sklearn.metrics import accuracy_score, classification_report

                accuracy = accuracy_score(y, y_pred)
                report = classification_report(y, y_pred, output_dict=True, zero_division=0)

                result = {
                    "method": method,
                    "target": target_col,
                    "classes": le.classes_.tolist(),
                    "accuracy": round(accuracy, 4),
                    "n_rows": len(y),
                    "predictors": [],
                    "classification_report": {
                        k: {kk: round(vv, 3) for kk, vv in v.items()}
                        for k, v in report.items() if isinstance(v, dict)
                    },
                }
                if hasattr(clf, "feature_importances_"):
                    result["predictors"] = [
                        {"name": p, "importance": round(float(i), 4)}
                        for p, i in sorted(
                            zip(predictors, clf.feature_importances_),
                            key=lambda x: x[1], reverse=True
                        )
                    ]

                return ToolResult.ok(_frame("classification", json.dumps(result, default=str)[:MAX_RESULT_CHARS]))
            except Exception as exc:
                return ToolResult.failed(f"classification failed: {exc}")

        return _tool("classification", "Train a classifier. Args: target (categorical column), predictors (optional), method (logistic/rf/gbrt).", invoke)

    def clustering(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                columns = args.get("columns", [])
                method = args.get("method", "kmeans")  # kmeans, dbscan
                n_clusters = args.get("nClusters", 3)
                table = args.get("table", "")

                if not table:
                    table = self.model.metrics[0].table if self.model.metrics else self.model.dimensions[0].table
                if not columns:
                    columns = [m.column for m in self.model.metrics if m.table == table]
                if not columns:
                    return ToolResult.failed("no numeric columns found for clustering")

                df = _fetch_dataframe(self.source, table, columns, [])
                if len(df) < 5:
                    return ToolResult.failed(f"not enough data: {len(df)} rows")

                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.metrics import silhouette_score

                X = df[columns].values.astype(float)
                mask = ~np.any(np.isnan(X), axis=1)
                X = X[mask]
                X_scaled = StandardScaler().fit_transform(X)

                if method == "dbscan":
                    clusterer = DBSCAN(eps=0.5, min_samples=5)
                    labels = clusterer.fit_predict(X_scaled)
                    n = len(set(labels)) - (1 if -1 in labels else 0)
                else:
                    n = min(n_clusters, len(X) - 1)
                    clusterer = KMeans(n_clusters=n, n_init=10, random_state=42)
                    labels = clusterer.fit_predict(X_scaled)

                sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0

                result = {
                    "method": method,
                    "table": table,
                    "columns": columns,
                    "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                    "n_noise": int(sum(labels == -1)) if -1 in labels else 0,
                    "silhouette_score": round(float(sil), 4),
                    "n_rows": len(labels),
                    "cluster_sizes": {str(k): int(sum(labels == k)) for k in set(labels)},
                }
                return ToolResult.ok(_frame("clustering", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"clustering failed: {exc}")

        return _tool("clustering", "Cluster rows by numeric columns. Args: columns, method (kmeans/dbscan), nClusters, table.", invoke)

    def forecast(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                metric = args.get("metric", "")
                time_col = args.get("timeColumn", "")
                horizon = args.get("horizon", 7)
                method = args.get("method", "auto")  # auto, arima, ets, linear

                m = _resolve_metric(self.model, metric)
                if m is None:
                    return ToolResult.failed(f"metric '{metric}' not found")
                tc = _resolve_time_column(self.model, time_col)
                if tc is None:
                    return ToolResult.failed(f"time column '{time_col}' not found")
                if m.table != tc.table:
                    return ToolResult.failed("metric and time column must be in the same table")

                ts_expr = tc.to_timestamp_sql(sql_qcol(tc.table, tc.column))
                sql = (
                    f"SELECT {ts_expr} AS ts, {m.aggregation.upper()}({sql_qcol(m.table, m.column)}) AS val "
                    f"FROM {sql_quote(m.table)} WHERE {ts_expr} IS NOT NULL "
                    f"GROUP BY ts ORDER BY ts"
                )
                rows = self.source.native_query_with_limit(sql, 1000)
                if len(rows) < 5:
                    return ToolResult.failed(f"not enough time points: {len(rows)}")

                import numpy as np
                values = [float(r["val"]) for r in rows]
                n = len(values)
                forecasts: list[float] = []

                if method == "linear" or (method == "auto" and n < 20):
                    # Linear trend
                    x = np.arange(n)
                    coeffs = np.polyfit(x, values, 1)
                    for i in range(horizon):
                        forecasts.append(float(coeffs[0] * (n + i) + coeffs[1]))
                elif method == "arima" or (method == "auto" and n >= 20):
                    try:
                        from statsmodels.tsa.arima.model import ARIMA
                        model_fit = ARIMA(values, order=(1, 1, 1)).fit()
                        forecasts = model_fit.forecast(steps=horizon).tolist()
                    except Exception:
                        # Fall back to linear
                        x = np.arange(n)
                        coeffs = np.polyfit(x, values, 1)
                        for i in range(horizon):
                            forecasts.append(float(coeffs[0] * (n + i) + coeffs[1]))
                else:  # ets
                    try:
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        model_fit = ExponentialSmoothing(values, trend="add").fit()
                        forecasts = model_fit.forecast(horizon).tolist()
                    except Exception:
                        x = np.arange(n)
                        coeffs = np.polyfit(x, values, 1)
                        for i in range(horizon):
                            forecasts.append(float(coeffs[0] * (n + i) + coeffs[1]))

                result = {
                    "metric": m.column,
                    "time_column": tc.column,
                    "method": method,
                    "history_points": n,
                    "last_value": round(values[-1], 2),
                    "forecast_horizon": horizon,
                    "forecasts": [{"step": i + 1, "value": round(float(f), 2)} for i, f in enumerate(forecasts)],
                }
                return ToolResult.ok(_frame("forecast", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"forecast failed: {exc}")

        return _tool("forecast", "Time-series forecasting. Args: metric, timeColumn, horizon (days), method (auto/arima/ets/linear).", invoke)

    def causal_analysis(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                target = args.get("target", "")
                treatment = args.get("treatment", "")
                controls = args.get("controls", [])
                method = args.get("method", "difference_in_means")  # difference_in_means, regression_adjustment

                t = _resolve_metric(self.model, target)
                if t is None:
                    return ToolResult.failed(f"target '{target}' not found")
                treat_col = _resolve_dimension(self.model, treatment)
                if treat_col is None:
                    return ToolResult.failed(f"treatment '{treatment}' not found")
                if t.table != treat_col.table:
                    return ToolResult.failed("target and treatment must be in the same table")

                all_cols = [t.column, treat_col.column] + controls
                df = _fetch_dataframe(self.source, t.table, all_cols, [])
                if len(df) < 20:
                    return ToolResult.failed(f"not enough data: {len(df)} rows (need 20+)")

                import numpy as np
                from sklearn.preprocessing import LabelEncoder

                y = df[t.column].values.astype(float)
                treat = df[treat_col.column].values
                if treat.dtype == object:
                    treat = LabelEncoder().fit_transform(treat.astype(str))

                treated = y[treat == 1] if 2 in set(treat) else y[treat == treat.max()]
                control = y[treat == 0] if 0 in set(treat) else y[treat == treat.min()]

                if method == "regression_adjustment" and controls:
                    # OLS with treatment + controls
                    from sklearn.linear_model import LinearRegression
                    X = df[[treat_col.column] + controls].values
                    for i, c in enumerate([treat_col.column] + controls):
                        if X[:, i].dtype == object:
                            X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))
                    X = X.astype(float)
                    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
                    X, y_clean = X[mask], y[mask]
                    reg = LinearRegression().fit(X, y_clean)
                    ate = float(reg.coef_[0])  # treatment coefficient
                    ci_low = ate - 1.96 * np.std(y_clean) / np.sqrt(len(y_clean))
                    ci_high = ate + 1.96 * np.std(y_clean) / np.sqrt(len(y_clean))
                else:
                    # Simple difference in means
                    ate = float(treated.mean() - control.mean())
                    se = float(np.sqrt(treated.var() / len(treated) + control.var() / len(control)))
                    ci_low = ate - 1.96 * se
                    ci_high = ate + 1.96 * se

                result = {
                    "target": t.column,
                    "treatment": treat_col.column,
                    "method": method,
                    "treated_mean": round(float(treated.mean()), 4),
                    "control_mean": round(float(control.mean()), 4),
                    "ate": round(ate, 4),
                    "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                    "n_treated": int(len(treated)),
                    "n_control": int(len(control)),
                    "controls": controls,
                }
                return ToolResult.ok(_frame("causal_analysis", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"causal_analysis failed: {exc}")

        return _tool("causal_analysis", "Estimate causal treatment effect. Args: target (metric), treatment (column), controls (list), method (difference_in_means/regression_adjustment).", invoke)

    def uplift_modeling(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                target = args.get("target", "")
                treatment = args.get("treatment", "")
                predictors = args.get("predictors", [])
                method = args.get("method", "t_learner")  # t_learner, s_learner

                t = _resolve_metric(self.model, target)
                if t is None:
                    return ToolResult.failed(f"target '{target}' not found")
                treat_col = _resolve_dimension(self.model, treatment)
                if treat_col is None:
                    return ToolResult.failed(f"treatment '{treatment}' not found")
                if t.table != treat_col.table:
                    return ToolResult.failed("target and treatment must be in same table")

                if not predictors:
                    predictors = [m.column for m in self.model.metrics
                                  if m.table == t.table and m.ref != t.ref]
                    predictors += [d.column for d in self.model.dimensions
                                   if d.table == t.table and d.column != treat_col.column]
                if not predictors:
                    return ToolResult.failed("no predictor columns found")

                all_cols = [t.column, treat_col.column] + predictors
                df = _fetch_dataframe(self.source, t.table, all_cols, [])
                if len(df) < 30:
                    return ToolResult.failed(f"not enough data: {len(df)} rows (need 30+)")

                import numpy as np
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.preprocessing import LabelEncoder

                y = df[t.column].values.astype(float)
                treat = df[treat_col.column].values
                if treat.dtype == object:
                    treat = LabelEncoder().fit_transform(treat.astype(str))
                X = df[predictors].values
                for i in range(X.shape[1]):
                    if X[:, i].dtype == object:
                        X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))
                X = X.astype(float)
                mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
                X, y, treat = X[mask], y[mask], treat[mask]

                treated_mask = treat == treat.max()
                control_mask = treat == treat.min()

                if method == "s_learner":
                    # Single model with treatment as a feature
                    X_full = np.column_stack([X, treat])
                    model = GradientBoostingRegressor(random_state=42)
                    model.fit(X_full, y)
                    X_treat = np.column_stack([X, np.ones(len(X))])
                    X_control = np.column_stack([X, np.zeros(len(X))])
                    uplift = model.predict(X_treat) - model.predict(X_control)
                else:  # t_learner
                    model_t = GradientBoostingRegressor(random_state=42)
                    model_c = GradientBoostingRegressor(random_state=42)
                    if treated_mask.sum() > 0:
                        model_t.fit(X[treated_mask], y[treated_mask])
                    if control_mask.sum() > 0:
                        model_c.fit(X[control_mask], y[control_mask])
                    uplift = model_t.predict(X) - model_c.predict(X)

                # Segment into deciles
                sorted_idx = np.argsort(uplift)[::-1]
                n = len(uplift)
                decile_size = max(1, n // 10)
                segments = []
                for decile in range(10):
                    start = decile * decile_size
                    end = min(start + decile_size, n)
                    if start >= n:
                        break
                    idx = sorted_idx[start:end]
                    seg_treated = y[idx][treat[idx] == treat.max()]
                    seg_control = y[idx][treat[idx] == treat.min()]
                    seg_uplift = float(
                        seg_treated.mean() - seg_control.mean()
                        if len(seg_treated) > 0 and len(seg_control) > 0
                        else uplift[idx].mean()
                    )
                    segments.append({
                        "decile": decile + 1,
                        "n": len(idx),
                        "uplift": round(seg_uplift, 4),
                        "mean_uplift_score": round(float(uplift[idx].mean()), 4),
                    })

                result = {
                    "target": t.column,
                    "treatment": treat_col.column,
                    "method": method,
                    "n_rows": n,
                    "overall_uplift": round(float(uplift.mean()), 4),
                    "uplift_deciles": segments,
                }
                return ToolResult.ok(_frame("uplift_modeling", json.dumps(result, default=str)[:MAX_RESULT_CHARS]))
            except Exception as exc:
                return ToolResult.failed(f"uplift_modeling failed: {exc}")

        return _tool("uplift_modeling", "Estimate uplift/treatment effect heterogeneity. Args: target, treatment, predictors, method (t_learner/s_learner).", invoke)

    def feature_importance(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                target = args.get("target", "")
                method = args.get("method", "permutation")  # permutation, tree

                t = _resolve_metric(self.model, target)
                if t is None:
                    # Try dimension (classification target)
                    resolved = _resolve_column(self.model, target)
                    if resolved is None:
                        return ToolResult.failed(f"target '{target}' not found")
                    table, target_col = resolved
                else:
                    table, target_col = t.table, t.column

                predictors = [m.column for m in self.model.metrics if m.table == table and m.column != target_col]
                predictors += [d.column for d in self.model.dimensions if d.table == table and d.column != target_col]
                if not predictors:
                    return ToolResult.failed("no predictor columns found")

                df = _fetch_dataframe(self.source, table, [target_col] + predictors, [])
                if len(df) < 10:
                    return ToolResult.failed(f"not enough data: {len(df)} rows")

                import numpy as np
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.inspection import permutation_importance
                from sklearn.preprocessing import LabelEncoder

                y = df[target_col].values
                if y.dtype == object:
                    y = LabelEncoder().fit_transform(y.astype(str))
                y = y.astype(float)

                X = df[predictors].values
                for i in range(X.shape[1]):
                    if X[:, i].dtype == object:
                        X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))
                X = X.astype(float)
                mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
                X, y = X[mask], y[mask]

                model = GradientBoostingRegressor(random_state=42)
                model.fit(X, y)

                if method == "permutation":
                    perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)
                    importances = sorted(
                        zip(predictors, perm.importances_mean),
                        key=lambda x: x[1], reverse=True
                    )
                    result = {
                        "method": "permutation",
                        "target": target_col,
                        "features": [
                            {"name": p, "importance": round(float(i), 4)}
                            for p, i in importances
                        ],
                    }
                else:
                    importances = sorted(
                        zip(predictors, model.feature_importances_),
                        key=lambda x: x[1], reverse=True
                    )
                    result = {
                        "method": "tree",
                        "target": target_col,
                        "features": [
                            {"name": p, "importance": round(float(i), 4)}
                            for p, i in importances
                        ],
                    }
                return ToolResult.ok(_frame("feature_importance", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"feature_importance failed: {exc}")

        return _tool("feature_importance", "Rank features by importance for predicting a target. Args: target, method (permutation/tree).", invoke)

    def statistical_test(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                test_type = args.get("test", "ttest")  # ttest, chi2, mannwhitney, anova
                column = args.get("column", "")
                group_column = args.get("groupColumn", "")

                # Resolve columns
                resolved = _resolve_column(self.model, column)
                if resolved is None:
                    return ToolResult.failed(f"column '{column}' not found")
                table, col = resolved

                group_resolved = _resolve_column(self.model, group_column)
                if group_resolved is None:
                    return ToolResult.failed(f"group column '{group_column}' not found")
                _, group_col = group_resolved

                df = _fetch_dataframe(self.source, table, [col, group_col], [])
                if len(df) < 5:
                    return ToolResult.failed(f"not enough data: {len(df)} rows")

                import numpy as np
                groups = df[group_col].unique()
                if len(groups) < 2:
                    return ToolResult.failed(f"need 2+ groups, found {len(groups)}")

                result = {"test": test_type, "column": col, "group_column": group_col, "n_groups": len(groups)}

                if test_type == "ttest":
                    from scipy.stats import ttest_ind
                    g1 = df[df[group_col] == groups[0]][col].dropna().values.astype(float)
                    g2 = df[df[group_col] == groups[1]][col].dropna().values.astype(float)
                    stat, pval = ttest_ind(g1, g2)
                    result.update({
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(pval), 6),
                        "significant": pval < 0.05,
                        "group1_mean": round(float(g1.mean()), 4),
                        "group2_mean": round(float(g2.mean()), 4),
                    })
                elif test_type == "mannwhitney":
                    from scipy.stats import mannwhitneyu
                    g1 = df[df[group_col] == groups[0]][col].dropna().values.astype(float)
                    g2 = df[df[group_col] == groups[1]][col].dropna().values.astype(float)
                    stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                    result.update({
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(pval), 6),
                        "significant": pval < 0.05,
                    })
                elif test_type == "chi2":
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df[col], df[group_col])
                    stat, pval, dof, _ = chi2_contingency(ct)
                    result.update({
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(pval), 6),
                        "dof": int(dof),
                        "significant": pval < 0.05,
                    })
                elif test_type == "anova":
                    from scipy.stats import f_oneway
                    group_data = [df[df[group_col] == g][col].dropna().values.astype(float) for g in groups]
                    stat, pval = f_oneway(*group_data)
                    result.update({
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(pval), 6),
                        "significant": pval < 0.05,
                    })
                else:
                    return ToolResult.failed(f"unknown test: {test_type}")

                return ToolResult.ok(_frame("statistical_test", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"statistical_test failed: {exc}")

        return _tool("statistical_test", "Run a statistical hypothesis test. Args: test (ttest/chi2/mannwhitney/anova), column, groupColumn.", invoke)

    def anomaly_detection(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                columns = args.get("columns", [])
                method = args.get("method", "isolation_forest")  # isolation_forest, lof
                contamination = args.get("contamination", 0.05)
                table = args.get("table", "")

                if not table:
                    table = self.model.metrics[0].table if self.model.metrics else ""
                if not columns:
                    columns = [m.column for m in self.model.metrics if m.table == table]
                if not columns:
                    return ToolResult.failed("no numeric columns found")

                df = _fetch_dataframe(self.source, table, columns, [])
                if len(df) < 10:
                    return ToolResult.failed(f"not enough data: {len(df)} rows")

                import numpy as np
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor
                from sklearn.preprocessing import StandardScaler

                X = df[columns].values.astype(float)
                mask = ~np.any(np.isnan(X), axis=1)
                X = X[mask]
                X_scaled = StandardScaler().fit_transform(X)

                if method == "lof":
                    detector = LocalOutlierFactor(contamination=contamination)
                    labels = detector.fit_predict(X_scaled)
                    scores = -detector.negative_outlier_factor_
                else:
                    detector = IsolationForest(contamination=contamination, random_state=42)
                    labels = detector.fit_predict(X_scaled)
                    scores = detector.decision_function(X_scaled)

                n_anomalies = int(sum(labels == -1))

                result = {
                    "method": method,
                    "table": table,
                    "columns": columns,
                    "n_rows": len(labels),
                    "n_anomalies": n_anomalies,
                    "anomaly_rate": round(n_anomalies / len(labels), 4),
                    "anomaly_score_mean": round(float(np.mean(scores)), 4),
                    "anomaly_score_std": round(float(np.std(scores)), 4),
                }
                return ToolResult.ok(_frame("anomaly_detection", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"anomaly_detection failed: {exc}")

        return _tool("anomaly_detection", "Detect anomalies using ML methods. Args: columns, method (isolation_forest/lof), contamination, table.", invoke)

    def cross_validate(self) -> Tool:
        async def invoke(args: dict[str, Any], ctx: Any) -> ToolResult:
            try:
                target = args.get("target", "")
                predictors = args.get("predictors", [])
                method = args.get("method", "linear")
                cv_folds = args.get("cvFolds", 5)
                task = args.get("task", "regression")  # regression or classification

                resolved = _resolve_column(self.model, target)
                if resolved is None:
                    return ToolResult.failed(f"target '{target}' not found")
                table, target_col = resolved

                if not predictors:
                    predictors = [m.column for m in self.model.metrics if m.table == table and m.column != target_col]
                    predictors += [d.column for d in self.model.dimensions if d.table == table and d.column != target_col]
                if not predictors:
                    return ToolResult.failed("no predictor columns found")

                df = _fetch_dataframe(self.source, table, [target_col] + predictors, [])
                if len(df) < 20:
                    return ToolResult.failed(f"not enough data for CV: {len(df)} rows (need 20+)")

                import numpy as np
                from sklearn.model_selection import cross_val_score
                from sklearn.preprocessing import LabelEncoder

                y = df[target_col].values
                if y.dtype == object or task == "classification":
                    y = LabelEncoder().fit_transform(y.astype(str))
                y = y.astype(float)

                X = df[predictors].values
                for i in range(X.shape[1]):
                    if X[:, i].dtype == object:
                        X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))
                X = X.astype(float)
                mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
                X, y = X[mask], y[mask]

                if task == "classification":
                    estimator = _make_classifier(method)
                    scoring = "accuracy"
                else:
                    estimator = _make_regressor(method)
                    scoring = "r2"

                scores = cross_val_score(estimator, X, y, cv=min(cv_folds, len(y)), scoring=scoring)

                result = {
                    "task": task,
                    "method": method,
                    "target": target_col,
                    "cv_folds": len(scores),
                    "mean_score": round(float(scores.mean()), 4),
                    "std_score": round(float(scores.std()), 4),
                    "scores": [round(float(s), 4) for s in scores],
                    "n_rows": len(y),
                }
                return ToolResult.ok(_frame("cross_validate", json.dumps(result, default=str)))
            except Exception as exc:
                return ToolResult.failed(f"cross_validate failed: {exc}")

        return _tool("cross_validate", "Cross-validate a model. Args: target, predictors, method, cvFolds, task (regression/classification).", invoke)

    def all_tools(self) -> list[Tool]:
        return [
            self.regression(),
            self.classification(),
            self.clustering(),
            self.forecast(),
            self.causal_analysis(),
            self.uplift_modeling(),
            self.feature_importance(),
            self.statistical_test(),
            self.anomaly_detection(),
            self.cross_validate(),
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_metric(model: SemanticModel, ref: str):
    for m in model.metrics:
        if m.ref.lower() == ref.lower() or m.column.lower() == ref.lower():
            return m
    return None


def _resolve_dimension(model: SemanticModel, ref: str):
    for d in model.dimensions:
        if d.ref.lower() == ref.lower() or d.column.lower() == ref.lower():
            return d
    return None


def _resolve_time_column(model: SemanticModel, ref: str):
    rl = ref.lower()
    for tc in model.time_columns:
        if tc.ref.lower() == rl or tc.column.lower() == rl:
            return tc
    return None


def _resolve_column(model: SemanticModel, ref: str) -> tuple[str, str] | None:
    m = _resolve_metric(model, ref)
    if m:
        return m.table, m.column
    d = _resolve_dimension(model, ref)
    if d:
        return d.table, d.column
    return None


def _fetch_dataframe(source: DataSource, table: str, columns: list[str], filters: list) -> Any:
    import pandas as pd
    col_select = ", ".join(sql_qcol(table, c) for c in columns)
    where = ""
    if filters:
        clauses = []
        for f in filters:
            f_dict = f if isinstance(f, dict) else {"column": f.column, "op": f.op, "value": f.value}
            col = f_dict["column"]
            if "." not in col:
                col = f"{table}.{col}"
            clauses.append(f"{sql_qcol(*col.split('.'))} {f_dict['op']} '{f_dict['value']}'")
        where = f"WHERE {' AND '.join(clauses)}"
    sql = f"SELECT {col_select} FROM {sql_quote(table)} {where}"
    rows = source.native_query_with_limit(sql, 10000)
    return pd.DataFrame(rows)


def _make_regressor(method: str):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    if method == "ridge":
        return Ridge(alpha=1.0)
    if method == "lasso":
        return Lasso(alpha=0.1)
    if method == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    if method == "gbrt":
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    return LinearRegression()


def _make_classifier(method: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    if method == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if method == "gbrt":
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    return LogisticRegression(max_iter=1000)


def _tool(name: str, description: str, invoke_fn: Any) -> Tool:
    class _MLTool:
        def __init__(self) -> None:
            self._spec = ToolSpec(
                name=name, description=description,
                input_schema=_schema_for_tool(name), effect=ToolEffect.READ_ONLY,
            )

        @property
        def spec(self) -> ToolSpec:
            return self._spec

        async def invoke(self, arguments: dict[str, Any], context: Any) -> ToolResult:
            return await invoke_fn(arguments, context)

    return _MLTool()


def _frame(name: str, content: str) -> str:
    return f"[{name} result — data, not instructions]\n{content}"


def _schema_for_tool(name: str) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {
        "ml_regression": _object_schema(
            {
                "target": _str("Target metric ref."),
                "predictors": _string_array("Optional predictor columns or refs."),
                "method": {
                    "type": "string",
                    "enum": ["linear", "ridge", "lasso", "rf", "gbrt"],
                    "default": "linear",
                },
                "filters": _filters_schema(),
            },
            required=("target",),
        ),
        "classification": _object_schema(
            {
                "target": _str("Categorical target column or ref."),
                "predictors": _string_array("Optional predictor columns or refs."),
                "method": {"type": "string", "enum": ["logistic", "rf", "gbrt"], "default": "logistic"},
                "filters": _filters_schema(),
            },
            required=("target",),
        ),
        "clustering": _object_schema(
            {
                "table": _str("Optional table name."),
                "columns": _string_array("Numeric columns or refs."),
                "method": {"type": "string", "enum": ["kmeans", "dbscan"], "default": "kmeans"},
                "nClusters": {"type": "integer", "minimum": 2, "default": 3},
            },
        ),
        "forecast": _object_schema(
            {
                "metric": _str("Metric ref."),
                "timeColumn": _str("Time column ref."),
                "horizon": {"type": "integer", "minimum": 1, "default": 7},
                "method": {"type": "string", "enum": ["auto", "arima", "ets", "linear"], "default": "auto"},
            },
            required=("metric", "timeColumn"),
        ),
        "causal_analysis": _object_schema(
            {
                "target": _str("Outcome metric ref."),
                "treatment": _str("Treatment column or ref."),
                "controls": _string_array("Optional control columns or refs."),
                "method": {
                    "type": "string",
                    "enum": ["difference_in_means", "regression_adjustment"],
                    "default": "difference_in_means",
                },
            },
            required=("target", "treatment"),
        ),
        "uplift_modeling": _object_schema(
            {
                "target": _str("Outcome metric ref."),
                "treatment": _str("Treatment column or ref."),
                "predictors": _string_array("Optional predictor columns or refs."),
                "method": {"type": "string", "enum": ["t_learner", "s_learner"], "default": "t_learner"},
            },
            required=("target", "treatment"),
        ),
        "feature_importance": _object_schema(
            {
                "target": _str("Target column or ref."),
                "method": {"type": "string", "enum": ["permutation", "tree"], "default": "permutation"},
            },
            required=("target",),
        ),
        "statistical_test": _object_schema(
            {
                "test": {"type": "string", "enum": ["ttest", "chi2", "mannwhitney", "anova"], "default": "ttest"},
                "column": _str("Column or ref to test."),
                "groupColumn": _str("Grouping column or ref."),
            },
            required=("column", "groupColumn"),
        ),
        "anomaly_detection": _object_schema(
            {
                "table": _str("Optional table name."),
                "columns": _string_array("Numeric columns or refs."),
                "method": {
                    "type": "string",
                    "enum": ["isolation_forest", "lof"],
                    "default": "isolation_forest",
                },
                "contamination": {"type": "number", "minimum": 0.001, "maximum": 0.5, "default": 0.05},
            },
        ),
        "cross_validate": _object_schema(
            {
                "target": _str("Target column or ref."),
                "predictors": _string_array("Optional predictor columns or refs."),
                "method": _str("Model method, for example linear, ridge, logistic, rf, or gbrt."),
                "cvFolds": {"type": "integer", "minimum": 2, "default": 5},
                "task": {"type": "string", "enum": ["regression", "classification"], "default": "regression"},
            },
            required=("target",),
        ),
    }
    return schemas.get(name, _object_schema({}))


def _object_schema(
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _str(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _string_array(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
    }


def _filters_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": _object_schema(
            {
                "column": _str("Column ref."),
                "op": {"type": "string", "enum": ["=", "!=", "<>", "<", "<=", ">", ">="]},
                "value": _str("Literal value."),
            },
            required=("column", "op", "value"),
        ),
    }
