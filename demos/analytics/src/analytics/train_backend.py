"""Pluggable training backends for row-level models (PR-8/PR-10).

By default the engine trains tree/cluster models on a *reservoir sample*
(``ANALYTICS_MAX_TRAIN_ROWS``) so big datasets trade a little accuracy for
speed and bounded memory. PR-10 makes that choice pluggable:

- ``bounded`` (default) — train on the reservoir sample, current behavior.
- ``full`` — train on the *entire* population in memory (no cap) when it fits.
- ``incremental`` — train with partial-fit (SGD) estimators in batches, so
  out-of-core-ish streaming training is possible for the full population.
- ``dask`` — out-of-core training via ``dask-ml`` ``Incremental`` (optional
  dependency; raises a clear error if ``dask``/``dask-ml`` are not installed).

Select a backend with ``ANALYTICS_TRAIN_BACKEND`` or by passing an instance to
``ModelsToolset(..., train_backend=...)``. A backend that sets
``uses_full_population = True`` also tells the toolset to skip the row cap when
assembling the *training* frame (serving/predict still honors the cap for speed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class TrainResult:
    """What a backend returns from ``fit``."""

    model: Any
    feature_importance: list[dict[str, Any]] | None = None
    method: str = ""
    algorithm: str = ""
    cv: float | None = None


class TrainBackend(Protocol):
    """Port a training backend must satisfy."""

    name: str
    uses_full_population: bool

    def fit(
        self,
        *,
        y: Any,
        x: Any,
        feature_cols: list[str],
        is_clf: bool,
        algo: str,
        random_state: int = 0,
    ) -> TrainResult: ...


def _rank_importances(names: list[str], values: Any) -> list[dict[str, Any]]:
    return [
        {"feature": n, "importance": round(float(v), 4)}
        for n, v in sorted(zip(names, values, strict=False), key=lambda p: -abs(p[1]))
    ]


def _rf_estimator(is_clf: bool, random_state: int) -> Any:
    if is_clf:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=200, random_state=random_state)


def _rf_method(is_clf: bool) -> str:
    if is_clf:
        return "RandomForestClassifier, 5-fold CV accuracy on historical data"
    return "RandomForestRegressor, 5-fold CV R^2 on historical data"


def _encode_codes(y: Any) -> Any:
    from demos.analytics.src.analytics.models_tools import _encode

    codes, _uniques = _encode(y)
    return codes


class BoundedTrainBackend:
    """Default backend: train a RandomForest on the (already capped) sample."""

    name = "bounded"
    uses_full_population = False

    def fit(
        self,
        *,
        y: Any,
        x: Any,
        feature_cols: list[str],
        is_clf: bool,
        algo: str,
        random_state: int = 0,
    ) -> TrainResult:
        xv = x.astype(float).values
        model = _rf_estimator(is_clf, random_state)
        if is_clf:
            model.fit(xv, _encode_codes(y))
        else:
            model.fit(xv, y.astype(float).values)
        return TrainResult(
            model=model,
            feature_importance=_rank_importances(feature_cols, model.feature_importances_),
            method=_rf_method(is_clf),
            algorithm=algo,
        )


class FullPopulationBackend:
    """Train a RandomForest on the full population (no reservoir cap)."""

    name = "full"
    uses_full_population = True

    def fit(
        self,
        *,
        y: Any,
        x: Any,
        feature_cols: list[str],
        is_clf: bool,
        algo: str,
        random_state: int = 0,
    ) -> TrainResult:
        xv = x.astype(float).values
        model = _rf_estimator(is_clf, random_state)
        if is_clf:
            model.fit(xv, _encode_codes(y))
        else:
            model.fit(xv, y.astype(float).values)
        return TrainResult(
            model=model,
            feature_importance=_rank_importances(feature_cols, model.feature_importances_),
            method=_rf_method(is_clf) + " (full population)",
            algorithm=algo,
        )


class IncrementalTrainBackend:
    """Train with partial-fit (SGD) estimators in batches over the full frame.

    Demonstrates the incremental/partial-fit strategy PR-10 calls for: the model
    is updated batch-by-batch, so it can stream the full population without
    holding it all in one contiguous array during fitting.
    """

    name = "incremental"
    uses_full_population = True

    def __init__(self, batch_size: int = 2000) -> None:
        self.batch_size = batch_size

    def fit(
        self,
        *,
        y: Any,
        x: Any,
        feature_cols: list[str],
        is_clf: bool,
        algo: str,
        random_state: int = 0,
    ) -> TrainResult:
        from sklearn.linear_model import SGDClassifier, SGDRegressor
        from sklearn.preprocessing import StandardScaler

        xv = x.astype(float).values
        if is_clf:
            codes = _encode_codes(y)
            classes = sorted(set(int(c) for c in codes))
            est = SGDClassifier(
                random_state=random_state, loss="log_loss", alpha=1e-3, average=True
            )
            yv = codes
            method = "StandardScaler+SGDClassifier (incremental partial_fit) on full population"
        else:
            est = SGDRegressor(random_state=random_state, alpha=1e-3, average=True)
            yv = y.astype(float).values
            classes = None
            method = "StandardScaler+SGDRegressor (incremental partial_fit) on full population"
        # Scale once on the full frame (it is already materialized in memory for
        # the incremental backend), then stream partial_fit batches on the scaled
        # array so SGD stays numerically stable.
        scaler = StandardScaler().fit(xv)
        xs = scaler.transform(xv)
        # Online SGD can emit transient numeric warnings on the first bootstrap
        # batch / rare outlier rows; the model converges to sane weights, so we
        # ignore those benign floating-point warnings during training + CV.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if is_clf:
                est.partial_fit(xs[:1], yv[:1], classes=classes)
            else:
                # Center/scale the target so online SGD gradients don't explode.
                y_mean = float(np.mean(yv))
                y_std = float(np.std(yv)) or 1.0
                ys = (yv - y_mean) / y_std
                est.partial_fit(xs[:1], ys[:1])
            for start in range(0, len(xs), self.batch_size):
                xe = xs[start : start + self.batch_size]
                if is_clf:
                    ye = yv[start : start + self.batch_size]
                else:
                    ye = ys[start : start + self.batch_size]
                est.partial_fit(xe, ye)
        model = _ScaledIncrementalModel(est)
        model.scaler = scaler
        if not is_clf:
            model.y_mean = y_mean
            model.y_std = y_std
        # CV on the pre-scaled full frame (stable std) rather than letting the
        # caller re-scale per-fold (tiny per-fold std can blow SGD up).
        try:
            from sklearn.metrics import accuracy_score, r2_score
            from sklearn.model_selection import train_test_split

            xtr, xte, ytr, yte = train_test_split(
                xs, ys if not is_clf else yv, test_size=0.2, random_state=random_state
            )
            probe = (
                type(est)(**est.get_params()) if not is_clf else est.__class__(**est.get_params())
            )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                probe.fit(xtr, ytr)
                preds = probe.predict(xte)
            if is_clf:
                cv_mean = accuracy_score(yte, preds)
            else:
                cv_mean = r2_score(yte, preds)
            cv = round(float(cv_mean), 3) if cv_mean == cv_mean else None
        except Exception:
            cv = None
        return TrainResult(
            model=model, feature_importance=None, method=method, algorithm="sgd", cv=cv
        )


class _ScaledIncrementalModel:
    """Wrap ``StandardScaler`` + an SGD estimator so partial-fit training is
    numerically stable on raw feature magnitudes, while ``predict`` transparently
    re-scales incoming features. Compatible with the existing predict path and
    sklearn ``cross_val_score`` (implements fit/predict + clone params).
    """

    def __init__(self, estimator: Any) -> None:
        from sklearn.preprocessing import StandardScaler

        self.estimator = estimator
        self.scaler = StandardScaler()
        # Target scaling (regression only) so online SGD stays numerically stable;
        # predictions are unscaled back to the original target units.
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, X: Any, y: Any) -> _ScaledIncrementalModel:
        Xs = self.scaler.fit_transform(X)
        self.estimator.fit(Xs, y)
        return self

    def predict(self, X: Any) -> Any:
        preds = self.estimator.predict(self.scaler.transform(X))
        if self.y_std != 1.0 or self.y_mean != 0.0:
            preds = preds * self.y_std + self.y_mean
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {"estimator": self.estimator}

    def set_params(self, **params: Any) -> _ScaledIncrementalModel:
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self


class DaskBackend:
    """Out-of-core training via ``dask-ml`` ``Incremental`` (optional dependency)."""

    name = "dask"
    uses_full_population = True

    def __init__(self, chunksize: int = 2000) -> None:
        try:
            import dask  # noqa: F401
            import dask_ml  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "DaskBackend requires the optional 'dask' and 'dask-ml' packages; "
                "install them or choose another ANALYTICS_TRAIN_BACKEND"
            ) from exc
        self.chunksize = chunksize

    def fit(
        self,
        *,
        y: Any,
        x: Any,
        feature_cols: list[str],
        is_clf: bool,
        algo: str,
        random_state: int = 0,
    ) -> TrainResult:  # pragma: no cover - depends on optional dask deps
        import dask.array as da
        from dask_ml.linear_model import Incremental
        from sklearn.linear_model import SGDClassifier, SGDRegressor

        xv = da.from_array(x.astype(float).values, chunks=(self.chunksize, x.shape[1]))
        if is_clf:
            codes = _encode_codes(y)
            classes = sorted(set(int(c) for c in codes))
            est = SGDClassifier(random_state=random_state, loss="log_loss")
            yv = da.from_array(codes, chunks=self.chunksize)
            method = "dask-ml Incremental SGDClassifier (out-of-core partial_fit)"
        else:
            est = SGDRegressor(random_state=random_state)
            yv = da.from_array(y.astype(float).values, chunks=self.chunksize)
            classes = None
            method = "dask-ml Incremental SGDRegressor (out-of-core partial_fit)"
        inc = Incremental(est, random_state=random_state)
        if is_clf:
            inc.partial_fit(xv, yv, classes=classes)
        else:
            inc.partial_fit(xv, yv)
        return TrainResult(model=inc, feature_importance=None, method=method, algorithm="sgd")


def get_train_backend(name: str | None = None) -> TrainBackend:
    """Resolve a backend by name (or ``ANALYTICS_TRAIN_BACKEND``, default ``bounded``)."""
    name = (name or os.getenv("ANALYTICS_TRAIN_BACKEND", "bounded")).lower()
    if name == "bounded":
        return BoundedTrainBackend()
    if name == "full":
        return FullPopulationBackend()
    if name == "incremental":
        return IncrementalTrainBackend()
    if name == "dask":
        return DaskBackend()
    raise ValueError(f"unknown train backend '{name}'; supported: bounded, full, incremental, dask")
