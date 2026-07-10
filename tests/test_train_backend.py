"""PR-10 verification: out-of-core / distributed row-level ML (§G G3).

Adds a pluggable training backend (PR-10) so row-level models can train on the
full population (or out-of-core via dask) instead of the bounded reservoir
sample, with no behavior change when no backend is configured.
"""

from __future__ import annotations

import numpy as np
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.train_backend import (
    BoundedTrainBackend,
    DaskBackend,
    FullPopulationBackend,
    IncrementalTrainBackend,
    get_train_backend,
)

pytest.importorskip("duckdb")
pytest.importorskip("sklearn")


def _train_meta(ts: ModelsToolset, target: str = "y") -> dict:
    """Train (or load) the model and return its metadata without the size-capped
    JSON envelope, so large ``train_stats`` samples don't get truncated."""
    _t, _c, _p, _m, meta, _cached, _trained_at = ts._train_or_load({"target": target})
    return meta


def _make_csv(tmp_path, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-10, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.7 + rng.normal(0, 0.5, n)
    csv = tmp_path / "facts.csv"
    rows = "\n".join(f"{a},{b},{c}" for a, b, c in zip(x1, x2, y, strict=False))
    csv.write_text("x1,x2,y\n" + rows + "\n")
    return csv


def _toolset(tmp_path, n, *, backend=None, max_train_rows=None):
    csv = _make_csv(tmp_path, n)
    src = CsvSource(named_csvs={"facts": csv})
    model = SemanticModel.from_profile(profile_dataset(src))
    return ModelsToolset(src, model, max_train_rows=max_train_rows, train_backend=backend)


def test_full_backend_trains_on_full_frame_and_improves(tmp_path):
    # Population well above the reservoir cap; strong signal so more rows help.
    n = 6000
    cap = 400

    bounded = _toolset(tmp_path, n, backend=BoundedTrainBackend(), max_train_rows=cap)
    full = _toolset(tmp_path, n, backend=FullPopulationBackend(), max_train_rows=cap)

    b = _train_meta(bounded)
    f = _train_meta(full)

    # Bounded trains on the reservoir sample; full trains on every row.
    assert b["n_rows"] <= cap
    assert f["n_rows"] == n
    assert "full population" in f["method"]
    # Training on the full population yields a better (or equal) fit.
    assert f["cv_r2"] > b["cv_r2"]


def test_default_backend_unchanged_bounded_sample(tmp_path):
    # No backend configured -> default bounded behavior, no full-population training.
    n = 6000
    cap = 400
    ts = _toolset(tmp_path, n, max_train_rows=cap)  # default backend
    assert ts.train_backend.name == "bounded"

    res = _train_meta(ts)
    assert res["n_rows"] <= cap
    assert "full population" not in res["method"]


def test_incremental_backend_trains_full_frame(tmp_path):
    n = 3000
    ts = _toolset(tmp_path, n, backend=IncrementalTrainBackend(), max_train_rows=200)
    res = _train_meta(ts)
    assert res["n_rows"] == n
    assert "incremental" in res["method"]


def test_get_train_backend_resolution_and_unknown():
    assert get_train_backend("bounded").name == "bounded"
    assert get_train_backend("full").name == "full"
    assert get_train_backend("incremental").name == "incremental"
    # Env default is bounded when unset.
    import os

    os.environ.pop("ANALYTICS_TRAIN_BACKEND", None)
    assert get_train_backend().name == "bounded"
    with pytest.raises(ValueError):
        get_train_backend("spark")


def test_dask_backend_requires_optional_deps():
    import importlib.util

    have_dask = importlib.util.find_spec("dask") is not None
    if have_dask:
        pytest.skip("dask is installed; cannot assert the missing-dep error")
    with pytest.raises(RuntimeError):
        DaskBackend()
    with pytest.raises(RuntimeError):
        get_train_backend("dask")
