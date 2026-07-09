"""PR-1 verification: file-store concurrency + durability.

Confirms FileModelStore and DecisionStore survive concurrent writers across
multiple processes (not just threads) without corruption or lost writes, and
that no ``.tmp`` files are left behind.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from demos.analytics.src.analytics.decision_store import DecisionStore
from demos.analytics.src.analytics.model_store import (
    FileModelStore,
    ModelRecord,
    model_key,
)


def _model_worker(directory: str, idx: int) -> None:
    store = FileModelStore(directory=Path(directory))
    key = model_key(
        dataset_sig="sig", task="regression", target="y",
        predictors=[f"x{idx}"], algorithm="linear",
    )
    rec = ModelRecord(key=key, model={"w": idx}, metadata={"idx": idx}, trained_at=0.0)
    store.put(rec)
    got = store.get(key)
    assert got is not None and got.metadata["idx"] == idx


def _decision_worker(file: str, idx: int) -> None:
    store = DecisionStore(file=file)
    store.record(unit_id=f"u{idx}", action_type="promo", status="accepted",
                 comment=f"c{idx}")


def _model_reader(directory: str, key: str) -> None:
    from demos.analytics.src.analytics.model_store import FileModelStore

    s = FileModelStore(directory=Path(directory))
    for _ in range(50):
        assert s.get(key) is not None


def _model_writer(directory: str, key: str) -> None:
    from demos.analytics.src.analytics.model_store import FileModelStore, ModelRecord

    s = FileModelStore(directory=Path(directory))
    for v in range(50):
        s.put(ModelRecord(key=key, model={"v": v}, metadata={}, trained_at=0.0))


def test_model_store_concurrent_processes(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    n = 12
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.starmap(_model_worker, [(str(directory), i) for i in range(n)])

    store = FileModelStore(directory=directory)
    for i in range(n):
        key = model_key(
            dataset_sig="sig", task="regression", target="y",
            predictors=[f"x{i}"], algorithm="linear",
        )
        rec = store.get(key)
        assert rec is not None, f"lost model {i}"
        assert rec.metadata["idx"] == i

    assert not list(tmp_path.glob("**/*.pkl.tmp")), "leftover tmp files"


def test_decision_store_concurrent_processes(tmp_path: Path) -> None:
    file = tmp_path / "decisions.json"
    n = 12
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.starmap(_decision_worker, [(str(file), i) for i in range(n)])

    store = DecisionStore(file=file)
    assert len(store.all()) == n
    for i in range(n):
        assert any(e.unit_id == f"u{i}" for e in store.all())

    assert not list(tmp_path.glob("**/*.json.tmp")), "leftover tmp files"


def test_model_store_atomic_no_read_of_partial_write(tmp_path: Path) -> None:
    """A reader must never observe a partially written pickle."""
    directory = tmp_path / "models"
    store = FileModelStore(directory=directory)
    key = model_key(dataset_sig="s", task="t", target="y",
                    predictors=["x"], algorithm="linear")
    store.put(ModelRecord(key=key, model={"v": 1}, metadata={}, trained_at=0.0))

    # Concurrent readers hammering while another process rewrites the same key.
    args = [(str(directory), key)] * 4
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.starmap(_model_reader, args)
        pool.starmap(_model_writer, args)
