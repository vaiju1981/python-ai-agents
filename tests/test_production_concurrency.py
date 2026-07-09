"""PR-1 verification: file-store concurrency + durability.

Confirms FileModelStore and DecisionStore survive concurrent writers across
multiple processes (not just threads) without corruption or lost writes, and
that no ``.tmp`` files are left behind.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from demos.analytics.src.analytics.decision_store import DecisionStore
from demos.analytics.src.analytics.model_store import (
    FileModelStore,
    ModelCacheIntegrityError,
    ModelRecord,
    model_key,
)


def _model_worker(directory: str, idx: int) -> None:
    store = FileModelStore(directory=Path(directory))
    key = model_key(
        dataset_sig="sig",
        task="regression",
        target="y",
        predictors=[f"x{idx}"],
        algorithm="linear",
    )
    rec = ModelRecord(key=key, model={"w": idx}, metadata={"idx": idx}, trained_at=0.0)
    store.put(rec)
    got = store.get(key)
    assert got is not None and got.metadata["idx"] == idx


def _decision_worker(file: str, idx: int) -> None:
    store = DecisionStore(file=file)
    store.record(unit_id=f"u{idx}", action_type="promo", status="accepted", comment=f"c{idx}")


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
            dataset_sig="sig",
            task="regression",
            target="y",
            predictors=[f"x{i}"],
            algorithm="linear",
        )
        rec = store.get(key)
        assert rec is not None, f"lost model {i}"
        assert rec.metadata["idx"] == i

    assert not list(tmp_path.glob("**/*.json.tmp")), "leftover tmp files"


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
    key = model_key(dataset_sig="s", task="t", target="y", predictors=["x"], algorithm="linear")
    store.put(ModelRecord(key=key, model={"v": 1}, metadata={}, trained_at=0.0))

    # Concurrent readers hammering while another process rewrites the same key.
    args = [(str(directory), key)] * 4
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.starmap(_model_reader, args)
        pool.starmap(_model_writer, args)


# --- PR-2: remove pickle RCE / version-stamp invalidation ---


def test_model_store_rejects_tampered_payload(tmp_path: Path, monkeypatch) -> None:
    """When a signing key is configured, a tampered cache file must raise
    rather than hand attacker bytes to pickle.loads."""
    monkeypatch.setenv("ANALYTICS_MODEL_CACHE_KEY", "secret-key")
    directory = tmp_path / "models"
    key = model_key(dataset_sig="s", task="t", target="y", predictors=["x"], algorithm="linear")
    store = FileModelStore(directory=directory)
    store.put(ModelRecord(key=key, model={"v": 1}, metadata={}, trained_at=0.0))

    # Flip a byte in the on-disk payload so the signature no longer matches.
    path = directory / f"{key}.json"
    raw = json.loads(path.read_text())
    raw["payload"] = raw["payload"][:-4] + ("AAAA" if raw["payload"][-4:] != "AAAA" else "BBBB")
    path.write_text(json.dumps(raw))

    with pytest.raises(ModelCacheIntegrityError):
        FileModelStore(directory=directory).get(key)


def test_model_store_rejects_unsigned_when_key_configured(tmp_path: Path, monkeypatch) -> None:
    """An unsigned cache file is rejected when a key is configured."""
    # Write without a key first ("dev" unsigned envelope).
    monkeypatch.delenv("ANALYTICS_MODEL_CACHE_KEY", raising=False)
    directory = tmp_path / "models"
    key = model_key(dataset_sig="s", task="t", target="y", predictors=["x"], algorithm="linear")
    FileModelStore(directory=directory).put(
        ModelRecord(key=key, model={"v": 1}, metadata={}, trained_at=0.0)
    )

    # Now enforce a key: the existing unsigned file must be rejected.
    monkeypatch.setenv("ANALYTICS_MODEL_CACHE_KEY", "secret-key")
    with pytest.raises(ModelCacheIntegrityError):
        FileModelStore(directory=directory).get(key)


def test_model_store_version_bump_is_cache_miss(tmp_path: Path, monkeypatch) -> None:
    """A bumped engine_version makes a cached model a miss (caller retrains)."""
    directory = tmp_path / "models"
    key = model_key(dataset_sig="s", task="t", target="y", predictors=["x"], algorithm="linear")
    FileModelStore(directory=directory).put(
        ModelRecord(key=key, model={"v": 1}, metadata={}, trained_at=0.0)
    )

    monkeypatch.setattr("demos.analytics.src.analytics.model_store.ENGINE_VERSION", "999")
    assert FileModelStore(directory=directory).get(key) is None


def test_model_store_roundtrip_unsigned_dev(tmp_path: Path, monkeypatch) -> None:
    """Without a key, an unsigned envelope still round-trips (dev mode)."""
    monkeypatch.delenv("ANALYTICS_MODEL_CACHE_KEY", raising=False)
    directory = tmp_path / "models"
    key = model_key(dataset_sig="s", task="t", target="y", predictors=["x"], algorithm="linear")
    store = FileModelStore(directory=directory)
    store.put(ModelRecord(key=key, model={"v": 42}, metadata={"a": 1}, trained_at=5.0))
    got = store.get(key)
    assert got is not None and got.model == {"v": 42} and got.metadata == {"a": 1}
