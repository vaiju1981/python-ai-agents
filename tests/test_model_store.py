"""Tests for the model store (train-once caching + retraining triggers)."""

from __future__ import annotations

from demos.analytics.src.analytics.model_store import (
    FileModelStore,
    InMemoryModelStore,
    ModelRecord,
    model_key,
)


def test_model_key_deterministic_and_order_insensitive() -> None:
    base = dict(task="regression", target="y", algorithm="rf")
    k1 = model_key(dataset_sig="d1", predictors=["a", "b"], **base)
    k2 = model_key(dataset_sig="d1", predictors=["b", "a"], **base)  # order flipped
    k3 = model_key(dataset_sig="d2", predictors=["a", "b"], **base)  # different data
    assert k1 == k2  # predictor order doesn't change the key
    assert k1 != k3  # a data change yields a new key → retrain


def test_inmemory_round_trip_and_ttl() -> None:
    store = InMemoryModelStore()
    store.put(ModelRecord(key="k", model={"w": 1}, metadata={"cv_r2": 0.9}, trained_at=1000.0))

    got = store.get("k")
    assert got is not None and got.metadata["cv_r2"] == 0.9
    # trained_at is far in the past, so any small TTL treats it as stale → miss.
    assert store.get("k", max_age=1.0) is None


def test_file_store_persists_across_instances(tmp_path) -> None:
    FileModelStore(tmp_path / "models").put(
        ModelRecord(key="abc", model=[1, 2, 3], metadata={"m": 1}, trained_at=1.0)
    )
    reopened = FileModelStore(tmp_path / "models")  # fresh instance, same dir
    got = reopened.get("abc")
    assert got is not None and got.model == [1, 2, 3]
    assert reopened.get("missing") is None
