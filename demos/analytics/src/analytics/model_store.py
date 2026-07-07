"""Model persistence + train-once caching for the analytics tools.

A trained model is stored under a deterministic **model key**, so repeated queries
reuse a fit instead of retraining. Retraining happens when the key changes (a new
dataset signature, target, predictors, or algorithm), when a cached model exceeds
its TTL, or when a caller forces it. See ``docs/MODEL_LIFECYCLE.md``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelRecord:
    key: str
    model: Any
    metadata: dict[str, Any]
    trained_at: float  # epoch seconds

    def age(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.trained_at


def model_key(
    *,
    dataset_sig: str,
    task: str,
    target: str,
    predictors: list[str],
    algorithm: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Deterministic key for a model. Same inputs → same key → cache hit."""
    payload = {
        "dataset_sig": dataset_sig,
        "task": task,
        "target": target,
        "predictors": sorted(predictors),
        "algorithm": algorithm,
        "params": params or {},
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ModelStore(Protocol):
    def get(self, key: str, *, max_age: float | None = None) -> ModelRecord | None: ...

    def put(self, record: ModelRecord) -> None: ...


@dataclass(slots=True)
class InMemoryModelStore:
    """Process-local cache. Good for a single session; lost on restart."""

    _store: dict[str, ModelRecord] = field(default_factory=dict)

    def get(self, key: str, *, max_age: float | None = None) -> ModelRecord | None:
        record = self._store.get(key)
        if record is None:
            return None
        if max_age is not None and record.age() > max_age:
            return None
        return record

    def put(self, record: ModelRecord) -> None:
        self._store[record.key] = record


@dataclass(slots=True)
class FileModelStore:
    """Pickle each model to ``<directory>/<key>.pkl``; survives across restarts if
    the directory is stable."""

    directory: Path

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, *, max_age: float | None = None) -> ModelRecord | None:
        path = self.directory / f"{key}.pkl"
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                record: ModelRecord = pickle.load(fh)
        except Exception:
            return None
        if max_age is not None and record.age() > max_age:
            return None
        return record

    def put(self, record: ModelRecord) -> None:
        path = self.directory / f"{record.key}.pkl"
        with path.open("wb") as fh:
            pickle.dump(record, fh)
