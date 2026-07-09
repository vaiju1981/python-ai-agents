"""Model persistence + train-once caching for the analytics tools.

A trained model is stored under a deterministic **model key**, so repeated queries
reuse a fit instead of retraining. Retraining happens when the key changes (a new
dataset signature, target, predictors, or algorithm), when a cached model exceeds
its TTL, or when a caller forces it. See ``docs/MODEL_LIFECYCLE.md``.

**Security / durability (PR-2):**
Cached models are written as a versioned JSON envelope. The fitted estimator
bytes are pickled, but ``pickle`` is *only ever* invoked after the envelope has
been integrity-checked:
  - ``engine_version`` + library versions are stamped on write; a mismatch makes
    the entry a silent cache miss (no stale/foreign-format load).
  - when ``ANALYTICS_MODEL_CACHE_KEY`` is set, every envelope carries an HMAC
    signature over its contents; a missing/invalid signature is *rejected* with a
    clear error and the bytes are never handed to ``pickle.loads``. This removes
    the deserialization RCE risk of a shared/writable cache directory.
With no key configured (local/dev), unsigned envelopes are still loaded, but the
version stamp still protects against format drift.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from demos.analytics.src.analytics.file_lock import atomic_write_bytes, file_lock

# Bumped on any incompatible change to the on-disk model envelope or the
# in-memory ModelRecord shape. A mismatch invalidates caches rather than
# mis-loading a stale format.
ENGINE_VERSION = "1"

# Environment variable holding the HMAC key used to authenticate cache files.
# When set, unsigned or tampered cache files are rejected instead of loaded.
CACHE_KEY_ENV = "ANALYTICS_MODEL_CACHE_KEY"
ENVELOPE_FORMAT = "analytics-model-v1"


class ModelCacheIntegrityError(Exception):
    """Raised when a cache file fails its signature/version check."""


def _cache_key() -> bytes | None:
    raw = os.environ.get(CACHE_KEY_ENV)
    if not raw:
        return None
    return raw.encode("utf-8")


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("sklearn", "numpy", "scipy", "pandas"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return versions


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


def _sign(material: bytes, key: bytes) -> str:
    return hmac.new(key, material, hashlib.sha256).hexdigest()


def _serialize_record(record: ModelRecord, key: bytes | None) -> bytes:
    """Build the signed, versioned JSON envelope (bytes for atomic write)."""
    model_bytes = pickle.dumps(record.model)
    payload_b64 = base64.b64encode(model_bytes).decode("ascii")
    envelope: dict[str, Any] = {
        "format": ENVELOPE_FORMAT,
        "engine_version": ENGINE_VERSION,
        "lib_versions": _lib_versions(),
        "metadata": record.metadata,
        "trained_at": record.trained_at,
        "payload": payload_b64,
        "signature": None,
    }
    if key is not None:
        # Signature covers everything except the signature field itself.
        material = json.dumps(envelope, sort_keys=True, default=str).encode("utf-8")
        envelope["signature"] = _sign(material, key)
    return json.dumps(envelope, sort_keys=True, default=str).encode("utf-8")


def _deserialize_record(data: bytes, key: bytes | None) -> ModelRecord | None:
    """Verify an envelope and return the ModelRecord, or ``None`` on a benign
    cache miss (format/version mismatch). Raises ``ModelCacheIntegrityError``
    on tampering when a key is configured."""
    try:
        envelope = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if envelope.get("format") != ENVELOPE_FORMAT:
        return None
    if envelope.get("engine_version") != ENGINE_VERSION:
        # Format/version drift: treat as a miss so the caller retrains rather
        # than loading a model serialized under a different contract.
        return None

    signature = envelope.get("signature")
    if key is not None:
        if not signature:
            raise ModelCacheIntegrityError("cache file is unsigned but a signing key is configured")
        check = dict(envelope)
        check["signature"] = None
        material = json.dumps(check, sort_keys=True, default=str).encode("utf-8")
        if not hmac.compare_digest(_sign(material, key), signature):
            raise ModelCacheIntegrityError("cache file signature does not match")
    elif signature:
        # A signed file loaded in an unsigned (dev) context: trust it, but note
        # that integrity was not verified here.
        warnings.warn(
            "cache file carries a signature but no signing key is configured; "
            "integrity was NOT verified. Set ANALYTICS_MODEL_CACHE_KEY to enforce.",
            stacklevel=2,
        )

    try:
        model_bytes = base64.b64decode(envelope["payload"])
        model = pickle.loads(model_bytes)  # only reached after the checks above
    except Exception:
        return None
    return ModelRecord(
        key="",  # filled in by the caller
        model=model,
        metadata=envelope.get("metadata", {}),
        trained_at=envelope.get("trained_at", 0.0),
    )


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
    """Versioned, signature-checked JSON cache under ``<directory>/<key>.json``.

    Survives restarts when the directory is stable, and is safe to share across
    multiple worker processes (combined with the cross-process lock from
    ``file_lock``).
    """

    directory: Path
    signing_key: bytes | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.signing_key is None:
            self.signing_key = _cache_key()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str, *, max_age: float | None = None) -> ModelRecord | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with file_lock(path):
                data = path.read_bytes()
            record = _deserialize_record(data, self.signing_key)
        except ModelCacheIntegrityError:
            raise
        except Exception:
            return None
        if record is None:
            return None
        if max_age is not None and record.age() > max_age:
            return None
        return ModelRecord(
            key=key,
            model=record.model,
            metadata=record.metadata,
            trained_at=record.trained_at,
        )

    def put(self, record: ModelRecord) -> None:
        path = self._path(record.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize_record(record, self.signing_key)
        with file_lock(path):
            atomic_write_bytes(path, data)
