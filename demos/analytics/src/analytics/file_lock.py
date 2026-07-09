"""Cross-process file locking for the JSON/pickle-backed stores.

The analytics engine persists some state to local files (model cache, decision
governance log). When multiple worker processes share a directory (multi-worker
app server, serverless replicas), in-process locks are not enough — concurrent
read-modify-write can corrupt or clobber files. This module provides a portable
exclusive lock built on POSIX ``flock`` over a sidecar ``.lock`` file, plus
atomic write helpers (write to ``.tmp`` then ``os.replace``).

Windows has no ``flock``; on that platform we fall back to an in-process
``threading.Lock`` only (no cross-process safety) and emit a clear warning so
the limitation is visible rather than silent.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from pathlib import Path

_FALLBACK_LOCK = threading.Lock()


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive cross-process lock keyed to ``path``.

    Uses a sidecar ``<path>.lock`` file so the lock never collides with the
    data file and survives atomic replacements of the data file.
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows only
        import warnings

        warnings.warn(
            "fcntl unavailable: file stores use an in-process lock only and are "
            "NOT safe for multiple processes (use a real DB backend on Windows).",
            stacklevel=2,
        )
        with _FALLBACK_LOCK:
            yield
        return

    with lock_path.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            except OSError:
                pass


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)
