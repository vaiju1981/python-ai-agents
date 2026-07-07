from __future__ import annotations

from pathlib import Path


class Workspace:
    """A per-run artifact workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("workspace paths must be relative and stay inside the workspace")
        return self.root / name

    def write_text(self, name: str, content: str) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read_text(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")
