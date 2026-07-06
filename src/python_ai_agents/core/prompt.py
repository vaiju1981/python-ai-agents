"""Minimal dependency-free prompt template with ``{named}`` placeholders."""

from __future__ import annotations

import re

__all__ = ["PromptTemplate"]

_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.\-]+)\}")


class PromptTemplate:
    """A prompt template with ``{named}`` placeholders.

    ``render`` requires every placeholder to have a value; a missing one is a
    programming error. ``variables`` exposes the placeholder names.
    """

    def __init__(self, template: str) -> None:
        self.template = template
        self._variables: list[str] = []
        seen: set[str] = set()
        for match in _PLACEHOLDER.finditer(template):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                self._variables.append(name)

    @property
    def variables(self) -> list[str]:
        """Placeholder names in first-seen order."""
        return list(self._variables)

    def render(self, values: dict[str, str]) -> str:
        """Substitutes every placeholder; raises ``KeyError`` if any value is missing."""

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(f"missing value for placeholder '{key}'")
            return values[key]

        return _PLACEHOLDER.sub(replace, self.template)
