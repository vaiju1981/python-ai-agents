"""Tokenizer seam and a zero-dependency heuristic estimator.

A ``Tokenizer`` estimates token counts for memory windowing and budgeting.
``HeuristicTokenizer`` uses the ~1-token-per-4-chars rule of thumb — good
enough for budgeting; swap in a provider tokenizer (e.g. ``tiktoken``) when
exact counts matter.
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["HeuristicTokenizer", "Tokenizer"]


class Tokenizer(Protocol):
    """Estimates the token count of a piece of text."""

    def count_tokens(self, text: str) -> int:
        ...


class HeuristicTokenizer:
    """Roughly one token per four characters (English-text rule of thumb)."""

    _CHARS_PER_TOKEN = 4

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + self._CHARS_PER_TOKEN - 1) // self._CHARS_PER_TOKEN
