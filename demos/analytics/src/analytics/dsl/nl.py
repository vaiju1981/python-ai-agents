"""NL -> DSL bridge via entity extraction (PR-D5, stretch).

A textual DSL is only as usable as the caller's ability to write it. ``nlp_api``
already extracts ``Metric/Dimension/Period/Operation/Denom`` entities (via
Dialogflow + local n-gram/edit-distance/pattern matching) and grounds them with
``NamingConventions``. We reuse that *pattern* locally (no Dialogflow dependency)
to turn a natural-language question into DSL text, then let PR-D4 execute it.
This closes the loop: NL -> entities -> DSL -> SQL -> answer.

Detectordesign (registry pattern, like ``nlp_api``'s
``DIMENSION_VALUES_DETECTION_MAPPER``):

* ``LocalEntityDetector`` -- n-gram + edit-distance matching over the engine's
  catalog / synonym vocabulary (borrowed from ``nlp_api``'s local string
  matching). No external dependency; runs everywhere.
* ``LLMEntityDetector`` -- optional pluggable detector that prompts an LLM to
  return structured entities. The LLM lives *outside* the execution path; it
  only proposes DSL. Gated behind ``PAA_RUN_OLLAMA_TESTS`` so it never runs in
  CI by default.

``nl_to_dsl`` degrades gracefully: if no entity is understood it raises a scoped
"could not understand" error rather than emitting a wrong query.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from demos.analytics.src.analytics.dsl.ast import DslFilter, DslQuery
from demos.analytics.src.analytics.dsl.catalog import MetricCatalog
from demos.analytics.src.analytics.dsl.grounding import NameResolver
from demos.analytics.src.analytics.semantic_model import SemanticModel


class NLDetectError(ValueError):
    """Raised when no usable entity could be extracted from the NL text."""


@dataclass
class Entities:
    """Structured entities extracted from a natural-language question."""

    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    filters: list[DslFilter] = field(default_factory=list)
    last_days: int | None = None
    between_start: str | None = None
    between_end: str | None = None
    order_by: str | None = None
    descending: bool = True
    limit: int | None = None


class EntityDetector(Protocol):
    """A pluggable NL entity detector."""

    def detect(self, text: str, engine: Any) -> Entities:
        ...


_PERIOD_RE = re.compile(
    r"last\s+(\d+)\s+(day|week|month)s?", re.IGNORECASE
)
_PERIOD_UNIT_RE = re.compile(r"last\s+(month|week|year)\b", re.IGNORECASE)
_YTD_RE = re.compile(r"\bytd\b", re.IGNORECASE)
_SINCE_DATE_RE = re.compile(r"since\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_BETWEEN_RE = re.compile(
    r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)
_FILTER_RE = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is|are|equals|=)\s+([a-zA-Z0-9_]+)", re.IGNORECASE
)


class LocalEntityDetector:
    """Local string-matching detector over the engine's known vocabulary."""

    def detect(self, text: str, engine: Any) -> Entities:
        model: SemanticModel = engine.model
        resolver: NameResolver = engine.resolver
        catalog: MetricCatalog | None = engine.catalog

        words = _tokenize(text)

        # ---- metrics & dimensions: classify each word via the resolver -----
        # A word is a metric if it grounds to one, else a dimension if it grounds
        # to one. This avoids the synonym-collision problem (e.g. ``region`` is a
        # dimension synonym, not a metric) and reuses PR-D3 grounding directly.
        metrics: list[str] = []
        dimensions: list[str] = []
        for w in words:
            try:
                resolver.resolve_metric(w, catalog, model)
                metrics.append(w)
                continue
            except Exception:
                pass
            try:
                resolver.resolve_dimension(w, model)
                dimensions.append(w)
            except Exception:
                pass

        # ---- filters: "<dimension> is/=/equals <value>" -------------------
        filters: list[DslFilter] = []
        for fm in _FILTER_RE.finditer(text):
            dim_tok, val = fm.group(1), fm.group(2)
            try:
                resolver.resolve_dimension(dim_tok, model)
            except Exception:
                continue
            filters.append(DslFilter(column=dim_tok, op="=", value=val))

        # ---- period --------------------------------------------------------
        ent = Entities(metrics=metrics, dimensions=dimensions, filters=filters)
        m = _PERIOD_RE.search(text)
        if m:
            n = int(m.group(1))
            ent.last_days = n * {"day": 1, "week": 7, "month": 30}[m.group(2).lower()]
        else:
            u = _PERIOD_UNIT_RE.search(text)
            if u:
                ent.last_days = {"month": 30, "week": 7, "year": 365}[u.group(1).lower()]
        if ent.last_days is None and _YTD_RE.search(text):
            ent.last_days = 365
        s = _SINCE_DATE_RE.search(text)
        if s:
            ent.between_start = s.group(1)
            ent.between_end = _today()
        b = _BETWEEN_RE.search(text)
        if b:
            ent.between_start, ent.between_end = b.group(1), b.group(2)

        if not (ent.metrics or ent.dimensions):
            raise NLDetectError(
                f"could not understand any metric or dimension in: {text!r}"
            )
        return ent


class LLMEntityDetector:
    """Optional LLM-backed detector (pluggable). Not run by default.

    Expects an injected ``callable`` that takes the NL text and returns a dict of
    entities. The LLM is intentionally *outside* the execution path.
    """

    def __init__(self, call: Any = None) -> None:
        self._call = call

    def detect(self, text: str, engine: Any) -> Entities:
        if self._call is None:
            raise NLDetectError("LLM detector not configured (no callable supplied)")
        raw = self._call(text)
        return Entities(
            metrics=list(raw.get("metrics", [])),
            dimensions=list(raw.get("dimensions", [])),
            filters=[DslFilter(**f) for f in raw.get("filters", [])],
            last_days=raw.get("last_days"),
            between_start=raw.get("between_start"),
            between_end=raw.get("between_end"),
        )


class OllamaEntityDetector:
    """LLM entity detector backed by a local Ollama server.

    Sends the question plus the engine's known vocabulary (metrics / dimensions /
    synonyms) to Ollama and parses the JSON entity set it returns. The model is
    intentionally *outside* the execution path: it only *proposes* the DSL, which
    ``nl_to_dsl`` then hands to the engine for grounding + safe planning.

    Defaults to ``ornith:latest`` (override via ``PAA_OLLAMA_MODEL`` or the
    ``model`` argument); ``gemma4:31b-cloud`` works equally well. Gated behind
    ``PAA_RUN_OLLAMA_TESTS=1`` in tests so CI never requires a running model.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("PAA_OLLAMA_MODEL", "ornith:latest")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def detect(self, text: str, engine: Any) -> Entities:
        prompt = _ollama_prompt(text, engine)
        raw = self._generate(prompt)
        data = _extract_json(raw)
        return Entities(
            metrics=list(data.get("metrics", [])),
            dimensions=list(data.get("dimensions", [])),
            filters=[DslFilter(**f) for f in data.get("filters", [])],
            last_days=data.get("last_days"),
            between_start=data.get("between_start"),
            between_end=data.get("between_end"),
        )

    def _generate(self, prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body.get("response", "")


def _ollama_prompt(text: str, engine: Any) -> str:
    """Build a self-contained prompt: instructions + the engine's vocabulary."""
    model: SemanticModel = engine.model
    resolver: NameResolver = engine.resolver
    metrics = sorted({m.ref for m in model.metrics} | set(resolver.synonyms.keys()))
    dims = sorted({d.ref for d in model.dimensions})
    vocab = json.dumps({"metrics": metrics, "dimensions": dims, "synonyms": resolver.synonyms})
    return (
        "You are an analytics query parser. Given a natural-language question, "
        "return a single JSON object with these keys:\n"
        "  metrics: list of metric refs chosen ONLY from the vocabulary\n"
        "  dimensions: list of dimension refs chosen ONLY from the vocabulary\n"
        "  filters: list of {column, op, value} (op in =, !=, <, <=, >, >=, IN)\n"
        "  last_days: integer or null\n"
        "  between_start: date 'YYYY-MM-DD' or null\n"
        "  between_end: date 'YYYY-MM-DD' or null\n"
        "Use only names from the provided vocabulary. Respond with JSON only.\n"
        f"Vocabulary: {vocab}\n"
        f"Question: {text}\n"
    )


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response (tolerates markdown)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise NLDetectError(f"LLM did not return JSON: {raw!r}")
    return json.loads(m.group(0))


def nl_to_dsl(text: str, engine: Any, detector: EntityDetector | None = None) -> str:
    """Extract entities from ``text`` and emit DSL text for PR-D4 to execute.

    The LLM (if used) only *proposes* the DSL; execution is delegated to the
    engine. Raises ``NLDetectError`` if nothing usable was understood.
    """
    if detector is None:
        detector = LocalEntityDetector()
    ent = detector.detect(text, engine)
    query = DslQuery(
        metrics=tuple(ent.metrics),
        dimensions=tuple(ent.dimensions),
        filters=tuple(ent.filters),
        last_days=ent.last_days,
        between_start=ent.between_start,
        between_end=ent.between_end,
        order_by=ent.order_by,
        descending=ent.descending,
        limit=ent.limit,
    )
    return query.to_text()


# -- helpers ----------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text) if t]


def _today() -> str:
    from datetime import date

    return date.today().isoformat()