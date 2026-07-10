"""Generic, source-agnostic analytics DSL engine.

Turns a compact textual query language (or a natural-language question) into
correct SQL executed against any ``DataSource``. Building blocks:

* calculated-metric catalog (semantic layer)  -- :mod:`.catalog`
* textual DSL parser -> query IR               -- :mod:`.parser`, :mod:`.ast`
* name grounding / synonym resolver            -- :mod:`.grounding`
* engine orchestrator + planner integration    -- :mod:`.engine`
* NL -> DSL bridge (stretch, local detector)   -- :mod:`.nl`
"""

from demos.analytics.src.analytics.dsl.ast import DslFilter, DslQuery
from demos.analytics.src.analytics.dsl.catalog import (
    BaseMetricDef,
    CalculatedMetricDef,
    CatalogError,
    CatalogStore,
    MetricCatalog,
    MetricDef,
    catalog_for_source,
)
from demos.analytics.src.analytics.dsl.engine import DslEngine, DslResult
from demos.analytics.src.analytics.dsl.grounding import NameResolver, UnresolvedNameError
from demos.analytics.src.analytics.dsl.nl import (
    LLMEntityDetector,
    LocalEntityDetector,
    NLDetectError,
    OllamaEntityDetector,
    nl_to_dsl,
)
from demos.analytics.src.analytics.dsl.parser import DslParseError, parse

__all__ = [
    "DslQuery",
    "DslFilter",
    "MetricCatalog",
    "MetricDef",
    "BaseMetricDef",
    "CalculatedMetricDef",
    "CatalogError",
    "CatalogStore",
    "catalog_for_source",
    "NameResolver",
    "UnresolvedNameError",
    "DslEngine",
    "DslResult",
    "parse",
    "DslParseError",
    "LocalEntityDetector",
    "LLMEntityDetector",
    "OllamaEntityDetector",
    "nl_to_dsl",
    "NLDetectError",
]
