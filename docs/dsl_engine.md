# Analytics DSL Engine — Implementation Plan

**Purpose:** A generic, source-agnostic **DSL engine** for the analytics engine
(`demos/analytics/src/analytics`). It lets a caller ask for **calculated
metrics** that do not physically exist in the source, by name, in a compact
textual query language, and get back correct SQL executed against *any*
`DataSource` (DuckDB today; warehouses via `warehouse_sources` already).

This is the Python analogue of two existing codebases we reviewed:

- **`api-multisite/.../ask/repo`** (Node.js) — a "CQL" DSL for Neo4j with a
  formula registry for derived/calculated metrics (`ADT = theoWin/visitCount`)
  and a `node-def` entity catalog. We borrow the *idea* of a named
  calculated-metric registry and a `d/m/g/p/s/l/c/r` query shape.
- **`nlp_api`** (Django) — Dialogflow + local string-matching NER that grounds
  `Metric/Dimension/Period/Operation` entities to schema via a
  `NamingConventions` triple-map and a `SOURCE_TYPES` capability matcher. We
  borrow the *name-grounding* and *capability-matching* ideas (not the
  Dialogflow dependency or the opaque external-API handoff).

We build directly on the existing Python engine rather than re-implementing it:
`SemanticModel`, `query_planner` (`QuerySpec`, `plan`, `JoinTree`,
`validate_plan`, inline `derivedMetrics`), `DataSource`/`warehouse_sources`,
`relationships` discovery, `dataset_fingerprint`. The gap is: (1) a **persistent
catalog of calculated metrics** referenced by name, (2) a **real textual DSL**
with a parser (the current `QuerySpec` is built programmatically only), and
(3) **name grounding / synonyms** so callers use friendly names.

**How to use this doc:** Implement one item, run its E2E gate, mark it ✅, commit
(referencing the item, e.g. `PR-D1`). Same cadence as `production_readiness.md`.

**Legend:** ✅ done · 🟡 in-progress · ⬜ planned

---

## Priority order

1. Calculated-metric catalog (semantic layer) (PR-D1)
2. Textual DSL parser → query IR (PR-D2)
3. Name grounding / synonym resolver (PR-D3)
4. DSL engine orchestrator + planner integration (PR-D4)
5. NL → DSL bridge via entity extraction (PR-D5) *(stretch)*

---

## PR-D1 — Calculated-metric catalog (semantic layer)  ✅ implemented

**Why:** Today a "calculated metric" only exists as an ad-hoc `derivedMetrics`
expression inside a single `QuerySpec` (`query_planner.py:58`). There is no
reusable, named, persisted metric like CQL's formula registry
(`api-multisite/.../lib/cypher-formulas/base-formulas.js`, e.g.
`ADT: ROUND(CASE WHEN visitCount=0 THEN 0 ELSE theoWin/visitCount END,2)`) or
`nlp_api`'s declared `METRICS`. Without a catalog, the same ratio is redefined
every query, computed metrics can't reference each other, and there's no single
source of truth for "what is average price".

**Implement (new module `dsl/catalog.py`):**
- `MetricDef` (Protocol/dataclass) with two concrete kinds:
  - `BaseMetricDef(table, column, aggregation, name)` — proxies a source column
    (delegates to `SemanticModel.metrics`).
  - `CalculatedMetricDef(name, expression, aggregation_hint=None,
    description=None)` — `expression` is a safe arithmetic/SQL-ish string that
    references *other metric names or `table.column` refs* (e.g.
    `"sales.amount / sales.quantity"`, or `"gross_margin"` where
    `gross_margin` is itself a calculated metric). Allow `CASE WHEN`/`NULLIF`
    for safe division-by-zero (mirrors CQL `ADT`).
- `MetricCatalog`:
  - `add(def)`, `get(name)`, `names()`, `to_dict()` / `from_dict()` (JSON on
    disk so it is reusable/persistable; atomic write via `file_lock`).
  - `resolve_sql(name, model) -> (sql_expr, alias)`: expands a calculated
    metric to its aggregated SQL form. Base metrics expand to
    `AGG("table"."col")`; calculated metrics expand their `expression` by
    substituting each referenced metric with its *aggregated* SQL (so ratios
    stay additive-safe — reuse `query_planner._expand_expr` logic, generalized
    to resolve by catalog name too, not only by `table.column`).
  - **Dependency resolution + cycle detection**: build the ref graph, topo-sort,
    and reject a metric that references itself (cycle) or an unknown metric with
    a scoped `CatalogError(name, reason)`.
  - Validate every referenced base column exists in the `SemanticModel` (delegates
    to `validate_plan`-style checks) so a broken catalog fails fast.
  - `override(other)` — customer/locale overrides, mirroring `nlp_api` catalog
    layering.
- **`dataset_sig`-keyed catalog selection (the "tailoring" mechanism):** the
  catalog is *not* global/hardcoded — it is selected per loaded dataset so the
  same generic engine auto-adapts to health, casino, retail, etc.
  - Reuse `dataset_fingerprint.fingerprint(source, row_count_aware=False)` to
    compute the dataset's `dataset_sig` (row-count-agnostic, so pure growth
    keeps the same catalog — consistent with PR-11).
  - A `CatalogStore` (or `MetricCatalog.for_source`) persists one catalog file
    per dataset under `<DSL_CATALOG_DIR>/<dataset_sig>.json` (atomic write via
    `file_lock`; created on first use and auto-seeded from the `SemanticModel`).
  - Selection order when building an engine for a source:
    1. the dataset-specific catalog (`<dataset_sig>.json`), if present;
    2. merged over a shared **base catalog** (common cross-domain calculated
       metrics + global synonyms), if configured;
    3. plus any per-customer `override` layer.
    So loading health data resolves the health-tailored catalog automatically;
    loading casino resolves the casino one — zero domain branching in the engine.
  - Base (source) metrics are always available by ref regardless of catalog, so
    even a brand-new dataset with no tailored catalog is fully queryable; the
    tailored catalog only *adds* calculated metrics / synonyms on top.

**Verify (E2E):** `tests/test_dsl_catalog.py`
- Define `avg_price = sales.amount / sales.quantity`; `catalog.resolve_sql`
  returns `SUM("sales"."amount")/SUM("sales"."quantity")` (additive-safe).
- Chained calculated metric: `gross = sales.amount - sales.cost`,
  `margin_pct = gross / sales.amount`; resolving `margin_pct` expands both
  levels and references only aggregations.
- `CASE WHEN`/`NULLIF` expression (`safe_ratio = NULLIF(sales.amount,0)/...`)
  round-trips without syntax error and is rejected if it references a non-metric
  column that isn't aggregatable.
- Cycle (`a = b`, `b = a`) → `CatalogError`; unknown ref → `CatalogError`.
- Persist to JSON and reload → identical catalog; override merges names.
- **`dataset_sig`-keyed selection:** two different datasets (health vs casino)
  yield two distinct catalog files under `<DSL_CATALOG_DIR>/<dataset_sig>.json`;
  `MetricCatalog.for_source` for the health source resolves the health catalog,
  for the casino source resolves the casino catalog, with no engine branching.
  A brand-new dataset with no tailored catalog still queries its base metrics by
  ref. Pure row growth (same `dataset_sig`) reuses the same catalog file.

---

## PR-D2 — Textual DSL parser → query IR  ✅ implemented

**Why:** The engine has no textual query language — `QuerySpec` is assembled in
Python only. `api-multisite` has CQL (`d/m/g/p/s/l/c/r`); `nlp_api` has an IR
dict but no real grammar. We want a small, parseable DSL (lexer + parser, no
external NLU dependency) so any caller (agent, UI, NL bridge) can express a query
as text and get a validated IR.

**Implement (new module `dsl/parser.py` + `dsl/ast.py`):**
- Define the language (CQL-shaped, explicit keywords, unambiguous):
  ```
  SELECT <metric>[, <metric> ...]
         [BY <dim>[, <dim> ...]]
         [WHERE <filter> [AND <filter> ...]]
         [SINCE <n> DAYS|WEEKS|MONTHS]
         [BETWEEN <date> AND <date>]
         [ORDER BY <metric|dim> [ASC|DESC]]
         [LIMIT <n>]
  ```
  - `<metric>` := a catalog name | a `table.column` base ref | an inline
    `expr AS alias` (already supported by `query_planner.derivedMetrics`).
  - `<filter>` := `<col|metric> <op> <value>` with `op` ∈
    `= != <> < <= > >= IN (a,b,...) LIKE 'x%'`.
- A real tokenizer + recursive-descent parser (no regex hack) producing a
  frozen `DslQuery` IR (metrics, dimensions, filters, time window, order, limit)
  that maps 1:1 onto `query_planner.QuerySpec` (+ carries inline derived metrics).
- Scoped parse errors (`DslParseError`) naming the offending token/clause.
- `DslQuery.to_spec(model, catalog)` → `QuerySpec`: resolves metric names via
  the catalog (PR-D1) and appends catalog-expanded expressions to
  `derivedMetrics`, resolves dimension/filter refs against the `SemanticModel`.

**Verify (E2E):** `tests/test_dsl_parser.py`
- Parse `SELECT sales.amount, sales.quantity BY sales.region WHERE sales.amount > 100 SINCE 30 DAYS ORDER BY sales.amount DESC LIMIT 10`
  → `DslQuery` with the right fields; `.to_spec` yields a `QuerySpec` whose
  `plan()` produces SQL matching the hand-built reference.
- Calculated metric by name: `SELECT avg_price BY sales.region` → spec carries
  the expanded `derivedMetrics` and `plan()` runs (PR-D1).
- Malformed input (`SELECT BY`, unknown `OP ~`, unbalanced `IN (...`) →
  `DslParseError` with the clause named.
- Round-trip: `DslQuery.to_text()` reproduces the normalized query.

---

## PR-D3 — Name grounding / synonym resolver  ✅ implemented

**Why:** Callers (and the NL bridge, PR-D5) use friendly names
(`revenue`, `region`, `last 30 days`), not `sales.amount` / `sales.region`.
`nlp_api` solves this with a `NamingConventions` triple-map (Dialogflow name ↔
internal name ↔ data-layer name) and `VALUE_SYNONYMS`. We want the same
grounding so the DSL accepts business vocabulary, with pluggable synonym sources
(env / JSON / per-customer override).

**Implement (new module `dsl/grounding.py`):**
- `NameResolver`:
  - `resolve_metric(token, catalog, model) -> MetricDef` and
    `resolve_dimension(token, model) -> Dimension` with layered matching:
    exact ref (`table.column`) → catalog name (case-insensitive) →
    synonym map → fuzzy (optional edit-distance, borrowed from
    `nlp_api` string_matching `editdistance`) behind a flag.
  - Synonym sources: built-in per-model (column name ↔ ref), a JSON/YAML
    `synonyms` file (`revenue: sales.amount`, `net win: sales.net_win`), and
    per-customer overrides (mirrors `nlp_api` settings layering).
  - `resolve_period(token) -> (last_days | (start,end))` for the `SINCE` /
    `BETWEEN` clauses (`30 days`, `last month`, `ytd`).
  - Scoped `UnresolvedNameError(name, kind)` when nothing matches.
- Wire into the PR-D2 parser: tokens are grounded *before* building the spec, so
  the DSL text can use business names throughout.
- `DIMENSION_VALUES_DETECTION_MAPPER`-style pluggable value detection for filter
  literals (e.g. `region = north` → `N`) — registry of per-dimension normalizers
  (optional; can land in PR-D5).

**Verify (E2E):** `tests/test_dsl_grounding.py`
- `SELECT revenue BY region WHERE region = north SINCE 30 days` grounds
  `revenue`→`sales.amount`, `region`→`sales.region`, `north`→`N`, `30 days`→
  `last_days=30`; produced SQL matches the literal-ref version.
- Synonym override (`net win: sales.net_win`) takes precedence; case/whitespace
  insensitive.
- Unknown token → `UnresolvedNameError` naming the kind (metric/dimension/period).
- Fuzzy fallback (flag on) maps `revenu` → `revenue` within a distance budget;
  fuzzy off → error.

---

## PR-D4 — DSL engine orchestrator + planner integration  ✅ implemented

**Why:** We now have the three pieces (catalog, parser, grounding) but no single
entry point that turns DSL text into results. This is the "plan → execute"
integration that `api-multisite` does via Cypher and `nlp_api` *fails* to do
(its IR is handed to an opaque external service). We reuse the existing
`query_planner.plan` / `plan_query` / `validate_plan` / `JoinTree` so multi-table
fan-out, partial-failure, and DuckDB execution are already solved.

**Implement (new module `dsl/engine.py`):**
- `DslEngine(source, model, catalog=None, synonyms=None, *, best_effort=False)`:
  - `query(dsl_text) -> DslResult`: parse (`parser`) → ground (`grounding`) →
    `to_spec` → `plan_query(model, spec, source, best_effort=...)` →
    `source.native_query(sql)`; returns `{rows, sql, spec, warnings}`.
  - Reuses `validate_plan`/`PlanResult` (PR-5) for scoped missing-table /
    schema-contract errors and best-effort degradation; reuses `JoinTree` for
    fan-out-safe multi-table joins; reuses `dataset_fingerprint` so repeated
    identical DSL against unchanged data is cacheable.
  - `explain(dsl_text) -> str`: return the planned SQL without executing (for
    agents / debugging) — mirrors `query_planner.plan` output.
  - `catalog` auto-seeded from the `SemanticModel` base metrics + any persisted
    calculated metrics; `synonyms` from `NameResolver`.
- A small `ModelsToolset`/`AnalyticsToolset` hook so an agent can call
  `dsl_query(text)` and get a defensible answer (trust grade + provenance) like
  the other tools — optional but consistent with the engine's answer model.

**Verify (E2E):** `tests/test_dsl_engine.py`
- End-to-end over a DuckDB `warehouse_source` and a CSV source:
  `engine.query("SELECT revenue BY region SINCE 30 days ORDER BY revenue DESC LIMIT 5")`
  returns rows whose aggregates equal a hand-computed pandas baseline on the same
  data.
- Calculated metric by name executes and matches the inline-expression result.
- Multi-table query joins via discovered relationships (fan-out-safe, no
  double-count) and matches the baseline.
- `best_effort=True` with a dropped table → `warnings` + partial rows (PR-5).
- `explain` returns valid SQL that, run directly, yields the same rows.

---

## PR-D5 — NL → DSL bridge via entity extraction *(stretch)*  ✅ implemented

**Why:** A textual DSL is only as usable as the caller's ability to write it.
`nlp_api` already extracts `Metric/Dimension/Period/Operation/Denom` entities
(via Dialogflow + local n-gram/edit-distance/pattern matching) and grounds them
with `NamingConventions`. We can reuse that *pattern* locally (no Dialogflow
dependency) to turn a natural-language question into DSL text, then let PR-D4
execute it. This closes the loop: NL → entities → DSL → SQL → answer.

**Implement (new module `dsl/nl.py`):**
- Pluggable **entity detectors** (registry pattern, like `nlp_api`
  `DIMENSION_VALUES_DETECTION_MAPPER`): a local string-matching detector
  (n-grams + edit-distance over catalog/synonym vocabulary) and an optional
  LLM-detector prompt that returns structured `{metrics, dimensions, filters,
  period, operation}`.
- `nl_to_dsl(text, engine) -> str`: extract entities → ground via PR-D3 → emit
  DSL text (reusing PR-D2 `to_text`) → return to caller (who runs PR-D4). Keep
  the LLM out of the execution path; it only proposes the DSL.
- Reuse `nlp_api`'s `patches`-style post-processing hooks for locale/period
  fixups if needed.

**Verify (E2E):** `tests/test_dsl_nl.py` *(gated; live LLM behind a flag)*
- Local detector: `"show revenue by region for the last 30 days"` → DSL text that
  PR-D4 executes and matches the baseline.
- `period` extraction (`last month`, `ytd`, `since 2024-01-01`) maps correctly.
- Optional LLM detector (flag `PAA_RUN_OLLAMA_TESTS=1`): produces DSL that
  executes to the expected aggregation.
- Detectors degrade gracefully (no entity → scoped "could not understand"
  rather than a wrong query).

---

## Usage example

```python
from demos.analytics.src.analytics.dsl import DslEngine, nl_to_dsl

engine = DslEngine(source, model, synonyms={"revenue": "sales.amount"})
result = engine.query("SELECT revenue BY region SINCE 30 days ORDER BY revenue DESC LIMIT 5")
print(result.rows, result.sql, result.warnings)

# Natural-language question (local detector, no external LLM):
dsl = nl_to_dsl("show revenue by region for the last 30 days", engine)
# Or expose it to an agent via the AnalyticsToolset tools `dsl_query` / `nl_query`.
```

## Caveats

- **`BETWEEN` end date is exclusive** (`<`), matching half-open interval
  conventions; use `SINCE` for inclusive trailing windows.
- **Time windows assume UTC** (`current_timestamp AT TIME ZONE 'UTC'`). Data
  stored in another timezone should be converted upstream or the `TimeColumn`
  encoding adjusted.
- **Value normalization** supports both static maps (`north` → `N`) and
  **callable normalizers** (`DIMENSION_VALUES_DETECTION_MAPPER` style), e.g.
  `value_synonyms={"sales.region": {"south": lambda v: v.upper()[:1]}}`.
- **LLM detector** is available via `OllamaEntityDetector` (local Ollama,
  `http://localhost:11434`). Model is `PAA_OLLAMA_MODEL` (default `ornith:latest`;
  `gemma4:31b-cloud` also works); run the gated test with
  `PAA_RUN_OLLAMA_TESTS=1`. The LLM only *proposes* DSL; execution is the engine's.
- **Best-effort caching** is keyed by `dataset_sig` + DSL text; it is disabled
  when `best_effort=True` (a dropped table depends on live source state).

## Explicitly OUT OF SCOPE (handled elsewhere / later)

- **Auth / multi-tenant scoping** of the DSL endpoint — owned by the fronting
  system (same as `production_readiness.md`).
- **Cypher / Neo4j execution** — the engine targets SQL `DataSource`s (DuckDB +
  warehouses). The CQL/Cypher ideas are borrowed for *catalog/DSL shape*, not for
  a graph backend.
- **Full Dialogflow hosting** — PR-D5 uses a local detector by default; Dialogflow
  remains an optional pluggable detector.
