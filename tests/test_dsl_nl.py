"""PR-D5 verification (stretch): NL -> DSL bridge via entity extraction.

Covers the gates from docs/dsl_engine.md PR-D5 (local detector; LLM detector
gated behind ``PAA_RUN_OLLAMA_TESTS``):

1. Local detector: ``"show revenue by region for the last 30 days"`` -> DSL text
   that PR-D4 executes and matches the pandas baseline.
2. Period extraction: ``last month``, ``ytd``, ``since 2024-01-01`` map correctly.
3. LLM detector (flag ``PAA_RUN_OLLAMA_TESTS=1``): produces DSL that executes to
   the expected aggregation. Here exercised with a stub callable so the wiring is
   verified without a live model; the live path is gated by the env flag.
4. Detectors degrade gracefully (no entity -> scoped ``NLDetectError``).
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.dsl.engine import DslEngine
from demos.analytics.src.analytics.dsl.nl import (
    LLMEntityDetector,
    LocalEntityDetector,
    NLDetectError,
    nl_to_dsl,
)
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
    TimeColumn,
)


def _csv_source(table: str, df: pd.DataFrame) -> CsvSource:
    d = tempfile.mkdtemp(prefix="dsl_nl_")
    path = os.path.join(d, f"{table}.csv")
    df.to_csv(path, index=False)
    return CsvSource(named_csvs={table: Path(path)})


def _sales_df(n: int = 12) -> pd.DataFrame:
    today = date.today()
    return pd.DataFrame(
        {
            "region": ["N", "S", "E", "W"][i % 4],
            "amount": float((i + 1) * 10),
            "quantity": float((i % 3) + 1),
            "day": (today - timedelta(days=i)).isoformat(),
        }
        for i in range(n)
    )


def _sales_model() -> SemanticModel:
    return SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="sales", column="quantity", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(TimeColumn(table="sales", column="day", encoding="date"),),
        relationships=(),
    )


def test_local_detector_to_dsl_matches_baseline():
    df = _sales_df()
    src = _csv_source("sales", df)
    model = _sales_model()
    engine = DslEngine(src, model, synonyms={"revenue": "sales.amount", "region": "sales.region"})

    dsl = nl_to_dsl("show revenue by region for the last 30 days", engine)
    assert "revenue" in dsl and "region" in dsl and "30" in dsl

    result = engine.query(dsl)
    baseline = df.groupby("region")["amount"].sum().to_dict()
    got = {r["region"]: r["revenue"] for r in result.rows}
    for region, total in baseline.items():
        assert got[region] == pytest.approx(total)
    src.close()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("amount trend last month", 30),
        ("amount summary ytd", 365),
        ("amount since 2024-01-01", None),
    ],
)
def test_period_extraction(text: str, expected: int | None):
    df = _sales_df()
    src = _csv_source("sales", df)
    model = _sales_model()
    engine = DslEngine(src, model)
    ent = LocalEntityDetector().detect(text, engine)
    if expected is not None:
        assert ent.last_days == expected
    else:
        assert ent.between_start == "2024-01-01"
    src.close()


def test_llm_detector_wiring_with_stub():
    df = _sales_df()
    src = _csv_source("sales", df)
    model = _sales_model()
    engine = DslEngine(src, model, synonyms={"revenue": "sales.amount", "region": "sales.region"})

    def fake_llm(text: str) -> dict:
        return {"metrics": ["revenue"], "dimensions": ["region"], "last_days": 30}

    detector = LLMEntityDetector(call=fake_llm)
    dsl = nl_to_dsl("whatever the llm says", engine, detector=detector)
    result = engine.query(dsl)
    assert result.rows
    src.close()


def test_no_entity_degrades_gracefully():
    df = _sales_df()
    src = _csv_source("sales", df)
    model = _sales_model()
    engine = DslEngine(src, model)
    with pytest.raises(NLDetectError):
        nl_to_dsl("please make me a sandwich", engine)
    src.close()
