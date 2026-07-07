"""Integration checks against local REDROCK analytics CSV exports.

The source files are intentionally not committed. These tests discover
``STATION_*_REDROCK.csv`` files from a local data directory, create bounded
sample copies, and exercise the same CSV profiling path used by the demo.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.schema_builder import refine_profile_with_llm
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents.adapters import DEFAULT_OLLAMA_TEST_MODELS, OllamaModelPort

DEFAULT_REDROCK_DATA_DIR = Path("/Users/vaijanath.rao/ga_cache/training_data")
DEFAULT_SAMPLE_ROWS = 250


def _redrock_data_dir() -> Path:
    return Path(os.environ.get("PAA_ANALYTICS_REDROCK_DATA", DEFAULT_REDROCK_DATA_DIR))


def _sample_rows() -> int:
    return int(os.environ.get("PAA_ANALYTICS_REDROCK_SAMPLE_ROWS", DEFAULT_SAMPLE_ROWS))


def _redrock_files() -> list[Path]:
    data_dir = _redrock_data_dir()
    if not data_dir.exists():
        pytest.skip(f"REDROCK data directory not found: {data_dir}")
    files = sorted(data_dir.glob("STATION_*_REDROCK.csv"))
    if not files:
        pytest.skip(f"no STATION_*_REDROCK.csv files found in {data_dir}")
    return files


def _live_redrock_models() -> tuple[str, ...]:
    configured = os.environ.get("PAA_ANALYTICS_REDROCK_MODELS")
    if configured:
        return tuple(model.strip() for model in configured.split(",") if model.strip())
    return DEFAULT_OLLAMA_TEST_MODELS


def _sample_csvs(tmp_path: Path, rows_per_file: int | None = None) -> dict[str, Path]:
    rows_per_file = rows_per_file or _sample_rows()
    sampled: dict[str, Path] = {}
    for source in _redrock_files():
        target = tmp_path / source.name
        with source.open(newline="") as inp, target.open("w", newline="") as out:
            reader = csv.reader(inp)
            writer = csv.writer(out)
            header = next(reader)
            writer.writerow(header)
            for idx, row in enumerate(reader):
                if idx >= rows_per_file:
                    break
                writer.writerow(row)
        sampled[source.stem] = target
    return sampled


@pytest.mark.integration
@pytest.mark.redrock
def test_redrock_csv_samples_profile_without_hardcoded_demo_schema(tmp_path) -> None:
    source = CsvSource(named_csvs=_sample_csvs(tmp_path))
    try:
        profile = profile_dataset(source)
        semantic = SemanticModel.from_profile(profile)
        table_names = {table.name for table in profile.tables}
        column_refs = {f"{column.table}.{column.name}" for column in profile.columns}

        assert table_names == {path.stem for path in _redrock_files()}
        assert all(table.rows == _sample_rows() for table in profile.tables)
        assert len(profile.columns) > 100
        assert semantic.metrics
        assert semantic.dimensions
        assert semantic.time_columns
        assert any(ref.endswith(".playerId") for ref in column_refs)
        assert any(ref.endswith(".casino") for ref in column_refs)
        assert not any(ref.startswith("sales.") for ref in column_refs)

        catalog = AnalyticsToolset(source, semantic).catalog_json()
        for table_name in table_names:
            assert table_name in catalog
        assert "sales.amount" not in catalog
    finally:
        source.close()


@pytest.mark.integration
@pytest.mark.redrock
@pytest.mark.ollama
@pytest.mark.skipif(
    os.environ.get("PAA_RUN_OLLAMA_TESTS") != "1"
    or os.environ.get("PAA_RUN_ANALYTICS_REDROCK_LLM") != "1",
    reason=(
        "set PAA_RUN_OLLAMA_TESTS=1 and PAA_RUN_ANALYTICS_REDROCK_LLM=1 "
        "to run live REDROCK schema refinement"
    ),
)
@pytest.mark.parametrize("model_name", _live_redrock_models())
def test_live_ollama_can_refine_redrock_csv_schema_sample(tmp_path, model_name: str) -> None:
    async def run() -> SemanticModel:
        model = OllamaModelPort(model_name, options={"temperature": 0}, timeout=240)
        if not await model.has_model():
            pytest.skip(f"Ollama model is not available: {model_name}")

        source = CsvSource(named_csvs=_sample_csvs(tmp_path, rows_per_file=min(_sample_rows(), 80)))
        try:
            profile = profile_dataset(source)
            refined = await refine_profile_with_llm(profile, model, max_columns=80)
            return SemanticModel.from_profile(refined)
        finally:
            source.close()

    semantic = anyio.run(run)

    assert semantic.metrics
    assert semantic.dimensions
    assert semantic.time_columns
