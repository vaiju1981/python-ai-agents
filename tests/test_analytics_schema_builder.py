"""Tests for LLM-assisted analytics schema refinement."""

from __future__ import annotations

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.schema_builder import refine_profile_with_llm
from demos.analytics.src.analytics.semantic_model import SemanticModel
from python_ai_agents import ModelRequest, ModelResponse


class SchemaHintModel:
    async def chat(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse.text_response(
            """
            {
              "columns": [
                {
                  "table": "transactions",
                  "column": "status_code",
                  "role": "dimension",
                  "description": "business status bucket"
                },
                {
                  "table": "transactions",
                  "column": "revenue_cents",
                  "role": "measure_additive",
                  "description": "transaction revenue in cents"
                },
                {
                  "table": "unknown",
                  "column": "made_up",
                  "role": "measure_additive"
                }
              ]
            }
            """
        )


def test_llm_schema_refinement_can_override_csv_roles(tmp_path) -> None:
    csv = tmp_path / "transactions.csv"
    csv.write_text("customer_id,status_code,revenue_cents\n1,A,101\n2,B,205\n3,A,309\n4,C,410\n")
    source = CsvSource(named_csvs={"transactions": csv})
    try:
        profile = profile_dataset(source)
        deterministic = SemanticModel.from_profile(profile)
        assert "transactions.status_code" in deterministic.entity_keys

        async def run():
            refined_profile = await refine_profile_with_llm(profile, SchemaHintModel())
            return SemanticModel.from_profile(refined_profile)

        semantic = anyio.run(run)
        assert "transactions.status_code" not in semantic.entity_keys
        assert any(dim.ref == "transactions.status_code" for dim in semantic.dimensions)
        assert any(metric.ref == "transactions.revenue_cents" for metric in semantic.metrics)
    finally:
        source.close()
