"""PR-7 verification: cross-answer lineage graph.

A derived answer (a forecast built on a reconciled metric) must be traceable
back through every upstream answer to its raw sources (SQL + dataset_sig).
"""

from __future__ import annotations

import anyio

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.lineage import LineageGraph
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents.core.tool import RequestContext


def _toolset(tmp_path, graph):
    csv = tmp_path / "sales.csv"
    # Six months of data so forecast has >= 4 periods; a date column for the
    # time dimension so the model exposes a TimeColumn for forecast.
    csv.write_text(
        "dt,region,amount\n"
        "2024-01-15,North,100\n2024-01-20,South,90\n"
        "2024-02-15,North,120\n2024-02-20,South,80\n"
        "2024-03-15,North,140\n2024-03-20,South,110\n"
        "2024-04-15,North,130\n2024-04-20,South,95\n"
        "2024-05-15,North,160\n2024-05-20,South,105\n"
        "2024-06-15,North,150\n2024-06-20,South,115\n"
    )
    src = CsvSource(named_csvs={"sales": csv})
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model, lineage=graph)
    models = ModelsToolset(src, model, dataset_sig="sig-x", lineage=graph)
    return src, tools, models


def _answer_id(result):
    return result.provenance["answerId"]


def test_lineage_traces_forecast_back_to_reconcile(tmp_path):
    graph = LineageGraph(tmp_path / "lineage.json")
    src, tools, models = _toolset(tmp_path, graph)

    def run(coro):
        return anyio.run(lambda: coro)

    # Answer A: reconcile.
    res_a = run(
        tools.reconcile().invoke(
            {"metric": "sales.amount", "expected": 0.0}, RequestContext.ephemeral()
        )
    )
    assert not res_a.error, res_a.content
    a_id = _answer_id(res_a)

    # Answer B: forecast consuming A (same conversation scope -> linked parent).
    res_b = run(
        models.forecast().invoke(
            {"metric": "sales.amount", "timeColumn": "sales.dt", "horizon": 3},
            RequestContext.ephemeral(),
        )
    )
    assert not res_b.error, res_b.content
    b_id = _answer_id(res_b)

    # B records A as an upstream parent.
    b_node = graph.node(b_id)
    assert a_id in b_node.parents

    # trace_lineage(B) walks upstream and returns A with its dataset_sig + SQL.
    chain = graph.trace_lineage(b_id)
    ids = [n.answer_id for n in chain]
    assert b_id in ids and a_id in ids
    a_node = graph.node(a_id)
    assert a_node.dataset_sig == b_node.dataset_sig
    assert a_node.sql  # the reconcile SQL is recorded
    assert "sales" in a_node.sql.lower()

    # Persistence: a freshly-loaded graph still has both nodes.
    reloaded = LineageGraph(tmp_path / "lineage.json")
    assert reloaded.node(a_id) is not None
    assert reloaded.node(b_id) is not None
    assert a_id in {n.answer_id for n in reloaded.trace_lineage(b_id)}
    src.close()


def test_lineage_node_missing_returns_empty_trace(tmp_path):
    graph = LineageGraph(tmp_path / "lineage.json")
    assert graph.trace_lineage("does-not-exist") == []
