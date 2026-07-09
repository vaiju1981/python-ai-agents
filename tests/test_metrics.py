"""PR-4 verification: structured metrics are emitted from the answer path.

Uses the in-memory metrics sink (no logging noise, easy assertions). Exercises
the real tool path via ``AnalyticsToolset`` / ``ModelsToolset`` and asserts the
expected counters land with the right tags.
"""

from __future__ import annotations

import pandas as pd
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.metrics import (
    InMemoryMetricsSink,
    get_sink,
    set_sink,
)
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset


@pytest.fixture
def sink():
    s = InMemoryMetricsSink()
    prev = get_sink()
    set_sink(s)
    yield s
    set_sink(prev)


def _csv_source(tmp_path):
    df = pd.DataFrame(
        {
            "region": ["N", "S", "N", "S"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "qty": [1, 2, 3, 4],
        }
    )
    csv = tmp_path / "sales.csv"
    df.to_csv(csv, index=False)
    return CsvSource(named_csvs={"sales": csv})


def _toolset(tmp_path):
    src = _csv_source(tmp_path)
    model = SemanticModel.from_profile(profile_dataset(src))
    return src, AnalyticsToolset(src, model)


def _big_csv_source(tmp_path):
    """Enough rows that summarize grades as TRUSTED (n >= TRUSTED_N, full coverage)."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 250
    df = pd.DataFrame(
        {
            "region": rng.choice(["N", "S"], n),
            "amount": rng.uniform(1.0, 100.0, n),
            "qty": rng.integers(1, 10, n),
        }
    )
    csv = tmp_path / "sales.csv"
    df.to_csv(csv, index=False)
    return CsvSource(named_csvs={"sales": csv})


def _big_toolset(tmp_path):
    src = _big_csv_source(tmp_path)
    model = SemanticModel.from_profile(profile_dataset(src))
    return src, AnalyticsToolset(src, model)


def test_summary_emits_trust_tier_and_latency(sink, tmp_path):
    import anyio

    from python_ai_agents.core.tool import RequestContext

    src, tools = _big_toolset(tmp_path)
    anyio.run(
        lambda: tools.summarize().invoke(
            {"metric": "sales.amount"}, RequestContext(session_id="s")
        )
    )

    # Trust tier recorded (DIRECTIONAL for a large, fully-covered sample with no
    # validation gates passed).
    assert sink.count("analytics.answer.by_trust_tier", tier="DIRECTIONAL") >= 1
    # Per-tool call + latency recorded.
    assert sink.count("analytics.tool.calls", tool="summarize") == 1
    lat = sink.values("analytics.tool.latency_seconds", tool="summarize")
    assert lat and lat[0] >= 0.0
    src.close()


def test_thin_evidence_abstention_recorded(sink, tmp_path):
    """A causal-style tool on thin evidence should record an abstention."""
    import anyio

    from python_ai_agents.core.tool import RequestContext

    df = pd.DataFrame(
        {
            "asset": ["A", "A", "B", "B"],
            "day": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-01", "2024-01-10"]),
            "coin": [10, 20, 30, 40],
        }
    )
    events = pd.DataFrame({"asset": ["A"], "day": pd.to_datetime(["2024-01-05"])})
    csv1 = tmp_path / "metric.csv"
    csv2 = tmp_path / "events.csv"
    df.to_csv(csv1, index=False)
    events.to_csv(csv2, index=False)
    src = CsvSource(named_csvs={"metric": csv1, "events": csv2})
    model = SemanticModel.from_profile(profile_dataset(src))
    tools = AnalyticsToolset(src, model)

    anyio.run(
        lambda: tools.matched_impact().invoke(
            {
                "valueCol": "metric.coin", "entityCol": "asset", "dateCol": "day",
                "treatmentTable": "events", "treatmentKey": "asset",
                "treatmentDateCol": "day",
            },
            RequestContext(session_id="s"),
        )
    )
    # Thin evidence → INSUFFICIENT → abstention counter bumped for that tool.
    assert sink.count("analytics.answer.abstained", tool="matched_impact") >= 1
    src.close()


def test_model_cache_hit_and_miss_metrics(sink, tmp_path):
    from demos.analytics.src.analytics.model_store import (
        FileModelStore,
        ModelRecord,
        model_key,
    )

    directory = tmp_path / "models"
    key = model_key(dataset_sig="s", task="t", target="y", predictors=["x"], algorithm="linear")
    store = FileModelStore(directory=directory)

    # First get → miss; put → write; second get → hit.
    assert store.get(key) is None
    store.put(ModelRecord(key=key, model={"v": 1}, metadata={}, trained_at=0.0))
    assert store.get(key) is not None

    assert sink.count("analytics.model_cache.misses") >= 1
    assert sink.count("analytics.model_cache.hits") == 1
    assert sink.count("analytics.model_cache.writes") == 1
