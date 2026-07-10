"""PR-9 verification: auto-apply feedback-loop trust-threshold tuning (§G G2).

Adds an opt-in policy (``ANALYTICS_TRUST_AUTO_TUNE=1``) that applies the
suggestion produced by ``DecisionStore.tune_trust_thresholds`` to the live
``ANALYTICS_TRUST_*`` thresholds, with clamps, a minimum-sample gate, and an
audit entry. The recommend-only path stays the default.
"""

from __future__ import annotations

import json

import pytest

from demos.analytics.src.analytics.audit_store import SqliteAuditStore
from demos.analytics.src.analytics.decision_store import DecisionStore
from demos.analytics.src.analytics.trust import (
    AUTO_TUNE_ENV,
    TRUST_BOUNDS,
    apply_trust_tuning,
    auto_tune_trust_thresholds,
    current_thresholds,
)

pytest.importorskip("anyio")


@pytest.fixture
def thresholds_guard(monkeypatch):
    """Snapshot and restore the live trust thresholds so global mutation in
    tests does not leak into other suites."""
    before = current_thresholds()
    yield
    for name, value in before.items():
        monkeypatch.setattr("demos.analytics.src.analytics.trust." + name, value)


def _suggest_raise(store: DecisionStore) -> dict:
    for _ in range(5):
        store.record_outcome("promo", "accepted", "TRUSTED", False, unit_id="u1")
    return store.tune_trust_thresholds(min_samples=3)


def test_auto_tune_applies_and_audits_when_enabled(tmp_path, thresholds_guard, monkeypatch):
    monkeypatch.setenv(AUTO_TUNE_ENV, "1")
    store = DecisionStore(tmp_path / "decisions.json")
    audit = SqliteAuditStore(tmp_path / "audit.db")
    suggestion = _suggest_raise(store)
    assert suggestion["action"] == "raise"

    before = current_thresholds()
    result = auto_tune_trust_thresholds(suggestion, min_samples=3, enabled=True, audit=audit)

    assert result["applied"] is True
    after = current_thresholds()
    # raise => thresholds get stricter (grow).
    assert after["TRUSTED_N"] > before["TRUSTED_N"]
    assert after["MIN_N"] > before["MIN_N"]
    # An audit entry was written.
    events = [e for e in audit.event_log() if e.event_type == "trust_tuning"]
    assert events, "expected a trust_tuning audit entry"
    detail = json.loads(events[0].detail)
    assert detail["old"] == {k: before[k] for k in before}
    assert detail["new"] == {k: after[k] for k in after}
    assert detail["evidence"]["samples"] >= 3


def test_auto_tune_off_keeps_recommend_only(tmp_path, thresholds_guard, monkeypatch):
    monkeypatch.delenv(AUTO_TUNE_ENV, raising=False)
    store = DecisionStore(tmp_path / "decisions.json")
    suggestion = _suggest_raise(store)
    before = current_thresholds()

    result = auto_tune_trust_thresholds(suggestion, min_samples=3, enabled=False)
    assert result["applied"] is False
    # Thresholds must be untouched when auto-tune is disabled.
    assert current_thresholds() == before


def test_auto_tune_respects_min_sample_gate(tmp_path, thresholds_guard, monkeypatch):
    monkeypatch.setenv(AUTO_TUNE_ENV, "1")
    store = DecisionStore(tmp_path / "decisions.json")
    # Only 2 labeled outcomes -> below a min_samples of 5.
    store.record_outcome("promo", "accepted", "TRUSTED", False, unit_id="u1")
    store.record_outcome("promo", "accepted", "TRUSTED", False, unit_id="u2")
    suggestion = store.tune_trust_thresholds(min_samples=3)
    before = current_thresholds()

    result = auto_tune_trust_thresholds(suggestion, min_samples=5, enabled=True)
    assert result["applied"] is False
    assert current_thresholds() == before


def test_auto_tune_clamps_within_bounds(tmp_path, thresholds_guard, monkeypatch):
    monkeypatch.setenv(AUTO_TUNE_ENV, "1")
    store = DecisionStore(tmp_path / "decisions.json")
    audit = SqliteAuditStore(tmp_path / "audit.db")

    # Pin thresholds at their upper clamp so a "raise" cannot move them.
    for name, (_lo, hi) in TRUST_BOUNDS.items():
        monkeypatch.setattr("demos.analytics.src.analytics.trust." + name, hi)

    store.record_outcome("promo", "accepted", "TRUSTED", False, unit_id="u1")
    suggestion = store.tune_trust_thresholds(min_samples=1)

    result = auto_tune_trust_thresholds(suggestion, min_samples=1, enabled=True, audit=audit)
    # Clamped to the bound: no net change, so not applied.
    assert result["applied"] is False
    after = current_thresholds()
    for name, (lo, hi) in TRUST_BOUNDS.items():
        assert lo <= after[name] <= hi


def test_apply_trust_tuning_is_bounded_under_repeated_raise(thresholds_guard, monkeypatch):
    for name, (lo, _hi) in TRUST_BOUNDS.items():
        monkeypatch.setattr("demos.analytics.src.analytics.trust." + name, lo)
    # Repeated raises must converge at the upper clamp, never exceed it.
    last = None
    for _ in range(50):
        last = apply_trust_tuning("raise")
    after = current_thresholds()
    for name, (lo, hi) in TRUST_BOUNDS.items():
        assert lo <= after[name] <= hi
    assert last is not None
    assert any(n in last["clamped"] for n in TRUST_BOUNDS)
